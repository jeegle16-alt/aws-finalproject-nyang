"""
src/runtime/batch_runner.py
───────────────────────────
일별 배치 실행 엔트리포인트.

S3에서 silver_events를 읽고, FE → Baseline → Model → Decoder를 순서대로 실행한 후
결과를 S3에 저장한다.

CLI:
  python -m src.runtime.batch_runner --dt 2024-01-15 [--config config.yaml]

출력 (S3):
  s3://{ml_bucket}/{outputs}/dt={dt}/state_out.parquet
  s3://{ml_bucket}/{outputs}/dt={dt}/model_out.parquet
  s3://{ml_bucket}/{outputs}/dt={dt}/metrics.json
  s3://{ml_bucket}/{baseline}/dt={dt}/baseline.parquet
  s3://{ml_bucket}/{artifacts_models}/iso_forest.pkl
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Optional

import boto3
import pandas as pd
import yaml

from src.steps.features import build_daily_features
from src.steps.baseline import add_time_context, compute_baseline
from src.steps.model import add_gate_and_context, run_model
from src.steps.decoder import decode

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# S3 I/O 유틸
# ──────────────────────────────────────────────────────────────────────────────

def _s3_client():
    return boto3.client("s3")


def _save_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    _s3_client().put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info(f"Saved parquet → s3://{bucket}/{key}  ({len(df)} rows)")


def _load_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    buf = io.BytesIO()
    _s3_client().download_fileobj(bucket, key, buf)
    buf.seek(0)
    return pd.read_parquet(buf)


def _save_json_to_s3(obj: dict, bucket: str, key: str) -> None:
    body = json.dumps(obj, ensure_ascii=False, default=str)
    _s3_client().put_object(Bucket=bucket, Key=key, Body=body.encode())
    logger.info(f"Saved json → s3://{bucket}/{key}")


def _save_pickle_to_s3(obj, bucket: str, key: str) -> None:
    body = pickle.dumps(obj)
    _s3_client().put_object(Bucket=bucket, Key=key, Body=body)
    logger.info(f"Saved pickle → s3://{bucket}/{key}")


def _load_pickle_from_s3(bucket: str, key: str):
    buf = io.BytesIO()
    _s3_client().download_fileobj(bucket, key, buf)
    buf.seek(0)
    return pickle.loads(buf.read())


def _key_exists(bucket: str, key: str) -> bool:
    try:
        _s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# silver_events 로드 (lookback 60일)
# ──────────────────────────────────────────────────────────────────────────────

def _load_silver_events(
    config: dict,
    dt_local: str,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """
    S3 silver_bucket 에서 dt_local 기준 lookback_days 일치 데이터를 로드.

    파티션 구조:
      s3://{silver_bucket}/silver_events/dt={YYYY-MM-DD}/part-*.parquet
    """
    silver_bucket = config["s3"]["silver_bucket"]
    end_dt   = datetime.strptime(dt_local, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=lookback_days - 1)

    s3 = _s3_client()
    dfs = []
    cur = start_dt
    while cur <= end_dt:
        date_str = cur.strftime("%Y-%m-%d")
        prefix   = f"silver_events/dt={date_str}/"
        try:
            resp = s3.list_objects_v2(Bucket=silver_bucket, Prefix=prefix)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".parquet"):
                    dfs.append(_load_parquet_from_s3(silver_bucket, key))
        except Exception as e:
            logger.warning(f"Skipped {date_str}: {e}")
        cur += timedelta(days=1)

    if not dfs:
        raise FileNotFoundError(
            f"No silver_events found for {dt_local} (lookback={lookback_days}d) "
            f"in s3://{silver_bucket}"
        )
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded silver_events: {len(df)} rows, {df['uuid'].nunique()} users")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 메트릭 집계
# ──────────────────────────────────────────────────────────────────────────────

def _build_metrics(state_df: pd.DataFrame, model_df: pd.DataFrame, dt_local: str) -> dict:
    metrics: dict = {
        "dt": dt_local,
        "n_users": int(state_df["uuid"].nunique()),
        "n_rows": len(state_df),
        "cat_state_counts": state_df["cat_state"].value_counts().to_dict(),
        "notify_level_counts": state_df["notify_level"].value_counts().to_dict(),
        "decoder_quality_counts": state_df["decoder_quality"].value_counts().to_dict(),
    }
    if "final_risk" in model_df.columns:
        risk = model_df["final_risk"].dropna()
        metrics["final_risk_mean"] = float(risk.mean()) if len(risk) else None
        metrics["final_risk_p50"]  = float(risk.quantile(0.50)) if len(risk) else None
        metrics["final_risk_p90"]  = float(risk.quantile(0.90)) if len(risk) else None
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 메인 실행 함수
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(
    dt_local: str,
    config: dict,
    *,
    lookback_days: int = 60,
    retrain_iso: bool = False,
    debug: bool = False,
) -> dict:
    """
    전체 배치 파이프라인 실행.

    Returns
    -------
    dict  metrics 요약
    """
    ml_bucket  = config["s3"]["ml_bucket"]
    out_prefix = config["paths"]["outputs"].rstrip("/")
    bl_prefix  = config["paths"]["baseline"].rstrip("/")
    art_prefix = config["paths"]["artifacts_models"].rstrip("/")
    iso_key    = f"{art_prefix}/iso_forest.pkl"

    # ── 1. silver_events 로드 ──────────────────────────────────────────────
    logger.info(f"[batch] Loading silver_events dt={dt_local}, lookback={lookback_days}d")
    silver = _load_silver_events(config, dt_local, lookback_days)

    # ── 2. Feature Engineering ────────────────────────────────────────────
    logger.info("[batch] FE start")
    feature_df = build_daily_features(silver, config, debug=debug)
    logger.info(f"[batch] FE done: {feature_df.shape}")

    # ── 3. Time context ───────────────────────────────────────────────────
    feature_df = add_time_context(feature_df)

    # ── 4. Gate & Context (baseline_fit_mask 생성) ────────────────────────
    logger.info("[batch] gate & context")
    feature_df = add_gate_and_context(feature_df, config)

    # ── 5. Baseline ───────────────────────────────────────────────────────
    logger.info("[batch] baseline")
    feature_df = compute_baseline(feature_df, config)

    # ── 6. Model (IsolationForest 로드 또는 신규 학습) ────────────────────
    iso_model: Optional[object] = None
    if not retrain_iso and _key_exists(ml_bucket, iso_key):
        logger.info(f"[batch] Loading IsolationForest from s3://{ml_bucket}/{iso_key}")
        try:
            iso_model = _load_pickle_from_s3(ml_bucket, iso_key)
        except Exception as e:
            logger.warning(f"Failed to load iso model, will retrain: {e}")
            iso_model = None

    logger.info("[batch] model")
    model_df, iso_model_new = run_model(feature_df, config, iso_model=iso_model, debug=debug)

    # IsolationForest 저장 (신규 학습된 경우)
    if iso_model is None and iso_model_new is not None:
        _save_pickle_to_s3(iso_model_new, ml_bucket, iso_key)

    # ── 7. Decoder ────────────────────────────────────────────────────────
    logger.info("[batch] decoder")
    state_df = decode(model_df, config, debug=debug)

    # ── 8. 결과 저장 ──────────────────────────────────────────────────────
    dt_tag = f"dt={dt_local}"

    state_key   = f"{out_prefix}/{dt_tag}/state_out.parquet"
    model_key   = f"{out_prefix}/{dt_tag}/model_out.parquet"
    bl_key      = f"{bl_prefix}/{dt_tag}/baseline.parquet"
    metrics_key = f"{out_prefix}/{dt_tag}/metrics.json"

    # state 출력 (cat_state / notify_level / decoder_quality 포함 전체)
    _save_parquet_to_s3(state_df, ml_bucket, state_key)

    # model 중간 결과
    _save_parquet_to_s3(model_df, ml_bucket, model_key)

    # baseline snapshot (유저별 최신 baseline 파라미터)
    _save_parquet_to_s3(feature_df, ml_bucket, bl_key)

    # metrics
    metrics = _build_metrics(state_df, model_df, dt_local)
    _save_json_to_s3(metrics, ml_bucket, metrics_key)

    logger.info(f"[batch] Done. n_users={metrics['n_users']}, n_rows={metrics['n_rows']}")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily batch runner")
    parser.add_argument("--dt",        required=True, help="Target date YYYY-MM-DD")
    parser.add_argument("--config",    default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--lookback",  type=int, default=60, help="Lookback days for silver_events")
    parser.add_argument("--retrain",   action="store_true", help="Force IsolationForest retraining")
    parser.add_argument("--debug",     action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    metrics = run_batch(
        dt_local=args.dt,
        config=config,
        lookback_days=args.lookback,
        retrain_iso=args.retrain,
        debug=args.debug,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
