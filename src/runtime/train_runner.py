"""
src/runtime/train_runner.py
────────────────────────────
주간 IsolationForest 재학습 실행기.

최근 N일치 feature_df (모델 중간 결과 model_out.parquet에서 복원하거나
S3 baseline 파티션에서 로드)를 사용해 IsolationForest를 학습하고
S3 artifacts 경로에 버전 태그와 함께 저장한다.

저장 경로:
  s3://{ml_bucket}/{artifacts_models}/model_id=iforest/v={TIMESTAMP}/model.pkl
  s3://{ml_bucket}/{artifacts_models}/_latest.json   ← 최신 버전 포인터

CLI:
  python -m src.runtime.train_runner --end-dt 2024-01-15 [--train-days 90] [--config config.yaml]
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import List

import boto3
import pandas as pd
import yaml

from src.steps.baseline import add_time_context, compute_baseline, BASE_COLS_CANDIDATES
from src.steps.model import (
    add_gate_and_context,
    compute_z_scores,
    train_isolation_forest,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# S3 I/O 유틸
# ──────────────────────────────────────────────────────────────────────────────

def _s3_client():
    return boto3.client("s3")


def _load_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    buf = io.BytesIO()
    _s3_client().download_fileobj(bucket, key, buf)
    buf.seek(0)
    return pd.read_parquet(buf)


def _save_pickle_to_s3(obj, bucket: str, key: str) -> None:
    body = pickle.dumps(obj)
    _s3_client().put_object(Bucket=bucket, Key=key, Body=body)
    logger.info(f"Saved pickle → s3://{bucket}/{key}")


def _save_json_to_s3(obj: dict, bucket: str, key: str) -> None:
    body = json.dumps(obj, ensure_ascii=False, default=str)
    _s3_client().put_object(Bucket=bucket, Key=key, Body=body.encode())
    logger.info(f"Saved json → s3://{bucket}/{key}")


def _list_partitions(bucket: str, prefix: str, start_dt: datetime, end_dt: datetime) -> List[str]:
    """dt=YYYY-MM-DD 파티션 키 목록 반환."""
    s3  = _s3_client()
    keys = []
    cur  = start_dt
    while cur <= end_dt:
        date_str = cur.strftime("%Y-%m-%d")
        part_prefix = f"{prefix}/dt={date_str}/"
        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=part_prefix)
            for obj in resp.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    keys.append(obj["Key"])
        except Exception as e:
            logger.debug(f"Skipped {date_str}: {e}")
        cur += timedelta(days=1)
    return keys


# ──────────────────────────────────────────────────────────────────────────────
# 학습 데이터 로드
# ──────────────────────────────────────────────────────────────────────────────

def _load_training_data(
    config: dict,
    end_dt: str,
    train_days: int,
) -> pd.DataFrame:
    """
    S3 baseline 파티션에서 train_days 일치 feature_df를 로드.

    baseline 파티션에는 compute_baseline 결과 전체가 저장되어 있다고 가정.
    없는 날짜는 무시하고 있는 것만 합쳐서 반환.
    """
    ml_bucket  = config["s3"]["ml_bucket"]
    bl_prefix  = config["paths"]["baseline"].rstrip("/")

    end   = datetime.strptime(end_dt, "%Y-%m-%d")
    start = end - timedelta(days=train_days - 1)

    keys = _list_partitions(ml_bucket, bl_prefix, start, end)
    if not keys:
        raise FileNotFoundError(
            f"No baseline parquet found for range {start.date()}~{end.date()} "
            f"in s3://{ml_bucket}/{bl_prefix}"
        )

    dfs = []
    for key in keys:
        try:
            dfs.append(_load_parquet_from_s3(ml_bucket, key))
        except Exception as e:
            logger.warning(f"Failed to load {key}: {e}")

    if not dfs:
        raise FileNotFoundError("All parquet loads failed.")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded training data: {len(df)} rows, {df['uuid'].nunique()} users, {len(keys)} partitions")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 메인 학습 함수
# ──────────────────────────────────────────────────────────────────────────────

def run_train(
    end_dt: str,
    config: dict,
    *,
    train_days: int = 90,
    debug: bool = False,
) -> str:
    """
    IsolationForest 학습 후 S3에 저장.

    Returns
    -------
    str  저장된 model S3 key
    """
    ml_bucket  = config["s3"]["ml_bucket"]
    art_prefix = config["paths"]["artifacts_models"].rstrip("/")
    version    = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_key  = f"{art_prefix}/model_id=iforest/v={version}/model.pkl"
    latest_key = f"{art_prefix}/_latest.json"

    # ── 1. 학습 데이터 로드 ────────────────────────────────────────────────
    logger.info(f"[train] Loading data end_dt={end_dt}, train_days={train_days}")
    df = _load_training_data(config, end_dt, train_days)

    # ── 2. 필요 컬럼 확인 후 gate & context ──────────────────────────────
    if "is_weekend" not in df.columns or "cold_stage" not in df.columns:
        df = add_time_context(df)

    if "baseline_fit_mask" not in df.columns:
        df = add_gate_and_context(df, config)

    # ── 3. Z-score 계산 ───────────────────────────────────────────────────
    logger.info("[train] Computing Z-scores")
    df, z_use_cols = compute_z_scores(df, config)

    if not z_use_cols:
        raise ValueError("[train] No Z-score columns available for training.")

    logger.info(f"[train] Z-cols ({len(z_use_cols)}): {z_use_cols[:10]} ...")

    # ── 4. IsolationForest 학습 ────────────────────────────────────────────
    logger.info("[train] Training IsolationForest")
    iso = train_isolation_forest(df, z_use_cols, config)

    # ── 5. 저장 ───────────────────────────────────────────────────────────
    _save_pickle_to_s3(iso, ml_bucket, model_key)

    latest_meta = {
        "version": version,
        "model_key": model_key,
        "end_dt": end_dt,
        "train_days": train_days,
        "z_cols": z_use_cols,
        "n_rows": len(df),
        "n_users": int(df["uuid"].nunique()),
    }
    _save_json_to_s3(latest_meta, ml_bucket, latest_key)

    logger.info(f"[train] Done. model saved → s3://{ml_bucket}/{model_key}")
    return model_key


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly IsolationForest training runner")
    parser.add_argument("--end-dt",     required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--train-days", type=int, default=90, help="Training window in days")
    parser.add_argument("--config",     default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--debug",      action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_key = run_train(
        end_dt=args.end_dt,
        config=config,
        train_days=args.train_days,
        debug=args.debug,
    )
    print(json.dumps({"model_key": model_key}, indent=2))


if __name__ == "__main__":
    main()
