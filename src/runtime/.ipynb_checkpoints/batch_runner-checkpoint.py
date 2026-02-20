"""
src/runtime/batch_runner.py
============================
EKS CronJob 엔트리포인트 (매일 02:00 KST).

[실행 흐름]
1. 오늘 silver raw (1일치) → FE → daily-feature S3 저장 (누적)
2. 최근 56일 daily-feature S3 로드
3. inference.run(feat_df=56일치) → state_out + model_out
4. model_out에서 오늘 baseline 추출 → baseline S3 저장
5. state_out S3 저장
6. RDS upsert (stub — ETL/RDS 구현 후 채울 것)

[TODO - ETL/RDS 구현 후]
- _upsert_rds() 함수 구현
- DB 스키마: uuid, date, cat_state, notify_level, updated_at
"""

import argparse
import io
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.runtime import inference
from src.steps import features

# ---------------------------------------------------------------------------
# 0. KST 기준 오늘 날짜
# ---------------------------------------------------------------------------

KST = timezone(timedelta(hours=9))
LT_WINDOW = 31  # LT(30) + shift(1) 여유 1일


def today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# 1. S3 헬퍼
# ---------------------------------------------------------------------------

def _s3():
    return boto3.client("s3")


def _parse_s3_uri(s3_uri: str):
    parts  = s3_uri.rstrip("/").replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def _read_parquet(bucket: str, key: str) -> pd.DataFrame:
    obj = _s3().get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _write_parquet(df: pd.DataFrame, bucket: str, key: str):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


# ---------------------------------------------------------------------------
# 2. 오늘 silver raw 로드 (1일치)
# ---------------------------------------------------------------------------

def load_today_silver(s3_silver_uri: str, target_date: str) -> pd.DataFrame:
    """오늘 날짜 파티션 1개만 로드."""
    bucket, prefix = _parse_s3_uri(s3_silver_uri)
    s3 = _s3()

    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet") and f"dt={target_date}" in key:
                keys.append(key)

    if not keys:
        raise FileNotFoundError(f"[BATCH] silver 없음: dt={target_date}")

    frames = []
    for key in sorted(keys):
        try:
            frames.append(_read_parquet(bucket, key))
        except Exception as e:
            print(f"  ⚠️ {key}: {e}")

    if not frames:
        raise RuntimeError(f"[BATCH] silver 읽기 실패: dt={target_date}")

    df = pd.concat(frames, ignore_index=True)
    print(f"[BATCH] silver 로드: {df.shape}, uuid: {df['uuid'].nunique()}명 (dt={target_date})")
    return df


# ---------------------------------------------------------------------------
# 3. daily-feature S3 저장 / 로드
# ---------------------------------------------------------------------------

def save_daily_feature(df: pd.DataFrame, s3_uri: str, target_date: str):
    """
    오늘 daily-feature를 S3에 저장 (누적).
    경로: {s3_uri}/dt={target_date}/daily-feature.parquet
    """
    bucket, prefix = _parse_s3_uri(s3_uri)
    key = f"{prefix}/dt={target_date}/daily-feature.parquet".lstrip("/")
    _write_parquet(df, bucket, key)
    print(f"[BATCH] daily-feature 저장: s3://{bucket}/{key} ({len(df)}행)")


def load_daily_features(s3_uri: str, target_date: str, lookback_days: int = LT_WINDOW) -> pd.DataFrame:
    """
    target_date 기준 최근 lookback_days일치 daily-feature 로드.
    없는 날짜는 스킵 (초기 서비스 기간 대응).
    """
    bucket, prefix = _parse_s3_uri(s3_uri)
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")

    frames = []
    missing_cnt = 0
    for i in range(lookback_days):
        dt = (target_dt - timedelta(days=lookback_days - 1 - i)).strftime("%Y-%m-%d")
        key = f"{prefix}/dt={dt}/daily-feature.parquet".lstrip("/")
        try:
            frames.append(_read_parquet(bucket, key))
        except Exception:
            missing_cnt += 1

    if missing_cnt:
        print(f"[BATCH] daily-feature 없는 날짜 {missing_cnt}개 스킵 (데이터 누적 중)")

    if not frames:
        raise FileNotFoundError(f"[BATCH] daily-feature 없음: {s3_uri}")

    df = pd.concat(frames, ignore_index=True)
    print(f"[BATCH] daily-feature 로드: {df.shape}, uuid: {df['uuid'].nunique()}명 ({lookback_days}일치)")
    return df


# ---------------------------------------------------------------------------
# 4. baseline S3 저장
# ---------------------------------------------------------------------------

def save_baseline(model_df: pd.DataFrame, s3_uri: str, target_date: str):
    """
    model_out에서 오늘 날짜 baseline 컬럼 추출 → S3 저장.
    경로: {s3_uri}/dt={target_date}/baseline.parquet
    """
    today_df = model_df[model_df["date"] == target_date].copy()
    if today_df.empty:
        print(f"[BATCH] ⚠️ baseline 저장 스킵: {target_date} 행 없음")
        return

    meta_cols = ["uuid", "date", "baseline_ready", "early_ready", "cold_stage"]
    mean_std_cols = [c for c in model_df.columns if
                     any(c.endswith(s) for s in [
                         "_mean_lt_g",    "_std_lt_g",
                         "_mean_st_g",    "_std_st_g",
                         "_mean_st_s",    "_std_st_s",
                         "_mean_early_g", "_std_early_g",
                         "_mean_early_s", "_std_early_s",
                         "_mean_final",   "_std_final",
                     ])]

    save_cols = [c for c in meta_cols + mean_std_cols if c in today_df.columns]

    bucket, prefix = _parse_s3_uri(s3_uri)
    key = f"{prefix}/dt={target_date}/baseline.parquet".lstrip("/")
    _write_parquet(today_df[save_cols], bucket, key)
    print(f"[BATCH] baseline 저장: s3://{bucket}/{key} ({len(today_df)}행, {len(save_cols)}컬럼)")


# ---------------------------------------------------------------------------
# 5. 결과 S3 저장
# ---------------------------------------------------------------------------

def save_output_to_s3(state_df: pd.DataFrame, s3_output_uri: str, target_date: str):
    """
    오늘 날짜 결과만 필터해서 저장.
    경로: {s3_output_uri}/dt={target_date}/state_out.csv
    """
    today_df = state_df[state_df["date"] == target_date].copy()
    if today_df.empty:
        print(f"[BATCH] ⚠️ {target_date} 결과 없음 — 저장 스킵")
        return

    csv_buf = io.StringIO()
    today_df.to_csv(csv_buf, index=False)

    bucket, prefix = _parse_s3_uri(s3_output_uri)
    key = f"{prefix}/dt={target_date}/state_out.csv".lstrip("/")

    _s3().put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    print(f"[BATCH] state_out 저장: s3://{bucket}/{key} ({len(today_df)}행)")

# ---------------------------------------------------------------------------
# 6. 메인
# ---------------------------------------------------------------------------

def main(args):
    target_date = args.target_date or today_kst()

    print("=" * 60)
    print(f"[BATCH] 실행: {target_date}")
    print(f"  silver       : {args.s3_silver_uri}")
    print(f"  daily-feature: {args.s3_daily_feature_uri}")
    print(f"  baseline     : {args.s3_baseline_uri}")
    print(f"  model        : {args.model_uri}")
    print(f"  output       : {args.s3_output_uri}")
    print("=" * 60)

    # 1. 오늘 silver raw 1일치 → FE → daily-feature 저장
    raw_df      = load_today_silver(args.s3_silver_uri, target_date)
    feat_today  = features.run(raw_df)
    save_daily_feature(feat_today, args.s3_daily_feature_uri, target_date)

    # 2. 최근 56일 daily-feature 로드
    feat_df = load_daily_features(args.s3_daily_feature_uri, target_date, LT_WINDOW)

    # 3. 추론 (FE 스킵 — feat_df 직접 전달)
    state_df, model_df = inference.run(feat_df=feat_df, model_uri=args.model_uri)

    # 4. baseline 저장
    save_baseline(model_df, args.s3_baseline_uri, target_date)

    # 5. state_out 저장
    save_output_to_s3(state_df, args.s3_output_uri, target_date)

    print(f"[BATCH] 완료 ✅  ({target_date})")


# ---------------------------------------------------------------------------
# 8. 진입점
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="NYAN-MATE 일별 추론 배치")

    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help="추론 기준 날짜 YYYY-MM-DD (기본: 오늘 KST)",
    )
    parser.add_argument(
        "--s3-silver-uri",
        type=str,
        default=os.environ.get("S3_SILVER_URI", "s3://silver-dummy/silver_events/"),
        help="silver parquet S3 URI",
    )
    parser.add_argument(
        "--s3-daily-feature-uri",
        type=str,
        default=os.environ.get(
            "S3_DAILY_FEATURE_URI",
            "s3://nyang-ml-apne2-dev/ml/daily-feature/",
        ),
        help="daily-feature 누적 저장 S3 URI",
    )
    parser.add_argument(
        "--s3-baseline-uri",
        type=str,
        default=os.environ.get(
            "S3_BASELINE_URI",
            "s3://nyang-ml-apne2-dev/ml/baseline/",
        ),
        help="baseline 저장 S3 URI",
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default=os.environ.get(
            "MODEL_URI",
            "s3://nyang-ml-apne2-dev/ml/artifacts/models/isolation_forest.pkl",
        ),
        help="iso.pkl S3 URI 또는 로컬 경로",
    )
    parser.add_argument(
        "--s3-output-uri",
        type=str,
        default=os.environ.get("S3_OUTPUT_URI", "s3://nyang-ml-apne2-dev/ml/outputs/"),
        help="결과 저장 S3 URI",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())