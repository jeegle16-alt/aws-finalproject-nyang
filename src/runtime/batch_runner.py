"""
src/runtime/batch_runner.py
============================
EKS CronJob 엔트리포인트 (매일 02:00 KST).

[실행 흐름]
1. S3 silver parquet 읽기 (오늘 기준 lookback_days일치)
2. inference.run() → state_out DataFrame
3. 오늘 날짜 결과만 필터 → S3 저장
4. RDS upsert (stub — ETL/RDS 구현 후 채울 것)

[TODO - ETL/RDS 구현 후]
- _upsert_rds() 함수 구현
- DB 스키마: uuid, date, cat_state, notify_level, updated_at
"""

import argparse
import io
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.runtime import inference

# ---------------------------------------------------------------------------
# 0. KST 기준 오늘 날짜
# ---------------------------------------------------------------------------

KST = timezone(timedelta(hours=9))


def today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# 1. S3 silver parquet 읽기
# ---------------------------------------------------------------------------

def _s3_client():
    return boto3.client("s3")


def _parse_s3_uri(s3_uri: str):
    parts  = s3_uri.rstrip("/").replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def load_silver_parquet(s3_silver_uri: str, lookback_days: int, target_date: str) -> pd.DataFrame:
    """
    target_date 기준 lookback_days일치 silver parquet 로드.
    dt=YYYY-MM-DD 파티션 구조 가정.

    rolling baseline 계산을 위해 오늘만이 아닌 과거 데이터 전부 포함.
    """
    bucket, prefix = _parse_s3_uri(s3_silver_uri)
    s3 = _s3_client()

    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    cutoff_dt = target_dt - timedelta(days=lookback_days)

    # 파티션 목록
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            m = re.search(r"dt=(\d{4}-\d{2}-\d{2})", key)
            if m:
                dt = datetime.strptime(m.group(1), "%Y-%m-%d")
                if cutoff_dt <= dt <= target_dt:
                    keys.append(key)

    if not keys:
        raise FileNotFoundError(
            f"[BATCH] silver parquet 없음: {s3_silver_uri} "
            f"({cutoff_dt.date()} ~ {target_dt.date()})"
        )

    print(f"[BATCH] {len(keys)}개 파티션 로드 ({cutoff_dt.date()} ~ {target_dt.date()})")
    frames = []
    for key in sorted(keys):
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            frames.append(pd.read_parquet(io.BytesIO(obj["Body"].read())))
        except Exception as e:
            print(f"  ⚠️ {key}: {e}")

    if not frames:
        raise RuntimeError("[BATCH] 파티션 읽기 전부 실패")

    df = pd.concat(frames, ignore_index=True)
    print(f"[BATCH] silver 로드 완료: {df.shape}, uuid: {df['uuid'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# 2. 결과 S3 저장
# ---------------------------------------------------------------------------

def save_output_to_s3(state_df: pd.DataFrame, s3_output_uri: str, target_date: str):
    """
    오늘 날짜 결과만 필터해서 저장.
    경로: {s3_output_uri}/dt={target_date}/state_out_v3.csv
    """
    # 오늘 날짜만 필터
    today_df = state_df[state_df["date"] == target_date].copy()
    if today_df.empty:
        print(f"[BATCH] ⚠️ {target_date} 결과 없음 — 저장 스킵")
        return

    csv_buf = io.StringIO()
    today_df.to_csv(csv_buf, index=False)

    bucket, prefix = _parse_s3_uri(s3_output_uri)
    key = f"{prefix}/dt={target_date}/state_out_v3.csv".lstrip("/")

    s3 = _s3_client()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    print(f"[BATCH] S3 저장 완료: s3://{bucket}/{key} ({len(today_df)}행)")


# ---------------------------------------------------------------------------
# 3. RDS upsert (stub)
#    TODO: ETL/RDS 구현 후 채울 것
# ---------------------------------------------------------------------------

def upsert_rds(state_df: pd.DataFrame, target_date: str):
    """
    TODO: RDS에 cat_state, notify_level upsert.

    예상 스키마:
        CREATE TABLE nyan_state (
            uuid        VARCHAR(64),
            date        DATE,
            cat_state   VARCHAR(16),
            notify_level VARCHAR(8),
            updated_at  TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (uuid, date)
        );

    구현 시 psycopg2 또는 SQLAlchemy 사용 예정.
    DB 접속 정보는 K8s Secret → 환경변수로 주입.
    """
    today_df = state_df[state_df["date"] == target_date]
    print(f"[BATCH] RDS upsert stub: {len(today_df)}행 (미구현 — 스킵)")
    # TODO:
    # import psycopg2
    # conn = psycopg2.connect(os.environ["DATABASE_URL"])
    # ...


# ---------------------------------------------------------------------------
# 4. 메인
# ---------------------------------------------------------------------------

def main(args):
    target_date = args.target_date or today_kst()

    print("=" * 60)
    print(f"[BATCH] 실행: {target_date}")
    print(f"  silver : {args.s3_silver_uri}")
    print(f"  model  : {args.model_uri}")
    print(f"  output : {args.s3_output_uri}")
    print(f"  lookback: {args.lookback_days}일")
    print("=" * 60)

    # 1. silver 로드 (rolling window 전체 기간)
    raw_df = load_silver_parquet(args.s3_silver_uri, args.lookback_days, target_date)

    # 2. 추론
    state_df = inference.run(raw_df=raw_df, model_uri=args.model_uri)

    # 3. S3 저장
    save_output_to_s3(state_df, args.s3_output_uri, target_date)

    # 4. RDS upsert (stub)
    upsert_rds(state_df, target_date)

    print(f"[BATCH] 완료 ✅  ({target_date})")


# ---------------------------------------------------------------------------
# 5. 진입점
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
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=int(os.environ.get("LOOKBACK_DAYS", "57")),
        help="rolling baseline 계산용 과거 데이터 일수 (LT_WINDOW+1=57)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
