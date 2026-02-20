"""
src/runtime/train.py
====================
SageMaker Training Job 엔트리포인트.

[실행 흐름]
1. S3에서 공개데이터셋 feature parquet 로드  (scripts/build_public_features.py로 미리 생성)
2. S3에서 사용자 ETL parquet 로드 → FE 실행  (있으면 합침, 없으면 공개만)
3. feature DataFrame 합치기
4. model.run(mode="train") → IsolationForest 학습
5. iso.pkl → /opt/ml/model/ 저장

[공개데이터셋 전처리]
  raw CSV 342개를 직접 읽지 않음 → OOM 방지
  로컬에서 scripts/build_public_features.py 실행 후
  daily_feature_public.parquet 1개를 S3에 올려두면 됨:
    aws s3 cp ./artifacts/daily_feature_public.parquet \
      s3://nyang-ml-apne2-dev/ml/inputs/public-dataset/daily_feature_public.parquet

[재학습 전략]
  초기:  공개 feature parquet만 사용
  이후:  공개 feature + 사용자 ETL feature (최근 30일) 합쳐서 재학습
"""

import argparse
import io
import os
import pickle
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.steps import features, model

# ---------------------------------------------------------------------------
# 0. 상수
# ---------------------------------------------------------------------------

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


def _list_keys(s3_uri: str, suffix: str = "") -> list:
    s3     = _s3()
    bucket, prefix = _parse_s3_uri(s3_uri)
    keys   = []
    pager  = s3.get_paginator("list_objects_v2")
    for page in pager.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(suffix):
                keys.append(obj["Key"])
    return keys


# ---------------------------------------------------------------------------
# 2. 공개데이터셋 feature parquet 로드  (parquet 1개 → 가볍게 읽기)
# ---------------------------------------------------------------------------

def load_public_features(s3_public_feature_uri: str) -> pd.DataFrame:
    """
    scripts/build_public_features.py로 미리 만들어둔
    daily_feature_public.parquet을 S3에서 읽기.

    s3_public_feature_uri 예:
      s3://nyang-ml-apne2-dev/ml/inputs/public-dataset/daily_feature_public.parquet
    """
    bucket, key = _parse_s3_uri(s3_public_feature_uri)

    # URI가 디렉토리면 parquet 파일 자동 탐색
    if not key.endswith(".parquet"):
        keys = _list_keys(s3_public_feature_uri, suffix=".parquet")
        if not keys:
            raise FileNotFoundError(f"[TRAIN] 공개 feature parquet 없음: {s3_public_feature_uri}")
        key = keys[0]

    df = _read_parquet(bucket, key)
    print(f"[TRAIN] 공개 feature 로드: {df.shape}, uuid: {df['uuid'].nunique()}명")
    return df


# ---------------------------------------------------------------------------
# 3. 사용자 ETL parquet 로드 → FE 실행
# ---------------------------------------------------------------------------

def load_user_features(s3_etl_uri: str, lookback_days: int = 30) -> Optional[pd.DataFrame]:
    """
    S3 silver parquet에서 최근 lookback_days일치 로드 후 FE 실행.
    파일 없으면 None 반환.
    """
    bucket, _ = _parse_s3_uri(s3_etl_uri)
    all_keys  = _list_keys(s3_etl_uri, suffix=".parquet")

    if not all_keys:
        print(f"[TRAIN] 사용자 ETL 없음 → 공개 feature만 사용")
        return None

    from datetime import date as _date
    cutoff  = _date.today() - timedelta(days=lookback_days)
    filtered = []
    for key in all_keys:
        m = re.search(r"dt=(\d{4}-\d{2}-\d{2})", key)
        if m and _date.fromisoformat(m.group(1)) >= cutoff:
            filtered.append(key)

    if not filtered:
        print(f"[TRAIN] ETL 있지만 최근 {lookback_days}일 내 파티션 없음 → 공개만 사용")
        return None

    print(f"[TRAIN] 사용자 ETL {len(filtered)}개 파티션 로드 중...")
    frames = []
    for key in sorted(filtered):
        try:
            frames.append(_read_parquet(bucket, key))
        except Exception as e:
            print(f"  ⚠️ {key}: {e}")

    if not frames:
        return None

    raw_df = pd.concat(frames, ignore_index=True)
    print(f"[TRAIN] ETL raw: {raw_df.shape}, uuid: {raw_df['uuid'].nunique()}명 → FE 실행 중...")

    feat_df = features.run(raw_df)
    print(f"[TRAIN] ETL feature: {feat_df.shape}")
    return feat_df


# ---------------------------------------------------------------------------
# 4. feature 합치기
# ---------------------------------------------------------------------------

def build_train_features(
    public_feat: pd.DataFrame,
    user_feat: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if user_feat is None:
        print("[TRAIN] 학습 데이터: 공개 feature만")
        return public_feat

    combined = pd.concat([public_feat, user_feat], ignore_index=True)
    print(
        f"[TRAIN] 학습 데이터: 공개 {public_feat['uuid'].nunique()}명 "
        f"+ 사용자 {user_feat['uuid'].nunique()}명 "
        f"= {combined['uuid'].nunique()}명"
    )
    return combined


# ---------------------------------------------------------------------------
# 5. 모델 저장
# ---------------------------------------------------------------------------

def save_model(iso, s3_model_uri: str):
    """메모리에서 바로 S3로 업로드."""
    import io
    if not s3_model_uri.endswith(".pkl"):
        s3_model_uri = s3_model_uri.rstrip("/") + "/isolation_forest.pkl"

    buf = io.BytesIO()
    pickle.dump(iso, buf)
    buf.seek(0)

    parts = s3_model_uri.replace("s3://", "").split("/", 1)
    _s3().put_object(Bucket=parts[0], Key=parts[1], Body=buf.getvalue())
    print(f"[TRAIN] S3 저장 완료: {s3_model_uri}")
    return s3_model_uri


# ---------------------------------------------------------------------------
# 6. 메인
# ---------------------------------------------------------------------------

def main(args):
    print("=" * 60)
    print("[TRAIN] 학습 시작")
    print(f"  공개 feature : {args.s3_public_feature_uri}")
    print(f"  사용자 ETL   : {args.s3_etl_uri or '없음 (초기 학습)'}")
    print(f"  lookback     : {args.lookback_days}일")
    print("=" * 60)

    # 1. 공개 feature 로드 (parquet 1개 - 가벼움)
    public_feat = load_public_features(args.s3_public_feature_uri)

    # 2. 사용자 ETL 로드 + FE (있으면)
    user_feat = load_user_features(args.s3_etl_uri, args.lookback_days) if args.s3_etl_uri else None

    # 3. 합치기
    feat_df = build_train_features(public_feat, user_feat)

    # 4. IF 학습
    print("[TRAIN] IsolationForest 학습 중...")
    _, iso = model.run(feat_df, mode="train")

    if iso is None:
        raise RuntimeError("[TRAIN] IsolationForest 학습 실패")

    # 5. S3 저장
    save_model(iso, args.s3_model_uri)
    print("[TRAIN] 완료 ✅")


# ---------------------------------------------------------------------------
# 7. 진입점
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="NYAN-MATE IsolationForest 학습")
    parser.add_argument(
        "--s3-public-feature-uri",
        type=str,
        default=os.environ.get(
            "S3_PUBLIC_FEATURE_URI",
            "s3://nyang-ml-apne2-dev/ml/inputs/public-dataset/daily_feature_public.parquet",
        ),
        help="공개데이터셋 feature parquet S3 URI (build_public_features.py로 생성)",
    )
    parser.add_argument(
        "--s3-etl-uri",
        type=str,
        default=os.environ.get("S3_ETL_URI", None),
        help="사용자 ETL silver parquet S3 URI (없으면 공개만)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=int(os.environ.get("LOOKBACK_DAYS", "30")),
        help="ETL lookback 일수 (default: 30)",
    )
    parser.add_argument(
        "--s3-model-uri",
        type=str,
        default=os.environ.get(
            "S3_MODEL_URI",
            "s3://nyang-ml-apne2-dev/ml/artifacts/models/isolation_forest.pkl",
        ),
        help="모델 S3 저장 경로",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())