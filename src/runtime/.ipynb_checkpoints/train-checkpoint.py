"""
src/runtime/train.py
====================
SageMaker Training Job 엔트리포인트.

[실행 흐름]
1. S3에서 공개데이터셋 feature parquet 로드
2. S3에서 사용자 ETL parquet 로드 → FE 실행
3. feature DataFrame 합치기
4. model.run(mode="train") → IsolationForest 학습
5. 모델을 S3 prefix 아래에 timestamp 버저닝으로 저장
"""

import argparse
import io
import os
import pickle
import re
import sys
import mlflow
import sklearn  # noqa: F401
from datetime import datetime, timedelta, timezone, date as _date
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.steps import features, model  # noqa: E402

# ---------------------------------------------------------------------------
# 0. 상수
# ---------------------------------------------------------------------------
KST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# 1. S3 헬퍼
# ---------------------------------------------------------------------------

def _s3():
    return boto3.client("s3")


def _parse_s3_uri(s3_uri: str):
    parts = s3_uri.rstrip("/").replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def _read_parquet(bucket: str, key: str) -> pd.DataFrame:
    obj = _s3().get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _list_keys(s3_uri: str, suffix: str = "") -> list:
    s3 = _s3()
    bucket, prefix = _parse_s3_uri(s3_uri)
    keys = []
    pager = s3.get_paginator("list_objects_v2")
    for page in pager.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(suffix):
                keys.append(obj["Key"])
    return keys


# ---------------------------------------------------------------------------
# 2. 공개데이터셋 feature parquet 로드
# ---------------------------------------------------------------------------

def load_public_features(s3_public_feature_uri: str) -> pd.DataFrame:
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

def load_user_features(
    s3_etl_uri: str,
    lookback_days: int = 30,
    as_of_date: Optional[_date] = None,
) -> Optional[pd.DataFrame]:
    if as_of_date is None:
        as_of_date = (datetime.now(KST) - timedelta(days=1)).date()

    bucket, _ = _parse_s3_uri(s3_etl_uri)
    all_keys = _list_keys(s3_etl_uri, suffix=".parquet")

    if not all_keys:
        print("[TRAIN] 사용자 ETL 없음 → 공개 feature만 사용")
        return None

    cutoff = as_of_date - timedelta(days=lookback_days)

    filtered = []
    for key in all_keys:
        m = re.search(r"dt=(\d{4}-\d{2}-\d{2})", key)
        if not m:
            continue
        part_dt = _date.fromisoformat(m.group(1))
        if cutoff <= part_dt <= as_of_date:
            filtered.append(key)

    if not filtered:
        print(f"[TRAIN] ETL 있지만 {cutoff}~{as_of_date} 범위 파티션 없음 → 공개만 사용")
        return None

    print(f"[TRAIN] 사용자 ETL 파티션 로드: {len(filtered)}개 ({cutoff} ~ {as_of_date})")
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

def build_train_features(public_feat: pd.DataFrame, user_feat: Optional[pd.DataFrame]) -> pd.DataFrame:
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
# 5. 모델 저장 (prefix → timestamp 버저닝)
# ---------------------------------------------------------------------------

def save_model(iso, s3_model_prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    s3_model_prefix = s3_model_prefix.rstrip("/") + "/"
    s3_model_uri = s3_model_prefix + f"isolation_forest_{ts}.pkl"

    buf = io.BytesIO()
    pickle.dump(iso, buf)
    buf.seek(0)

    bucket, key = _parse_s3_uri(s3_model_uri)
    _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"[TRAIN] S3 저장 완료: {s3_model_uri}")
    return s3_model_uri


# ---------------------------------------------------------------------------
# 6. 메인
# ---------------------------------------------------------------------------

def main(args):
    # -----------------------------
    # MLflow (hardcoded)
    # -----------------------------
    MLFLOW_APP_ARN = "arn:aws:sagemaker:ap-northeast-2:211946439649:mlflow-app/app-G6EK75LWNP4K"
    MLFLOW_EXPERIMENT = "nyang-data-gen"
    AWS_REGION = "ap-northeast-2"

    os.environ["AWS_REGION"] = AWS_REGION
    os.environ["MLFLOW_TRACKING_AWS_SIGV4"] = "true"

    mlflow.set_tracking_uri(MLFLOW_APP_ARN)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="train-isolation-forest"):
        as_of_date = (
            _date.fromisoformat(args.as_of_date)
            if args.as_of_date
            else (datetime.now(KST) - timedelta(days=1)).date()
        )
        cutoff = as_of_date - timedelta(days=args.lookback_days)

        mlflow.log_param("job_type", "train")
        mlflow.log_param("s3_public_feature_uri", args.s3_public_feature_uri)
        mlflow.log_param("s3_etl_uri", args.s3_etl_uri or "")
        mlflow.log_param("lookback_days", int(args.lookback_days))
        mlflow.log_param("as_of_date", str(as_of_date))
        mlflow.log_param("cutoff_date", str(cutoff))
        mlflow.log_param("s3_model_prefix", args.s3_model_prefix)
        mlflow.log_param("timezone", "KST")

        # 1) 공개 feature 로드
        public_feat = load_public_features(args.s3_public_feature_uri)
        mlflow.log_metric("public_rows", int(len(public_feat)))
        mlflow.log_metric("public_users", int(public_feat["uuid"].nunique()))
        mlflow.log_metric("public_cols", int(public_feat.shape[1]))

        # 2) 사용자 ETL 로드 + FE
        user_feat = None
        if args.s3_etl_uri:
            all_keys = _list_keys(args.s3_etl_uri, suffix=".parquet")
            part_cnt_total = 0
            part_cnt_used = 0
            for k in all_keys:
                m = re.search(r"dt=(\d{4}-\d{2}-\d{2})", k)
                if not m:
                    continue
                part_cnt_total += 1
                dt = _date.fromisoformat(m.group(1))
                if cutoff <= dt <= as_of_date:
                    part_cnt_used += 1

            mlflow.log_metric("etl_partitions_total", int(part_cnt_total))
            mlflow.log_metric("etl_partitions_used", int(part_cnt_used))

            user_feat = load_user_features(
                args.s3_etl_uri,
                lookback_days=args.lookback_days,
                as_of_date=as_of_date,
            )

        if user_feat is None:
            mlflow.log_metric("user_feat_rows", 0)
            mlflow.log_metric("user_feat_users", 0)
            mlflow.log_metric("user_feat_cols", 0)
        else:
            mlflow.log_metric("user_feat_rows", int(len(user_feat)))
            mlflow.log_metric("user_feat_users", int(user_feat["uuid"].nunique()))
            mlflow.log_metric("user_feat_cols", int(user_feat.shape[1]))

        # 3) 합치기
        feat_df = build_train_features(public_feat, user_feat)
        mlflow.log_metric("train_rows", int(len(feat_df)))
        mlflow.log_metric("train_users", int(feat_df["uuid"].nunique()))
        mlflow.log_metric("train_cols", int(feat_df.shape[1]))

        # 4) 학습
        _, iso = model.run(feat_df, mode="train")
        if iso is None:
            raise RuntimeError("[TRAIN] IsolationForest 학습 실패")

        # 5) 저장
        saved_uri = save_model(iso, args.s3_model_prefix)
        mlflow.log_param("saved_model_uri", saved_uri)

        print(f"[TRAIN] 완료 ✅ as_of_date={as_of_date}, model_saved={saved_uri}")


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
        help="공개데이터셋 feature parquet S3 URI",
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
        "--s3-model-prefix",
        type=str,
        default=os.environ.get("S3_MODEL_PREFIX", "s3://nyang-ml-apne2-dev/ml/artifacts/models/"),
        help="모델 S3 저장 prefix (예: s3://.../ml/artifacts/models/)",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=os.environ.get("AS_OF_DATE", None),
        help="학습 기준일 YYYY-MM-DD (기본: KST 기준 어제)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())