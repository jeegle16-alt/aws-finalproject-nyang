"""
src/runtime/inference.py
========================
FE → MODEL(infer) → DECODER 파이프라인 오케스트레이터.
batch_runner.py에서 호출.

[역할]
- iso.pkl 로드 (S3 or 로컬)
- raw_df 받아서 3 step 순서대로 실행
- state_out DataFrame 반환

[설계 원칙]
- 상태 없음 (stateless): 매 호출마다 iso 로드
- 베이스라인은 raw_df 안의 rolling window로 매일 새로 계산 (파일 저장 X)
- S3 저장 / RDS upsert는 batch_runner 담당 (여기선 안 함)
"""

import io
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.steps import features, model, decoder


# ---------------------------------------------------------------------------
# 1. 모델 로드
# ---------------------------------------------------------------------------

def load_model_from_s3(s3_model_uri: str) -> IsolationForest:
    """
    s3://bucket/path/isolation_forest.pkl 로드.
    s3_model_uri가 디렉토리면 isolation_forest.pkl 자동 append.
    """
    if not s3_model_uri.endswith(".pkl"):
        s3_model_uri = s3_model_uri.rstrip("/") + "/isolation_forest.pkl"

    parts  = s3_model_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key    = parts[1]

    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    iso = pickle.loads(obj["Body"].read())
    print(f"[INFER] 모델 로드 완료: s3://{bucket}/{key}")
    return iso


def load_model_from_local(model_path: str) -> IsolationForest:
    with open(model_path, "rb") as f:
        iso = pickle.load(f)
    print(f"[INFER] 모델 로드 완료 (로컬): {model_path}")
    return iso


def load_model(model_uri: str) -> IsolationForest:
    """s3:// 또는 로컬 경로 자동 판별."""
    if model_uri.startswith("s3://"):
        return load_model_from_s3(model_uri)
    return load_model_from_local(model_uri)


# ---------------------------------------------------------------------------
# 2. 파이프라인 실행
# ---------------------------------------------------------------------------

def run(
    model_uri: str,
    raw_df: Optional[pd.DataFrame] = None,
    feat_df: Optional[pd.DataFrame] = None,
    iso: Optional[IsolationForest] = None,
) -> tuple:
    """
    추론 파이프라인 실행.

    Parameters
    ----------
    model_uri : str
        iso.pkl 위치 (s3:// 또는 로컬 경로)
        iso가 직접 전달된 경우 무시됨
    raw_df : pd.DataFrame, optional
        silver parquet raw 이벤트. feat_df 없을 때 FE 실행.
    feat_df : pd.DataFrame, optional
        이미 FE된 daily-feature (56일치). 있으면 FE 스킵.
    iso : IsolationForest, optional
        이미 로드된 모델. 없으면 model_uri에서 로드.

    Returns
    -------
    state_df : pd.DataFrame
        state_out — uuid, date, cat_state, notify_level 포함
    model_df : pd.DataFrame
        model_out — baseline 컬럼 포함 (baseline 저장용)
    """
    if feat_df is None and raw_df is None:
        raise ValueError("[INFER] raw_df 또는 feat_df 중 하나는 필요합니다.")

    # 모델 로드
    if iso is None:
        iso = load_model(model_uri)

    # Step 1: Feature Engineering (feat_df 없을 때만)
    if feat_df is None:
        print(f"[INFER] raw_df: {raw_df.shape}, uuid: {raw_df['uuid'].nunique()}")
        feat_df = features.run(raw_df)
    else:
        print(f"[INFER] feat_df: {feat_df.shape}, uuid: {feat_df['uuid'].nunique()}")

    # Step 2: Baseline + Risk (추론 모드)
    model_df, _ = model.run(feat_df, mode="infer", iso=iso)

    # Step 3: State Decoding
    state_df = decoder.run(model_df)

    print(f"[INFER] 완료: {state_df.shape}")
    return state_df, model_df