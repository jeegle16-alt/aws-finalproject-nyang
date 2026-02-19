"""
src/runtime/inference.py
─────────────────────────
SageMaker 엔드포인트 핸들러.

입력 JSON:
  {"dt": "YYYY-MM-DD"}                           → batch_runner.run_batch 호출
  {"features": {col: val, ...}, "uuid": "..."}   → 단일 유저 즉시 예측 (stub)

출력 JSON:
  {"status": "ok", "dt": "...", "output_s3": "s3://..."}
"""
from __future__ import annotations

import io
import json
import logging
import os
import pickle

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SageMaker 핸들러
# ──────────────────────────────────────────────────────────────────────────────

def model_fn(model_dir: str):
    """
    SageMaker: /opt/ml/model 에서 model artifacts 로드.

    model_dir 에는 iso_forest.pkl 과 config.yaml 이 있다고 가정.
    """
    import yaml
    from sklearn.ensemble import IsolationForest

    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    iso_path = os.path.join(model_dir, "iso_forest.pkl")
    iso_model: IsolationForest | None = None
    if os.path.exists(iso_path):
        with open(iso_path, "rb") as f:
            iso_model = pickle.load(f)
        logger.info(f"[model_fn] Loaded IsolationForest from {iso_path}")
    else:
        logger.warning("[model_fn] iso_forest.pkl not found – model will retrain on first call")

    return {"config": config, "iso": iso_model, "model_dir": model_dir}


def input_fn(request_body: str, content_type: str = "application/json") -> dict:
    """
    SageMaker: 요청 본문 파싱.

    지원 형식:
      {"dt": "YYYY-MM-DD"}                         → 배치 모드
      {"dt": "YYYY-MM-DD", "lookback_days": 30}    → 배치 모드 (lookback 설정)
    """
    if "application/json" not in content_type:
        raise ValueError(f"Unsupported content_type: {content_type}")
    return json.loads(request_body)


def predict_fn(input_data: dict, model: dict) -> dict:
    """
    SageMaker: 예측 실행.

    batch_runner.run_batch 를 호출한다.
    S3 I/O가 포함되므로 IAM Role에 적절한 권한이 필요하다.
    """
    from src.runtime.batch_runner import run_batch

    config       = model["config"]
    iso_model    = model.get("iso")
    dt_local     = input_data.get("dt")
    lookback     = int(input_data.get("lookback_days", 60))

    if not dt_local:
        raise ValueError("'dt' field is required in request body.")

    ml_bucket  = config["s3"]["ml_bucket"]
    out_prefix = config["paths"]["outputs"].rstrip("/")
    output_s3  = f"s3://{ml_bucket}/{out_prefix}/dt={dt_local}/state_out.parquet"

    metrics = run_batch(
        dt_local=dt_local,
        config=config,
        lookback_days=lookback,
        retrain_iso=(iso_model is None),
    )

    return {
        "status": "ok",
        "dt": dt_local,
        "output_s3": output_s3,
        "metrics": metrics,
    }


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    """
    SageMaker: 응답 직렬화.
    """
    return json.dumps(prediction, ensure_ascii=False, default=str)
