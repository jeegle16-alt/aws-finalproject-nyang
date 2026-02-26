"""
lambda/trigger_training.py
===========================
EventBridge → Lambda → SageMaker Training Job 트리거.

[트리거 조건]
  EventBridge: cron(0 18 ? * SUN *)  # 매주 일요일 18:00 UTC = 월요일 03:00 KST

[환경변수 (Lambda 콘솔에서 설정) - 필수]
  SAGEMAKER_ROLE_ARN   : SageMaker 실행 역할 ARN
  S3_PUBLIC_URI        : 공개데이터셋 S3 URI
  S3_ETL_URI           : 사용자 ETL URI (없으면 실행 금지)
  S3_OUTPUT_MODEL_URI  : 모델 저장 prefix (예: s3://.../ml/artifacts/models/)
  ECR_IMAGE_URI        : 학습 컨테이너 이미지 URI
[환경변수 - 선택]
  INSTANCE_TYPE        : SageMaker 인스턴스 (default: ml.m5.xlarge)
  LOOKBACK_DAYS        : default 30
"""

import json
import os
from datetime import datetime, timedelta, timezone

import boto3

KST = timezone(timedelta(hours=9))


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return v.strip()


def _get_env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or str(v).strip() == "" else v.strip()


def handler(event, context):
    sm = boto3.client("sagemaker")

    job_name = f"nyang-train-{datetime.now(KST).strftime('%Y%m%d-%H%M%S')}"

    # ✅ 필수 env
    role_arn = _require_env("SAGEMAKER_ROLE_ARN")
    ecr_image_uri = _require_env("ECR_IMAGE_URI")
    s3_public_uri = _require_env("S3_PUBLIC_URI")
    s3_etl_uri = _require_env("S3_ETL_URI")
    s3_model_prefix = _require_env("S3_OUTPUT_MODEL_URI").rstrip("/") + "/"

    # 선택 env
    instance_type = _get_env("INSTANCE_TYPE", "ml.m5.xlarge")
    lookback_days = _get_env("LOOKBACK_DAYS", "30")

    # ✅ 컨테이너로 전달할 하이퍼파라미터
    hyperparameters = {
        "s3-public-uri": s3_public_uri,
        "s3-etl-uri": s3_etl_uri,
        "lookback-days": str(lookback_days),
        "s3-model-prefix": s3_model_prefix,
    }

    print(f"[TRIGGER] Training Job start: {job_name}")
    print(f"[TRIGGER] s3_public_uri={s3_public_uri}")
    print(f"[TRIGGER] s3_etl_uri={s3_etl_uri}")
    print(f"[TRIGGER] s3_model_prefix={s3_model_prefix}")
    print(f"[TRIGGER] instance_type={instance_type}")
    print(f"[TRIGGER] hyperparameters={hyperparameters}")

    response = sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=role_arn,
        AlgorithmSpecification={
            "TrainingImage": ecr_image_uri,
            "TrainingInputMode": "File",
        },
        # SageMaker가 기본 산출물(tar.gz 등)을 저장할 위치 (prefix)
        OutputDataConfig={
            "S3OutputPath": s3_model_prefix,
        },
        ResourceConfig={
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 3600,
        },
        HyperParameters=hyperparameters,
        EnableManagedSpotTraining=False,
    )

    job_arn = response["TrainingJobArn"]
    print(f"[TRIGGER] ARN: {job_arn}")

    return {
        "statusCode": 200,
        "body": json.dumps({"job_name": job_name, "job_arn": job_arn}),
    }