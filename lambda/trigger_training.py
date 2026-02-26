"""
lambda/trigger_training.py
===========================
EventBridge → Lambda → SageMaker Training Job 트리거.

[트리거 조건]
  EventBridge: cron(0 18 ? * SUN *)  # 매주 일요일 18:00 UTC = 월요일 03:00 KST

[환경변수 (Lambda 콘솔에서 설정)]
  SAGEMAKER_ROLE_ARN   : SageMaker 실행 역할 ARN (필수)
  S3_PUBLIC_URI        : 공개데이터셋 S3 URI (필수)
  S3_ETL_URI           : 사용자 ETL URI (✅ 필수 - 없으면 실행 금지)
  S3_OUTPUT_MODEL_URI  : 학습된 모델 저장 경로 (필수)
  ECR_IMAGE_URI        : 학습 컨테이너 이미지 URI (필수)
  INSTANCE_TYPE        : SageMaker 인스턴스 (default: ml.m5.xlarge)
"""

import json
import os
from datetime import datetime, timezone, timedelta

import boto3

KST = timezone(timedelta(hours=9))


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return v.strip()


def handler(event, context):
    sm = boto3.client("sagemaker")

    job_name = f"nyang-train-{datetime.now(KST).strftime('%Y%m%d-%H%M%S')}"

    # ✅ 전부 필수로 강제 (더미/기본값 제거)
    role_arn        = _require_env("SAGEMAKER_ROLE_ARN")
    ecr_image_uri   = _require_env("ECR_IMAGE_URI")
    s3_public_uri   = _require_env("S3_PUBLIC_URI")
    s3_etl_uri      = _require_env("S3_ETL_URI")          # ✅ 핵심: 반드시 있어야 함
    s3_output_model = _require_env("S3_OUTPUT_MODEL_URI")

    # 선택
    instance_type   = os.environ.get("INSTANCE_TYPE", "ml.m5.xlarge").strip()

    hyperparameters = {
        "s3-public-uri": s3_public_uri,
        "s3-etl-uri": s3_etl_uri,     # ✅ 항상 포함
        "lookback-days": "30",
    }

    print(f"[TRIGGER] Training Job 시작: {job_name}")
    print(f"[TRIGGER] s3_public_uri={s3_public_uri}")
    print(f"[TRIGGER] s3_etl_uri={s3_etl_uri}")
    print(f"[TRIGGER] s3_output_model={s3_output_model}")
    print(f"[TRIGGER] instance_type={instance_type}")

    response = sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=role_arn,
        AlgorithmSpecification={
            "TrainingImage": ecr_image_uri,
            "TrainingInputMode": "File",
        },
        OutputDataConfig={
            "S3OutputPath": s3_output_model,
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
