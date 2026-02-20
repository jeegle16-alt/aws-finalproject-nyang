"""
lambda/trigger_training.py
===========================
EventBridge → Lambda → SageMaker Training Job 트리거.

[트리거 조건]
  EventBridge: cron(0 18 ? * SUN *)  # 매주 일요일 18:00 UTC = 월요일 03:00 KST

[환경변수 (Lambda 콘솔에서 설정)]
  SAGEMAKER_ROLE_ARN   : SageMaker 실행 역할 ARN
  S3_PUBLIC_URI        : 공개데이터셋 S3 URI
  S3_ETL_URI           : 사용자 ETL URI (없으면 공개만 학습)
  S3_OUTPUT_MODEL_URI  : 학습된 모델 저장 경로
  ECR_IMAGE_URI        : 학습 컨테이너 이미지 URI
  INSTANCE_TYPE        : SageMaker 인스턴스 (default: ml.m5.xlarge)
"""

import json
import os
from datetime import datetime, timezone, timedelta

import boto3

KST = timezone(timedelta(hours=9))


def handler(event, context):
    sm = boto3.client("sagemaker")

    job_name = f"nyang-train-{datetime.now(KST).strftime('%Y%m%d-%H%M%S')}"

    role_arn        = os.environ["SAGEMAKER_ROLE_ARN"]
    ecr_image_uri   = os.environ["ECR_IMAGE_URI"]
    s3_public_uri   = os.environ.get("S3_PUBLIC_URI",    "s3://nyang-ml-apne2-dev/ml/inputs/public-dataset/")
    s3_etl_uri      = os.environ.get("S3_ETL_URI",       "")
    s3_output_model = os.environ.get("S3_OUTPUT_MODEL_URI", "s3://nyang-ml-apne2-dev/ml/artifacts/models/")
    instance_type   = os.environ.get("INSTANCE_TYPE",    "ml.m5.xlarge")

    hyperparameters = {
        "s3-public-uri": s3_public_uri,
        "lookback-days": "30",
    }
    if s3_etl_uri:
        hyperparameters["s3-etl-uri"] = s3_etl_uri

    response = sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=role_arn,
        AlgorithmSpecification={
            "TrainingImage":    ecr_image_uri,
            "TrainingInputMode": "File",
        },
        OutputDataConfig={
            "S3OutputPath": s3_output_model,
        },
        ResourceConfig={
            "InstanceType":    instance_type,
            "InstanceCount":   1,
            "VolumeSizeInGB":  30,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 3600,
        },
        HyperParameters=hyperparameters,
        EnableManagedSpotTraining=False,
    )

    job_arn = response["TrainingJobArn"]
    print(f"[TRIGGER] Training Job 시작: {job_name}")
    print(f"[TRIGGER] ARN: {job_arn}")

    return {
        "statusCode": 200,
        "body": json.dumps({"job_name": job_name, "job_arn": job_arn}),
    }