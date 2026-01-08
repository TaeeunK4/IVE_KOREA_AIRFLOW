import pandas as pd
import os
from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from scripts.s3_upload_csv import s3_upload_csv
from scripts.classify_industry_gemini_1 import classify_industry_1
from scripts.classify_industry_gemini_2 import classify_industry_2
from scripts.classify_industry_gemini_3 import classify_industry_3

# --- 설정 구간 ---
BUCKET_NAME = "ivekorea-airflow-practice-taeeunk"
LOCAL_PATH = "/opt/airflow/data"

default_args = {
    "owner" : "Taeeun",
    "start_date" : days_ago(1),
    "catchup" : False,
}

with DAG(
    dag_id='gemini_industry_classification_sample',
    default_args=default_args,
    schedule_interval=None, # 필요 시 설정
    catchup=False,
    tags=['gemini', 's3', 'ai_enrichment']
) as dag:
    with TaskGroup("sample_upload") as sample_upload:
        upload_sample = PythonOperator(
            task_id = "upload_sample",
            python_callable = s3_upload_csv,
            op_kwargs = {
                "local_base_path" : os.path.join(LOCAL_PATH, "industry_sample"),
                "file_names" : ["ive_industry_sample.csv"],
                "s3_folder" : "ive_sample",
                "bucket_name" : BUCKET_NAME,
                "aws_credentials" : {
                    "AWS_ACCESS_KEY_ID" : "{{var.value.AWS_ACCESS_KEY_ID}}",
                    "AWS_SECRET_ACCESS_KEY" : "{{var.value.AWS_SECRET_ACCESS_KEY}}",
                    "AWS_ACCESS_REGION" : "{{var.value.AWS_ACCESS_REGION}}",
                    },
                "target_columns" : [
                    "ADS_IDX", "NAME"
                    ]
            }
        )
    with TaskGroup("classify_sample") as classify_sample:        
        classify_sample_1 = PythonOperator(
            task_id = "classify_sample_1",
            python_callable = classify_industry_1,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "S3_KEY" : "ive_sample/ive_industry_sample.csv",
                "OUTPUT_S3_KEY" : "ive_sample/ive_industry_sample_result_1.csv"
            }
        )
        classify_sample_2 = PythonOperator(
            task_id = "classify_sample_2",
            python_callable = classify_industry_2,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "S3_KEY" : "ive_sample/ive_industry_sample.csv",
                "OUTPUT_S3_KEY" : "ive_sample/ive_industry_sample_result_2.csv"
            }
        )
        classify_sample_3 = PythonOperator(
            task_id = "classify_sample_3",
            python_callable = classify_industry_3,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "S3_KEY" : "ive_sample/ive_industry_sample.csv",
                "OUTPUT_S3_KEY" : "ive_sample/ive_industry_sample_result_3.csv"
            }
        )
    sample_upload >> classify_sample