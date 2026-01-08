import pandas as pd
from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from scripts.classify_industry_gemini_3_left import s3_temp_delete, extract_null, split_prior_classify, classify_industry_3_left, merge_after_classify

# 설정값
BUCKET_NAME = "ivekorea-airflow-practice-taeeunk"
TEMP_INPUT_DIR = "ive_temp_batch/left/input/"
TEMP_OUTPUT_DIR = "ive_temp_batch/left/output/"
CLASSIFIED_OUTPUT_KEY = "ive_industry_classify/ive_industry_classified_left.csv"
UNIQUE_MASTER_KEY = "ive_sample/unique_master_key_left.csv"


default_args = {
    "owner" : "Taeeun",
    "start_date" : days_ago(1),
    "catchup" : False,
}

with DAG(
    dag_id="gemini_industry_classification_left_v3",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["gemini", "s3", "classify", "left"]
) as dag:
    with TaskGroup("ive_industry_classify_left") as ive_industry_classify_left:
        temp_clear_task = PythonOperator(
            task_id = "delete_temp_data",
            python_callable = s3_temp_delete,
            op_kwargs = {
                "BUCKET_NAME": BUCKET_NAME,
                "TEMPS" : ["ive_temp_batch/left/"]
            }
        )
        extract_null_task = PythonOperator(
            task_id = "extract_null_data",
            python_callable = extract_null,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "S3_KEY" : "ive_industry_classify/ive_industry_classified.csv",
                "RETRY_KEY" : "ive_industry_classify/retry/ive_industry_retry.csv",
            }
        )
        split_task = PythonOperator(
            task_id = "split_left_data",
            python_callable = split_prior_classify,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "S3_KEY" : "ive_industry_classify/retry/ive_industry_retry.csv",
                "BATCH_SIZE" : 100,
                "TEMP_INPUT_DIR" : TEMP_INPUT_DIR,
                "TEMP_OUTPUT_DIR" : TEMP_OUTPUT_DIR,
                "UNIQUE_MASTER_KEY" : UNIQUE_MASTER_KEY
            }

        )
        classify_task = PythonOperator.partial(
            task_id = "classify_splited_data",
            python_callable = classify_industry_3_left,
            max_active_tis_per_dag = 50
        ).expand_kwargs(
            split_task.output 
        )
        merge_task = PythonOperator(
            task_id = "merge_classified_data",
            python_callable = merge_after_classify,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "TEMP_OUTPUT_DIR" : TEMP_OUTPUT_DIR,
                "CLASSIFIED_OUTPUT_KEY" : CLASSIFIED_OUTPUT_KEY,
                "UNIQUE_MASTER_KEY" : UNIQUE_MASTER_KEY
            }
        )

    temp_clear_task >> extract_null_task >> split_task >> classify_task >> merge_task