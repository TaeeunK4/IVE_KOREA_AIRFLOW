import pandas as pd
from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from scripts.IVE_GEMENI_LLM_CLASSIFIER_V3_2.py import S3_TEMP_DELETE, EXTRACT_NULL, SPLIT_PRIOR_CLASSIFY, CLASSIFY_INDUSTRY_V3_2, MERGE_AFTER_CLASSIFY

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
    dag_id="IVE_GEMINI_LLM_CLASSIFIER_2_PIPELINE",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["gemini", "s3", "classify", "left"]
) as dag:
    with TaskGroup("IVE_INDUSTRY_CLASSIFY_2_GROUP") as IVE_INDUSTRY_CLASSIFY_2_GROUP:
        temp_clear_task = PythonOperator(
            task_id = "Delte_temp_data",
            python_callable = S3_TEMP_DELETE,
            op_kwargs = {
                "BUCKET_NAME": BUCKET_NAME,
                "TEMPS" : ["ive_temp_batch/left/"]
            }
        )
        extract_null_task = PythonOperator(
            task_id = "Extract_null_data",
            python_callable = EXTRACT_NULL,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "S3_KEY" : "ive_industry_classify/ive_industry_classified.csv",
                "RETRY_KEY" : "ive_industry_classify/retry/ive_industry_retry.csv",
            }
        )
        split_task = PythonOperator(
            task_id = "Split_left_data",
            python_callable = SPLIT_PRIOR_CLASSIFY,
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
            task_id = "Classify_splited_data",
            python_callable = CLASSIFY_INDUSTRY_V3_2,
            max_active_tis_per_dag = 50
        ).expand_kwargs(
            split_task.output 
        )
        merge_task = PythonOperator(
            task_id = "Merge_classified_data",
            python_callable = MERGE_AFTER_CLASSIFY,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "TEMP_OUTPUT_DIR" : TEMP_OUTPUT_DIR,
                "CLASSIFIED_OUTPUT_KEY" : CLASSIFIED_OUTPUT_KEY,
                "UNIQUE_MASTER_KEY" : UNIQUE_MASTER_KEY
            }
        )
        temp_clear_task >> extract_null_task >> split_task >> classify_task >> merge_task

    with TaskGroup("TRIGGER_TO_LLM_CLASSIFIER_GROUP") as TRIGGER_TO_LLM_CLASSIFIER_GROUP:
        trigger_classifier = TriggerDagRunOperator(
        task_id="Trigger_to_llm_3",
        trigger_dag_id="IVE_GEMINI_LLM_CLASSIFIER_3_PIPELINE",
        wait_for_completion=False,
        poke_interval=60,
        reset_dag_run=True,
        dag=dag,
    )
    IVE_INDUSTRY_CLASSIFY_2_GROUP >> TRIGGER_TO_LLM_CLASSIFIER_GROUP