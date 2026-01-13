import pandas as pd
from pathlib import Path
from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from cosmos import DbtTaskGroup, ProjectConfig, ProfileConfig, ExecutionConfig
from cosmos.profiles import SnowflakeUserPasswordProfileMapping
from airflow.utils.dates import days_ago
from scripts.IVE_GEMENI_LLM_CLASSIFIER_V3_3 import S3_TEMP_DELETE, EXTRACT_NULL, SPLIT_PRIOR_CLASSIFY, CLASSIFY_INDUSTRY_V3_3, MERGE_AFTER_CLASSIFY
from scripts.IVE_GEMINI_LLM_CLASSIFIER_FINALIZE import FINALIZE_CLASSIFIED_DATA
from scripts.IVE_GEMENI_LLM_CLASSIFIER_MAPPING_PARQUET import MAPPING_S3_PARQUET

BUCKET_NAME = "ivekorea-airflow-practice-taeeunk"
TEMP_INPUT_DIR = "ive_temp_batch/left/final/input/"
TEMP_OUTPUT_DIR = "ive_temp_batch/left/final/output/"
CLASSIFIED_OUTPUT_KEY = "ive_industry_classify/ive_industry_classified_left_final.csv"
UNIQUE_MASTER_KEY = "ive_sample/unique_master_key_left_final.csv"
SNOWFLAKE_CONN_ID = "snowflake_con"
DBT_PROJECT_PATH = Path("/opt/airflow/dbt_project")

profile_config = ProfileConfig(
    profile_name = "ive_dbt_project",
    target_name = "dev",
    profile_mapping = SnowflakeUserPasswordProfileMapping(
        conn_id = SNOWFLAKE_CONN_ID,
        profile_args = {"database" : "IVE_DATA", "schema" : "MAPPING_DATA"}
    )
)

default_args = {
    "owner" : "Taeeun",
    "start_date" : days_ago(1),
    "catchup" : False,
}

with DAG(
    dag_id="IVE_GEMINI_LLM_CLASSIFIER_3_PIPELINE",
    default_args=default_args,
    schedule_interval=None,
    template_searchpath = [
        '/opt/airflow/dbt_project/snowflake_queries'
    ],  
    catchup=False,
    tags=["gemini", "s3", "classify", "left"]
) as dag:
    with TaskGroup("IVE_INDUSTRY_CLASSIFY_3_GROUP") as IVE_INDUSTRY_CLASSIFY_3_GROUP:
        temp_clear_task = PythonOperator(
            task_id = "Delete_temp_data",
            python_callable = S3_TEMP_DELETE,
            op_kwargs = {
                "BUCKET_NAME": BUCKET_NAME,
                "TEMPS" : ["ive_temp_batch/left/final/"]
            }
        )
        extract_null_task = PythonOperator(
            task_id = "Extract_null_data",
            python_callable = EXTRACT_NULL,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "S3_KEY" : "ive_industry_classify/ive_industry_classified_left.csv",
                "RETRY_KEY" : "ive_industry_classify/retry/ive_industry_retry_final.csv",
            }
        )
        split_task = PythonOperator(
            task_id = "Split_left_data",
            python_callable = SPLIT_PRIOR_CLASSIFY,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "S3_KEY" : "ive_industry_classify/retry/ive_industry_retry_final.csv",
                "BATCH_SIZE" : 100,
                "TEMP_INPUT_DIR" : TEMP_INPUT_DIR,
                "TEMP_OUTPUT_DIR" : TEMP_OUTPUT_DIR,
                "UNIQUE_MASTER_KEY" : UNIQUE_MASTER_KEY
            }

        )
        classify_task = PythonOperator.partial(
            task_id = "Classify_splited_data",
            python_callable = CLASSIFY_INDUSTRY_V3_3,
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
    with TaskGroup("IVE_INDUSTRY_FINALIZE_MERGE_GROUP") as IVE_INDUSTRY_FINALIZE_MERGE_GROUP:
        finalize_task = PythonOperator(
            task_id = "Finalize_industry_data",
            python_callable = FINALIZE_CLASSIFIED_DATA,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "CLASSIFIED_KEYS" : [
                    "ive_industry_classify/ive_industry_classified.csv",
                    "ive_industry_classify/ive_industry_classified_left.csv",
                    "ive_industry_classify/ive_industry_classified_left_final.csv",
                ],
                "FINAL_OUTPUT_KEY" : "ive_industry_classify/ive_industry_mapping.csv"
            }
        )
        load_mapping_data = SnowflakeOperator(
            task_id = "Load_snowflake_mapping_data",
            snowflake_conn_id = SNOWFLAKE_CONN_ID,
            sql = [
               """
                CREATE OR REPLACE SCHEMA {{ var.value.DATABASE_NAME }}.{{ var.value.SCHEMA_MAPPING_NAME }};
               """,
               """
                CREATE OR REPLACE TABLE {{ var.value.DATABASE_NAME }}.{{ var.value.SCHEMA_MAPPING_NAME }}.IVE_INDUSTRY_MAPPING (
                    NAME VARCHAR, INDUSTRY VARCHAR
                );
                """,
                """
                COPY INTO {{ var.value.DATABASE_NAME }}.{{ var.value.SCHEMA_MAPPING_NAME }}.IVE_INDUSTRY_MAPPING
                FROM @{{ var.value.DATABASE_NAME }}.{{ var.value.SCHEMA_NAME }}.{{ var.value.STAGE_NAME }}/ive_industry_classify/
                FILES = ('ive_industry_mapping.csv')
                FILE_FORMAT = (
                    TYPE = 'CSV', 
                    SKIP_HEADER = 1, 
                    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
                )
                ON_ERROR = 'ABORT_STATEMENT'
                """ 
            ]
        )
        Dbt_Snowflake_clean_join = DbtTaskGroup(
            group_id = "DBT_SNOWFLAKE_INDUSTRY_JOIN_GROUP",
            project_config = ProjectConfig(DBT_PROJECT_PATH),
            profile_config = profile_config,
            execution_config = ExecutionConfig(dbt_executable_path = "/usr/local/bin/dbt"),
            operator_args= {"install_deps": True},
        )
        snowflake_s3_upload_final = SnowflakeOperator(
            task_id = "Snowflake_s3_upload",
            snowflake_conn_id = SNOWFLAKE_CONN_ID,
            sql = "IVE_ANALYTICS_FINAL_S3_UPLOAD.sql"
        )
        mapping_final_data = PythonOperator(
            task_id = "Mapping_parquet_data",
            python_callable = MAPPING_S3_PARQUET,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "S3_KEY" : "ive_analytic/IVE_ANALYTICS_FINAL.parquet",
                "LOCAL_PATH" : "/opt/airflow/data/IVE_ANALYTICS_FINAL.parquet"
            }
        )
        finalize_task >> load_mapping_data >> Dbt_Snowflake_clean_join >> snowflake_s3_upload_final >> mapping_final_data
    
    with TaskGroup("TRIGGER_TO_ANOVA_TEST_GROUP") as TRIGGER_TO_ANOVA_TEST_GROUP:
        trigger_classifier = TriggerDagRunOperator(
        task_id="Trigger_to_anova_test",
        trigger_dag_id="IVE_MLFLOW_ANOVA_TEST_PIPELINE",
        wait_for_completion=False,
        poke_interval=60,
        reset_dag_run=True,
        dag=dag,
    )
    IVE_INDUSTRY_CLASSIFY_3_GROUP >> IVE_INDUSTRY_FINALIZE_MERGE_GROUP >> TRIGGER_TO_ANOVA_TEST_GROUP