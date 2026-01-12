import itertools
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from scripts.anova_welchs_kruskal import CLEAR_MLFLOW_RUNS, ANOVA_LOAD_DATA, RUN_ANOVA_TEST, LABELING_BY_JUDGEMENT

BUCKET_NAME = "ivekorea-airflow-practice-taeeunk"
LOCAL_PATH = "/opt/airflow/data/IVE_ANALYTICS_FINAL.parquet"
EXPERIMENT_NAME = "ANOVA_TEST"
default_args = {
    "owner" : "Taeeun",
    "start_date" : days_ago(1),
    "catchup" : False,
}

GROUP_COLS = ["INDUSTRY", "OS_TYPE", "REJOIN_TYPE", "START_QUARTER", "LIMIT_TYPE"]
DV_COLS = ["CVR", "1000_W_EFFICIENCY", "ABS"]
all_mapped_params = []
for g, d in itertools.product(GROUP_COLS, DV_COLS):
    all_mapped_params.append({
        "GROUP_COL": g,
        "DV_COL": d,
        "LOCAL_PATH": LOCAL_PATH,
        "alpha": 0.05,
        "EXPERIMENT_NAME" : EXPERIMENT_NAME
    })

with DAG(
    dag_id='ANOVA_TEST_PIPELINE',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["ANOVA", "WELCHS", "KRUSKAL", "TEST"]
) as dag:
    with TaskGroup("ive_anova_test") as ive_anova_test:
        clear_experiment = PythonOperator(
            task_id = "clear_experiment",
            python_callable = CLEAR_MLFLOW_RUNS,
            op_kwargs = {
                "EXPERIMENT_NAME" : EXPERIMENT_NAME
            }
        )        
        
        load_data_task = PythonOperator(
            task_id = "anova_prior_load",
            python_callable = ANOVA_LOAD_DATA,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "S3_KEY" : "ive_analytic/IVE_ANALYTICS_FINAL.parquet",
                "LOCAL_PATH" : LOCAL_PATH,
                "EXPERIMENT_NAME" : EXPERIMENT_NAME
            }
        )
        anova_task = PythonOperator.partial(
            task_id = "anova_test_task",
            python_callable = RUN_ANOVA_TEST,
            max_active_tis_per_dag = 2
        ).expand(
            op_kwargs = all_mapped_params
        )
        clear_experiment >> load_data_task >> anova_task
    with TaskGroup("ive_anova_labeling") as ive_anova_labeling:
        labeling_s3_upload_task = PythonOperator(
            task_id = "anova_labeling_upload",
            python_callable = LABELING_BY_JUDGEMENT,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "LOCAL_PATH" : "/opt/airflow/data/IVE_ANALYTICS_FINAL.parquet",
                "EXPERIMENT_NAME" : EXPERIMENT_NAME
            }
        )
    ive_anova_test >> ive_anova_labeling