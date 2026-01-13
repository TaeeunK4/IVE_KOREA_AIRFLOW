from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from scripts.IVE_SPLIT_CLUSTER import SPLIT_CLUSTER

BUCKET_NAME = "ivekorea-airflow-practice-taeeunk"
LOCAL_PATH = "/opt/airflow/data/IVE_ANALYTICS_CLUSTER.parquet"
default_args = {
    "owner" : "Taeeun",
    "start_date" : days_ago(1),
    "catchup" : False,
}

with DAG(
    dag_id='IVE_SPLIT_CLUSTER_PIPELINE',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:
    with TaskGroup("SPLIT_CLUSTERING_GROUP") as SPLIT_CLUSTERING_GROUP:
        split_cluster = PythonOperator(
            task_id = "Split_task",
            python_callable = SPLIT_CLUSTER,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "S3_KEY" : "ive_ml/Clustering/IVE_ANALYTICS_CLUSTER.parquet",
                "LOCAL_PATH" : LOCAL_PATH
            }
        )
    with TaskGroup("TRIGGER_TO_PRED_TOP10_GROUP") as TRIGGER_TO_PRED_TOP10_GROUP:
        trigger_classifier = TriggerDagRunOperator(
        task_id="Trigger_to_pred_top10",
        trigger_dag_id="IVE_PRED_TOP10_PIPELINE",
        wait_for_completion=False,
        poke_interval=60,
        reset_dag_run=True,
        dag=dag,
    )        
    SPLIT_CLUSTERING_GROUP >> TRIGGER_TO_PRED_TOP10_GROUP