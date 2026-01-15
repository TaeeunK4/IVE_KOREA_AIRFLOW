from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from scripts.IVE_CATBOOST_PRED_TOP10 import PREDICT_TOP10_HIGHCVR, PREDICT_TOP10_HIGHEFF, PREDICT_TOP10_HIGHATS, CONCAT_TOP10, CONCAT_MASTER_TABLEAU_FILE


BUCKET_NAME = "ivekorea-airflow-practice-taeeunk"

default_args = {
    "owner" : "Taeeun",
    "start_date" : days_ago(1),
    "catchup" : False,
}

with DAG(
    dag_id='IVE_PRED_TOP10_PIPELINE',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,  
    tags = ["S3", "SPLIT", "CLUSTER", "UPLOAD"]
) as dag:
    with TaskGroup("PREDICT_TOP_10_BY_CATBOOST_GROUP") as PREDICT_TOP_10_BY_CATBOOST_GROUP:
        process_highcvr_task = PythonOperator.partial(
            task_id = "Predict_top_10_highcvr",
            python_callable = PREDICT_TOP10_HIGHCVR,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME
            }
        ).expand(
            op_args=[[i] for i in range(7)]
        )
        process_higheff_task = PythonOperator.partial(
            task_id = "Predict_top_10_higheff",
            python_callable = PREDICT_TOP10_HIGHEFF,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME
            }
        ).expand(
            op_args=[[i] for i in range(7)]
        )
        process_highats_task = PythonOperator.partial(
            task_id = "Predict_top_10_highats",
            python_callable = PREDICT_TOP10_HIGHATS,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME
            }
        ).expand(
            op_args=[[i] for i in range(7)]
        )
        process_highcvr_task >> process_higheff_task >> process_highats_task

    with TaskGroup("CONCAT_TOP10_RESULTS_WITH_METRICS_GROUP") as CONCAT_TOP10_RESULTS_WITH_METRICS_GROUP:
        metrics = ['highcvr', 'higheff', 'highats']
        for metric in metrics:
            concat_top_10 = PythonOperator(
                task_id = f"Concat_{metric}",
                python_callable = CONCAT_TOP10,
                op_kwargs = {
                    "METRIC_SUFFIX" : metric,
                    "BUCKET_NAME" : BUCKET_NAME
                }
            )

    with TaskGroup("CONCAT_MATER_FILE_TABLEAU_GROUP") as CONCAT_MATER_FILE_TABLEAU_GROUP:
        concat_master_file = PythonOperator(
        task_id='Concat_master_tableau_csv',
        python_callable = CONCAT_MASTER_TABLEAU_FILE,
        op_kwargs = {
            'BUCKET_NAME': BUCKET_NAME
            }
        )
    PREDICT_TOP_10_BY_CATBOOST_GROUP >> CONCAT_TOP10_RESULTS_WITH_METRICS_GROUP >> CONCAT_MATER_FILE_TABLEAU_GROUP