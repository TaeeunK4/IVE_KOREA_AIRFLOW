from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from scripts.IVE_GMM_ML import GMM_LOAD_DATA, GMM_SEARCH_N, GMM_CLUSTERING
from scripts.IVE_CATBOOST_ML import CAT_LOAD_DATA, CATBOOST_TOTAL_PROCESS
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

BUCKET_NAME = "ivekorea-airflow-practice-taeeunk"
LOCAL_PATH = "/opt/airflow/data/IVE_ANALYTICS_LABEL_DATA.parquet"
EXPERIMENT_NAME_GMM = "IVE_GMM_CLUSTERING"
EXPERIMENT_NAME_CAT = "IVE_CAT_REGRESSOR"

default_args = {
    "owner" : "Taeeun",
    "start_date" : days_ago(1),
    "catchup" : False,
}

with DAG(
    dag_id='IVE_ML_GMM_CATBOOST_PIPELINE',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["ML", "GMM", "CATBOOST", "MLFLOW"]
) as dag:
    with TaskGroup("IVE_GMM_CLUSTERING_GROUP") as IVE_GMM_CLUSTERING_GROUP:
        GMM_LOAD = PythonOperator(
            task_id = "Load_analytics_data",
            python_callable = GMM_LOAD_DATA,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "S3_KEY" : "ive_analytic/IVE_ANALYTICS_LABEL_DATA.parquet",
                "LOCAL_PATH" : LOCAL_PATH
            }
        )
        GMM_SEARCH = PythonOperator(
            task_id = "Search_op_n",
            python_callable = GMM_SEARCH_N,
            op_kwargs = {
                "LOCAL_PATH" : GMM_LOAD.output,
                "SCALER_TYPE" : RobustScaler,
                "EXPERIMENT_NAME" : EXPERIMENT_NAME_GMM
            }
        )
        GMM_CLUSTERING_DATA = PythonOperator(
            task_id = "Gmm_clustering",
            python_callable = GMM_CLUSTERING,
            op_kwargs = {
                "BUCKET_NAME" : BUCKET_NAME,
                "LOCAL_PATH" : GMM_LOAD.output,
                "SCALER_TYPE" : RobustScaler,
                "EXPERIMENT_NAME" : EXPERIMENT_NAME_GMM
            }
        )
    GMM_LOAD >> GMM_SEARCH >> GMM_CLUSTERING_DATA
    
    with TaskGroup("IVE_CATBOOST_REGRESSOR_GROUP") as IVE_CATBOOST_REGRESSOR_GROUP:
            CAT_LOAD = PythonOperator(
                task_id = "Load_cluster_data",
                python_callable = CAT_LOAD_DATA,
                op_kwargs = {
                    "BUCKET_NAME" : BUCKET_NAME,
                    "S3_KEY" : "ive_ml/Clustering/IVE_ANALYTICS_CLUSTER.parquet",
                    "LOCAL_PATH" : "/opt/airflow/data/IVE_ANALYTICS_CLUSTER.parquet"
                }
            )
            CAT_REG_CLUSTER_DATA = PythonOperator(
                task_id = "Cat_reg_total_process",
                python_callable = CATBOOST_TOTAL_PROCESS,
                op_kwargs = {
                    "BUCKET_NAME" : BUCKET_NAME,
                    "LOCAL_PATH" : CAT_LOAD.output,
                    "EXPERIMENT_NAME" : EXPERIMENT_NAME_CAT
                }
            )
            CAT_LOAD >> CAT_REG_CLUSTER_DATA
    with TaskGroup("TRIGGER_TO_SPLIT_CLUSTER_GROUP") as TRIGGER_TO_SPLIT_CLUSTER_GROUP:
        trigger_classifier = TriggerDagRunOperator(
        task_id="Trigger_to_split_cluster",
        trigger_dag_id="IVE_SPLIT_CLUSTER_PIPELINE",
        wait_for_completion=False,
        poke_interval=60,
        reset_dag_run=True,
        dag=dag,
    )
    IVE_GMM_CLUSTERING_GROUP >> IVE_CATBOOST_REGRESSOR_GROUP >> TRIGGER_TO_SPLIT_CLUSTER_GROUP