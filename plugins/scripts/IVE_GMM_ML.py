import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from typing import Type
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# s3 data load
def GMM_LOAD_DATA(BUCKET_NAME, S3_KEY, LOCAL_PATH):
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    file_obj = s3_hook.get_key(S3_KEY, BUCKET_NAME)
    file_obj.download_file(LOCAL_PATH)
    return LOCAL_PATH

# search op_n
def GMM_SEARCH_N(LOCAL_PATH: str,
                 SCALER_TYPE: Type, EXPERIMENT_NAME: str, **context):
    df = pd.read_parquet(LOCAL_PATH)
    # groupby master_label -> x features : cvr, abs, 1000_w_efficiency
    df_label = df.groupby('MASTER_LABEL')[['CVR', 'ABS', '1000_W_EFFICIENCY']].mean().reset_index()
    df_label.columns = [
        'MASTER_LABEL', 'avg_CVR', 'avg_ABS', 'avg_1000_W_EFFICIENCY']

    X = df_label[['avg_CVR', 'avg_ABS', 'avg_1000_W_EFFICIENCY']]
    # all num -> scaling
    scaler = SCALER_TYPE()
    X_scaled = scaler.fit_transform(X)
    # 2 ~ 10 -> what is op_n?
    ns = range(2, 11)
    results = {}

    # mlflow connect
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    # run : gmm_n-search
    with mlflow.start_run(run_name="GMM_N_search"):
        # check all n's bic_score
        for n in ns:
            GMM_SEARCH_MODEL = GaussianMixture(n_components = n, random_state=42).fit(X_scaled)
            BIC_SCORE = GMM_SEARCH_MODEL.bic(X_scaled)
            results[n] = BIC_SCORE
            mlflow.log_metric("BIC", BIC_SCORE, step = n)
        # op_n = min(bic_score) -> push xcom
        OP_N = min(results, key=results.get)
        context['ti'].xcom_push(key = 'op_n_components', value = OP_N)

# gmm clustering by op_n
def GMM_CLUSTERING(BUCKET_NAME: str, LOCAL_PATH: str,
                   SCALER_TYPE: Type, EXPERIMENT_NAME: str, **context):
    # pull xcom's op_n
    OP_N = context['ti'].xcom_pull(key = 'op_n_components', task_ids = 'GMM_CLUSTERING_FINAL_DATA.SEARCH_OP_N')
    OP_N = int(OP_N)
    df = pd.read_parquet(LOCAL_PATH)

    df_label = df.groupby('MASTER_LABEL')[['CVR', 'ABS', '1000_W_EFFICIENCY']].mean().reset_index()
    df_label.columns = [
        'MASTER_LABEL', 'avg_CVR', 'avg_ABS', 'avg_1000_W_EFFICIENCY']

    X = df_label[['avg_CVR', 'avg_ABS', 'avg_1000_W_EFFICIENCY']]
    scaler = SCALER_TYPE()
    X_scaled = scaler.fit_transform(X)

    # mlflow connect
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    # run : gmm_clustering by op_n
    with mlflow.start_run(run_name=f"GMM_CLUSTERING_{OP_N}"):
        GMM_MODEL = GaussianMixture(n_components = OP_N, random_state = 42)
        GMM_LABEL = GMM_MODEL.fit_predict(X_scaled)
        # cluster_id record
        df_label['GMM_CLUSTER'] = GMM_LABEL
        # clustering result check by pca plot
        pca = PCA(n_components = 2)
        PCA_GMM_RESULT = pca.fit_transform(X_scaled)
        df_label['PCA_x'] = PCA_GMM_RESULT[:, 0]
        df_label['PCA_y'] = PCA_GMM_RESULT[:, 1]

        plt.figure(figsize = (12, 8))
        sns.scatterplot(
            x = 'PCA_x', y = 'PCA_y',
            hue = 'GMM_CLUSTER',
            palette = 'Set1',
            data = df_label,
            alpha = 0.7
        )
        plt.title(f'GMM Clustering PCA [{OP_N} Clusters]')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plot_path = "GMM_PCA_SCATTER.png"
        plt.savefig(plot_path) 
        # mlflow plot record
        mlflow.log_artifact(plot_path)
        plt.close()
        # mlflow parmas record
        mlflow.log_params({
            "OP_N": OP_N,
            "SCALER_TYPE": f"{SCALER_TYPE}"
        })
        # mlflow model record
        mlflow.sklearn.log_model(GMM_MODEL, "GMM_MODEL")
    # direct s3 upload
    df_label_upload = df_label[['MASTER_LABEL', 'GMM_CLUSTER']]
    df_result = pd.merge(df, df_label_upload, how='left', on='MASTER_LABEL')

    pq_buffer = io.BytesIO()
    df_result.to_parquet(pq_buffer, index=False, engine='pyarrow', compression='snappy')
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    s3_hook.load_bytes(
        bytes_data=pq_buffer.getvalue(),
        key="ive_ml/Clustering/IVE_ANALYTICS_CLUSTER.parquet",
        bucket_name=BUCKET_NAME,
        replace=True
    )