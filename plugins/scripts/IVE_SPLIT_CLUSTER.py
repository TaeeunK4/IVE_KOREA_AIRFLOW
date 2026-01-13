import io
import pandas as pd
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def SPLIT_CLUSTER(BUCKET_NAME, S3_KEY, LOCAL_PATH, **kwargs):
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    file_obj = s3_hook.get_key(S3_KEY, BUCKET_NAME)
    file_obj.download_file(LOCAL_PATH)
    
    df = pd.read_parquet(LOCAL_PATH)
    unique_clusters = sorted(df['GMM_CLUSTER'].unique())
    for i in unique_clusters:
        df_1 = df[df['GMM_CLUSTER'] == i]
        pq_buffer = io.BytesIO()
        df_1.to_parquet(pq_buffer, index=False, engine='pyarrow', compression='snappy')

        s3_hook.load_bytes(
            bytes_data=pq_buffer.getvalue(),
            key=f"ive_ml/Clustering/IVE_ANALYTICS_CLUSTER_{i}.parquet",
            bucket_name=BUCKET_NAME,
            replace=True
        )