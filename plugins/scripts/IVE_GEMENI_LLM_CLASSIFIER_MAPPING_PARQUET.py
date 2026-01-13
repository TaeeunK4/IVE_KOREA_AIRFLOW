import io
import pandas as pd
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def MAPPING_S3_PARQUET(BUCKET_NAME, S3_KEY, LOCAL_PATH, **kwargs):
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    file_obj = s3_hook.get_key(S3_KEY, BUCKET_NAME)
    file_obj.download_file(LOCAL_PATH)
    
    actual_columns = [
        "KEY", "DATE", "TIME", "MDA", "NAME", "SAVE_WAY", "GUIDE", "OS_TYPE", "REJOIN_TYPE", "SHAPE", "RPT_TYPE",
        "MDA_ARR", "START_QUARTER", "START_TIME", "START_DATE", "END_DATE", "INDUSTRY", "TIME_CLK", "TIME_TURN",
        "ADV_COST", "REWARD_COST", "MDA_COST", "CVR", "ABS", "1000_W_EFFICIENCY", "CONTRACT_PRICE", "SCH_KEY", "LIMIT_TYPE"
    ]
    
    df = pd.read_parquet(LOCAL_PATH)
    df.columns = actual_columns
    
    pq_buffer = io.BytesIO()
    df.to_parquet(pq_buffer, index=False, engine='pyarrow', compression='snappy')

    s3_hook.load_bytes(
        bytes_data=pq_buffer.getvalue(),
        key="ive_analytic/IVE_ANALYTICS_FINAL.parquet",
        bucket_name=BUCKET_NAME,
        replace=True
    )