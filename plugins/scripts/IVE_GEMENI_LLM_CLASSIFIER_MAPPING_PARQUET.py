import io
import pandas as pd
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# snowflake -> s3 problem : all column's name edit by col_0, col_1
# column's name mapping
def MAPPING_S3_PARQUET(BUCKET_NAME, S3_KEY, LOCAL_PATH, **kwargs):
    # s3 connect
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    file_obj = s3_hook.get_key(S3_KEY, BUCKET_NAME)
    file_obj.download_file(LOCAL_PATH)
    
    # column's name
    actual_columns = [
        "KEY", "DATE", "TIME", "MDA", "NAME", "SAVE_WAY", "GUIDE", "OS_TYPE", "REJOIN_TYPE", "SHAPE", "RPT_TYPE",
        "MDA_ARR", "START_QUARTER", "START_TIME", "START_DATE", "END_DATE", "INDUSTRY", "TIME_CLK", "TIME_TURN",
        "ADV_COST", "REWARD_COST", "MDA_COST", "CVR", "ATS", "1000_W_EFFICIENCY", "CONTRACT_PRICE", "SCH_KEY", "LIMIT_TYPE"
    ]
    # data load
    df = pd.read_parquet(LOCAL_PATH)
    # mapping
    df.columns = actual_columns
    
    # s3 re-upload
    pq_buffer = io.BytesIO()
    df.to_parquet(pq_buffer, index=False, engine='pyarrow', compression='snappy')

    s3_hook.load_bytes(
        bytes_data=pq_buffer.getvalue(),
        key="ive_analytic/IVE_ANALYTICS_FINAL.parquet",
        bucket_name=BUCKET_NAME,
        replace=True
    )