import os
import io
import pyarrow.parquet as pq
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
    table = pq.read_table(LOCAL_PATH)

    # mapping
    new_table = table.rename_columns(actual_columns)
    
    # local save -> s3 upload
    OUTPUT_PATH = LOCAL_PATH + "_processed"
    pq.write_table(new_table, OUTPUT_PATH, compression='snappy')

    s3_hook.load_file(
        filename = OUTPUT_PATH,
        key = "ive_analytic/IVE_ANALYTICS_FINAL.parquet",
        bucket_name = BUCKET_NAME,
        replace=True
    )

    if os.path.exists(LOCAL_PATH): os.remove(LOCAL_PATH)
    if os.path.exists(OUTPUT_PATH): os.remove(OUTPUT_PATH)