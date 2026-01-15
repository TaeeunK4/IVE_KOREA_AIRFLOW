import io
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# cvr weight top 10
def PREDICT_TOP10_HIGHCVR(CLUSTER_ID: int, BUCKET_NAME: str):
    # s3 connect
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    # data load
    data_key = f"ive_ml/Clustering/IVE_ANALYTICS_CLUSTER_{CLUSTER_ID}.parquet"
    data_obj = s3_hook.get_key(data_key, BUCKET_NAME)
    df = pd.read_parquet(io.BytesIO(data_obj.get()['Body'].read()))
    
    # model load
    model_key = f"ive_ml/Models/Cluster_{CLUSTER_ID}_cat_re_models.pkl"
    model_obj = s3_hook.get_key(model_key, BUCKET_NAME)
    model_content = model_obj.get()['Body'].read()
    _model = pickle.loads(model_content)
    
    # x : shape, mda, start_time -> cvr, eff, ats predict
    unique_conditions = df[['SHAPE', 'MDA', 'START_TIME']].drop_duplicates()
    result_df = unique_conditions.copy()
    result_df['MDA'] = result_df['MDA'].astype(str)
    
    targets = {
            'CVR': 'Pred_CVR',
            '1000_W_EFFICIENCY': 'Pred_EFF',
            'ATS': 'Pred_ATS'
        }

    for model_key_name, col_name in targets.items():
        target_model = _model[model_key_name]
            
        # predict impossible -> float
        if hasattr(target_model, 'predict'):
            result_df[col_name] = target_model.predict(unique_conditions)
        else:
            result_df[col_name] = float(target_model)
            print(f"Cluster {CLUSTER_ID}: {model_key_name} is a constant value ({target_model}).")
    
    # scaling range 0 ~ 100, cvr weight -> score
    if not result_df.empty:
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaled_vals = scaler.fit_transform(result_df[['Pred_CVR', 'Pred_EFF', 'Pred_ATS']])
        result_df['CVR_scaled'] = scaled_vals[:, 0]
        result_df['EFF_scaled'] = scaled_vals[:, 1]
        result_df['ATS_scaled'] = scaled_vals[:, 2]
        result_df['score'] = result_df['CVR_scaled']*0.5 + result_df['EFF_scaled']*0.25 + result_df['ATS_scaled']*0.25
        # sort by score -> rank, cluster record
        top_10 = result_df.sort_values('score', ascending=False).head(10).copy()
        top_10['Rank'] = range(1, len(top_10) + 1)
        top_10['GMM_CLUSTER'] = CLUSTER_ID
        
        # s3 upload
        output_key = f"ive_ml/Pred_Top/Cluster_{CLUSTER_ID}_top10_highcvr.csv"
        s3_hook.load_string(
            string_data=top_10.to_csv(index=False),
            key=output_key,
            bucket_name=BUCKET_NAME,
            replace=True
        )
        print(f"Successfully processed Cluster {CLUSTER_ID}")
    else:
        print(f"Cluster {CLUSTER_ID} has no valid data after filtering.")

# eff weight top 10
def PREDICT_TOP10_HIGHEFF(CLUSTER_ID: int, BUCKET_NAME: str):
    # s3 connect    
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    # data load
    data_key = f"ive_ml/Clustering/IVE_ANALYTICS_CLUSTER_{CLUSTER_ID}.parquet"
    data_obj = s3_hook.get_key(data_key, BUCKET_NAME)
    df = pd.read_parquet(io.BytesIO(data_obj.get()['Body'].read()))

    # model load
    model_key = f"ive_ml/Models/Cluster_{CLUSTER_ID}_cat_re_models.pkl"
    model_obj = s3_hook.get_key(model_key, BUCKET_NAME)
    model_content = model_obj.get()['Body'].read()
    _model = pickle.loads(model_content)
    
    # x : shape, mda, start_time -> cvr, eff, ats predict
    unique_conditions = df[['SHAPE', 'MDA', 'START_TIME']].drop_duplicates()
    result_df = unique_conditions.copy()
    result_df['MDA'] = result_df['MDA'].astype(str)
    
    targets = {
            'CVR': 'Pred_CVR',
            '1000_W_EFFICIENCY': 'Pred_EFF',
            'ATS': 'Pred_ATS'
        }

    for model_key_name, col_name in targets.items():
        target_model = _model[model_key_name]
            
        # predict impossible -> float
        if hasattr(target_model, 'predict'):
            result_df[col_name] = target_model.predict(unique_conditions)
        else:
            result_df[col_name] = float(target_model)
            print(f"Cluster {CLUSTER_ID}: {model_key_name} is a constant value ({target_model}).")

    # scaling range 0 ~ 100, cvr weight -> score    
    if not result_df.empty:
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaled_vals = scaler.fit_transform(result_df[['Pred_CVR', 'Pred_EFF', 'Pred_ATS']])
        result_df['CVR_scaled'] = scaled_vals[:, 0]
        result_df['EFF_scaled'] = scaled_vals[:, 1]
        result_df['ATS_scaled'] = scaled_vals[:, 2]
        result_df['score'] = result_df['CVR_scaled']*0.25 + result_df['EFF_scaled']*0.5 + result_df['ATS_scaled']*0.25
        # sort by score -> rank, cluster record        
        top_10 = result_df.sort_values('score', ascending=False).head(10).copy()
        top_10['Rank'] = range(1, len(top_10) + 1)
        top_10['GMM_CLUSTER'] = CLUSTER_ID
        
        # s3 upload
        output_key = f"ive_ml/Pred_Top/Cluster_{CLUSTER_ID}_top10_higheff.csv"
        s3_hook.load_string(
            string_data=top_10.to_csv(index=False),
            key=output_key,
            bucket_name=BUCKET_NAME,
            replace=True
        )
        print(f"Successfully processed Cluster {CLUSTER_ID}")
    else:
        print(f"Cluster {CLUSTER_ID} has no valid data after filtering.")

# ats weight top 10
def PREDICT_TOP10_HIGHATS(CLUSTER_ID: int, BUCKET_NAME: str):
    # s3 connect    
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    # data load
    data_key = f"ive_ml/Clustering/IVE_ANALYTICS_CLUSTER_{CLUSTER_ID}.parquet"
    data_obj = s3_hook.get_key(data_key, BUCKET_NAME)
    df = pd.read_parquet(io.BytesIO(data_obj.get()['Body'].read()))
    
    # model load
    model_key = f"ive_ml/Models/Cluster_{CLUSTER_ID}_cat_re_models.pkl"
    model_obj = s3_hook.get_key(model_key, BUCKET_NAME)
    model_content = model_obj.get()['Body'].read()
    _model = pickle.loads(model_content)
    
    # x : shape, mda, start_time -> cvr, eff, ats predict
    unique_conditions = df[['SHAPE', 'MDA', 'START_TIME']].drop_duplicates()
    result_df = unique_conditions.copy()
    result_df['MDA'] = result_df['MDA'].astype(str)
    
    targets = {
            'CVR': 'Pred_CVR',
            '1000_W_EFFICIENCY': 'Pred_EFF',
            'ATS': 'Pred_ATS'
        }

    for model_key_name, col_name in targets.items():
        target_model = _model[model_key_name]
            
        # predict impossible -> float
        if hasattr(target_model, 'predict'):
            result_df[col_name] = target_model.predict(unique_conditions)
        else:
            result_df[col_name] = float(target_model)
            print(f"Cluster {CLUSTER_ID}: {model_key_name} is a constant value ({target_model}).")

    # scaling range 0 ~ 100, cvr weight -> score      
    if not result_df.empty:
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaled_vals = scaler.fit_transform(result_df[['Pred_CVR', 'Pred_EFF', 'Pred_ATS']])
        result_df['CVR_scaled'] = scaled_vals[:, 0]
        result_df['EFF_scaled'] = scaled_vals[:, 1]
        result_df['ATS_scaled'] = scaled_vals[:, 2]
        result_df['score'] = result_df['CVR_scaled']*0.25 + result_df['EFF_scaled']*0.25 + result_df['ATS_scaled']*0.5
        # sort by score -> rank, cluster record          
        top_10 = result_df.sort_values('score', ascending=False).head(10).copy()
        top_10['Rank'] = range(1, len(top_10) + 1)
        top_10['GMM_CLUSTER'] = CLUSTER_ID
        
        # s3 upload
        output_key = f"ive_ml/Pred_Top/Cluster_{CLUSTER_ID}_top10_highats.csv"
        s3_hook.load_string(
            string_data=top_10.to_csv(index=False),
            key=output_key,
            bucket_name=BUCKET_NAME,
            replace=True
        )
        print(f"Successfully processed Cluster {CLUSTER_ID}")
    else:
        print(f"Cluster {CLUSTER_ID} has no valid data after filtering.")
# concat by list[highcvr, higheff, highats]
def CONCAT_TOP10(METRIC_SUFFIX: list, BUCKET_NAME: str, **context):
    # s3 connect
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    prefix = "ive_ml/Pred_Top/"

    # get keys
    all_keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix=prefix)
    target_keys = [key for key in all_keys if key.endswith(f"_{METRIC_SUFFIX}.csv")]
    
    if not target_keys:
        print(f"No files found for metric: {METRIC_SUFFIX}")
        return

    df_list = []
    for key in target_keys:
        data_obj = s3_hook.get_key(key, BUCKET_NAME)
        temp_df = pd.read_csv(io.BytesIO(data_obj.get()['Body'].read()))
        df_list.append(temp_df)
    
    # concat list
    final_df = pd.concat(df_list, ignore_index=True)
    
    # s3 upload
    output_key = f"ive_ml/Pred_Top/Concat_Top/Final_Top10_{METRIC_SUFFIX}.csv"
    s3_hook.load_string(
        string_data=final_df.to_csv(index=False),
        key=output_key,
        bucket_name=BUCKET_NAME,
        replace=True
    )
    print(f"Successfully created: {output_key}")

def CONCAT_MASTER_TABLEAU_FILE(BUCKET_NAME: str, **kwargs):
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    
    metrics = ["highcvr", "higheff", "highats"]
    label_map = {"highcvr": "이익", "higheff": "비용", "highats": "안정성"}
    all_data = []

    for m in metrics:
        key = f"ive_ml/Pred_Top/Concat_Top/Final_Top10_{m}.csv"
        
        file_obj = s3_hook.get_key(key, BUCKET_NAME)
        temp_df = pd.read_csv(io.BytesIO(file_obj.get()['Body'].read()))
        
        temp_df['Metric_Type'] = label_map[m]
        all_data.append(temp_df)

    master_df = pd.concat(all_data, ignore_index=True)
    
    output_key = "ive_ml/Pred_Top/Concat_Top/Master_Top10_Tableau.csv"
    csv_buffer = io.StringIO()
    master_df.to_csv(csv_buffer, index=False)
    
    s3_hook.load_string(
        string_data=csv_buffer.getvalue(),
        key=output_key,
        bucket_name=BUCKET_NAME,
        replace=True
    )
    print(f"Master file created and uploaded to {output_key}")