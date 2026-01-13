import io
import pickle
import pandas as pd
import optuna
import gc
import mlflow
import mlflow.catboost
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def CAT_LOAD_DATA(BUCKET_NAME, S3_KEY, LOCAL_PATH):
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    file_obj = s3_hook.get_key(S3_KEY, BUCKET_NAME)
    file_obj.download_file(LOCAL_PATH)
    return LOCAL_PATH

def objective(trial, X, y, cat_features): 
    if y.nunique() <= 1:
        return 0.0
    
    params = {
        "iterations": 1500,
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "task_type": "GPU",
        "gpu_ram_part": 0.7,
        "devices": "0",
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "logging_level": "Silent",
    }

    with mlflow.start_run(nested=True):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        model = CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=100,
            cat_features=cat_features
        )
        
        val_rmse = model.get_best_score()['validation']['RMSE']
        train_rmse = model.get_best_score()['learn']['RMSE']
        
        # 안정성 페널티 적용 (L자형 수렴 유도)
        gap = abs(val_rmse - train_rmse)
        stability_score = val_rmse + (gap * 0.7)
        
        mlflow.log_params(params)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("rmse_gap", gap)
        
        return stability_score

def CATBOOST_REGRESSOR_OP_MODEL(LOCAL_PATH: str, BUCKET_NAME: str,
                             CLUSTER: str, EXPERIMENT_NAME: str):
    df = pd.read_parquet(LOCAL_PATH)
    target_cluster = int(CLUSTER)
    cluster_df = df[df['GMM_CLUSTER'] == target_cluster].copy()
    
    # 공통 설정
    cat_features = ['SHAPE', 'MDA', 'START_TIME']
    targets = ['CVR', '1000_W_EFFICIENCY', 'ABS']
    cluster_models_pack = {} # 모델들을 담을 바구니
    
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 한 클러스터 내에서 3개의 타겟에 대해 루프 수행
    for target in targets:
        X = cluster_df[cat_features]
        y = cluster_df[target]
        
        if y.nunique() <= 1:
        # [수정] 모델 대신 실제 값(상수)을 저장합니다.
        # 나중에 예측 시 expm1을 적용한다면 여기서 미리 log1p를 취해 저장하는 것이 편리합니다.
            const_value = y.iloc[0] 
            print(f"⚠️ {target}의 값이 모두 {const_value}로 일정합니다. 상수로 저장합니다.")
            cluster_models_pack[target] = const_value
            continue

        run_name = f"Cluster_{CLUSTER}_{target}_Optimization"
        
        with mlflow.start_run(run_name=run_name, nested=True):
            # 1. Optuna 튜닝
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, X, y, cat_features), n_trials=5)

            # 2. 최종 모델 학습 (L자형 수렴 기록용)
            best_model = CatBoostRegressor(
                iterations=1500, **study.best_params, 
                cat_features=cat_features, task_type="GPU", devices="0"
            )
            X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42)
            best_model.fit(X_t, y_t, eval_set=(X_v, y_v), early_stopping_rounds=100, logging_level='Silent')

            # 3. L자형 그래프용 지표 기록
            eval_results = best_model.get_evals_result()
            for i, (t_loss, v_loss) in enumerate(zip(eval_results['learn']['RMSE'], eval_results['validation']['RMSE'])):
                mlflow.log_metric(f"final_{target}_train_rmse", t_loss, step=i)
                mlflow.log_metric(f"final_{target}_val_rmse", v_loss, step=i)

            # 4. 딕셔너리에 모델 저장
            cluster_models_pack[target] = best_model
            mlflow.catboost.log_model(best_model, f"Model_{CLUSTER}_{target}")

    # 5. 한 클러스터의 모든 모델이 담긴 딕셔너리를 S3에 업로드
    if not cluster_models_pack:
        print(f"❌ No models trained for Cluster {CLUSTER}. Skipping S3 upload.")
        return
    
    model_buffer = io.BytesIO()
    pickle.dump(cluster_models_pack, model_buffer) # { 'CVR': model, ... } 형태로 저장
    
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    s3_key = f"ive_ml/Models/Cluster_{CLUSTER}_cat_re_models.pkl"
    
    s3_hook.load_bytes(
        bytes_data=model_buffer.getvalue(),
        key=s3_key,
        bucket_name=BUCKET_NAME,
        replace=True
    )
    print(f"Successfully uploaded integrated models for Cluster {CLUSTER} to S3.")

def CATBOOST_TOTAL_PROCESS(LOCAL_PATH, BUCKET_NAME, EXPERIMENT_NAME, **context):

    # ti = context['ti']
    # # GMM_SEARCH_N 태스크에서 push한 op_n_components 값을 가져옴
    # op_n = ti.xcom_pull(key='op_n_components', task_ids='GMM_CLUSTERING_FINAL_DATA.SEARCH_OP_N')
    # op_n = int(op_n)
    # clusters = range(op_n) # 보통 0부터 op_n-1까지이므로 range(op_n) 사용
    clusters = [0, 1, 2]
    # 하나의 태스크 내부에서 클러스터별로 순차 학습
    for cluster_id in clusters:
        print(f">>> Starting Tuning for Cluster {cluster_id}")
        CATBOOST_REGRESSOR_OP_MODEL(
            LOCAL_PATH=LOCAL_PATH, 
            BUCKET_NAME=BUCKET_NAME, 
            CLUSTER=str(cluster_id), 
            EXPERIMENT_NAME=EXPERIMENT_NAME
        )
        gc.collect()