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

# load output -> direct input another def
def CAT_LOAD_DATA(BUCKET_NAME: str, S3_KEY: str, LOCAL_PATH: str):
    # s3 connect
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    file_obj = s3_hook.get_key(S3_KEY, BUCKET_NAME)
    file_obj.download_file(LOCAL_PATH)
    return LOCAL_PATH

# min(rmse) + min(val & train gap) + early_stopping -> min rmse + protect over-fitting + L shape loss graph 
def FIND_OP_HYPERPARAMETER(trial, X, y, cat_features, cluster_id, target_name, task_type): 
    # train, valid data split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    if y_train.nunique() <= 1:
        print(f"⚠️ {target_name} C{cluster_id}: 학습셋 타겟이 단일 값입니다. Trial 중단.")
        return 100.0
    
    # only depth, learning_rate edit
    params = {
        "iterations": 1500,
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "task_type": task_type,
        "gpu_ram_part": 0.7 if task_type == "GPU" else None,
        "devices": "0" if task_type == "GPU" else None,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "logging_level": "Silent",
    }
    trial_run_name = f"Trial_{trial.number}_{target_name}_C{cluster_id}"
    with mlflow.start_run(run_name=trial_run_name, nested=True):
        # early_stopping_rounds -> L shape loss graph
        model = CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=100,
            cat_features=cat_features
        )
        
        # min rmse
        val_rmse = model.get_best_score()['validation']['RMSE']
        train_rmse = model.get_best_score()['learn']['RMSE']
        
        # val & train gap : protect over-fitting
        gap = abs(val_rmse - train_rmse)
        stability_score = val_rmse + (gap * 0.7)
        
        # mlflow record
        mlflow.log_params(params)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("rmse_gap", gap)
        mlflow.log_metric("total_score", stability_score)
        
        return stability_score

# shape, mda, start_time 3 features -> cvr, 1000_w_efficiency_ ats predict by using 
def CATBOOST_REGRESSOR_OP_MODEL(LOCAL_PATH: str, BUCKET_NAME: str,
                             CLUSTER: str, EXPERIMENT_NAME: str):
    # data load
    df = pd.read_parquet(LOCAL_PATH)
    target_cluster = int(CLUSTER)
    cluster_df = df[df['GMM_CLUSTER'] == target_cluster].copy()

    ROW_COUNT = len(cluster_df)
    DYNAMIC_TASK_TYPE = "GPU" if ROW_COUNT >= 64 else "CPU"
    print(f">>> Cluster {CLUSTER} (Rows: {ROW_COUNT}) -> Mode: {DYNAMIC_TASK_TYPE}")
    
    # features
    cat_features = ['SHAPE', 'MDA', 'START_TIME']
    targets = ['CVR', '1000_W_EFFICIENCY', 'ATS']
    cluster_models_pack = {}

    # mlflow connect
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1 cluster -> cvr, 1000_w_efficiency, ats 3 times 
    for target in targets:
        X = cluster_df[cat_features]
        y = cluster_df[target]
        
        # data lack -> constant_value
        if y.nunique() <= 1:
            const_value = y.iloc[0] 
            print(f"⚠️ {target}의 값이 모두 {const_value}로 일정합니다. 상수로 저장합니다.")
            cluster_models_pack[target] = const_value
            continue

        run_name = f"Cluster_{CLUSTER}_{target}_Optimization"
        with mlflow.start_run(run_name=run_name, nested=True):
            # def FIND_OP_HYPERPARAMETER : 5 times trial
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: FIND_OP_HYPERPARAMETER(trial, X, y, cat_features, CLUSTER, target, DYNAMIC_TASK_TYPE), n_trials=5)

            # select best model
            best_run_name = f"BestModel_{target}_C{CLUSTER}"
            with mlflow.start_run(run_name=best_run_name, nested=True):
                best_model = CatBoostRegressor(
                    iterations=1500, **study.best_params, 
                    cat_features=cat_features, task_type=DYNAMIC_TASK_TYPE, devices="0" if DYNAMIC_TASK_TYPE == "GPU" else None
                )
                X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42)

                if y_t.nunique() <= 1:
                    print(f"⚠️ {target} C{CLUSTER}: 최종 학습셋 분할 후 단일 값 발견. 학습을 건너뛰고 상수로 저장합니다.")
                    cluster_models_pack[target] = y_t.iloc[0]
                    continue

                best_model.fit(X_t, y_t, eval_set=(X_v, y_v), early_stopping_rounds=100, logging_level='Silent')

                # learn rmse/ validation rmse mapping -> check loss graph, over-fitting
                eval_results = best_model.get_evals_result()
                for i, (t_loss, v_loss) in enumerate(zip(eval_results['learn']['RMSE'], eval_results['validation']['RMSE'])):
                    mlflow.log_metric(f"final_{target}_train_rmse", t_loss, step=i)
                    mlflow.log_metric(f"final_{target}_val_rmse", v_loss, step=i)

                # mlflow best model record
                cluster_models_pack[target] = best_model
                mlflow.catboost.log_model(best_model, f"Model_{CLUSTER}_{target}")

    if not cluster_models_pack:
        print(f"❌ No models trained for Cluster {CLUSTER}. Skipping S3 upload.")
        return
    
    # model s3 upload : {cvr : model, 1000_w_efficiency : model, ats : model}
    model_buffer = io.BytesIO()
    pickle.dump(cluster_models_pack, model_buffer)
    
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    s3_key = f"ive_ml/Models/Cluster_{CLUSTER}_cat_re_models.pkl"
    
    s3_hook.load_bytes(
        bytes_data=model_buffer.getvalue(),
        key=s3_key,
        bucket_name=BUCKET_NAME,
        replace=True
    )
    print(f"Successfully uploaded integrated models for Cluster {CLUSTER} to S3.")

# all cluster process def
def CATBOOST_TOTAL_PROCESS(LOCAL_PATH, BUCKET_NAME, EXPERIMENT_NAME, **context):
    # OP_N = context['ti'].xcom_pull(key = 'op_n_components', task_ids = 'IVE_GMM_CLUSTERING_GROUP.Search_op_n')
    # OP_N = int(OP_N)
    # clusters = list(range(0, OP_N))
    clusters = [5,6]
    for cluster_id in clusters:
        print(f">>> Starting Tuning for Cluster {cluster_id}")
        CATBOOST_REGRESSOR_OP_MODEL(
            LOCAL_PATH=LOCAL_PATH, 
            BUCKET_NAME=BUCKET_NAME, 
            CLUSTER=str(cluster_id), 
            EXPERIMENT_NAME=EXPERIMENT_NAME
        )
        gc.collect()