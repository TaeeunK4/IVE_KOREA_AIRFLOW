import os
import pingouin as pg
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# experiment clear
def CLEAR_MLFLOW_RUNS(EXPERIMENT_NAME):
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment:
        experiment_id = experiment.experiment_id
        runs = client.search_runs(experiment_ids=[experiment_id])
        
        for run in runs:
            client.delete_run(run.info.run_id)
# data path load
def ANOVA_LOAD_DATA(BUCKET_NAME, S3_KEY, LOCAL_PATH):
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    file_obj = s3_hook.get_key(S3_KEY, BUCKET_NAME)
    file_obj.download_file(LOCAL_PATH)
    return LOCAL_PATH
# parquet data -> anova, welchs, kruskal test
def RUN_ANOVA_TEST(GROUP_COL: str, DV_COL: str,
                   LOCAL_PATH: str, alpha: float, EXPERIMENT_NAME: str, **kwargs):
    df = pd.read_parquet(LOCAL_PATH, columns=[GROUP_COL, DV_COL])

    # if len(df) > 200000:
    #     df = df.sample(n=200000, random_state=42)
    # mlflow & experiment connect
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # normal, homogeneous check
    is_normal = pg.normality(data=df, dv=DV_COL, group=GROUP_COL)['normal'].all()
    is_homogeneous = pg.homoscedasticity(data=df, dv=DV_COL, group=GROUP_COL)['equal_var'].iloc[0]

    with mlflow.start_run(run_name=f"{DV_COL}_by_{GROUP_COL}"):
        n = len(df)
        
        # CHOOSE METHOD
        if not is_normal:
            # Kruskal-Wallis (Epsilon)
            res = pg.kruskal(data = df,
                             dv = DV_COL,
                             between = GROUP_COL)
            p_val = res['p-unc'].iloc[0]
            h_stat = res['H'].iloc[0]
            epsilon_sq = h_stat / ((n**2 - 1) / (n + 1))
            method, es_name, es_val = "Kruskal-Wallis", "epsilon_squared", epsilon_sq

            if epsilon_sq < 0.01:
                jug_effect_size = "Very small"
            elif epsilon_sq < 0.06:
                jug_effect_size = "Small"
            elif epsilon_sq < 0.14:
                jug_effect_size = "Medium"
            else:
                jug_effect_size = "Big"

        elif not is_homogeneous:
            # Welch's ANOVA (Omega)
            res = pg.welch_anova(data = df,
                                 dv = DV_COL,
                                 between = GROUP_COL)
            p_val = res['p-unc'].iloc[0]
            f_stat = res['F'].iloc[0]
            df_bet = res['ddof1'].iloc[0]
            omega_sq = (df_bet * (f_stat - 1)) / (df_bet * (f_stat - 1) + n)
            method, es_name, es_val = "Welch's ANOVA", "omega_squared", max(0, omega_sq)

            if omega_sq < 0.01:
                jug_effect_size = "Very small"
            elif omega_sq < 0.06:
                jug_effect_size = "Small"
            elif omega_sq < 0.14:
                jug_effect_size = "Medium"
            else:
                jug_effect_size = "Big"
        else:
            # One-way ANOVA (Eta)
            res = pg.anova(data = df,
                           dv = DV_COL,
                           between = GROUP_COL)
            p_val = res['p-unc'].iloc[0]
            eta_sq = res['np2'].iloc[0]
            method, es_name, es_val = "One-way ANOVA", "eta_squared", eta_sq

            if eta_sq < 0.01:
                jug_effect_size = "Very small"
            elif eta_sq < 0.06:
                jug_effect_size = "Small"
            elif eta_sq < 0.14:
                jug_effect_size = "Medium"
            else:
                jug_effect_size = "Big"            
        # Judgement
        if p_val < alpha:
            jug = "REJECT"
        else:
            jug = "FAIL TO REJECT"

        # 3. MLflow record
        mlflow.log_params({
            "IV": GROUP_COL,
            "DV": DV_COL,
            "METHOD": method,
            "EFFECT_SIZE_TYPE": es_name,
            "JUDGEMENT": jug,
            "JUDGEMENT_EFFECT_SIZE": jug_effect_size
        })
        mlflow.log_metrics({
            "p_value": p_val,
            "effect_size": es_val
        })
        
        print(f"Finished {GROUP_COL}: {DV_COL}: {method}, p={p_val:.4f}")
# p_value < alpha & effect_size != very small -> labeling x
def LABELING_BY_JUDGEMENT(BUCKET_NAME: str, LOCAL_PATH: str,
                          EXPERIMENT_NAME: str, **kwargs):
    # IV all category check
    all_variable_info = {
        'INDUSTRY': ['F&B/식품', '커머스/유통', '뷰티/헬스', '가전/가구', '게임', '서비스', '금융/보험', '교육/공공'],
        'OS_TYPE': ['WEB', 'ANDROID', 'IOS'],
        'START_QUARTER': ['1Q', '2Q', '3Q', '4Q'],
        'REJOIN_TYPE': ['NONE', 'ADS_CODE_DAILY_UPDATE', 'REJOINABLE'],
        'LIMIT_TYPE' : ['UNLIMITED', 'LIMITED']
    }
    # mlflow & experiment connect
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    # p_value < alpha & effect_size != very small check
    kept_vars = []
    for iv in all_variable_info.keys():
        iv_results = runs[runs['params.IV'] == iv]
        
        if len(iv_results) >= 3:
            is_rejected = (iv_results['params.JUDGEMENT'] == 'REJECT').all()
            is_impactful = (iv_results['params.JUDGEMENT_EFFECT_SIZE'] != 'VERY SMALL').all()
            
            if is_rejected and is_impactful:
                kept_vars.append(iv)
    # p_value < alpha & effect_size != very small IV labeling
    master_map = pd.MultiIndex.from_product(
        [all_variable_info[v] for v in kept_vars],
        names=kept_vars
    ).to_frame(index=False)
    
    master_map['MASTER_LABEL'] = range(len(master_map))
    # origin data load and merge
    df = pd.read_parquet(LOCAL_PATH)
    final_labeled_df = pd.merge(df, master_map, on=kept_vars, how='left')
    del df

    TMP_OUTPUT = "/tmp/final_labeled_data.parquet"
    final_labeled_df.to_parquet(TMP_OUTPUT, index=False, engine='pyarrow', compression='snappy')
    del final_labeled_df

    # s3 upload parquet    
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    s3_hook.load_file(
        filename=TMP_OUTPUT,
        key="ive_analytic/IVE_ANALYTICS_LABEL_DATA.parquet",
        bucket_name=BUCKET_NAME,
        replace=True
    )
    if os.path.exists(TMP_OUTPUT):
        os.remove(TMP_OUTPUT)