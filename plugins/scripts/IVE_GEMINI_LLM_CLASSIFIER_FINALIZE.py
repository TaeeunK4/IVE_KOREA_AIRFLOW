import pandas as pd
import io
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# classified_keys -> one file -> mapping -> merge with origin_data -> s3 upload
def FINALIZE_CLASSIFIED_DATA(BUCKET_NAME, CLASSIFIED_KEYS, FINAL_OUTPUT_KEY):
    # s3 connect
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    all_dfs = []
    for key in CLASSIFIED_KEYS:
        content = s3_hook.read_key(key, BUCKET_NAME)
        all_dfs.append(pd.read_csv(io.StringIO(content)))
    # classified industry merge
    merged_industry_df = pd.concat(all_dfs).drop_duplicates(subset=['NAME'], keep='last')
    # failed data : Null + '기타' -> drop
    merged_industry_df = merged_industry_df.dropna(subset=['INDUSTRY'])
    merged_industry_df = merged_industry_df[merged_industry_df['INDUSTRY'] != '기타']

    # LLM이 예시 이름으로 지정하는 문제 발생 : 각 예시에 맞는 산업군으로 mapping
    result_mapping = {
        # 1. 금융/보험
        '핀테크': '금융/보험',
        
        # 2. 커머스/유통
        '쇼핑': '커머스/유통', '의류': '커머스/유통', '의류/잡화': '커머스/유통', '쇼핑/유통': '커머스/유통', '가전/가구 부속품': '커머스/유통', '생활': '커머스/유통',
        
        # 3. 서비스
        '숙박': '서비스', '소프트웨어': '서비스', '플랫폼': '서비스', '꽃배달': '서비스', '부동산/임대': '서비스', '부동산': '서비스',
        
        # 6. 뷰티/헬스
        '건강기능식품': '뷰티/헬스', '건강/헬스': '뷰티/헬스', '헬스/뷰티': '뷰티/헬스', '위생/헬스': '뷰티/헬스',
        
        # 7. F&B/식품
        '카페': 'F&B/식품',
        
        # 8. 가전/가구
        '가구/가전': '가전/가구', '가구': '가전/가구'
    }
    merged_industry_df['INDUSTRY'] = merged_industry_df['INDUSTRY'].replace(result_mapping)
    
    # Not match 8 industry -> drop
    main_categories = ['금융/보험', '커머스/유통', '서비스', '게임', '교육/공공', '뷰티/헬스', 'F&B/식품', '가전/가구']
    merged_industry_df = merged_industry_df[merged_industry_df['INDUSTRY'].isin(main_categories)]
    
    csv_buffer = io.StringIO()
    merged_industry_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    s3_hook.load_string(csv_buffer.getvalue(),
                        FINAL_OUTPUT_KEY, BUCKET_NAME,
                        replace=True)
    
    print(f"Final dataset created with {len(merged_industry_df)} rows. Uploaded to {FINAL_OUTPUT_KEY}")