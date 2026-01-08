import pandas as pd
import io
from google import genai
from google.genai import types
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable

PROMPT_TEMPLATE = '''
당신은 광고 데이터 분류 전문가입니다. 
제공된 [names_batch]의 'NAME'을 검색하고, 실제 비즈니스 성격을 파악하여 'INDUSTRY'를 지정하십시오.

[데이터 정제 규칙]
- 'NAME'에 포함된 "[정답입력]", "맞추기", "날짜(2025...)", "검색 후..." 등의 수식어는 무시하고 "핵심 아이템 이름"에만 집중하십시오.

['INDUSTRY' 분류 가이드]
1. 금융/보험: 은행, 카드, 증권, 보험, 핀테크
2. 커머스/유통: 의류, 잡화, 액세서리(폰케이스, 매트, 필름, 라이너), 가전/가구의 부속품 및 소모품, 오픈마켓, 쇼핑
3. 서비스: 전문직(법률, 세무), 배달(꽃배달), 숙박, 플랫폼, 소프트웨어, 오프라인 체험 장소
4. 게임: 게임 타이틀 그 자체(모바일/PC 게임), 게임 배급사
5. 교육/공공: 학원, 인강, 정부/지자체 사업, 자격증
6. 뷰티/헬스: 화장품, 건강기능식품, 병원, 다이어트, 위생용품(생리대, 기저귀)
7. F&B/식품: 식재료(쌀, 고기), 프랜차이즈 음식점, 카페, 가공식품, 음료
8. 가전/가구: 가전제품 본체, 자동차 본체, 완제품 가구, PC 본체(조립PC 포함), 제조 장비
9. 기타: 위 범주에 없거나 업체 확인이 불가능한 경우

[판단 우선순위 지침]
- (중요) 'INDUSTRY' 분류 가이드에 있는 9가지 종류에서 무조건 하나를 지정해야 합니다. 이외의 'INDUSTRY'를 지정해서는 안 됩니다.
- (중요) '게임용PC'는 게임이 아니라 [가전/가구]입니다.
- (중요) '보드게임카페'는 게임이 아니라 오프라인 [서비스]입니다.
- (중요) '자동차 매트/키링/필름'은 자동차가 아니라 [커머스/유통]입니다.
- (중요) '법무법인/변호사'는 [서비스]입니다.

[출력 형식 및 절대 원칙]
- 출력은 오직 'ADS_IDX' | 'NAME' | 'INDUSTRY' 형식만 허용합니다.
- 전달받은 'ADS_IDX'를 절대 수정하거나 생략하지 마십시오. 반드시 'NAME'과 짝을 맞춰 그대로 출력하십시오.
- 서론, 결론, 설명은 절대 금지하며 데이터 행만 출력하십시오.

['INDUSTRY' 분류 및 출력 예시]
1 | [정답입력]첨단 파스타 맛집(3) | F&B/식품
2 | 삼성온라인공식몰 판매자더보기 클릭후 2080 맞추기 12.2 | 가전/제조
3 | [정답입력]원주꽃집 꽃배달 | 서비스
4 | 플레이도쿠: 블록 퍼즐 게임 | 게임
5 | 건설기초안전교육 (플레이스 보고 퀴즈 맞추기)_203403 | 교육/공공
6 | KB PAY 연말쇼핑 총알 장전! | 금융/보험
7 | [정답입력]  미추홀구정형외과 | 뷰티/헬스
8 | [정답입력] 택배테이프 1개 | 커머스/유통

[names_batch]
{names_list}
'''
# s3 temp file|folder delete
def s3_temp_delete(BUCKET_NAME: str, TEMPS: list):
    # s3 connect
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    # delte TEMPS file|folder
    for temp in TEMPS:
        keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix=temp)
        if keys:
            s3_hook.delete_objects(bucket=BUCKET_NAME, keys=keys)
            print(f"Successfully deleted {len(keys)} objects from s3://{BUCKET_NAME}/{temp}")
        else:
            print(f"No objects found to delete in s3://{BUCKET_NAME}/{temp}")

# s3 -> split by batch_size -> file + params s3 upload
def split_prior_classify(BUCKET_NAME: str, S3_KEY: str, BATCH_SIZE: int,
                            TEMP_INPUT_DIR: str, TEMP_OUTPUT_DIR: str, UNIQUE_MASTER_KEY: str):
    # s3 connect
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    file_obj = s3_hook.get_key(S3_KEY, BUCKET_NAME)
    # clean data load
    df = pd.read_csv(io.StringIO(file_obj.get()['Body'].read().decode('utf-8')))
    # only index + name/ index -> protect loss name : unique_master_key
    unique_df = df[['NAME']].drop_duplicates().reset_index(drop=True)
    unique_df['ADS_IDX'] = unique_df.index + 1 
    # unique_master_key s3 upload
    master_buffer = io.StringIO()
    unique_df.to_csv(master_buffer, index=False, encoding='utf-8-sig')
    s3_hook.load_string(master_buffer.getvalue(),
                        UNIQUE_MASTER_KEY, BUCKET_NAME,
                        replace=True)
    # split by batch_size
    CLASSIFY_REF_PARAMS = []
    for i in range(0, len(unique_df), BATCH_SIZE):
        batch_df = unique_df.iloc[i:i+BATCH_SIZE]
        BATCH_IDX = i // BATCH_SIZE
        
        CLASSIFY_IN_KEY = f"{TEMP_INPUT_DIR}batch_{BATCH_IDX}.csv"
        CLASSIFY_OUT_KEY = f"{TEMP_OUTPUT_DIR}batch_{BATCH_IDX}_res.csv"
        
        # splited file s3 upload
        csv_buffer = io.StringIO()
        batch_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        s3_hook.load_string(csv_buffer.getvalue(),
                            CLASSIFY_IN_KEY, BUCKET_NAME,
                            replace=True)
        # save classify_mapped_params
        CLASSIFY_REF_PARAMS.append({
            "op_kwargs": {
                "BUCKET_NAME": BUCKET_NAME,
                "S3_KEY": CLASSIFY_IN_KEY,
                "OUTPUT_S3_KEY": CLASSIFY_OUT_KEY
            }
        })
    return CLASSIFY_REF_PARAMS

# s3 splited_data -> classify -> s3 upload
def classify_industry_3(BUCKET_NAME: str, S3_KEY: str, OUTPUT_S3_KEY: str):
    # airflow variables -> get api key
    GEMINI_API_KEY = Variable.get("GOOGLE_AI_API_KEY")
    # s3 connect
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    file_obj = s3_hook.get_key(S3_KEY, BUCKET_NAME)
    file_content = file_obj.get()['Body'].read().decode('utf-8')
    # splited data load
    df = pd.read_csv(io.StringIO(file_content))

    # data preprocessing for gemini
    input_rows = []
    for _, row in df.iterrows():
        input_rows.append(f"{row['ADS_IDX']} | {row['NAME']}")
    names_list_str = "\n".join(input_rows)
    # load prompt
    final_prompt = PROMPT_TEMPLATE.format(names_list=names_list_str)

    # run gemini api
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=final_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
            top_p=0.95
        )
    )
    
    # result_text parsing
    result_text = response.text.strip()
    result_data = []
    for line in result_text.split('\n'):
        # delete '|' and '---', unnecessary 'ADS_IDX'
        if '|' in line and '---' not in line and 'ADS_IDX' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 3:
                result_data.append(parts)
    result_df = pd.DataFrame(result_data, columns=['ADS_IDX', 'NAME', 'INDUSTRY'])

    # result_df -> s3 upload
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    s3_hook.load_string(
        csv_buffer.getvalue(),
        OUTPUT_S3_KEY, BUCKET_NAME,
        replace=True
    )
    
    print(f"Successfully saved classification results to s3://{BUCKET_NAME}/{OUTPUT_S3_KEY}")

# s3 classified data -> merge with master_key -> name/industry : for merge with list data
def merge_after_classify(BUCKET_NAME: str, TEMP_OUTPUT_DIR: str,
                  CLASSIFIED_OUTPUT_KEY: str, UNIQUE_MASTER_KEY: str):
    # s3 connect
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    # result concat
    keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix=TEMP_OUTPUT_DIR)
    all_res_dfs = [pd.read_csv(io.StringIO(s3_hook.read_key(k, BUCKET_NAME))) for k in keys if k.endswith('.csv')]
    gemini_df = pd.concat(all_res_dfs)
    # protect duplicated
    gemini_df = gemini_df.drop_duplicates(subset=['ADS_IDX'])
    # unique_master_key load
    master_content = s3_hook.read_key(UNIQUE_MASTER_KEY, BUCKET_NAME)
    master_df = pd.read_csv(io.StringIO(master_content))
    # protect loss name by unique_master_key
    mapped_industry = pd.merge(
        master_df[['ADS_IDX', 'NAME']], 
        gemini_df[['ADS_IDX', 'INDUSTRY']], 
        on='ADS_IDX', 
        how='left'
    )
    # mapped_industry -> s3 upload
    csv_buffer = io.StringIO()
    mapped_industry[['NAME', 'INDUSTRY']].to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    s3_hook.load_string(csv_buffer.getvalue(),
                        CLASSIFIED_OUTPUT_KEY, BUCKET_NAME,
                        replace=True)