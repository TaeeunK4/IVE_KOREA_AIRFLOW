import pandas as pd
import io
from google import genai
from google.genai import types
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable

GEMINI_API_KEY = Variable.get("GOOGLE_AI_API_KEY")

PROMPT_TEMPLATE = '''
당신은 광고 데이터 분류 전문가입니다. 
제공된 [names_batch]의 'NAME'을 바탕으로 실제 비즈니스 성격을 파악하여 'INDUSTRY'를 지정하십시오.

[데이터 정제 규칙]
- 'NAME'에 포함된 "[정답입력]", "맞추기", "날짜(2025...)", "검색 후..." 등의 수식어는 무시하고 "핵심 아이템 이름"에만 집중하십시오.

[카테고리 분류 상세 가이드]
1. 금융/보험
2. 커머스/유통
3. 서비스
4. 게임
5. 교육/공공
6. 뷰티/헬스
7. F&B/식품
8. 가전/가구
9. 기타: 위 범주에 없거나 업체 확인이 불가능한 경우

[판단 지침]
- (중요) 'INDUSTRY' 분류 가이드에 있는 9가지 종류에서 무조건 하나를 지정해야 합니다.

[출력 형식]
- 출력은 오직 'ADS_IDX' | 'NAME' | 'INDUSTRY' 형식만 허용합니다.
- 서론, 결론, 설명은 절대 금지하며 데이터 행만 출력하십시오.

[names_batch]
{names_list}
'''
def classify_industry_1(BUCKET_NAME: str, S3_KEY: str, OUTPUT_S3_KEY: str):
    
    # 1. S3에서 원본 CSV 파일 읽기
    s3_hook = S3Hook(aws_conn_id='AWS_CON')
    file_obj = s3_hook.get_key(S3_KEY, BUCKET_NAME)
    file_content = file_obj.get()['Body'].read().decode('utf-8')
    
    df = pd.read_csv(io.StringIO(file_content))
    adv_names = df['NAME'].tolist()

    # 2. Gemini용 프롬프트 준비
    input_rows = []
    for _, row in df.iterrows():
        input_rows.append(f"{row['ADS_IDX']} | {row['NAME']}")
    names_list_str = "\n".join(input_rows)
    final_prompt = PROMPT_TEMPLATE.format(names_list=names_list_str)

    # 3. Gemini API 호출 (Google Search 사용)
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=final_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
    )
    
    # 4. 결과 텍스트 파싱 (NAME | INDUSTRY -> DataFrame)
    result_text = response.text.strip()
    result_data = []

    for line in result_text.split('\n'):
        # 구분선(---), 빈 줄, 또는 헤더(ADS_IDX 포함 줄)는 제외하고 데이터만 추출
        if '|' in line and '---' not in line and 'ADS_IDX' not in line:
            parts = [p.strip() for p in line.split('|')]
            # 정확히 3개의 컬럼(ID, NAME, INDUSTRY)이 있을 때만 수집
            if len(parts) == 3:
                result_data.append(parts)

    # 결과 데이터프레임 생성
    result_df = pd.DataFrame(result_data, columns=['ADS_IDX', 'NAME', 'INDUSTRY'])

    # 5. 결과를 CSV 문자열로 변환하여 S3에 저장
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    
    s3_hook.load_string(
        string_data=csv_buffer.getvalue(),
        key=OUTPUT_S3_KEY,
        bucket_name=BUCKET_NAME,
        replace=True  # 동일 파일명이 있을 경우 덮어쓰기
    )
    
    print(f"Successfully saved classification results to s3://{BUCKET_NAME}/{OUTPUT_S3_KEY}")