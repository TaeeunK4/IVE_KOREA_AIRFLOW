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
1. 금융/보험: 은행, 카드, 증권, 보험, 핀테크
2. 커머스/유통: 의류, 잡화, 액세서리(폰케이스, 매트, 필름, 라이너), 가전/가구의 부속품 및 소모품, 오픈마켓
3. 서비스: 전문직(법률, 세무), 배달(꽃배달), 숙박, 플랫폼, 소프트웨어, 오프라인 체험 장소(카페, 노래방)
4. 게임: 게임 타이틀 그 자체(모바일/PC 게임), 게임 배급사
5. 교육/공공: 학원, 인강, 정부/지자체 사업, 자격증
6. 뷰티/헬스: 화장품, 건강기능식품, 병원, 다이어트, 위생용품(생리대, 기저귀)
7. F&B/식품: 식재료(쌀, 고기), 프랜차이즈 음식점, 가공식품, 음료
8. 가전/가구: 가전제품 본체, 자동차 본체, 완제품 가구, PC 본체(조립PC 포함)
9. 기타: 위 범주에 없거나 업체 확인이 불가능한 경우

[판단 우선순위 지침]
- (중요) 'INDUSTRY' 분류 가이드에 있는 9가지 종류에서 무조건 하나를 지정해야 합니다.
- (중요) '게임용PC'는 게임이 아니라 [가전/가구]입니다.
- (중요) '보드게임카페'는 게임이 아니라 오프라인 [서비스]입니다.
- (중요) '자동차 매트/키링/필름'은 자동차가 아니라 [커머스/유통]입니다.
- (중요) '법무법인/변호사'는 [서비스]입니다.

[출력 형식]
- 출력은 오직 'ADS_IDX' | 'NAME' | 'INDUSTRY' 형식만 허용합니다.
- 서론, 결론, 설명은 절대 금지하며 데이터 행만 출력하십시오.

[names_batch]
{names_list}
'''
def classify_industry_2(BUCKET_NAME: str, S3_KEY: str, OUTPUT_S3_KEY: str):
    
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