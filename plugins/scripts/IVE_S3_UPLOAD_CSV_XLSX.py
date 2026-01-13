import os
import boto3
import pandas as pd
from io import StringIO

def S3_UPLOAD_CSV_XLSX(
        local_base_path: str, file_names: list,  
        s3_folder: str, bucket_name: str,
        aws_credentials: dict = None, target_columns: list = None):
    
    # aws_credentials -> dict.get -> id, key, region
    s3_client = boto3.client(
        "s3",
        aws_access_key_id = aws_credentials.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = aws_credentials.get("AWS_SECRET_ACCESS_KEY"),
        region_name = aws_credentials.get("AWS_ACCESS_REGION")
    )
    print(f"🚀 [Start] '{s3_folder}' 폴더로 업로드 작업을 시작합니다.")

    # xlsx, csv files -> 1 csv file
    for file_name in file_names:
        # local file location and name
        local_path = os.path.join(local_base_path, file_name)

        if not os.path.exists(local_path):
            print(f"❌ [Error] 파일을 찾을 수 없습니다: {local_path}")
            continue

        try:
            # name + ext -> name/ ext
            name, ext = os.path.splitext(file_name)
            ext = ext.lower()

            df = None

            if ext == ".xlsx":
                print(f"🔄 [Converting] {file_name} -> CSV 변환 중...")
                # read_excel -> memory
                df = pd.read_excel(local_path, engine="openpyxl")
            elif ext == ".csv":
                if target_columns:
                    # read_csv -> memory
                    df = pd.read_csv(local_path, encoding='')
                else:
                    # direct upload_csv
                    target_key = f"{s3_folder}/{file_name}"
                    s3_client.upload_file(local_path, bucket_name, target_key)
                    print(f"✅ [Uploaded] {file_name} -> s3://{bucket_name}/{target_key}")
            else:
                print(f"⚠️ [Skip] 지원하지 않는 파일 형식입니다: {file_name}")
                continue

            if df is not None:
                if target_columns:
                    valid_cols = [col for col in target_columns if col in df.columns]
                    if valid_cols:
                        # columns filtering
                        df = df[valid_cols]
                        print(f"✂️ [Filter] {file_name}: {len(valid_cols)}개 컬럼만 선택됨")
                    else:
                        print(f"⚠️ [Warning] 요청한 컬럼이 파일에 하나도 없어서 전체를 업로드합니다.")
                
                # memory -> to_csv -> s3
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")

                target_key = f"{s3_folder}/{name}.csv"
                s3_client.put_object(
                    Bucket = bucket_name,
                    Key = target_key,
                    Body = csv_buffer.getvalue()
                )
                print(f"✅ [Uploaded] {file_name} -> s3://{bucket_name}/{target_key}")

        except Exception as e:
            print(f"❌ [Fail] {file_name} 업로드 실패: {e}")
            # raise e
        
    print(f"✨ [Done] '{s3_folder}' 작업 완료.\n")