import os
from dotenv import load_dotenv

load_dotenv()

GCP_PROJECT_ID  = os.getenv("GCP_PROJECT_ID")
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
GCP_REGION      = os.getenv("GCP_REGION", "asia-northeast3")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.0-flash")

# Backend API — AI 파이프라인이 여기서 데이터를 가져온다
BACKEND_API_URL      = os.getenv("BACKEND_API_URL", "http://localhost:8000")
BACKEND_DATA_DIR     = os.getenv("BACKEND_DATA_DIR", "../digital-twin-backend/data")

# 출력 경로
OUTPUT_DIR            = "./output"
SYNTHETIC_OUTPUT_PATH = f"{OUTPUT_DIR}/synthetic_customers.parquet"
PERSONAS_OUTPUT_PATH  = f"{OUTPUT_DIR}/personas.json"

# 하위 호환: Excel 경로 (00_generate_dummy_excel.py 전용)
DATA_DIR   = "./data"
EXCEL_PATH = f"{DATA_DIR}/Digital Customer Twin.xlsx"

# 클러스터링 설정
N_PERSONAS             = 7
RANDOM_STATE           = 42
SYNTHETIC_N_CUSTOMERS  = 1000