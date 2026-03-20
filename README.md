# digital-twin-ai

> **⚠️ 전역 시스템 제약조건 및 코드 컨벤션**
> 본 프로젝트는 엔터프라이즈 B2B SaaS 아키텍처를 지향하며, 엄격한 코드 컨벤션을 따릅니다.
> 자세한 사항은 루트 디렉토리의 [`docs/conventions.md`](../docs/conventions.md)를 반드시 확인하세요.
> 주요 AI 파이프라인 제약: **모듈형 패키지 구조, Ruff & Mypy Strict, 비동기 호출을 위한 진입점(Entrypoint) 기반 실행**

Samsung Digital Customer Twin — ML 페르소나 모델링 파이프라인

## 개요

실제 고객 데이터의 분포를 학습해 합성 데이터를 생성하고, UMAP + K-Means로 페르소나 클러스터를 도출한다.
Gemini (e.g., Gemini-3-Flash)가 각 클러스터를 마케팅 리서치용 페르소나 프로파일로 변환한다.

## 기술 스택

- Python 3.12+ / pandas / numpy / scikit-learn / pydantic
- **UMAP** — 고차원 피처 2D 압축
- **K-Means** — 페르소나 클러스터링
- **Gemini** — 페르소나 설명 및 스토리 생성
- GCP (Cloud Storage, BigQuery)

## 패키지 구조 (`src/digital_twin_ai/`)

- `data_generation.py`: 더미 엑셀 및 합성 데이터 생성 로직
- `feature_engineering.py`: 정규화 및 피처 벡터 구성
- `clustering.py`: UMAP 차원 축소 및 K-Means 클러스터링
- `persona_modeling.py`: Gemini 기반 페르소나 프로파일 및 스토리 생성
- `pipeline.py`: 전체 파이프라인 통합 실행 (Entrypoint)

## 시작하기

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # GEMINI_API_KEY, GCP 설정
```

### 파이프라인 실행 (Python)

```python
from digital_twin_ai import run_pipeline

config = {
    "random_state": 42,
    "n_synthetic_customers": 1000,
    "n_personas": 7,
    "excel_path": "./data/Digital Customer Twin.xlsx",
    "output_dir": "./output",
    "gemini_api_key": "YOUR_API_KEY",
    "gemini_model_name": "gemini-3-flash"
}

results = run_pipeline(config)
print(f"Pipeline results: {results}")
```

최종 출력: `./output/personas.json`, `./output/clustered_customers.parquet` 등

## 설정 (`config/gcp_config.py`)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `N_PERSONAS` | 7 | 생성할 페르소나 수 |
| `SYNTHETIC_N_CUSTOMERS` | 1000 | 합성 고객 수 |
| `RANDOM_STATE` | 42 | 재현성 시드 |

## 개발 도구

- **Ruff**: Linting 및 코드 스타일 가이드 준수
- **Mypy**: 정적 타입 체크 (Strict 모드 권장)
