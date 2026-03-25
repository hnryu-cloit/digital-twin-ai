"""
persona_modeling.py - Logic for generating persona profiles and stories using Gemini.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List

import google.generativeai as genai
import pandas as pd


def _safe_float(val: Any, default: float = 0.0) -> float:
    if pd.isna(val) or isinstance(val, float) and math.isnan(val):
        return default
    return float(val)


class PersonaManager:
    """Class for generating and managing persona profiles and individual stories."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.0-flash"):
        self.model = None
        self.model_name = model_name
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction="당신은 삼성 디지털 트윈 시스템의 마케팅 리서치 전문가입니다. 통계 데이터를 바탕으로 정교한 페르소나 프로파일을 생성하며, 반드시 지정된 JSON 형식을 엄수합니다.",
                generation_config={"response_mime_type": "application/json"}
            )

    def extract_cluster_stats(self, df: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
        """Extract representative stats for a cluster."""
        cluster = df[df["persona_cluster"] == cluster_id]

        def get_mean(col: str, rnd: int = 2) -> float:
            if cluster.empty or col not in cluster:
                return 0.0
            return _safe_float(round(cluster[col].mean(), rnd))

        def get_mode(col: str) -> str:
            if cluster.empty or col not in cluster or cluster[col].mode().empty:
                return "Unknown"
            return str(cluster[col].mode()[0])

        gender_ratio_male = 0.5
        if not cluster.empty and "usr_gndr" in cluster:
            gender_ratio_male = _safe_float(round((cluster["usr_gndr"] == "M").mean(), 2))

        return {
            "cluster_id": int(cluster_id),
            "size": int(len(cluster)),
            "avg_age": get_mean("usr_age", 1),
            "gender_ratio_male": gender_ratio_male,
            "avg_ltv": get_mean("ltv_r", 0),
            "avg_retention": get_mean("retention_score", 3),
            "avg_purchase_count": get_mean("pchs_cnt", 1),
            "avg_premium_count": get_mean("premium_cnt", 1),
            "repurchase_rate": get_mean("cum_repchs_flg", 2),
            "health_app_rate": get_mean("samsunghealth_flag", 2),
            "wallet_rate": get_mean("samsungwallet_flag", 2),
            "smartthings_rate": get_mean("smartthings_flag", 2),
            "avg_game_ratio": get_mean("app_game_ratio", 2),
            "avg_social_ratio": get_mean("app_social_ratio", 2),
            "avg_daily_usage_min": get_mean("avg_daily_usage_min", 0),
            "top_region": get_mode("usr_cnty_ap2"),
        }

    def generate_persona_profile(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a high-level persona profile using Gemini."""
        prompt = f"""
다음은 삼성 디지털 트윈 시스템의 고객 클러스터 통계입니다.
이 데이터를 바탕으로 마케팅 리서치용 페르소나 프로파일을 생성해주세요.

클러스터 통계:
{json.dumps(stats, ensure_ascii=False, indent=2)}

다음 JSON 형식 스키마를 엄격하게 준수하여 응답하세요:
{{
  "cluster_id": {stats.get('cluster_id', 0)},
  "persona_name": "페르소나 이름 (예: MZ 얼리어답터)",
  "persona_name_en": "Persona Name in English",
  "age_range": "예: 25-35",
  "description": "페르소나 설명 2-3문장",
  "key_characteristics": ["특성1", "특성2", "특성3"],
  "purchase_intent": {round(_safe_float(stats.get('avg_ltv', 0)) / 10000, 1)},
  "brand_attitude": {round(_safe_float(stats.get('avg_retention', 0)) * 100, 1)},
  "marketing_acceptance": {round(_safe_float(stats.get('repurchase_rate', 0)) * 100, 1)},
  "future_value": {round(min(_safe_float(stats.get('avg_ltv', 0)) / 200, 100), 1)},
  "preferred_channel": "SNS 숏폼 | 영상 캠페인 | 텍스트 브리핑 중 하나",
  "keywords": ["키워드1", "키워드2", "키워드3"],
  "interests": ["관심사1", "관심사2", "관심사3"],
  "segment_tags": ["태그1", "태그2"],
  "churn_risk": {round((1 - _safe_float(stats.get('avg_retention', 0))) * 100, 1)},
  "size": {stats.get('size', 0)}
}}
"""
        if self.model is None:
            return self._build_fallback_persona(stats)

        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text.strip())
        except Exception as e:
            print(f"Persona Generation Error: {e}")
            return self._build_fallback_persona(stats)

    def generate_individual_stories(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate individual persona stories for a batch of customers."""
        if self.model is None or batch_df.empty:
            return []

        # Replace NaN with 0 or empty string safely before creating dict
        safe_df = batch_df.copy()
        for col in safe_df.columns:
            if safe_df[col].dtype == "float64":
                safe_df[col] = safe_df[col].fillna(0.0)
            elif safe_df[col].dtype == "object":
                safe_df[col] = safe_df[col].fillna("")
        batch_data = safe_df.to_dict(orient="records")

        prompt = f"""
다음은 가상 고객 중 일부의 수치 데이터입니다.
각 데이터 행을 바탕으로 삼성 디지털 트윈 시스템에서 사용할 수 있는 '살아있는 가상 페르소나' 상세 정보를 생성해주세요.

입력 데이터:
{json.dumps(batch_data, ensure_ascii=False, indent=1)}

출력 형식 스키마 (배열 형태의 JSON):
[
  {{
    "index": 고객인덱스,
    "name": "성함(가명)",
    "job": "직업 또는 라이프스타일 역할",
    "personality": "성격 및 가치관 (1-2문장)",
    "samsung_experience": "삼성 제품 연결 경험 및 소감 (가전과 모바일의 연결성 위주)"
  }}
]
"""
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text.strip())
        except Exception as e:
            print(f"Story Generation Error: {e}")
            return []

    def _build_fallback_persona(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        age_value = int(round(_safe_float(stats.get("avg_age", 30))))
        age_start = max(18, age_value - 5)
        age_end = min(70, age_value + 5)

        avg_game_ratio = _safe_float(stats.get("avg_game_ratio", 0))
        smartthings_rate = _safe_float(stats.get("smartthings_rate", 0))
        gender_ratio_male = _safe_float(stats.get("gender_ratio_male", 0.5))

        if avg_game_ratio >= 0.25:
            persona_name = "디지털 엔터테인먼트 헤비유저"
            persona_name_en = "Digital Entertainment Power User"
            preferred_channel = "영상 캠페인"
            keywords = ["엔터테인먼트", "몰입감", "고성능"]
        elif smartthings_rate >= 0.3:
            persona_name = "커넥티드 라이프 실용층"
            persona_name_en = "Connected Life Pragmatist"
            preferred_channel = "텍스트 브리핑"
            keywords = ["스마트홈", "연결성", "실용성"]
        else:
            persona_name = "밸런스형 갤럭시 사용자"
            persona_name_en = "Balanced Galaxy User"
            preferred_channel = "SNS 숏폼"
            keywords = ["편의성", "브랜드 신뢰", "일상 활용"]

        if gender_ratio_male >= 0.55:
            gender_hint = "남성 비중이 높은"
        elif gender_ratio_male <= 0.45:
            gender_hint = "여성 비중이 높은"
        else:
            gender_hint = "성별 구성이 비교적 균형적인"

        top_region = stats.get("top_region", "Unknown")
        avg_purchase_count = _safe_float(stats.get("avg_purchase_count", 0))
        avg_premium_count = _safe_float(stats.get("avg_premium_count", 0))
        avg_daily_usage_min = _safe_float(stats.get("avg_daily_usage_min", 0))
        avg_ltv = _safe_float(stats.get("avg_ltv", 0))
        avg_retention = _safe_float(stats.get("avg_retention", 0))
        repurchase_rate = _safe_float(stats.get("repurchase_rate", 0))
        avg_social_ratio = _safe_float(stats.get("avg_social_ratio", 0))
        health_app_rate = _safe_float(stats.get("health_app_rate", 0))

        return {
            "cluster_id": int(stats.get("cluster_id", 0)),
            "persona_name": persona_name,
            "persona_name_en": persona_name_en,
            "age_range": f"{age_start}-{age_end}",
            "description": (
                f"{top_region} 중심의 {gender_hint} 사용자군으로, "
                f"평균 연령은 {age_value}세이며 재구매와 일상 활용성이 함께 관측됩니다."
            ),
            "key_characteristics": [
                f"평균 구매 횟수 {avg_purchase_count}",
                f"평균 프리미엄 구매 {avg_premium_count}",
                f"평균 일 사용시간 {avg_daily_usage_min}분",
            ],
            "purchase_intent": round(min(max(avg_ltv / 10000, 0), 100), 1),
            "brand_attitude": round(min(max(avg_retention * 100, 0), 100), 1),
            "marketing_acceptance": round(min(max(repurchase_rate * 100, 0), 100), 1),
            "future_value": round(min(avg_ltv / 200, 100), 1),
            "preferred_channel": preferred_channel,
            "keywords": keywords,
            "interests": [
                f"게임 비중 {avg_game_ratio}",
                f"소셜 비중 {avg_social_ratio}",
                f"삼성 Health 사용률 {health_app_rate}",
            ],
            "segment_tags": [top_region, preferred_channel, f"cluster-{stats.get('cluster_id', 0)}"],
            "churn_risk": round((1 - avg_retention) * 100, 1),
            "size": int(stats.get("size", 0)),
        }
