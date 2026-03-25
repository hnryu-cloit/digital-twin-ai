"""
persona_modeling.py - Logic for generating persona profiles and stories using Gemini.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import google.generativeai as genai
import pandas as pd


class PersonaManager:
    """Class for generating and managing persona profiles and individual stories."""

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash"):
        self.model = None
        self.model_name = model_name
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)

    def extract_cluster_stats(self, df: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
        """Extract representative stats for a cluster."""
        cluster = df[df["persona_cluster"] == cluster_id]

        return {
            "cluster_id": int(cluster_id),
            "size": int(len(cluster)),
            "avg_age": float(round(cluster["usr_age"].mean(), 1)),
            "gender_ratio_male": float(round((cluster["usr_gndr"] == "M").mean(), 2)),
            "avg_ltv": float(round(cluster["ltv_r"].mean(), 0)),
            "avg_retention": float(round(cluster["retention_score"].mean(), 3)),
            "avg_purchase_count": float(round(cluster["pchs_cnt"].mean(), 1)),
            "avg_premium_count": float(round(cluster["premium_cnt"].mean(), 1)),
            "repurchase_rate": float(round(cluster["cum_repchs_flg"].mean(), 2)),
            "health_app_rate": float(round(cluster["samsunghealth_flag"].mean(), 2)),
            "wallet_rate": float(round(cluster["samsungwallet_flag"].mean(), 2)),
            "smartthings_rate": float(round(cluster["smartthings_flag"].mean(), 2)),
            "avg_game_ratio": float(round(cluster["app_game_ratio"].mean(), 2)),
            "avg_social_ratio": float(round(cluster["app_social_ratio"].mean(), 2)),
            "avg_daily_usage_min": float(round(cluster["avg_daily_usage_min"].mean(), 0)),
            "top_region": str(cluster["usr_cnty_ap2"].mode()[0]) if len(cluster) > 0 else "Unknown",
        }

    def generate_persona_profile(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a high-level persona profile using Gemini."""
        prompt = f"""
다음은 삼성 디지털 트윈 시스템의 고객 클러스터 통계입니다.
이 데이터를 바탕으로 마케팅 리서치용 페르소나 프로파일을 JSON 형식으로 생성해주세요.

클러스터 통계:
{json.dumps(stats, ensure_ascii=False, indent=2)}

다음 JSON 형식으로 반환해주세요 (다른 텍스트 없이 JSON만):
{{
  "cluster_id": {stats['cluster_id']},
  "persona_name": "페르소나 이름 (예: MZ 얼리어답터)",
  "persona_name_en": "Persona Name in English",
  "age_range": "예: 25-35",
  "description": "페르소나 설명 2-3문장",
  "key_characteristics": ["특성1", "특성2", "특성3"],
  "purchase_intent": {round(stats['avg_ltv'] / 10000, 1)},
  "brand_attitude": {round(stats['avg_retention'] * 100, 1)},
  "marketing_acceptance": {round(stats['repurchase_rate'] * 100, 1)},
  "future_value": {round(min(stats['avg_ltv'] / 200, 100), 1)},
  "preferred_channel": "SNS 숏폼 | 영상 캠페인 | 텍스트 브리핑 중 하나",
  "keywords": ["키워드1", "키워드2", "키워드3"],
  "interests": ["관심사1", "관심사2", "관심사3"],
  "segment_tags": ["태그1", "태그2"],
  "churn_risk": {round((1 - stats['avg_retention']) * 100, 1)},
  "size": {stats['size']}
}}
"""
        if self.model is None:
            return self._build_fallback_persona(stats)

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            return json.loads(text)
        except Exception:
            return self._build_fallback_persona(stats)

    def generate_individual_stories(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate individual persona stories for a batch of customers."""
        if self.model is None:
            return []

        batch_data = batch_df.to_dict(orient="records")

        prompt = f"""
다음은 가상 고객 중 일부의 수치 데이터입니다.
각 데이터 행을 바탕으로 삼성 디지털 트윈 시스템에서 사용할 수 있는 '살아있는 가상 페르소나' 상세 정보를 생성해주세요.

각 페르소나에 대해 다음 정보를 추가해주세요:
1. 이름(가명)
2. 직업 또는 라이프스타일 역할
3. 삼성 제품 사용 소감 (특히 가전과 모바일의 연결성 위주)
4. 성격 및 가치관 (1-2문장)

입력 데이터:
{json.dumps(batch_data, ensure_ascii=False, indent=1)}

출력 형식 (JSON 리스트만 반환):
[
  {{
    "index": 고객인덱스,
    "name": "성함",
    "job": "직업",
    "personality": "성격 설명",
    "samsung_experience": "삼성 제품 연결 경험 및 소감"
  }}
]
"""
        try:
            response = self.model.generate_content(prompt)
            content = response.text.strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except Exception:
            return []

    def _build_fallback_persona(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        age_value = int(round(stats["avg_age"]))
        age_start = max(18, age_value - 5)
        age_end = min(70, age_value + 5)

        if stats["avg_game_ratio"] >= 0.25:
            persona_name = "디지털 엔터테인먼트 헤비유저"
            persona_name_en = "Digital Entertainment Power User"
            preferred_channel = "영상 캠페인"
            keywords = ["엔터테인먼트", "몰입감", "고성능"]
        elif stats["smartthings_rate"] >= 0.3:
            persona_name = "커넥티드 라이프 실용층"
            persona_name_en = "Connected Life Pragmatist"
            preferred_channel = "텍스트 브리핑"
            keywords = ["스마트홈", "연결성", "실용성"]
        else:
            persona_name = "밸런스형 갤럭시 사용자"
            persona_name_en = "Balanced Galaxy User"
            preferred_channel = "SNS 숏폼"
            keywords = ["편의성", "브랜드 신뢰", "일상 활용"]

        gender_ratio = stats["gender_ratio_male"]
        if gender_ratio >= 0.55:
            gender_hint = "남성 비중이 높은"
        elif gender_ratio <= 0.45:
            gender_hint = "여성 비중이 높은"
        else:
            gender_hint = "성별 구성이 비교적 균형적인"

        return {
            "cluster_id": int(stats["cluster_id"]),
            "persona_name": persona_name,
            "persona_name_en": persona_name_en,
            "age_range": f"{age_start}-{age_end}",
            "description": (
                f"{stats['top_region']} 중심의 {gender_hint} 사용자군으로, "
                f"평균 연령은 {stats['avg_age']}세이며 재구매와 일상 활용성이 함께 관측됩니다."
            ),
            "key_characteristics": [
                f"평균 구매 횟수 {stats['avg_purchase_count']}",
                f"평균 프리미엄 구매 {stats['avg_premium_count']}",
                f"평균 일 사용시간 {stats['avg_daily_usage_min']}분",
            ],
            "purchase_intent": round(min(max(stats["avg_ltv"] / 10000, 0), 100), 1),
            "brand_attitude": round(min(max(stats["avg_retention"] * 100, 0), 100), 1),
            "marketing_acceptance": round(min(max(stats["repurchase_rate"] * 100, 0), 100), 1),
            "future_value": round(min(stats["avg_ltv"] / 200, 100), 1),
            "preferred_channel": preferred_channel,
            "keywords": keywords,
            "interests": [
                f"게임 비중 {stats['avg_game_ratio']}",
                f"소셜 비중 {stats['avg_social_ratio']}",
                f"삼성 Health 사용률 {stats['health_app_rate']}",
            ],
            "segment_tags": [stats["top_region"], preferred_channel, f"cluster-{stats['cluster_id']}"],
            "churn_risk": round((1 - stats["avg_retention"]) * 100, 1),
            "size": int(stats["size"]),
        }
