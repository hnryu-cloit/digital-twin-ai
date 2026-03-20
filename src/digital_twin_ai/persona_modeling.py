"""
persona_modeling.py - Logic for generating persona profiles and stories using Gemini.
"""

import pandas as pd
import numpy as np
import json
import google.generativeai as genai
from typing import List, Dict, Any, Optional


class PersonaManager:
    """Class for generating and managing persona profiles and individual stories."""

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash"):
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
        response = self.model.generate_content(prompt)
        text = response.text.strip()
        
        # Parse JSON block
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        return json.loads(text)

    def generate_individual_stories(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate individual persona stories for a batch of customers."""
        batch_data = batch_df.to_dict(orient='records')
        
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
  }},
  ...
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
        except Exception as e:
            print(f"Error generating stories: {e}")
            return []
