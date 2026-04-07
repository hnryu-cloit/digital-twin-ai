"""
Centralized prompt templates for all Gemini AI calls.
"""
from __future__ import annotations

import json


# ── 설문 설계 ──────────────────────────────────────────────────────────────────

SURVEY_DESIGN_SYSTEM = (
    "당신은 삼성 디지털 트윈 시스템의 수석 마케팅 리서치 전문가입니다. "
    "세그먼트 분석 결과와 리서치 목적을 깊이 이해하고, "
    "타겟 페르소나의 심리와 행동 패턴을 반영한 최적의 설문 문항을 설계합니다. "
    "반드시 지정된 JSON 형식만 출력합니다."
)


def build_survey_prompt(
    project_purpose: str,
    survey_type: str,
    question_count: int,
    segment_info: list[dict],
    filter_summary: str,
    target_count: int,
    template: dict,
    user_prompt: str,
) -> str:
    segments_text = "\n".join(
        f"  - {s.get('name', '')}: {s.get('count', 0)}명"
        for s in segment_info
    ) or "  - 전체 모집단"

    required_blocks = ", ".join(template.get("required_blocks", [])) or "자유 구성"
    template_id = template.get("template_id", "")

    return f"""다음 리서치 맥락에 맞는 설문 문항 {question_count}개를 설계하세요.

## 리서치 목적
{project_purpose}

## 추가 요청사항
{user_prompt}

## 타겟 세그먼트 (세그먼트 분석 결과)
필터 조건: {filter_summary}
총 대상: {target_count}명
{segments_text}

## 설문 유형
{survey_type}

## 필수 구성 블록
{required_blocks}
{'(템플릿: ' + template_id + ')' if template_id else ''}

## 출력 형식 (JSON 배열만 출력)
[
  {{
    "text": "문항 텍스트",
    "type": "단일선택|복수선택|리커트척도|주관식",
    "options": ["선택지1", "선택지2"],
    "rationale": "이 문항이 리서치 목적과 타겟 세그먼트에 어떻게 기여하는지 1~2문장",
    "evidence": [
      {{"label": "반영 근거", "value": "구체적 내용"}}
    ]
  }}
]

주의사항:
- 주관식 options는 반드시 []
- 리커트척도 options: ["매우 그렇다", "그렇다", "보통", "아니다", "전혀 아니다"]
- 타겟 세그먼트의 특성(관심사, 구매 패턴, 가치관)을 문항에 직접 반영
- 정확히 {question_count}개 생성"""

# ── 페르소나 시뮬레이션 응답 ──────────────────────────────────────────────────

SIMULATION_SYSTEM = (
    "당신은 삼성 고객 디지털 트윈 시뮬레이터입니다. "
    "주어진 페르소나 프로필을 완전히 체화하여 설문에 응답합니다. "
    "페르소나의 나이, 직업, 세그먼트, 구매 이력, 관심사, 가치관을 철저히 반영해야 합니다. "
    "반드시 지정된 JSON 형식만 출력합니다."
)


def build_simulation_prompt(
    persona: dict,
    questions: list[dict],
    project_purpose: str,
    project_name: str,
) -> str:
    questions_text = "\n".join(
        f"{i+1}. [{q.get('id','q')}] ({q.get('type','단일선택')}) {q.get('text','')}\n"
        f"   선택지: {json.dumps(q.get('options', []), ensure_ascii=False)}"
        for i, q in enumerate(questions)
    )

    return f"""다음 페르소나로서 설문에 응답하세요.

## 페르소나 프로필
- 이름: {persona.get('name', '')}
- 나이: {persona.get('age', '')}세 / 성별: {persona.get('gender', '')}
- 직업: {persona.get('occupation', '')} ({persona.get('occupation_category', '')})
- 세그먼트: {persona.get('segment', '')}
- 거주 지역: {persona.get('region', '')}
- 관심사: {', '.join(persona.get('interests', []))}
- 핵심 키워드: {', '.join(persona.get('keywords', []))}
- 선호 채널: {persona.get('preferred_channel', '')}
- 구매 채널: {persona.get('buy_channel', '')}
- 구매 이력: {', '.join(persona.get('purchase_history', []))}
- 구매 의향 점수: {persona.get('purchase_intent', 0)}/100
- 브랜드 태도 점수: {persona.get('brand_attitude', 0)}/100
- 마케팅 수용도: {persona.get('marketing_acceptance', 0)}/100
- 페르소나 설명: {persona.get('profile', '')}

## 리서치 컨텍스트
프로젝트: {project_name}
목적: {project_purpose}

## 설문 문항
{questions_text}

## 출력 형식 (JSON 배열만 출력, 문항 수와 동일한 개수)
[
  {{
    "question_id": "문항 ID",
    "selected_option": "선택한 답변 (선택지 중 하나, 복수선택은 ', '로 구분, 주관식은 자유 서술)",
    "rationale": "이 페르소나가 이렇게 응답한 이유 2~3문장 (페르소나 특성과 연결)",
    "cot": [
      "사고 과정 1단계",
      "사고 과정 2단계",
      "사고 과정 3단계"
    ],
    "integrity_score": 0.0
  }}
]

주의사항:
- selected_option은 반드시 주어진 선택지 중에서 선택 (주관식 제외)
- cot는 이 페르소나가 어떤 논리로 해당 답변에 도달했는지 3단계로 서술
- integrity_score는 페르소나 프로필과 응답의 일관성 점수 (85~99 사이 float)
- 페르소나의 세그먼트 특성이 응답에 반드시 드러나야 함
- 정확히 {len(questions)}개 응답"""


# ── 분석 결과 리포트 ──────────────────────────────────────────────────────────

REPORT_SYSTEM = (
    "당신은 삼성 마케팅 전략 컨설턴트입니다. "
    "디지털 트윈 시뮬레이션 결과를 분석하여 실행 가능한 전략적 인사이트를 도출합니다. "
    "반드시 지정된 JSON 형식만 출력합니다."
)

REPORT_TYPES = {
    "strategy": "전략 리포트",
    "insight": "핵심 인사이트 리포트",
    "segment": "세그먼트 분석 리포트",
    "summary": "경영진 요약 리포트",
}


def build_report_prompt(
    report_type: str,
    project_name: str,
    project_purpose: str,
    dominant_segment: str,
    persona_count: int,
    response_count: int,
    segment_cards: list[dict],
    top_responses: list[dict],
    keyword_items: list[dict],
    detailed_distribution: list[dict],
    segment_response_summary: dict | None = None,
    persona_response_samples: list[dict] | None = None,
) -> str:
    type_label = REPORT_TYPES.get(report_type, "전략 리포트")

    segments_text = "\n".join(
        f"  - {s.get('segment', s.get('name', ''))}: 구매의향 {s.get('purchase_intent', s.get('avgPurchaseIntent', 'N/A'))}%, "
        f"브랜드태도 {s.get('brand_attitude', s.get('avgBrandAttitude', 'N/A'))}%"
        for s in segment_cards[:5]
    ) or "  - 데이터 없음"

    keywords_text = ", ".join(
        f"{k.get('keyword', '')}({k.get('frequency', 0)})"
        for k in keyword_items[:10]
    ) or "없음"

    # 세그먼트별 응답 패턴 섹션
    response_pattern_text = ""
    if segment_response_summary:
        lines = []
        for seg_name, questions in list(segment_response_summary.items())[:5]:
            lines.append(f"  [{seg_name}]")
            for q_data in list(questions.values())[:3]:
                top_opt = q_data.get("top_option", "")
                rationales = q_data.get("top_rationales", [])
                if top_opt:
                    lines.append(f"    - 최다 선택: '{top_opt}'")
                if rationales:
                    lines.append(f"      근거: {rationales[0][:80]}")
        response_pattern_text = "\n".join(lines) if lines else "  - 데이터 없음"
    else:
        top_responses_text = "\n".join(
            f"  [{r.get('persona_name', '')} / {r.get('segment', '')}] "
            f"Q: {r.get('question_text', '')[:50]} → {r.get('selected_option', '')}"
            for r in top_responses[:10]
        ) or "  - 응답 데이터 없음"
        response_pattern_text = top_responses_text

    # 페르소나 응답 샘플 섹션
    persona_samples_text = ""
    if persona_response_samples:
        lines = []
        for sample in persona_response_samples[:5]:
            name = sample.get("persona_name", "")
            seg = sample.get("segment", "")
            answers = sample.get("answers", [])
            lines.append(f"  {name} ({seg})")
            for ans in answers[:2]:
                opt = ans.get("selected_option", "")
                rationale = (ans.get("rationale") or "")[:80]
                lines.append(f"    → {opt}: {rationale}")
        persona_samples_text = "\n".join(lines)

    persona_section = f"\n\n## 페르소나별 응답 샘플 (상위 5명)\n{persona_samples_text}" if persona_samples_text else ""

    return f"""다음 시뮬레이션 결과를 바탕으로 [{type_label}]를 작성하세요.

## 프로젝트 정보
- 이름: {project_name}
- 목적: {project_purpose}
- 총 분석 페르소나: {persona_count}명
- 총 시뮬레이션 응답: {response_count}건
- 주요 세그먼트: {dominant_segment}

## 세그먼트별 분석 결과
{segments_text}

## 세그먼트별 실제 응답 패턴 (선택 이유 포함)
{response_pattern_text}{persona_section}

## 핵심 키워드
{keywords_text}

## 리포트 유형별 작성 지침
{_get_report_type_instruction(report_type)}

중요: 세그먼트별 실제 응답 패턴과 페르소나 응답 샘플에 나타난 구체적인 근거(rationale)를 인사이트에 직접 인용하세요.
evidence의 source_question_id는 위 detailed_distribution의 question_id 중 가장 관련 있는 것을 사용하세요. 관련 질문이 없으면 null로 설정하세요.

## 출력 형식 (JSON만 출력)
{{
  "title": "리포트 제목",
  "sections": [
    {{
      "id": "section_id",
      "title": "섹션 제목",
      "content": "분석 내용 (3~5문장, 구체적 수치와 응답 근거 포함)",
      "evidence": [
        {{
          "label": "근거 항목",
          "value": "수치/내용",
          "source_question_id": "q-001"
        }}
      ],
      "action": "실행 가능한 권장사항 1문장"
    }}
  ],
  "kpis": [
    {{"label": "KPI명", "value": "수치"}}
  ],
  "key_findings": ["핵심 발견사항 1", "핵심 발견사항 2", "핵심 발견사항 3"]
}}"""


def _get_report_type_instruction(report_type: str) -> str:
    instructions = {
        "strategy": (
            "마케팅 전략 관점에서 작성. "
            "섹션 구성: 종합 요약 / 전략적 기회 / 타겟 우선순위 / 실행 권고안. "
            "각 섹션에 구체적인 채널·메시지·타이밍 전략을 포함."
        ),
        "insight": (
            "리서치 인사이트 관점에서 작성. "
            "섹션 구성: 핵심 발견사항 / 예상과의 차이점 / 세그먼트별 반응 패턴 / 추가 조사 필요 영역. "
            "데이터에서 도출한 놀라운 발견이나 역설적 패턴을 강조."
        ),
        "segment": (
            "세그먼트 심층 분석 관점에서 작성. "
            "섹션 구성: 세그먼트 프로파일링 / 구매 의향 격차 분석 / 채널 선호도 / 세그먼트별 맞춤 접근법. "
            "각 세그먼트의 고유한 특성과 공략 포인트를 구체적으로 서술."
        ),
        "summary": (
            "경영진 요약(Executive Summary) 관점에서 작성. "
            "섹션 구성: 리서치 결론 / 비즈니스 임팩트 / 우선 실행 과제. "
            "간결하고 명확하게, 의사결정에 필요한 핵심만 포함."
        ),
    }
    return instructions.get(report_type, instructions["strategy"])