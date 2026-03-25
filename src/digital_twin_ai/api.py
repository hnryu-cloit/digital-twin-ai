from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from .persona_modeling import PersonaManager
from .pipeline import run_pipeline


class PersonaGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str | None = None
    project_id: str
    random_state: int = 42
    n_synthetic_customers: int = Field(default=1000, ge=100)
    n_personas: int = Field(default=7, ge=2, le=20)
    excel_path: str = "./data/Digital Customer Twin.xlsx"
    output_dir: str = "./output"
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-3-flash"


class PersonaGenerateResponse(BaseModel):
    resource: str
    job_id: str | None = None
    project_id: str
    personas: list[dict]
    artifacts: dict[str, str]
    metadata: dict


class SurveyGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str | None = None
    project_id: str
    user_prompt: str
    survey_type: str
    question_count: int = Field(default=5, ge=1, le=20)
    template: dict = Field(default_factory=dict)
    segment_context: dict = Field(default_factory=dict)
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-3-flash"


class SurveyGenerateResponse(BaseModel):
    resource: str
    job_id: str | None = None
    project_id: str
    questions: list[dict]
    metadata: dict


class ReportGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str | None = None
    project_id: str
    project_name: str
    purpose: str = ""
    persona_count: int = 0
    response_count: int = 0
    target_responses: int = 0
    response_progress: int = 0
    dominant_segment: str = "데이터 없음"
    top_question: str = "집계 중"
    keyword_items: list[dict] = Field(default_factory=list)
    age_buckets: list[dict] = Field(default_factory=list)
    segment_cards: list[dict] = Field(default_factory=list)
    question_strength_data: list[dict] = Field(default_factory=list)
    detailed_distribution: list[dict] = Field(default_factory=list)
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-3-flash"


class ReportGenerateResponse(BaseModel):
    resource: str
    job_id: str | None = None
    project_id: str
    report: dict
    metadata: dict


app = FastAPI(title="digital-twin-ai", version="0.1.0")

QUESTION_TYPE_REASON = {
    "단일선택": "핵심 선호를 빠르게 비교하기 위한 단일 선택 문항입니다.",
    "복수선택": "복수 동기를 함께 수집해 우선순위와 조합 패턴을 파악하기 위한 문항입니다.",
    "리커트척도": "태도 강도와 변화 폭을 정량적으로 측정하기 위한 척도 문항입니다.",
    "주관식": "정량 응답으로 포착되지 않는 표현과 우려 요인을 수집하기 위한 서술형 문항입니다.",
}


def _fallback_question_templates(survey_type: str) -> list[tuple[str, list[str]]]:
    templates = {
        "concept": [
            ("이 컨셉의 첫인상은 어떻습니까?", ["매우 긍정적", "긍정적", "보통", "부정적", "매우 부정적"]),
            ("가장 매력적으로 느껴지는 요소는 무엇입니까?", ["핵심 기능", "디자인", "브랜드 신뢰", "가격 경쟁력"]),
            ("실제 구매를 고려하게 만드는 요인은 무엇입니까?", ["성능", "편의성", "가격", "추천/후기"]),
            ("이 컨셉에서 가장 우려되는 점은 무엇입니까?", ["가격", "복잡성", "차별성 부족", "신뢰성"]),
        ],
        "ad": [
            ("광고 메시지가 제품의 강점을 명확히 전달합니까?", ["매우 그렇다", "그렇다", "보통", "아니다", "전혀 아니다"]),
            ("광고를 본 뒤 기억에 남는 요소는 무엇입니까?", ["카피", "비주얼", "혜택", "브랜드"]),
            ("광고 노출 후 구매 의향 변화는 어떻습니까?", ["매우 상승", "상승", "변화 없음", "하락"]),
            ("광고 메시지에서 보완이 필요한 부분은 무엇입니까?", ["차별성", "신뢰성", "혜택 설명", "타깃 적합성"]),
        ],
    }
    return templates.get(
        survey_type,
        [
            ("이 주제에 대해 얼마나 관심이 있습니까?", ["매우 높다", "높다", "보통", "낮다", "매우 낮다"]),
            ("가장 중요하게 보는 판단 기준은 무엇입니까?", ["품질", "가격", "편의성", "브랜드"]),
            ("구매 또는 참여를 결정하게 만드는 계기는 무엇입니까?", ["추천", "경험", "혜택", "필요성"]),
            ("개선이 필요하다고 느끼는 지점은 무엇입니까?", ["기능", "가격", "설명", "접근성"]),
        ],
    )


def _compose_generation_prompt(
    user_prompt: str,
    survey_type: str,
    question_count: int,
    template: dict,
    segment_context: dict,
) -> str:
    template_id = template.get("template_id", "template-not-set")
    required_blocks = ", ".join(template.get("required_blocks", [])) or "none"
    segment_summary = json.dumps(segment_context, ensure_ascii=False) if segment_context else "없음"
    return (
        f"{question_count}개 설문 문항을 생성하세요.\n"
        f"유형: {survey_type}\n"
        f"사용자 요청: {user_prompt}\n"
        f"리서치 템플릿 ID: {template_id}\n"
        f"필수 블록: {required_blocks}\n"
        f"세그먼트 분석 컨텍스트: {segment_summary}"
    )


def _build_fallback_questions(user_prompt: str, survey_type: str, question_count: int) -> list[dict]:
    questions: list[dict] = []
    templates = _fallback_question_templates(survey_type)
    for index in range(question_count):
        text, options = templates[index % len(templates)]
        question_type = "단일선택"
        questions.append(
            {
                "id": f"q-{uuid.uuid4().hex[:8]}",
                "text": f"{user_prompt} - {text}",
                "type": question_type,
                "options": options,
                "order": index + 1,
                "status": "draft",
                "generation_source": "fallback",
                "ai_rationale": QUESTION_TYPE_REASON[question_type],
                "ai_evidence": [
                    {"label": "survey_type", "value": survey_type},
                    {"label": "user_prompt", "value": user_prompt},
                ],
            }
        )
    return questions


def _generate_survey_questions(request: SurveyGenerateRequest) -> list[dict]:
    manager = PersonaManager(api_key=request.gemini_api_key, model_name=request.gemini_model_name)
    generated = None
    if manager.model is not None:
        try:
            prompt = _compose_generation_prompt(
                user_prompt=request.user_prompt,
                survey_type=request.survey_type,
                question_count=request.question_count,
                template=request.template,
                segment_context=request.segment_context,
            )
            full_prompt = f"""{prompt}

다음 JSON 배열만 출력하세요:
[{{"text": "문항 텍스트", "type": "단일선택|복수선택|리커트척도|주관식", "options": ["선택지1", "선택지2"], "rationale": "문항이 필요한 이유 1~2문장", "evidence": [{{"label": "근거 항목", "value": "반영한 정보"}}]}}]

주의:
- 주관식 문항의 options는 빈 배열 []
- 리커트척도는 ["매우 그렇다", "그렇다", "보통", "아니다", "전혀 아니다"]
- rationale은 사용자 요청/템플릿/세그먼트 컨텍스트와 어떻게 연결되는지 설명
- evidence에는 실제 반영한 컨텍스트만 2~4개 포함
- 정확히 {request.question_count}개 생성"""
            response = manager.model.generate_content(full_prompt)
            text = response.text.strip()
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if isinstance(parsed, list) and parsed:
                    generated = []
                    for index, item in enumerate(parsed, start=1):
                        if isinstance(item, dict) and "text" in item:
                            question_type = item.get("type", "단일선택")
                            generated.append(
                                {
                                    "id": f"q-{uuid.uuid4().hex[:8]}",
                                    "text": item["text"],
                                    "type": question_type,
                                    "options": item.get("options", []) if question_type != "주관식" else [],
                                    "order": index,
                                    "status": "draft",
                                    "generation_source": "gemini",
                                    "ai_rationale": item.get("rationale", "").strip() or QUESTION_TYPE_REASON.get(question_type, ""),
                                    "ai_evidence": item.get("evidence", []),
                                }
                            )
        except Exception:
            generated = None

    return generated or _build_fallback_questions(request.user_prompt, request.survey_type, request.question_count)


def _generate_report_payload(request: ReportGenerateRequest) -> dict:
    summary_content = (
        f"{request.project_name} 프로젝트는 현재 {request.persona_count}명의 페르소나와 "
        f"{request.response_count}건의 응답을 기반으로 집계되었습니다."
    )
    findings_content = (
        f"가장 큰 세그먼트는 {request.dominant_segment}이며, "
        f"우선 검토 문항은 '{request.top_question or '집계 중'}' 입니다."
    )
    action_content = (
        f"{request.dominant_segment} 세그먼트를 기준으로 메시지와 채널 전략을 우선 정렬하고, "
        f"응답 진행률 {request.response_progress}% 구간에서 이탈 요인을 추가 확인합니다."
    )
    return {
        "title": f"{request.project_name} 리포트",
        "type": "strategy",
        "format": "PDF",
        "size": "4.2MB",
        "sections": [
            {
                "id": "summary",
                "title": "종합 분석 요약",
                "content": summary_content,
            },
            {
                "id": "findings",
                "title": "전략적 핵심 인사이트",
                "content": findings_content,
                "evidence": [
                    {"label": "최대 세그먼트", "value": request.dominant_segment},
                    {"label": "우선 문항", "value": request.top_question or "집계 중"},
                    {"label": "응답 진행률", "value": f"{request.response_progress}%"},
                ],
                "action": action_content,
            },
            {
                "id": "detail",
                "title": "데이터 기반 상세 분석",
                "content": f"{request.purpose or '조사 목적'} 기준으로 세그먼트·키워드·문항 응답 분포를 함께 해석할 수 있습니다.",
            },
            {
                "id": "segment",
                "title": "세그먼트 기회 매트릭스",
                "content": f"상위 세그먼트 {len(request.segment_cards)}개를 대상으로 구매 채널, 제품군, 지역 단위 액션을 정리했습니다.",
            },
        ],
        "kpis": [
            {"label": "응답 진행률", "value": f"{request.response_progress}%"},
            {"label": "목표 응답 수", "value": str(request.target_responses)},
            {"label": "총 페르소나 수", "value": str(request.persona_count)},
            {"label": "총 시뮬레이션 응답", "value": str(request.response_count)},
        ],
        "charts": [
            {"id": "keyword-radar", "type": "radar", "title": "상위 키워드 레이더", "data": [
                {
                    "subject": item.get("keyword", ""),
                    "dominant": item.get("frequency", 0),
                    "baseline": 50,
                    "fullMark": 100,
                }
                for item in request.keyword_items
            ]},
            {"id": "question-strength", "type": "area", "title": "문항별 우세 응답 강도", "data": request.question_strength_data},
            {"id": "age-distribution", "type": "bar", "title": "연령대별 분석 대상 규모", "data": request.age_buckets},
            {"id": "question-distribution", "type": "distribution", "title": request.top_question or "응답 분포", "data": request.detailed_distribution},
            {"id": "segment-cards", "type": "segment", "title": "세그먼트 기회 매트릭스", "data": request.segment_cards},
        ],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/internal/personas/generate", response_model=PersonaGenerateResponse, status_code=status.HTTP_200_OK)
def generate_personas(request: PersonaGenerateRequest) -> PersonaGenerateResponse:
    try:
        metadata = run_pipeline(request.model_dump(exclude={"job_id", "project_id"}))
        personas_path = Path(metadata["outputs"]["personas"])
        personas = json.loads(personas_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI persona generation failed: {error}",
        ) from error

    return PersonaGenerateResponse(
        resource="personas",
        job_id=request.job_id,
        project_id=request.project_id,
        personas=personas,
        artifacts=metadata.get("outputs", {}),
        metadata=metadata,
    )


@app.post("/internal/surveys/generate-draft", response_model=SurveyGenerateResponse, status_code=status.HTTP_200_OK)
def generate_survey_draft(request: SurveyGenerateRequest) -> SurveyGenerateResponse:
    questions = _generate_survey_questions(request)
    return SurveyGenerateResponse(
        resource="survey_questions",
        job_id=request.job_id,
        project_id=request.project_id,
        questions=questions,
        metadata={
            "question_count": len(questions),
            "template_id": request.template.get("template_id"),
            "template_version": request.template.get("template_version"),
            "segment_source": request.segment_context.get("source"),
        },
    )


@app.post("/internal/reports/generate", response_model=ReportGenerateResponse, status_code=status.HTTP_200_OK)
def generate_report(request: ReportGenerateRequest) -> ReportGenerateResponse:
    report = _generate_report_payload(request)
    return ReportGenerateResponse(
        resource="report",
        job_id=request.job_id,
        project_id=request.project_id,
        report=report,
        metadata={
            "persona_count": request.persona_count,
            "response_count": request.response_count,
            "response_progress": request.response_progress,
        },
    )
