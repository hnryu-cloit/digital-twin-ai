"""
Async parallel Gemini simulation engine with SSE streaming.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import AsyncGenerator

import google.generativeai as genai

from .prompts import SIMULATION_SYSTEM, build_simulation_prompt


# ── 단일 페르소나 응답 생성 ──────────────────────────────────────────────────────

async def _generate_single_response(
    persona: dict,
    questions: list[dict],
    project_purpose: str,
    project_name: str,
    model: genai.GenerativeModel,
) -> dict:
    """Run one persona through the survey and return structured responses."""
    loop = asyncio.get_event_loop()
    prompt = build_simulation_prompt(persona, questions, project_purpose, project_name)

    def _call() -> str:
        response = model.generate_content(
            [{"role": "user", "parts": [prompt]}],
            generation_config=genai.types.GenerationConfig(temperature=0.85),
        )
        return response.text.strip()

    try:
        raw = await loop.run_in_executor(None, _call)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            answers = json.loads(match.group())
        else:
            answers = []
    except Exception as exc:
        answers = []
        raw = str(exc)

    return {
        "persona_id": persona.get("id", ""),
        "persona_name": persona.get("name", ""),
        "segment": persona.get("segment", ""),
        "answers": answers,
        "raw": raw if not answers else None,
    }


# ── 배치 병렬 실행 ───────────────────────────────────────────────────────────────

async def _run_batch(
    batch: list[dict],
    questions: list[dict],
    project_purpose: str,
    project_name: str,
    model: genai.GenerativeModel,
) -> list[dict]:
    tasks = [
        _generate_single_response(p, questions, project_purpose, project_name, model)
        for p in batch
    ]
    return await asyncio.gather(*tasks)


# ── SSE 스트리밍 제너레이터 ─────────────────────────────────────────────────────

async def run_simulation_stream(
    personas: list[dict],
    questions: list[dict],
    project_context: dict,
    api_key: str,
    model_name: str = "gemini-3-flash",
    batch_size: int = 5,
) -> AsyncGenerator[str, None]:
    """
    Yields SSE-formatted strings.

    Event types:
      - progress : {"type":"progress","done":N,"total":M,"persona_name":"..."}
      - result   : {"type":"result","persona_id":"...","persona_name":"...","segment":"...","answers":[...]}
      - error    : {"type":"error","persona_id":"...","message":"..."}
      - done     : {"type":"done","total":M}
    """
    project_purpose = project_context.get("purpose", "")
    project_name = project_context.get("name", "")
    total = len(personas)

    # Gemini 초기화
    genai.configure(api_key=api_key)
    system_prompt = SIMULATION_SYSTEM
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
    )

    done_count = 0

    for batch_start in range(0, total, batch_size):
        batch = personas[batch_start : batch_start + batch_size]
        results = await _run_batch(batch, questions, project_purpose, project_name, model)

        for res in results:
            done_count += 1

            if res["answers"]:
                event = {
                    "type": "result",
                    "persona_id": res["persona_id"],
                    "persona_name": res["persona_name"],
                    "segment": res["segment"],
                    "answers": res["answers"],
                }
            else:
                event = {
                    "type": "error",
                    "persona_id": res["persona_id"],
                    "persona_name": res["persona_name"],
                    "message": res.get("raw") or "응답 파싱 실패",
                }

            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # 배치 완료 후 진행상황 이벤트
        progress_event = {
            "type": "progress",
            "done": done_count,
            "total": total,
        }
        yield f"data: {json.dumps(progress_event, ensure_ascii=False)}\n\n"

        # 배치 간 짧은 대기 (rate-limit 완충)
        if batch_start + batch_size < total:
            await asyncio.sleep(0.3)

    yield f"data: {json.dumps({'type': 'done', 'total': total}, ensure_ascii=False)}\n\n"