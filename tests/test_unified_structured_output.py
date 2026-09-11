"""Structured contract/repair boundaries, without any live LLM calls."""

import asyncio
import copy
import json
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal

import httpx
import pytest
from hypothesis import given, strategies as st
from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.ai_client import AIClient, LLMUnavailableError, ToolLLMResponse
from app.core.copilot.final_response import CopilotFinalResponse, finish_copilot_response, parse_final_response
from app.core.structured_output import StructuredOutputParseError, validate_structured_output
from tests.structured_output_support import (
    CONFIG, SCHEMA_REQUIRED, completion, failure,
    structured_provider as structured_provider,
)

GOOD = '{"answer":"李华有\\"小男人\\"性格。\\n证据来自第一章。","suggestions":[]}'
BAD = '{"answer":"李华有"小男人"性格。","suggestions":[]}'
MESSAGES = [{"role": "system", "content": "Use evidence only."},
            {"role": "user", "content": "Analyse the supplied chapter."}]


@pytest.mark.parametrize("raw", [
    BAD, '{"answer":"unfinished', '```json\n' + BAD + '\n```',
    'Analysis:\n' + BAD, '{"answer":"ok","suggestions":[{"kind":"update_entity","delta":[] }]}',
    '{"answer":"ok","suggestions":null}', '{"answer":"ok","suggestions":[3]}',
    '{"answer":true}', '{"answer":"   "}', '{"answer":"ok","cited_evidence_indices":[-1]}',
    '{"answer":"ok","cited_evidence_indices":["1"]}',
    '{"answer":"ok","suggestions":[{"kind":"update_entity","title":123}]}',
    '{"answer":"ok","suggestions":[{"kind":"update_entity","delta":{"x":NaN}}]}',
    '{"answer":"ok","suggestions":[{}],"suggestions":[]}',
    '[]', 'null', '123', '"a string"', '{}',
])
def test_invalid_structured_output_is_never_plain_text_success(raw):
    with pytest.raises(StructuredOutputParseError):
        parse_final_response(raw, allow_plain_text=False)


@pytest.mark.parametrize("wrap", ["{}", "```json\n{}\n```", "Analysis:\n{}\nEnd."])
def test_valid_quotes_newlines_and_empty_suggestions_are_preserved(wrap):
    result = parse_final_response(wrap.format(GOOD), allow_plain_text=False)
    assert result == json.loads(GOOD) | {"cited_evidence_indices": []}


def test_task_plain_text_needs_structure_but_chat_does_not():
    assert parse_final_response("你好，我可以帮你整理人物。")["suggestions"] == []
    with pytest.raises(StructuredOutputParseError):
        parse_final_response("发现三个人物。", allow_plain_text=False)


@pytest.mark.parametrize("text", ["```text\n人物、地点、事件\n```", "```\n一段普通说明\n```"])
def test_non_json_markdown_in_ordinary_chat_is_not_a_broken_envelope(text):
    assert parse_final_response(text)["answer"] == text


@pytest.mark.asyncio
async def test_valid_existing_response_does_not_request_or_acquire_slot(structured_provider):
    @asynccontextmanager
    async def forbidden_slot():
        raise AssertionError("Valid output must not wait for an LLM slot")
        yield

    result = await finish_copilot_response(
        client=AIClient(), messages=MESSAGES, llm_config=CONFIG, user_id=42,
        response=ToolLLMResponse(content=GOOD, finish_reason="stop"),
        request_context=forbidden_slot(),
    )
    assert result["answer"] == json.loads(GOOD)["answer"]
    assert structured_provider.requests == []


@pytest.mark.asyncio
async def test_repair_uses_schema_and_same_context_and_preserves_suggestions(structured_provider):
    p = structured_provider
    suggestion = {"kind": "update_entity", "title": "性格", "target_id": "12",
                  "cited_evidence_indices": [0], "delta": {"description": "谨慎"}}
    good = json.dumps({"answer": "人物谨慎。", "suggestions": [suggestion]}, ensure_ascii=False)
    bad = good.replace('"人物谨慎。"', '"人物有"谨慎"性格。"')
    p.replies.extend([failure(400, SCHEMA_REQUIRED), completion(good)])
    messages = copy.deepcopy(MESSAGES)
    events = []

    @asynccontextmanager
    async def slot():
        events.append("acquire")
        try:
            yield
        finally:
            events.append("release")

    result = await finish_copilot_response(
        client=AIClient(), messages=messages, llm_config=CONFIG, user_id=42,
        response=ToolLLMResponse(content=bad), request_context=slot(),
    )
    assert result["suggestions"] == [suggestion]
    assert messages == MESSAGES
    assert events == ["acquire", "release"]
    assert len(p.requests) == 2
    assert p.requests[1]["response_format"]["json_schema"]["schema"] == CopilotFinalResponse.model_json_schema()
    assert any(m["content"] == bad for m in p.requests[1]["messages"])
    assert "tools" not in p.requests[1]
    assert all(r["model"] == CONFIG.model for r in p.requests)
    assert p.usage[0][1]["user_id"] == 42
    assert p.usage[0][1]["billing_source"] == "selfhost"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["length", "content_filter"])
async def test_truncation_and_refusal_do_not_initiate_repair(structured_provider, reason):
    with pytest.raises(StructuredOutputParseError) as exc:
        await finish_copilot_response(
            client=AIClient(), messages=MESSAGES, llm_config=CONFIG, user_id=1,
            response=ToolLLMResponse(content=GOOD, finish_reason=reason),
        )
    assert exc.value.finish_reason == reason
    assert structured_provider.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [1, 2, 3])
async def test_final_budget_includes_existing_answer(structured_provider, monkeypatch, budget):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "copilot_final_max_attempts", budget)
    p = structured_provider
    p.replies.extend(completion(BAD) for _ in range(budget - 1))
    with pytest.raises(StructuredOutputParseError):
        await finish_copilot_response(
            client=AIClient(), messages=MESSAGES, llm_config=CONFIG, user_id=1,
            response=ToolLLMResponse(content=BAD),
        )
    assert len(p.requests) == budget - 1


class Extraction(BaseModel):
    model_config = ConfigDict(strict=True)
    names: list[str]


@pytest.mark.asyncio
async def test_other_structured_tasks_share_validation_and_feedback(structured_provider):
    p = structured_provider
    p.replies.extend([failure(400, SCHEMA_REQUIRED), completion('{"names":123}'),
                      completion('{"names":["李华"]}')])
    result = await AIClient().generate_structured(
        "Extract names", Extraction, llm_config=CONFIG, max_retries=2,
    )
    assert result.names == ["李华"]
    assert [r["response_format"]["type"] for r in p.requests] == ["json_object", "json_schema", "json_schema"]
    assert any(m["content"] == '{"names":123}' for m in p.requests[2]["messages"])
    assert p.requests[2]["messages"][-1]["role"] == "user"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
async def test_provider_errors_keep_existing_budget_without_formatting_repair(structured_provider, status):
    p = structured_provider
    p.replies.extend(failure(status, "synthetic provider error") for _ in range(3))
    with pytest.raises(LLMUnavailableError):
        await AIClient().generate_structured("Extract", Extraction, llm_config=CONFIG, max_retries=3)
    assert len(p.requests) == 3
    assert p.requests[0] == p.requests[1] == p.requests[2]


@pytest.mark.asyncio
async def test_read_timeout_is_not_json_repair(structured_provider):
    p = structured_provider
    p.replies.extend(httpx.ReadTimeout("synthetic timeout") for _ in range(3))
    with pytest.raises(LLMUnavailableError):
        await AIClient().generate_structured("Extract", Extraction, llm_config=CONFIG)
    assert len(p.requests) == 3
    assert p.requests[0] == p.requests[1] == p.requests[2]


@pytest.mark.asyncio
async def test_transient_provider_error_still_recovers_for_existing_structured_tasks(structured_provider):
    p = structured_provider
    p.replies.extend([failure(503, "temporary outage"), completion('{"names":["李华"]}')])
    result = await AIClient().generate_structured("Extract", Extraction, llm_config=CONFIG)
    assert result.names == ["李华"]
    assert len(p.requests) == 2
    assert p.requests[0] == p.requests[1]


@pytest.mark.asyncio
async def test_final_generation_does_not_add_transport_retries(structured_provider, monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "copilot_final_max_attempts", 5)
    p = structured_provider
    p.replies.append(failure(401, "invalid key"))
    with pytest.raises(LLMUnavailableError):
        await finish_copilot_response(
            client=AIClient(), messages=MESSAGES, llm_config=CONFIG, user_id=1,
        )
    assert len(p.requests) == 1


@pytest.mark.asyncio
async def test_cancel_during_repair_releases_slot_and_provider(structured_provider):
    p = structured_provider
    p.replies.append(asyncio.CancelledError())
    events = []

    @asynccontextmanager
    async def slot():
        events.append("acquire")
        try:
            yield
        finally:
            events.append("release")

    with pytest.raises(asyncio.CancelledError):
        await finish_copilot_response(
            client=AIClient(), messages=MESSAGES, llm_config=CONFIG, user_id=1,
            response=ToolLLMResponse(content=BAD), request_context=slot(),
        )
    assert events == ["acquire", "release"]
    assert len(p.requests) == 1


@pytest.mark.asyncio
async def test_schema_errors_log_position_not_story_or_credentials(structured_provider, caplog, monkeypatch):
    # Alembic's fileConfig in earlier migration tests disables existing loggers.
    monkeypatch.setattr(logging.getLogger("app.core.structured_output"), "disabled", False)
    caplog.set_level(logging.WARNING, logger="app.core.structured_output")
    p = structured_provider
    secret = "PRIVATE-NOVEL-DETAIL"
    p.replies.append(completion('{"answer":"' + secret + '"bad"}'))
    with pytest.raises(StructuredOutputParseError) as exc:
        await AIClient().generate_structured("Extract", CopilotFinalResponse, llm_config=CONFIG, max_retries=1)
    assert any(getattr(r, "column", None) is not None for r in caplog.records)
    assert secret not in caplog.text
    assert secret not in "".join(traceback.format_exception(exc.value))
    assert CONFIG.api_key not in caplog.text
    assert exc.value.__context__ is None


@pytest.mark.asyncio
async def test_truncated_structured_generation_does_not_repeat_same_budget(structured_provider):
    p = structured_provider
    p.replies.append(completion('{"names":[]}', finish="length"))
    with pytest.raises(StructuredOutputParseError) as exc:
        await AIClient().generate_structured("Extract", Extraction, llm_config=CONFIG, max_retries=3)
    assert exc.value.finish_reason == "length"
    assert len(p.requests) == 1


def test_validation_retains_error_location_without_salvaging_inner_object():
    with pytest.raises(StructuredOutputParseError) as exc:
        validate_structured_output('{"broken":"x"x","inner":{"answer":"ok"}}', CopilotFinalResponse)
    assert isinstance(exc.value.last_error, json.JSONDecodeError)


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1).filter(lambda s: s.strip()))
def test_json_roundtrip_preserves_unicode_content(answer):
    raw = json.dumps({"answer": answer, "suggestions": []})
    assert parse_final_response(raw, allow_plain_text=False)["answer"] == answer


def test_shared_validator_preserves_existing_strict_json_date_semantics():
    class DatedResult(BaseModel):
        model_config = ConfigDict(strict=True)
        day: date

    result = validate_structured_output('{"day":"2026-09-10"}', DatedResult)
    assert result.day == date(2026, 9, 10)


def test_shared_validator_preserves_pydantic_numeric_semantics():
    class NumericResult(BaseModel):
        value: Decimal

    raw = '{"value":12345678901234567890.123456789}'
    assert validate_structured_output(raw, NumericResult) == NumericResult.model_validate_json(raw)


@pytest.mark.parametrize("value", [0, -1, 6])
def test_final_attempt_configuration_rejects_invalid_budgets(value):
    from app.config import Settings
    with pytest.raises(ValidationError):
        Settings(copilot_final_max_attempts=value, _env_file=None)


@pytest.mark.asyncio
async def test_missing_choices_are_not_a_successful_structured_result(structured_provider):
    p = structured_provider
    p.replies.append(httpx.Response(200, json={
        "id": "empty", "object": "chat.completion", "created": 0,
        "model": CONFIG.model, "choices": [],
    }))
    with pytest.raises(StructuredOutputParseError):
        await AIClient().generate_structured("Extract", Extraction, llm_config=CONFIG, max_retries=1)
    assert len(p.requests) == 1
