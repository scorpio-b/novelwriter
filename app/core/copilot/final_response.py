"""One final-answer contract for tool-loop, wrap-up and one-shot Copilot calls."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from app.config import get_settings
from app.core.ai_client import AIClient, ToolLLMResponse
from app.core.llm_config import ResolvedLlmConfig
from app.core.structured_output import (
    StructuredOutputParseError,
    log_validation_failure,
    looks_like_json,
    repair_messages,
    validate_structured_output,
)
from .tool_call_recovery import contains_tool_call_markup, strip_tool_call_markup

EvidenceIndex = Annotated[StrictInt, Field(ge=0)]


class SuggestionDraft(BaseModel):
    # Keep business-specific delta fields and legacy target IDs for the compiler.
    model_config = ConfigDict(strict=True, extra="allow")
    kind: str
    title: str = ""
    summary: str = ""
    cited_evidence_indices: list[EvidenceIndex] = Field(default_factory=list)
    target_resource: str = "entity"
    target_id: int | str | None = None
    delta: dict[str, Any] = Field(default_factory=dict)


class CopilotFinalResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    answer: str = Field(min_length=1)
    cited_evidence_indices: list[EvidenceIndex] = Field(default_factory=list)
    suggestions: list[SuggestionDraft] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def require_nonblank_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Final answer must not be blank")
        return value

    def as_result(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "cited_evidence_indices": self.cited_evidence_indices,
            "suggestions": [s.model_dump(exclude_unset=True) for s in self.suggestions],
        }


def parse_final_response(
    text: str, *, allow_plain_text: bool = True, finish_reason: str | None = None,
) -> dict[str, Any]:
    if contains_tool_call_markup(text):
        text = strip_tool_call_markup(text)
    if allow_plain_text and text.strip() and not looks_like_json(text):
        # Preserve ordinary chat. Never reinterpret broken JSON as plain prose.
        text = CopilotFinalResponse(answer=text).model_dump_json()
    return validate_structured_output(
        text, CopilotFinalResponse, finish_reason=finish_reason,
    ).as_result()


async def finish_copilot_response(
    *,
    client: AIClient,
    messages: list[dict[str, Any]],
    llm_config: ResolvedLlmConfig,
    user_id: int,
    response: ToolLLMResponse | None = None,
    allow_plain_text: bool = False,
    request_context: AbstractAsyncContextManager | None = None,
) -> dict[str, Any]:
    attempts = get_settings().copilot_final_max_attempts
    if response is not None:
        error: StructuredOutputParseError | None = None
        try:
            return parse_final_response(
                response.content or "", allow_plain_text=allow_plain_text,
                finish_reason=response.finish_reason,
            )
        except StructuredOutputParseError as exc:
            log_validation_failure(exc, attempt=1)
            error = exc
        if response.finish_reason in ("length", "content_filter") or attempts <= 1:
            raise error
        attempts -= 1
        messages = repair_messages(messages, response.content or "")

    # A valid existing answer needs neither another request nor another queue slot.
    async with request_context if request_context is not None else nullcontext():
        result = await client.generate_structured(
            prompt="", response_model=CopilotFinalResponse, messages=messages,
            system_prompt="Produce the final Copilot answer using the existing evidence. Do not call tools.",
            llm_config=llm_config, user_id=user_id, role="default",
            max_tokens=4000, temperature=0.4, max_retries=attempts,
            retry_transport_errors=False,
        )
    return result.as_result()
