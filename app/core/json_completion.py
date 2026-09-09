"""Negotiate structured output without changing prompts or validation semantics."""

from typing import Any

from openai import AsyncOpenAI


def _requires_json_schema(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) not in (400, 422):
        return False
    message = str(exc).casefold()
    return (
        "response_format" in message
        and "json_schema" in message
        and any(hint in message for hint in ("must be", "must use", "only supports", "only supported"))
    )


class JsonCompletion:
    """Format selection lives only for one logical operation, never across users."""

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self.response_format: dict[str, Any] = {"type": "json_object"}

    async def create(self, client: AsyncOpenAI, **kwargs: Any) -> Any:
        try:
            return await client.chat.completions.create(
                **kwargs, response_format=self.response_format,
            )
        except Exception as exc:
            if self.response_format["type"] != "json_object" or not _requires_json_schema(exc):
                raise
        # LM Studio explicitly rejects json_object before inference. Retry once
        # with the actual schema; Pydantic remains the final output validator.
        self.response_format = {
            "type": "json_schema",
            "json_schema": {"name": "structured_response", "schema": self.schema},
        }
        return await client.chat.completions.create(
            **kwargs, response_format=self.response_format,
        )
