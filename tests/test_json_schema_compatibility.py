"""LM Studio's schema-only response format through the real OpenAI client."""

import json

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.core.ai_client import AIClient, LLMUnavailableError, StructuredOutputParseError
from app.core.llm_config import ResolvedLlmConfig


class Character(BaseModel):
    name: str
    age: int


class World(BaseModel):
    characters: list[Character]
    synopsis: str | None = None


LM_STUDIO_ERROR = "'response_format.type' must be 'json_schema' or 'text'"
CONFIG = ResolvedLlmConfig(
    base_url="http://lm-studio.test/v1", api_key="test-only", model="local-model",
    billing_source_hint="selfhost", source="selfhost_settings",
)


@pytest.fixture
def provider(monkeypatch):
    requests = []
    results = []
    clients = []

    def handle(request):
        requests.append(json.loads(request.content))
        result = results.pop(0)
        if isinstance(result, tuple):
            status, message = result
            return httpx.Response(status, json={"error": {"message": message}})
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": "local-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result},
                         "finish_reason": "stop"}],
        })

    def make_client(**kwargs):
        client = AsyncOpenAI(**kwargs, max_retries=0,
                             http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        clients.append(client)
        return client

    monkeypatch.setattr("app.core.ai_client.AsyncOpenAI", make_client)
    monkeypatch.setattr("app.core.ai_client.ensure_ai_available_fresh_session", lambda **kw: None)
    yield requests, results
    assert all(client.is_closed() for client in clients)


@pytest.mark.asyncio
async def test_schema_only_provider_uses_the_actual_nested_model(provider):
    requests, results = provider
    results.extend([(400, LM_STUDIO_ERROR), '{"characters":[{"name":"Lin","age":31}]}'])
    world = await AIClient().generate_structured("Extract characters", World, llm_config=CONFIG)
    assert world.characters[0].name == "Lin"
    assert len(requests) == 2
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert requests[1]["response_format"]["type"] == "json_schema"
    assert requests[1]["response_format"]["json_schema"]["schema"] == World.model_json_schema()
    assert requests[1]["messages"] == requests[0]["messages"]
    assert requests[1]["model"] == CONFIG.model


@pytest.mark.asyncio
async def test_schema_format_survives_pydantic_retry_but_does_not_leak_between_calls(provider):
    requests, results = provider
    results.extend([(400, LM_STUDIO_ERROR), '{"characters":"wrong"}', '{"characters":[]}'])
    await AIClient().generate_structured("Extract", World, llm_config=CONFIG)
    assert [r["response_format"]["type"] for r in requests] == ["json_object", "json_schema", "json_schema"]
    results.append('{"characters":[]}')
    await AIClient().generate_structured("Extract", World, llm_config=CONFIG)
    assert requests[-1]["response_format"]["type"] == "json_object"


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ['not JSON', '{"characters":"wrong"}'])
async def test_schema_mode_still_validates_output(provider, reply):
    requests, results = provider
    results.extend([(400, LM_STUDIO_ERROR), reply])
    with pytest.raises(StructuredOutputParseError):
        await AIClient().generate_structured("Extract", World, llm_config=CONFIG, max_retries=1)
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status,message", [
    (401, LM_STUDIO_ERROR),
    (403, "access denied"),
    (400, "context length exceeded"),
    (400, "response_format json_object is not supported"),
    (422, "invalid json_schema: missing required field"),
    (503, LM_STUDIO_ERROR),
])
async def test_unrelated_provider_errors_do_not_switch_formats(provider, status, message):
    requests, results = provider
    results.append((status, message))
    with pytest.raises(LLMUnavailableError):
        await AIClient().generate_structured("Extract", World, llm_config=CONFIG, max_retries=1)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_schema_rejection_does_not_start_an_unbounded_format_loop(provider):
    requests, results = provider
    results.extend([(400, LM_STUDIO_ERROR), (400, "json_schema is not supported")])
    with pytest.raises(LLMUnavailableError):
        await AIClient().generate_structured("Extract", World, llm_config=CONFIG, max_retries=1)
    assert len(requests) == 2
