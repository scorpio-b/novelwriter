"""Real SDK requests with a deterministic, network-free HTTP provider."""

import json
from dataclasses import dataclass, field

import httpx
import pytest
from openai import AsyncOpenAI

from app.core.llm_config import ResolvedLlmConfig

CONFIG = ResolvedLlmConfig(
    base_url="http://structured.test/v1", api_key="synthetic-key", model="local-test-model",
    billing_source_hint="selfhost", source="selfhost_settings",
)
SCHEMA_REQUIRED = "'response_format.type' must be 'json_schema' or 'text'"


def completion(content=None, *, finish="stop", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return httpx.Response(200, json={
        "id": "mock-response", "object": "chat.completion", "created": 0,
        "model": CONFIG.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    })


def failure(status, message):
    return httpx.Response(status, json={"error": {"message": message}})


@dataclass
class Provider:
    replies: list = field(default_factory=list)
    requests: list = field(default_factory=list)
    clients: list = field(default_factory=list)
    usage: list = field(default_factory=list)
    inspect_request: object = None

    def handle(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        if self.inspect_request:
            self.inspect_request(body)
        if not self.replies:
            raise AssertionError("Unexpected extra LLM request")
        result = self.replies.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def make_client(self, **kwargs):
        client = AsyncOpenAI(
            **kwargs, max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
        )
        self.clients.append(client)
        return client


@pytest.fixture
def structured_provider(monkeypatch):
    provider = Provider()
    monkeypatch.setattr("app.core.ai_client.AsyncOpenAI", provider.make_client)
    monkeypatch.setattr("app.core.ai_client.ensure_ai_available_fresh_session", lambda **kw: None)
    monkeypatch.setattr("app.core.ai_client._record_usage", lambda *a, **kw: provider.usage.append((a, kw)))
    yield provider
    assert all(client.is_closed() for client in provider.clients)
