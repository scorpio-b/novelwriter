"""Run-level regression: real routing, SDK, validators, compiler and persistence."""

import json

import pytest

from app.config import get_settings
from app.core.copilot.service import create_run, execute_copilot_run, open_or_reuse_session
from app.models import Chapter, CopilotRun, WorldEntity
from tests.copilot.test_resource_lifecycle import (
    _seed_runtime, runtime_database as runtime_database,
)
from tests.structured_output_support import (
    CONFIG, SCHEMA_REQUIRED, completion, failure,
    structured_provider as structured_provider,
)

BAD = '{"answer":"人物有"谨慎"性格。","suggestions":[]}'
GOOD = '{"answer":"人物谨慎。","cited_evidence_indices":[],"suggestions":[]}'


def seed_research(factory):
    novel_id, _, _, _, _ = _seed_runtime(factory)
    with factory() as db:
        session, _ = open_or_reuse_session(db, novel_id, 1, "research", "whole_book", None, "zh", "")
        run = create_run(db, session, 1, "探索全书的人物与关系，依据已有章节给出建议")
        return novel_id, run.run_id


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["natural", "wrap_up", "one_shot"])
async def test_all_final_paths_use_schema_and_persist_validated_result(
    runtime_database, structured_provider, monkeypatch, path,
):
    engine, factory, _ = runtime_database
    novel_id, run_id = seed_research(factory)
    p = structured_provider
    p.inspect_request = lambda request: assert_no_db_lease(engine)
    if path == "natural":
        p.replies.append(completion(BAD))
    elif path == "wrap_up":
        monkeypatch.setattr(get_settings(), "copilot_max_tool_rounds", 0)
    else:
        p.replies.append(failure(400, "tools not supported"))
    p.replies.extend([failure(400, SCHEMA_REQUIRED), completion(GOOD)])

    await execute_copilot_run(run_id, novel_id, 1, CONFIG)

    with factory() as db:
        run = db.query(CopilotRun).filter_by(run_id=run_id).one()
        assert run.status == "completed"
        assert run.answer == "人物谨慎。"
        assert run.suggestions_json == []
        assert isinstance(run.evidence_json, list)
        assert db.query(Chapter).one().content == "张三在宗门修行。"
        assert db.query(WorldEntity).one().description == "主角"
    assert p.replies == []
    assert p.requests[-1]["response_format"]["type"] == "json_schema"
    assert "CopilotFinalResponse" in json.dumps(p.requests[-1]["response_format"])
    assert "tools" not in p.requests[-1]
    assert len(p.requests) == (2 if path == "wrap_up" else 3)


def assert_no_db_lease(engine):
    assert engine.pool.checkedout() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["natural", "wrap_up", "one_shot"])
async def test_invalid_final_output_errors_without_nested_fallback_or_world_mutation(
    runtime_database, structured_provider, monkeypatch, path,
):
    _, factory, _ = runtime_database
    novel_id, run_id = seed_research(factory)
    p = structured_provider
    if path == "wrap_up":
        monkeypatch.setattr(get_settings(), "copilot_max_tool_rounds", 0)
    if path == "one_shot":
        p.replies.append(failure(400, "tools not supported"))
    p.replies.extend([completion(BAD), completion(BAD)])

    await execute_copilot_run(run_id, novel_id, 1, CONFIG)

    with factory() as db:
        run = db.query(CopilotRun).filter_by(run_id=run_id).one()
        assert run.status == "error"
        assert not run.answer
        assert not run.suggestions_json
        assert db.query(WorldEntity).count() == 1
        assert db.query(WorldEntity).one().description == "主角"
        assert db.query(Chapter).one().content == "张三在宗门修行。"
        if path == "natural":
            assert run.workspace_json["final_answer_draft"] == BAD
            assert run.workspace_json["messages"]
    assert len(p.requests) == (3 if path == "one_shot" else 2)


@pytest.mark.asyncio
async def test_tool_evidence_is_preserved_for_repair_and_not_searched_again(
    runtime_database, structured_provider,
):
    _, factory, _ = runtime_database
    novel_id, run_id = seed_research(factory)
    p = structured_provider
    p.replies.extend([
        completion(tool_calls=[{"id": "find-1", "type": "function", "function": {
            "name": "find", "arguments": '{"query":"张三"}',
        }}], finish="tool_calls"),
        completion(BAD), completion(GOOD),
    ])
    await execute_copilot_run(run_id, novel_id, 1, CONFIG)
    assert len(p.requests) == 3
    assert "response_format" not in p.requests[0]
    assert "response_format" not in p.requests[1]
    assert "tools" not in p.requests[2]
    old_tools = [m for m in p.requests[1]["messages"] if m["role"] == "tool"]
    assert old_tools
    assert old_tools == [m for m in p.requests[2]["messages"] if m["role"] == "tool"]
    with factory() as db:
        run = db.query(CopilotRun).filter_by(run_id=run_id).one()
        assert run.status == "completed"
        assert run.workspace_json["tool_call_count"] == 1
        assert run.workspace_json["tool_journal"]
        assert json.loads(run.workspace_json["final_answer_draft"])["answer"] == "人物谨慎。"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 429, 503])
async def test_business_run_does_not_retry_provider_failure_as_formatting(
    runtime_database, structured_provider, status,
):
    _, factory, _ = runtime_database
    novel_id, run_id = seed_research(factory)
    p = structured_provider
    p.replies.extend([completion(BAD), failure(status, "provider unavailable")])
    await execute_copilot_run(run_id, novel_id, 1, CONFIG)
    assert len(p.requests) == 2
    with factory() as db:
        run = db.query(CopilotRun).filter_by(run_id=run_id).one()
        assert run.status == "error"
        assert not run.answer


@pytest.mark.asyncio
async def test_truncated_answer_cannot_be_completed_or_restarted_as_one_shot(
    runtime_database, structured_provider,
):
    _, factory, _ = runtime_database
    novel_id, run_id = seed_research(factory)
    p = structured_provider
    p.replies.append(completion(GOOD, finish="length"))
    await execute_copilot_run(run_id, novel_id, 1, CONFIG)
    assert len(p.requests) == 1
    with factory() as db:
        assert db.query(CopilotRun).filter_by(run_id=run_id).one().status == "error"


@pytest.mark.asyncio
async def test_ordinary_chat_needs_no_extra_request_or_queue_slot(
    runtime_database, structured_provider, monkeypatch,
):
    _, factory, _ = runtime_database
    novel_id, run_id = seed_research(factory)
    with factory() as db:
        run = db.query(CopilotRun).filter_by(run_id=run_id).one()
        run.prompt = "你好"
        db.commit()
    events = []

    async def acquire():
        events.append("acquire")

    monkeypatch.setattr("app.core.llm_semaphore.acquire_llm_slot", acquire)
    monkeypatch.setattr("app.core.llm_semaphore.release_llm_slot", lambda: events.append("release"))
    structured_provider.replies.append(completion("你好，我可以帮助你整理人物。"))
    await execute_copilot_run(run_id, novel_id, 1, CONFIG)
    assert len(structured_provider.requests) == 1
    assert events == ["acquire", "release"]
    with factory() as db:
        run = db.query(CopilotRun).filter_by(run_id=run_id).one()
        assert run.status == "completed"
        assert run.answer == "你好，我可以帮助你整理人物。"


@pytest.mark.asyncio
async def test_lost_lease_before_repair_does_not_call_model_or_commit(
    runtime_database, structured_provider, monkeypatch,
):
    _, factory, _ = runtime_database
    novel_id, run_id = seed_research(factory)
    p = structured_provider
    p.replies.append(completion(BAD))
    events = []

    async def acquire():
        events.append("acquire")
        if events.count("acquire") == 2:
            with factory() as db:
                run = db.query(CopilotRun).filter_by(run_id=run_id).one()
                run.lease_owner = "new-worker"
                db.commit()

    monkeypatch.setattr("app.core.llm_semaphore.acquire_llm_slot", acquire)
    monkeypatch.setattr("app.core.llm_semaphore.release_llm_slot", lambda: events.append("release"))
    await execute_copilot_run(run_id, novel_id, 1, CONFIG)
    assert len(p.requests) == 1
    assert events == ["acquire", "release", "acquire", "release"]
    with factory() as db:
        run = db.query(CopilotRun).filter_by(run_id=run_id).one()
        assert run.lease_owner == "new-worker"
        assert run.status == "running"
        assert not run.answer


@pytest.mark.asyncio
async def test_repaired_suggestions_still_require_live_target_validation_and_approval(
    runtime_database, structured_provider,
):
    _, factory, _ = runtime_database
    novel_id, run_id = seed_research(factory)
    p = structured_provider
    good = json.dumps({"answer": "存在一项待核对建议。", "suggestions": [{
        "kind": "update_entity", "title": "补充性格", "summary": "建议核对",
        "target_resource": "entity", "target_id": 999,
        "cited_evidence_indices": [999], "delta": {"description": "谨慎"},
    }]}, ensure_ascii=False)
    p.replies.extend([completion(BAD), completion(good)])
    await execute_copilot_run(run_id, novel_id, 1, CONFIG)
    with factory() as db:
        run = db.query(CopilotRun).filter_by(run_id=run_id).one()
        assert run.status == "completed"
        assert len(run.suggestions_json) == 1
        assert run.suggestions_json[0]["preview"]["actionable"] is False
        assert run.suggestions_json[0]["apply"] is None
        assert db.query(WorldEntity).count() == 1
        assert db.query(WorldEntity).one().description == "主角"
