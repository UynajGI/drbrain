"""CLI contracts for durable autoresearch operator controls."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from drbrain.cli.autoresearch_commands import autoresearch_app
from drbrain.loop.store import RunLedger
from drbrain.loop.transitions import TransitionService

runner = CliRunner()


def _manual_review_run(tmp_path) -> tuple[dict[str, object], str, str]:
    run_dir = tmp_path / "autoresearch"
    ledger = RunLedger(run_dir / "ledger.sqlite3")
    run = ledger.get_or_create_run("CLI governed topic")
    transitions = TransitionService(ledger)
    transitions.start_run(run.run_id)
    step_id = transitions.begin_cycle(run.run_id, cycle=1, worker_id="worker", lease_seconds=60)
    transitions.mark_manual_review(run.run_id, step_id=step_id, reason="interrupted external work")
    return {"autoresearch": {"run_dir": str(run_dir)}}, run.run_id, step_id


def test_status_reports_manual_review_steps_as_json(tmp_path):
    cfg, run_id, step_id = _manual_review_run(tmp_path)

    result = runner.invoke(autoresearch_app, ["status", run_id, "--json"], obj={"config": cfg})

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_id"] == run_id
    assert payload["manual_review_steps"] == [step_id]


def test_status_never_creates_a_ledger_for_an_unknown_run_dir(tmp_path):
    run_dir = tmp_path / "missing-autoresearch"

    result = runner.invoke(
        autoresearch_app,
        ["status", "unknown topic"],
        obj={"config": {"autoresearch": {"run_dir": str(run_dir)}}},
    )

    assert result.exit_code == 1
    assert not (run_dir / "ledger.sqlite3").exists()


def test_trace_and_audit_commands_read_the_existing_run(tmp_path):
    cfg, run_id, _ = _manual_review_run(tmp_path)

    trace = runner.invoke(autoresearch_app, ["trace", run_id], obj={"config": cfg})
    audit = runner.invoke(autoresearch_app, ["audit", run_id], obj={"config": cfg})

    assert trace.exit_code == 0, trace.output
    assert json.loads(trace.output)["run_id"] == run_id
    assert audit.exit_code == 0, audit.output
    assert json.loads(audit.output)["run_id"] == run_id


def test_evidence_command_reads_the_existing_run_without_starting_it(tmp_path):
    cfg, run_id, _ = _manual_review_run(tmp_path)

    result = runner.invoke(autoresearch_app, ["evidence", run_id], obj={"config": cfg})

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_id"] == run_id
    assert payload["claims"] == []


def test_preflight_reports_mcp_tools_that_would_be_hidden(tmp_path):
    cfg = {
        "autoresearch": {
            "run_dir": str(tmp_path / "autoresearch"),
            "mcp_servers": [
                {
                    "id": "catalog",
                    "command": "catalog-mcp",
                    "trusted": True,
                    "allowed_tools": ["search"],
                }
            ],
            "step_capabilities": {"retrieve": ["mcp:catalog:search"]},
        }
    }

    result = runner.invoke(autoresearch_app, ["preflight", "--json"], obj={"config": cfg})

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["servers"][0]["status"] == "blocked"
    assert payload["servers"][0]["issues"] == ["side_effect must be classified for durable use"]


def test_pause_and_cancel_commands_change_the_existing_run(tmp_path):
    cfg, run_id, _ = _manual_review_run(tmp_path)

    paused = runner.invoke(autoresearch_app, ["pause", run_id, "--json"], obj={"config": cfg})
    cancelled = runner.invoke(
        autoresearch_app,
        ["cancel", run_id, "--reason", "operator ended the topic", "--json"],
        obj={"config": cfg},
    )

    assert paused.exit_code == 0, paused.output
    assert json.loads(paused.output)["status"] == "paused"
    pause_trace = runner.invoke(autoresearch_app, ["trace", run_id], obj={"config": cfg})
    assert pause_trace.exit_code == 0, pause_trace.output
    assert json.loads(pause_trace.output)["events"][-1]["actor"] == "operator"
    assert cancelled.exit_code == 0, cancelled.output
    assert json.loads(cancelled.output)["status"] == "cancelled"


def test_resolve_manual_review_is_explicit_and_auditable(tmp_path):
    cfg, run_id, step_id = _manual_review_run(tmp_path)

    result = runner.invoke(
        autoresearch_app,
        [
            "resolve-manual-review",
            run_id,
            "--step",
            step_id,
            "--reason",
            "operator checked the remote job",
            "--json",
        ],
        obj={"config": cfg},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["manual_review_steps"] == []
