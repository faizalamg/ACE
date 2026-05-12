"""Regression tests for the user-scope GSD consensus runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER_PATH = Path.home() / ".copilot" / "skills" / "gsd-consensus" / "consensus_runner.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("gsd_consensus_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_diversity_slot_models_returns_dynamic_slots(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(
        runner,
        "get_diversity_slots",
        lambda: [
            {"slot": "A", "model": "qwen/qwen3", "family": "qwen"},
            {"slot": "B", "model": "minimax/minimax-m2.5", "family": "minimax"},
        ],
    )

    selected = runner.select_diversity_slot_models()

    assert selected == {
        "advocate": "diversity-slot-a",
        "critic": "diversity-slot-b",
    }


def test_actual_diversity_collision_fails_fast():
    runner = load_runner()
    results = {
        "advocate": {"success": True, "model_actual": "qwen/qwen3"},
        "critic": {"success": True, "model_actual": "qwen/qwen3-coder"},
    }

    try:
        runner.fail_if_actual_diversity_invalid(results)
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected SystemExit(2) for colliding actual model families")


def test_call_model_records_response_model(monkeypatch):
    runner = load_runner()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return b'{"model":"minimax/minimax-m2.5","choices":[{"message":{"content":"ok"}}]}'

    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda req, timeout: FakeResponse())

    result = runner.call_model("diversity-slot-b", "system", "user")

    assert result["model_requested"] == "diversity-slot-b"
    assert result["model_actual"] == "minimax/minimax-m2.5"
