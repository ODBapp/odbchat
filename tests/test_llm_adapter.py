import pytest

from server.llm_adapter import LLM


def test_llm_status_reports_health(monkeypatch):
    # Force predictable ping result
    monkeypatch.setattr(LLM, "_ping_backend", lambda: (True, None))
    LLM.last_error = None
    status = LLM.get_status()
    assert status["reachable"] is True
    assert status["healthy"] is True


def test_llm_status_unhealthy(monkeypatch):
    monkeypatch.setattr(LLM, "_ping_backend", lambda: (False, "boom"))
    LLM.last_error = "ReadTimeout"
    status = LLM.get_status()
    assert status["reachable"] is False
    assert status["healthy"] is False
    assert "ReadTimeout" in status["last_error"]
