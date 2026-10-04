from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

import pytest
from pytest import MonkeyPatch

from jev_eval import TypeSafeProvider, provider_from_env


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_evaluate_posts_and_parses(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any) -> _Response:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["auth"] = request.get_header("Authorization")
        return _Response({"model": "jev-latest", "answers": {"u": {"noul": 0.9}}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = TypeSafeProvider("k").evaluate({"text": "x"}, {"u": {"type": "noul"}})

    assert out["answers"]["u"]["noul"] == 0.9
    assert captured["url"].endswith("/v1/systemone")
    assert captured["body"]["model"] == "jev-latest"
    assert captured["auth"] == "Bearer k"


def test_retries_on_429(monkeypatch: MonkeyPatch) -> None:
    calls = {"n": 0}

    def flaky(request: Any) -> _Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "rate limited", {}, None)  # type: ignore[arg-type]
        return _Response({"answers": {}})

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    assert TypeSafeProvider("k").evaluate({}, {}) == {"answers": {}}
    assert calls["n"] == 2


def test_provider_from_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="JEV_API_KEY"):
        provider_from_env()
    monkeypatch.setenv("JEV_API_KEY", "k")
    monkeypatch.setenv("TYPESAFE_AI_BASE_URL", "http://localhost:9/v1/")
    provider = provider_from_env("jev-1")
    assert provider.model == "jev-1"
    assert provider._base_url == "http://localhost:9/v1"
