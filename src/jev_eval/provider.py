"""Jev provider: POST /v1/systemone. Zero runtime deps (stdlib urllib)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

_RETRYABLE = frozenset({429, 529})
_REJECTED = frozenset({400, 413, 422})
DEFAULT_MODEL = "jev-latest"
DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"


class Provider(Protocol):
    """Anything that can answer typed questions about a state."""

    def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]: ...


class RequestRejected(RuntimeError):
    """Jev refused this one request (e.g. a state over the context limit); the run carries on."""


def _post_json(url: str, headers: dict[str, str], body: dict[str, Any], attempts: int = 4) -> dict[str, Any]:
    """POST JSON with exponential backoff on 429/529 (the retry the docs recommend)."""
    data = json.dumps(body).encode("utf-8")
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(  # noqa: S310 - fixed https base_url
            url, data=data, method="POST", headers={"Content-Type": "application/json", **headers}
        )
        try:
            with urllib.request.urlopen(request) as response:  # noqa: S310
                parsed: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return parsed
        except urllib.error.HTTPError as err:
            if err.code in _RETRYABLE and attempt < attempts:
                time.sleep(0.25 * 2 ** (attempt - 1))
                continue
            if err.code in _REJECTED:
                raise RequestRejected(f"HTTP {err.code}: {err.read().decode('utf-8', 'replace')[:200]}") from err
            raise
    raise RuntimeError("unreachable")


class TypeSafeProvider:
    """First-party Jev client (verified against docs.typesafe.ai/api)."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL) -> None:
        self._api_key = api_key
        self.model = model
        self._base_url = base_url

    def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return _post_json(
            f"{self._base_url}/systemone",
            {"Authorization": f"Bearer {self._api_key}"},
            {"model": self.model, "state": state, "questions": questions},
        )


def provider_from_env(model: str = DEFAULT_MODEL) -> TypeSafeProvider:
    """Build a provider from JEV_API_KEY, honoring TYPESAFE_AI_BASE_URL for a self-host/proxy/mock."""
    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise RuntimeError("JEV_API_KEY is not set (get a key at https://typesafe.ai/)")
    base_url = os.environ.get("TYPESAFE_AI_BASE_URL") or DEFAULT_BASE_URL
    return TypeSafeProvider(key, model=model, base_url=base_url.rstrip("/"))
