"""Small, explicitly configured client for the course NRP model service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from openai import OpenAI

BASE_URL = "https://ellm.nrp-nautilus.io/v1"
CHAT_MODEL = "gpt-oss"
EMBED_MODEL = "qwen3-embedding"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
TIMEOUT = 60.0


class MissingKey(RuntimeError):
    """The project has no usable NRP API token."""


class InvalidResponse(RuntimeError):
    """The model service returned an incomplete or unexpected response."""


def _from_env_file(name: str) -> str | None:
    """Read simple KEY=value entries, including Windows UTF-8 BOM files."""
    try:
        with open(ENV_FILE, encoding="utf-8-sig") as env_file:
            for line in env_file:
                key, separator, value = line.strip().partition("=")
                if separator and key.strip() == name:
                    return value.strip().strip("'\"") or None
    except OSError:
        return None
    return None


def _api_key() -> str:
    """Environment takes precedence; reread the project file on every call."""
    key = (os.environ.get("NRP_API_KEY", "").strip() or _from_env_file("NRP_API_KEY") or "").strip()
    if not key or key == "replace-with-your-nrp-token":
        raise MissingKey(
            f"Set NRP_API_KEY in your environment or {ENV_FILE}. "
            "Get a token at https://nrp.ai/llmtoken. Never commit it."
        )
    return key


def client() -> OpenAI:
    """Create a client; callers own its lifetime and must close it."""
    return OpenAI(api_key=_api_key(), base_url=BASE_URL, timeout=TIMEOUT, max_retries=0)


def chat(
    prompt: str,
    *,
    system: str | None = None,
    model: str = CHAT_MODEL,
    temperature: float = 0.7,
    **kwargs,
) -> str:
    """Send one prompt with optional standing instructions."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return converse(messages, model=model, temperature=temperature, **kwargs)


def converse(messages: list[dict], *, model: str = CHAT_MODEL, **kwargs) -> str:
    """Send messages and reject empty model responses before further processing."""
    with client() as service:
        response = service.chat.completions.create(model=model, messages=messages, **kwargs)
    if not response.choices or not response.choices[0].message.content:
        raise InvalidResponse("The chat service returned no answer.")
    content = response.choices[0].message.content.strip()
    if not content:
        raise InvalidResponse("The chat service returned an empty answer.")
    return content


def embed(texts: str | Iterable[str], *, model: str = EMBED_MODEL) -> list[list[float]]:
    """Return vectors in input order, checking for missing or duplicate indices."""
    texts = [texts] if isinstance(texts, str) else list(texts)
    if not texts:
        return []
    with client() as service:
        response = service.embeddings.create(model=model, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in ordered] != list(range(len(texts))):
        raise InvalidResponse("The embedding service returned incomplete vectors.")
    vectors = [item.embedding for item in ordered]
    if any(not vector or len(vector) != len(vectors[0]) for vector in vectors):
        raise InvalidResponse("The embedding service returned inconsistent vector dimensions.")
    return vectors


def models() -> list[str]:
    """List model IDs available to this token."""
    with client() as service:
        return sorted(model.id for model in service.models.list().data)
