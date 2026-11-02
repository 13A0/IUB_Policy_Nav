"""Thin helpers around the NRP managed LLM service.

Every lab in this course imports from here so that the endpoint, the key
handling and the default model live in exactly one place.

    from nrp_llm import chat, embed, client

The API is OpenAI-compatible, so anything you learn here transfers directly to
the OpenAI, Together, Groq or vLLM clients.

Endpoint : https://ellm.nrp-nautilus.io/v1
Token    : https://nrp.ai/llmtoken  ->  export NRP_API_KEY=...
"""
from __future__ import annotations

import os
from typing import Iterable

from openai import OpenAI

BASE_URL = "https://ellm.nrp-nautilus.io/v1"

# Sensible defaults for coursework. Override per call when a lab asks you to.
CHAT_MODEL = "gpt-oss"              # fast, reliable, 131K context
EMBED_MODEL = "qwen3-embedding"     # embeddings only — never call it for chat


class MissingKey(RuntimeError):
    """Raised with instructions rather than a bare KeyError."""


ENV_FILE = os.path.expanduser("~/doc/genai_appsAI_Lab/.env")


def _from_env_file(name: str) -> str | None:
    """Read one KEY=value out of ~/persistent/.env.

    The workspace writes the key there and sources it from ~/.bashrc, so a
    login shell already has it. A Streamlit app started from the Coder app
    button, or a Jupyter kernel started by VS Code, may not go through
    .bashrc at all -- so read the file directly rather than assuming.
    """
    try:
        with open(ENV_FILE) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == name:
                    return v.strip().strip("'\"") or None
    except OSError:
        pass
    return None


def _api_key() -> str:
    """Resolve the key, preferring ~/persistent/.env over the environment.

    The file is checked FIRST on purpose. A process inherits its environment
    once, at start, so a Streamlit app or Jupyter kernel launched at workspace
    boot keeps whatever key was current then -- for the whole session. Edit
    .env to fix a bad key and the running app would carry on using the old one
    and keep failing, because the stale environment copy won.

    Reading the file on every call means a corrected key takes effect on the
    next request, with no restart and no new terminal. The environment stays
    as a fallback for anyone who deliberately exports a key.
    """
    key = (_from_env_file("OPENAI_API_KEY")
           or _from_env_file("NRP_API_KEY")
           or os.environ.get("NRP_API_KEY")
           or os.environ.get("OPENAI_API_KEY"))
    if not key:
        raise MissingKey(
            "No API key found.\n"
            f"  1. Get a token at https://nrp.ai/llmtoken\n"
            f"  2. Put it in {ENV_FILE} as:  OPENAI_API_KEY=sk-...\n"
            "  3. Save the file. Nothing to restart: it is re-read on each call\n"
            "Never paste the token into a notebook or commit it to a repo."
        )
    return key


# The OpenAI client defaults to a 600s timeout, so a slow or wedged endpoint
# looks exactly like a hung script for ten minutes. Fail in 60 instead.
TIMEOUT = 60.0


def client() -> OpenAI:
    """An OpenAI client pointed at NRP."""
    return OpenAI(api_key=_api_key(), base_url=BASE_URL, timeout=TIMEOUT)


def chat(prompt: str, *, system: str | None = None, model: str = CHAT_MODEL,
         temperature: float = 0.7, **kw) -> str:
    """One-shot prompt in, string out. The 90% case for these labs."""
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    resp = client().chat.completions.create(
        model=model, messages=messages, temperature=temperature, **kw)
    return resp.choices[0].message.content


def converse(messages: list[dict], *, model: str = CHAT_MODEL, **kw) -> str:
    """Full message list in, string out — use once you need multi-turn memory."""
    resp = client().chat.completions.create(model=model, messages=messages, **kw)
    return resp.choices[0].message.content


def embed(texts: str | Iterable[str], *, model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed one string or many. Always returns a list of vectors."""
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    resp = client().embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def models() -> list[str]:
    """Every model id this token may call."""
    return sorted(m.id for m in client().models.list().data)


if __name__ == "__main__":
    print("Models available to you:")
    for m in models():
        print("  -", m)
    print("\nSmoke test:", chat("Reply with exactly: ok", temperature=0))
