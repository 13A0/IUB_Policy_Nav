"""Shared helpers for every CSCI-B 490 lab.

    from coursekit import chat, embed, models

One module, one endpoint, one place to change a model name.
"""
from .nrp_llm import (BASE_URL, CHAT_MODEL, EMBED_MODEL, MissingKey,  # noqa: F401
                      chat, client, converse, embed, models)
