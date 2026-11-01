"""Put the repo root on sys.path so notebooks can `from coursekit import chat`.

Notebooks live two directories down (unitN/notebooks/), so a plain import will
not find the package. One line at the top of a notebook fixes it:

    import sys; sys.path.insert(0, "../..")
    from coursekit import chat

This module exists for scripts that would rather not hardcode that.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure() -> str:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    return ROOT
