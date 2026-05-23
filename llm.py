"""Shared LM management for DSPy.

DSPy's global ``dspy.configure(lm=...)`` pins the LM to whichever thread
first called it. Streamlit reruns the script on a different ScriptRunner
thread when the user clicks Restart, which would then raise
``RuntimeError: dspy.settings can only be changed by the thread that
initially configured it``.

Instead we keep the ``dspy.LM`` instance as a module-level singleton and
expose ``lm_context()`` — a context manager that scopes the LM for the
current thread only. Every DSPy predictor call site wraps itself in this
context, so the global settings stay untouched and restarts are safe.
"""

from __future__ import annotations

import dspy


_lm: dspy.LM | None = None
_api_key: str | None = None


def set_api_key(api_key: str) -> None:
    """Register the Gemini key. Resets the cached LM if it changed."""
    global _lm, _api_key
    key = (api_key or "").strip()
    if key != _api_key:
        _lm = None
        _api_key = key


def get_lm() -> dspy.LM:
    """Return the singleton ``dspy.LM`` for the current API key."""
    global _lm
    if _lm is None:
        if not _api_key:
            raise ValueError(
                "API key not set. Call llm.set_api_key(api_key) first "
                "(usually done inside build_graph)."
            )
        _lm = dspy.LM(
            model="gemini/gemini-2.5-flash",
            api_key=_api_key,
            temperature=0.7,
        )
    return _lm


def lm_context():
    """Thread-local context manager that scopes the LM for DSPy.

    Usage::

        with lm_context():
            result = my_predictor(...)
    """
    return dspy.settings.context(lm=get_lm())
