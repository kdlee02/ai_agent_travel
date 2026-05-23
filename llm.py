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
import dspy.dsp.utils.settings as _dspy_settings_module


_lm: dspy.LM | None = None
_api_key: str | None = None


def _reset_dspy_thread_owner() -> None:
    """Clear DSPy's thread-ownership lock so the current thread can configure.

    DSPy records which thread first called ``dspy.configure`` and refuses
    subsequent calls from any other thread.  Streamlit runs each rerun on a
    fresh ScriptRunner thread, so after a Restart the new thread would be
    blocked.  Resetting the owner here lets the current thread take over
    cleanly.
    """
    _dspy_settings_module.config_owner_thread_id = None
    _dspy_settings_module.config_owner_async_task = None


def set_api_key(api_key: str) -> None:
    """Register the Gemini key. Resets the cached LM if it changed."""
    global _lm, _api_key
    if api_key != _api_key:
        _lm = None
        _api_key = api_key
        # Allow the current (possibly new) thread to own DSPy configuration.
        _reset_dspy_thread_owner()


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

    Also resets DSPy's thread-ownership lock if the current thread is not
    the registered owner, so Streamlit reruns on new ScriptRunner threads
    don't raise ``RuntimeError: dspy.settings can only be changed by the
    thread that initially configured it``.

    Usage::

        with lm_context():
            result = my_predictor(...)
    """
    import threading
    if (
        _dspy_settings_module.config_owner_thread_id is not None
        and _dspy_settings_module.config_owner_thread_id != threading.get_ident()
    ):
        _reset_dspy_thread_owner()
    return dspy.settings.context(lm=get_lm())
