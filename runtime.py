"""Runtime helpers: Hermes home resolution and a minimal child-process environment.

Core review required two things of this module:

* the Hermes home must come from ``hermes_constants.get_hermes_home()`` rather than
  ``os.environ``, so a multi-profile gateway writes transcripts and models to the
  profile that is actually running;
* child processes must never inherit the full Hermes environment, which carries
  provider API keys.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

# Environment variables a child may keep. Everything else is dropped.
_BASE_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TZ")

# Only these extra keys are forwarded, per child kind.
_CAPTURE_KEEP = ("XDG_RUNTIME_DIR", "PULSE_SERVER", "PULSE_RUNTIME_PATH", "DISPLAY", "WAYLAND_DISPLAY")


def hermes_home() -> Path:
    """Resolve the Hermes home for the running profile, preferring core's resolver."""
    try:  # core is imported lazily so the plugin also works outside a Hermes process
        from hermes_constants import get_hermes_home as _core_home  # type: ignore

        return Path(_core_home()).expanduser().resolve()
    except Exception:
        env_home = os.environ.get("HERMES_HOME", "").strip()
        if env_home:
            return Path(env_home).expanduser().resolve()
        return (Path.home() / ".hermes").resolve()


def minimal_env(kind: str = "generic", extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an explicit environment for a child process.

    ``kind`` selects the allowlist: ``"capture"`` adds the audio session variables,
    ``"download"`` adds the Hugging Face cache and token, ``"generic"`` adds nothing.
    """
    env: dict[str, str] = {}
    for key in _BASE_KEEP:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value

    if kind == "capture":
        for key in _CAPTURE_KEEP:
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
    elif kind == "download":
        for key in ("HF_HOME", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            value = os.environ.get(key)
            if value is not None:
                env[key] = value

    if extra:
        for key, value in extra.items():
            env[key] = str(value)
    return env
