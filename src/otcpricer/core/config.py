"""Runtime configuration.

Numerical defaults (grid sizes, path counts, tolerances) were previously
hardcoded as literals scattered through the engines and the Flask layer, which
made them impossible to tune without editing pricing code. They live here
instead, overridable by environment variable so a run can be made faster for
an interactive quote or more accurate for a validation report.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "DataSettings",
    "NumericalSettings",
    "ServerSettings",
    "Settings",
    "get_settings",
]


class NumericalSettings(BaseModel):
    """Defaults for the numerical engines."""

    model_config = {"frozen": True}

    pde_space_nodes: int = Field(400, ge=50, description="PDE spatial grid nodes")
    pde_steps_per_year: int = Field(
        252, ge=12, description="PDE time steps per year; 252 aligns with trading days"
    )
    rannacher_steps: int = Field(
        4,
        ge=0,
        description=(
            "Fully-implicit start-up steps damping the non-smooth payoff before "
            "Crank-Nicolson takes over; 2-4 is the standard choice"
        ),
    )
    mc_paths: int = Field(50_000, ge=1_000, description="Default Monte Carlo path count")
    mc_seed: int = Field(42, description="Default RNG seed; fixed so quotes are reproducible")
    iv_tolerance: float = Field(1e-8, gt=0, description="Implied-vol solver price tolerance")
    iv_max_iterations: int = Field(100, ge=1, description="Implied-vol solver iteration cap")


class DataSettings(BaseModel):
    """Market data sourcing."""

    model_config = {"frozen": True}

    source: str = Field(
        "snapshot",
        description=(
            "'akshare' to hit the live feed, 'snapshot' to read data/snapshots, "
            "'csv' for a user-supplied file. Defaults to snapshot so tests and "
            "CI never depend on the network."
        ),
    )
    snapshot_dir: str = Field("data/snapshots", description="Offline market data snapshots")
    wind_dir: str = Field(
        "",
        description=(
            "Directory holding the six Wind panel files. Empty by default: the "
            "panel is institutional data and is not committed, so an "
            "unconfigured checkout must not assume it is present."
        ),
    )
    cache_ttl_seconds: int = Field(300, ge=0, description="How long a live quote pull stays cached")
    request_timeout_seconds: float = Field(20.0, gt=0, description="Per-request network timeout")
    max_concurrent_requests: int = Field(
        8, ge=1, description="Concurrency cap when fanning out an option chain pull"
    )


class ServerSettings(BaseModel):
    """How the web application is served.

    These exist because the entry point was ``app.run(debug=True, port=5000)``
    -- the Flask development server with the debugger on. That is fine for
    developing and is not something to hand to anyone else: the reloader
    double-executes module state, the debugger exposes an interactive console
    on any traceback, and the server is single-threaded, so one 10-second
    report blocks every other request.

    :mod:`serve` reads these. ``python app.py`` still runs the development
    server deliberately, so the two paths cannot be confused.
    """

    model_config = {"frozen": True}

    host: str = Field(
        "127.0.0.1",
        description=(
            "Interface to bind. Loopback by default: this tool reports prices "
            "with no authentication of any kind, so binding 0.0.0.0 is an "
            "explicit decision rather than a default."
        ),
    )
    port: int = Field(5000, ge=1, le=65535, description="Port to serve on")
    threads: int = Field(
        8,
        ge=1,
        description=(
            "Waitress worker threads. Serves concurrent requests; it does not "
            "make any single request faster -- the PDE time loop is Python "
            "level and GIL-bound, measured at 0.55x speedup when the solves "
            "within one request were threaded."
        ),
    )
    debug: bool = Field(
        False,
        description=(
            "Development server with the interactive debugger. False here and "
            "overridable only through OTCPRICER_DEBUG, so production never "
            "gets it by inheriting a default."
        ),
    )
    reload_templates: bool = Field(
        True,
        description=(
            "Re-read the page template when it changes on disk. On by "
            "default, which is a deliberate departure from what 'production' "
            "usually means: Flask caches templates whenever the debugger is "
            "off, so this entry point served a stale page after every "
            "front-end edit, silently -- a cached page renders perfectly. "
            "The cost is one stat() per request against one template. Set "
            "OTCPRICER_RELOAD_TEMPLATES=0 to cache instead."
        ),
    )


class Settings(BaseModel):
    """Top-level settings object."""

    model_config = {"frozen": True}

    numerical: NumericalSettings = Field(default_factory=NumericalSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    log_level: str = Field("INFO")
    log_format: str = Field("text")


def _env_int(name: str) -> int | None:
    """Read an integer environment override, or ``None`` if unset."""
    raw = os.getenv(name)
    return None if raw is None else int(raw)


def _env_float(name: str) -> float | None:
    """Read a float environment override, or ``None`` if unset."""
    raw = os.getenv(name)
    return None if raw is None else float(raw)


def _env_bool(name: str) -> bool | None:
    """Read a boolean environment override, or ``None`` if unset.

    Only the listed spellings count as true. An unrecognised value is an
    error rather than a silent false: ``OTCPRICER_DEBUG=no`` meaning "yes,
    debug" would be the worst possible way to learn about this function.
    """
    raw = os.getenv(name)
    if raw is None:
        return None
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean; use one of 1/true/yes/on or 0/false/no/off")


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    """Strip unset overrides so pydantic falls back to each field default."""
    return {k: v for k, v in values.items() if v is not None}


def _build_settings() -> Settings:
    """Assemble settings from ``OTCPRICER_*`` environment overrides.

    Flat variable names map onto nested fields, e.g. ``OTCPRICER_MC_PATHS``
    sets ``numerical.mc_paths`` and ``OTCPRICER_DATA_SOURCE`` sets
    ``data.source``. Anything unset keeps its model default.
    """
    numerical = NumericalSettings(
        **_drop_none(
            {
                "pde_space_nodes": _env_int("OTCPRICER_PDE_SPACE_NODES"),
                "pde_steps_per_year": _env_int("OTCPRICER_PDE_STEPS_PER_YEAR"),
                "rannacher_steps": _env_int("OTCPRICER_RANNACHER_STEPS"),
                "mc_paths": _env_int("OTCPRICER_MC_PATHS"),
                "mc_seed": _env_int("OTCPRICER_MC_SEED"),
                "iv_max_iterations": _env_int("OTCPRICER_IV_MAX_ITERATIONS"),
                "iv_tolerance": _env_float("OTCPRICER_IV_TOLERANCE"),
            }
        )
    )
    data = DataSettings(
        **_drop_none(
            {
                "source": os.getenv("OTCPRICER_DATA_SOURCE"),
                "snapshot_dir": os.getenv("OTCPRICER_SNAPSHOT_DIR"),
                "wind_dir": os.getenv("OTCPRICER_WIND_DIR"),
                "cache_ttl_seconds": _env_int("OTCPRICER_CACHE_TTL_SECONDS"),
                "max_concurrent_requests": _env_int("OTCPRICER_MAX_CONCURRENT_REQUESTS"),
                "request_timeout_seconds": _env_float("OTCPRICER_REQUEST_TIMEOUT_SECONDS"),
            }
        )
    )
    server = ServerSettings(
        **_drop_none(
            {
                "host": os.getenv("OTCPRICER_HOST"),
                "port": _env_int("OTCPRICER_PORT"),
                "threads": _env_int("OTCPRICER_THREADS"),
                "debug": _env_bool("OTCPRICER_DEBUG"),
                "reload_templates": _env_bool("OTCPRICER_RELOAD_TEMPLATES"),
            }
        )
    )
    return Settings(
        numerical=numerical,
        data=data,
        server=server,
        **_drop_none(
            {
                "log_level": os.getenv("OTCPRICER_LOG_LEVEL"),
                "log_format": os.getenv("OTCPRICER_LOG_FORMAT"),
            }
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, built once from the environment.

    Cached, so settings are read at first use rather than at import time;
    tests can clear the cache with ``get_settings.cache_clear()``.
    """
    return _build_settings()
