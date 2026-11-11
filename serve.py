"""Serve the web application for use by someone other than its author.

Why this file exists
--------------------
The entry point was ``app.run(debug=True, port=5000)`` -- the Flask development
server with the interactive debugger on. That is the right thing while writing
the code and the wrong thing to hand anyone, for three separate reasons:

**The debugger is a remote console.** Any unhandled traceback renders a page
with an interactive Python prompt. This application has no authentication of
any kind, so on anything but loopback that is a shell.

**The reloader double-executes module state.** ``app.py`` builds
process-wide caches on import -- ``_SURFACE_CACHE``, ``_LOCAL_VOL_CACHE``, and
``_WIND_PANEL``, which holds 34 MB of parsed parquet. Under the reloader those
are built twice and one copy is orphaned.

**It is single-threaded.** ``/api/snowball_report`` takes 11 seconds at its
default settings and a surface-path risk computation takes 31. On the
development server one of those blocks every other request for its duration,
including the page load.

So: waitress. Not gunicorn -- it does not run on Windows, which is where this
is developed and launched.

What this does not fix
----------------------
Stated plainly, because a production-looking server invites the assumption that
the rest is production too:

- **No authentication, no authorisation, no rate limiting.** Anyone who can
  reach the port can price anything and can make the process spend 31 seconds
  of CPU per request. The default bind is loopback for exactly that reason.
- **No request timeouts.** A pathological grid request runs to completion.
- **No HTTPS.** Terminate TLS in front of it if it ever leaves loopback.
- **Threads do not make a request faster.** They serve concurrent requests.
  Within one request the solves are GIL-bound -- threading them was measured at
  0.55x, i.e. 1.8 times *slower* on a 32-core machine, because the PDE time
  loop is Python-level.

Usage
-----
``python serve.py``, or ``run_app.ps1`` / ``run_app.bat`` which pin the
interpreter. Configuration is by environment variable, in the same
``OTCPRICER_*`` scheme as everything else:

===========================  ==========================================
``OTCPRICER_HOST``           interface to bind (default ``127.0.0.1``)
``OTCPRICER_PORT``           port (default ``5000``)
``OTCPRICER_THREADS``        waitress worker threads (default ``8``)
``OTCPRICER_RELOAD_TEMPLATES`` re-read the page when it changes (default on)
``OTCPRICER_DEBUG``          ``1`` to use the development server instead
===========================  ==========================================

``python app.py`` still starts the development server directly. That is
deliberate: two entry points with visibly different names, rather than one
whose behaviour depends on a variable.
"""

from __future__ import annotations

import os
import sys

# The library lives under src/, so a clone that has not been installed cannot
# import it. Appended rather than inserted: an installed copy wins, so a
# developer running against a `pip install -e .` checkout does not silently
# get this directory instead.
#
# Until now `python app.py` failed from a fresh clone with
# "ModuleNotFoundError: No module named 'otcpricer'" -- the tests worked only
# because pyproject sets pythonpath = ["src"] for pytest. For a tool meant to
# be handed over, the entry point has to run.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.append(_SRC)

from otcpricer.core.config import get_settings
from otcpricer.core.logging import get_logger

__all__ = ["main"]

_log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Serve the application. Returns a process exit code.

    Args:
        argv: Unused; present so this is callable from a test without
            touching ``sys.argv``.

    Returns:
        0 on a clean shutdown, 1 if waitress is not installed.
    """
    del argv
    settings = get_settings()
    server = settings.server

    # Imported here rather than at module scope so the module can be imported
    # and inspected -- including by a test -- on a machine without waitress,
    # and so the failure names the fix.
    try:
        from waitress import serve
    except ImportError:
        print(
            "waitress is not installed, and it is what serves this "
            "application.\n"
            "    pip install waitress\n"
            "or install the project's pinned set:\n"
            "    pip install -r requirements.txt\n"
            "\n"
            "To use the Flask development server instead -- for development "
            "only, it carries an interactive debugger and is single-threaded "
            "-- run 'python app.py'.",
            file=sys.stderr,
        )
        return 1

    from app import app as application

    # Flask caches templates unless the debugger is on, so with the debugger
    # off -- which is the entire point of this file -- a front-end edit is
    # served from the cache until the process restarts. It cost a false
    # verification while this was being built: a tab reorder was committed,
    # the page fetched over HTTP to check it, and the old order came back.
    #
    # jinja_env.auto_reload is set as well as the config key. The config key
    # alone is read when Flask builds its Jinja environment, and that has
    # already happened by the time app.py finishes importing.
    application.config["TEMPLATES_AUTO_RELOAD"] = server.reload_templates
    application.jinja_env.auto_reload = server.reload_templates

    if server.debug:
        # Reachable only through OTCPRICER_DEBUG. Announced loudly, because a
        # development server that is mistaken for this one is the failure mode
        # this file exists to prevent.
        print(
            "OTCPRICER_DEBUG is set: starting the Flask DEVELOPMENT server.\n"
            "  The interactive debugger is enabled and this process serves one "
            "request at a time.\n"
            "  Do not leave this reachable by anyone else.",
            file=sys.stderr,
        )
        application.run(host=server.host, port=server.port, debug=True)
        return 0

    banner = (
        f"  Option Pricing Tool -- serving on http://{server.host}:{server.port}\n"
        f"  waitress, {server.threads} threads, debugger off\n"
        f"  No authentication: {'loopback only' if server.host.startswith('127.') else 'BOUND PUBLICLY'}"
    )
    print("=" * 72)
    print(banner)
    print("=" * 72)
    _log.info(
        "serving",
        extra={"host": server.host, "port": server.port, "threads": server.threads},
    )

    serve(application, host=server.host, port=server.port, threads=server.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
