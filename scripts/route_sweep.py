"""Call every registered route once and report what came back.

Why this exists
---------------
The test suite covers the pricing thoroughly and the routes thinly: a panel
calling an endpoint that 500s looks, from the browser, exactly like a panel
whose JavaScript is broken. Several of this project's defects were of that
shape -- ``/api/cn_fdm`` served a price 44.9% wrong with ``calc_greeks=false``,
and an echoed knock-out barrier disagreed with the one priced -- and none of
them announced itself as an error.

So this sweeps ``app.url_map`` rather than a hand-kept list, and **refuses to
pass if any route goes unexercised**. A new endpoint added without a body here
fails the sweep instead of being quietly skipped, which is the whole point:
the failure mode being guarded against is an omission, so the omission has to
be loud.

It also treats a 200 carrying an ``error`` key as a failure. Flask routes in
this application return 400 for bad input, so a 200 with an error in it means
something was swallowed.

What it does not do
-------------------
It does not check any number. Two engines, the convergence studies and the
measured error budgets do that, and duplicating it here would be a second
place to update. This answers one question: does every route still answer.

Usage::

    python scripts/route_sweep.py

Exits 0 when every route answered cleanly, 1 otherwise.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Run from a clone without an install, the same bootstrap app.py and serve.py
# use. Without it this fails with ModuleNotFoundError on otcpricer, since
# pyproject.toml puts src/ on the path for pytest only.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import app as application  # noqa: E402

SURFACE = {"underlying": "510300", "source": "snapshot", "r": 0.02}
VANILLA = {"S": 100, "K": 100, "T": 1.0, "r": 0.02, "sigma": 0.2, "q": 0.0, "option_type": "call"}
SNOWBALL = {
    "S0": 100,
    "T": 1.0,
    "KO_pct": 1.03,
    "KI_pct": 0.75,
    "coupon_pa": 0.20,
    "r": 0.02,
    "q": 0.0,
    "sigma": 0.2,
    "basis_pa": 0.0,
}

#: One request body per route. Doubles as a worked example of what each
#: endpoint wants, which is why the bodies are spelled out rather than
#: generated -- several of these fields are required with no default on
#: purpose, and a generator would paper over exactly that.
BODIES: dict[str, dict | None] = {
    "/api/calculate": VANILLA,
    "/api/iv": {**VANILLA, "market_price": 8.0},
    "/api/portfolio": {
        "positions": [{**VANILLA, "qty": 1, "entry_price": 8.0, "product": "vanilla"}]
    },
    "/api/mc_distribution": {**VANILLA, "n_simulations": 20000},
    "/api/sensitivity": VANILLA,
    "/api/vol_smile": {
        "S": 100,
        "T": 0.5,
        "r": 0.02,
        "q": 0.0,
        "rows": [
            {"K": 90, "price": 12.5},
            {"K": 100, "price": 6.2},
            {"K": 110, "price": 2.4},
        ],
    },
    "/api/smile_presets": {"S": 100, "T": 0.5, "r": 0.02, "q": 0.0, "sigma": 0.2},
    "/api/vol_surface/fit": SURFACE,
    "/api/vol_surface/query": {**SURFACE, "strikes": [3.9, 4.1, 4.3], "maturities": [0.25, 1.0]},
    "/api/vol_surface/snapshots": None,  # GET
    "/api/local_vol/surface": {**SURFACE, "horizon": 1.0, "n_spots": 9, "n_times": 5},
    "/api/local_vol/validate": SURFACE,
    "/api/european": {**SURFACE, "K": 4.1, "T": 0.5, "flag": "c"},
    "/api/heston/calibrate": SURFACE,
    "/api/heston/european": {**SURFACE, "K": 4.1, "T": 0.5, "flag": "c"},
    "/api/heston/snowball": {**SURFACE, "S0": 100, "T": 1.0, "paths": 20000},
    # monitoring is observations a year, or the string 'continuous'. Required
    # with no default: the two conventions differ by 22% on the legacy test
    # case, and the code that used each never compared them.
    "/api/shark_fin": {
        "S": 100,
        "K": 100,
        "H": 120,
        "T": 1.0,
        "r": 0.02,
        "sigma": 0.2,
        "q": 0.0,
        "direction": "up_out_call",
        "monitoring": 252,
        "rebate": 0.0,
    },
    "/api/snowball": {**SNOWBALL, "n_paths": 20000},
    "/api/snowball_pde": SNOWBALL,
    "/api/snowball_crosscheck": {**SNOWBALL, "n_paths": 20000},
    "/api/snowball_risk": SNOWBALL,
    "/api/snowball_risk_crosscheck": {**SNOWBALL, "n_paths": 20000},
    "/api/snowball_quote": SNOWBALL,
    "/api/snowball_report": {**SNOWBALL, "quote": True, "mc": True, "n_paths": 20000},
    "/api/carry": {"as_of": "2024-12-25", "tail": "hold_last"},
    "/api/forward": {
        "S": 100,
        "T": 1.0,
        "r": 0.02,
        "q": 0.0,
        "notional": 1000,
        "direction": "long",
    },
    "/api/irs": {
        "notional": 1_000_000,
        "fixed_rate": 0.025,
        "T": 5.0,
        "fixed_freq": 2,
        "float_freq": 4,
    },
    "/api/cn_fdm": {**SNOWBALL, "calc_greeks": False},
    "/api/export": {"rows": [{"a": 1, "b": 2}], "filename": "sweep"},
    "/api/marketdata/mark": {"underlying": "510300"},
    "/api/marketdata/publish": {
        "underlying": "510300",
        "source": "snapshot",
        "r": 0.02,
        "force": True,
    },
    "/api/marketdata/upload": None,  # raw CSV, see RAW_BODIES
}

#: Routes that take something other than a JSON object.
#:
#: The upload endpoint wants a CSV. A two-row file is enough -- what is
#: being swept is whether the route answers, and the column contract has
#: its own tests in tests/test_marketdata.py.
RAW_BODIES = {
    "/api/marketdata/upload": "\n".join(
        [
            "code,last,previous_settlement,strike,expiry,days_to_expiry,as_of",
            "510300C2611M04100,0.1,0.101,4.1,2026-11-26,29,2026-10-27",
            "510300P2611M04100,0.05,0.051,4.1,2026-11-26,29,2026-10-27",
            "",
        ]
    ).encode("utf-8"),
}

#: The two routes that answer GET rather than POST.
GETS = {"/", "/api/vol_surface/snapshots"}


def expected_refusals() -> dict[str, tuple[int, str]]:
    """Routes that cannot answer 200 here, and the text their refusal must carry.

    ``/api/carry`` needs the Wind panel, which is institutional data and
    deliberately not committed; ``tests/test_carry.py`` asserts this exact
    refusal. Counting it as a failure would mean the sweep only passes on a
    machine holding licensed data.

    The message is still checked rather than waved past, because a refusal
    that does not name its own fix is a different defect -- and this project
    already had one of those, in launchers that aborted on a missing package
    without saying which.
    """
    if os.environ.get("OTCPRICER_WIND_DIR"):
        return {}
    return {"/api/carry": (400, "OTCPRICER_WIND_DIR")}


def main() -> int:
    refusals = expected_refusals()
    client = application.app.test_client()

    # Publishing a mark and storing an upload are the only two routes here
    # that write anything. A diagnostic sweep should not leave a published
    # mark behind -- somebody would later read it as a decision -- so both
    # directories are pointed at a temporary one for the duration.
    scratch = tempfile.TemporaryDirectory()
    application.MARK_DIR = Path(scratch.name) / "marks"
    application.UPLOAD_DIR = Path(scratch.name) / "uploads"

    registered = sorted(
        rule.rule
        for rule in application.app.url_map.iter_rules()
        if rule.rule != "/static/<path:filename>"
    )
    unexercised = [r for r in registered if r != "/" and r not in BODIES]
    if unexercised:
        print("ROUTES WITH NO REQUEST BODY IN THIS SWEEP:")
        for route in unexercised:
            print("  ", route)
        print()

    rows = []
    for route in registered:
        start = time.perf_counter()
        if route in GETS:
            response = client.get(route)
        elif route in RAW_BODIES:
            response = client.post(route, data=RAW_BODIES[route], content_type="text/csv")
        else:
            response = client.post(route, json=BODIES.get(route))
        elapsed = time.perf_counter() - start

        detail = ""
        text = response.data.decode("utf-8", "replace")
        expected = refusals.get(route)
        if expected and response.status_code == expected[0]:
            if expected[1] not in text:
                detail = "refused without naming the fix: " + " ".join(text[:180].split())
        elif response.status_code != 200:
            detail = " ".join(text[:220].split())
        else:
            payload = response.get_json(silent=True)
            if isinstance(payload, dict) and "error" in payload:
                detail = "200 carrying an error key: " + str(payload["error"])[:180]
        rows.append((route, response.status_code, elapsed, detail))

    bad = 0
    for route, status, elapsed, detail in rows:
        clean = not detail and (status == 200 or route in refusals)
        if not clean:
            bad += 1
        tag = "  (expected refusal, reason named)" if route in refusals and clean else ""
        mark = "ok " if clean else "BAD"
        print(f"{mark} {route:<34} {status:3d}  {elapsed:6.2f}s  {detail}{tag}")

    print()
    scratch.cleanup()
    print(f"{len(rows)} routes, {bad} not clean, {len(unexercised)} without a body")
    return 1 if (bad or unexercised) else 0


if __name__ == "__main__":
    sys.exit(main())
