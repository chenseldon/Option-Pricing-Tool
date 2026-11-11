"""Tests for the entry points and the packaging around them.

None of this is pricing. It is the layer that decides whether anyone other
than the author can start the thing, and it had four defects that no pricing
test could see:

**`python app.py` did not run from a clone.** It failed with
``ModuleNotFoundError: No module named 'otcpricer'``. The test suite never
noticed because ``pyproject.toml`` puts ``src/`` on the path for pytest only.

**The launchers gated on a package the code does not import.** Both probed for
``py_vollib`` and aborted if it was absent -- four phases after the pricing
moved to ``otcpricer.models.bsm``. They also pointed at a different interpreter
from the one the project is tested on, so the documented launch path was not
the verified one.

**`pyarrow` was an undeclared dependency.** ``marketdata/wind.py`` reads
parquet and nothing declared it, so it worked here only because conda happened
to supply it.

**The server was the Flask development server with the debugger on**, which is
an interactive Python console on any traceback, for an application with no
authentication.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _clean_env() -> dict[str, str]:
    """The environment a fresh clone would have.

    ``PYTHONPATH`` is stripped deliberately: with it set, every entry point
    imports fine and the defect these tests exist for is invisible.
    """
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


class TestTheEntryPointsRunFromAClone:
    """Without an install, and without PYTHONPATH."""

    @pytest.mark.parametrize("module", ["app", "serve"])
    def test_it_imports_with_nothing_on_the_path(self, module: str) -> None:
        """The defect, stated as the test that catches it.

        Both entry points append ``src/`` themselves. Appended rather than
        inserted, so an installed copy still wins and a developer working
        against ``pip install -e .`` does not silently get the checkout.
        """
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            env=_clean_env(),
            cwd=ROOT,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr

    def test_the_bootstrap_does_not_shadow_an_installed_copy(self) -> None:
        """``src`` must land at the end of sys.path, not the front."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import serve, sys, os;"
                "src = os.path.join(os.getcwd(), 'src');"
                "print(sys.path.index(src) > 0)",
            ],
            capture_output=True,
            text=True,
            env=_clean_env(),
            cwd=ROOT,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"


class TestTheDependencyProbe:
    """scripts/check_deps.py, which both launchers call.

    It is a file rather than an inlined ``python -c`` because PowerShell
    stripped the quotes out of the here-string version: the probe died with a
    SyntaxError and the launcher reported that the *interpreter* could not run
    it. A file has no quoting layer to get wrong.
    """

    def test_this_interpreter_passes(self) -> None:
        """Skipped where the server is genuinely absent.

        The probe asks whether this interpreter can *run the application*, and
        without waitress the honest answer is no. A reader running the suite
        from a library install has no reason to have a web server, so that is
        a skip rather than a failure -- but CI installs waitress precisely so
        this does run there.
        """
        pytest.importorskip("waitress", reason="no server installed; nothing to verify")
        result = subprocess.run(
            [sys.executable, "scripts/check_deps.py"],
            capture_output=True,
            text=True,
            env=_clean_env(),
            cwd=ROOT,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "deps-ok" in result.stdout

    def test_it_names_what_is_missing_rather_than_failing_vaguely(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A launcher that says "dependencies broken" sends the reader nowhere.

        Run in process with one import refused, which is what a partial
        install looks like. The first version of this test shelled out and
        exec'd the file, which failed on its own ``__file__`` reference -- a
        reminder that a script and a module are not interchangeable.
        """
        import builtins

        from scripts import check_deps

        real_import = builtins.__import__

        def refuse_waitress(name: str, *args: object, **kwargs: object) -> object:
            if name == "waitress":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", refuse_waitress)
        assert check_deps.main() == 1
        printed = capsys.readouterr().out

        assert "deps-missing" in printed
        assert "waitress" in printed
        assert "pip install -r requirements.txt" in printed
        # And it points at the alternative rather than leaving a dead end.
        assert "app.py" in printed

    def test_it_separates_optional_from_required(self) -> None:
        """A partial environment should read as partial, not as broken.

        QuantLib is cross-validation only and numba is an optional JIT path;
        neither should stop the application starting, so neither is required.
        """
        from scripts.check_deps import OPTIONAL, REQUIRED

        required = {name for name, _ in REQUIRED}
        optional = {name for name, _ in OPTIONAL}
        assert "waitress" in required, "the server is not optional"
        assert "otcpricer" in required
        assert {"QuantLib", "numba", "pyarrow"} <= optional
        assert not required & optional


class TestServerSettings:
    def test_the_debugger_is_off_by_default(self) -> None:
        """The whole reason serve.py exists.

        The interactive debugger renders a Python console on any unhandled
        traceback, and this application authenticates nothing.
        """
        from otcpricer.core.config import ServerSettings

        assert ServerSettings().debug is False

    def test_it_binds_loopback_by_default(self) -> None:
        """Binding publicly is a decision, not a default."""
        from otcpricer.core.config import ServerSettings

        assert ServerSettings().host.startswith("127.")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1", True),
            ("true", True),
            ("YES", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("", False),
        ],
    )
    def test_the_debug_override_reads_the_spellings_people_use(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
    ) -> None:
        from otcpricer.core.config import get_settings

        monkeypatch.setenv("OTCPRICER_DEBUG", raw)
        get_settings.cache_clear()
        try:
            assert get_settings().server.debug is expected
        finally:
            get_settings.cache_clear()

    def test_an_unrecognised_boolean_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``OTCPRICER_DEBUG=maybe`` must not quietly mean "no".

        Silently defaulting to false would be the safe direction, but it is
        still a value the caller wrote and did not get. For a flag that
        switches on a remote console, the failure has to be loud.
        """
        from otcpricer.core.config import get_settings

        monkeypatch.setenv("OTCPRICER_DEBUG", "maybe")
        get_settings.cache_clear()
        try:
            with pytest.raises(ValueError, match="not a boolean"):
                get_settings()
        finally:
            get_settings.cache_clear()

    def test_the_port_is_a_port(self) -> None:
        from pydantic import ValidationError

        from otcpricer.core.config import ServerSettings

        assert ServerSettings(port=8080).port == 8080
        with pytest.raises(ValidationError):
            ServerSettings(port=70000)


class TestServeModule:
    def test_it_reports_a_missing_waitress_with_the_fix(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Importable and runnable on a machine without the server installed.

        That is why the import is inside ``main`` rather than at module scope:
        a reader should be able to read the file and run the probe without
        first installing what it needs.
        """
        import builtins

        import serve

        real_import = builtins.__import__

        def refuse_waitress(name: str, *args: object, **kwargs: object) -> object:
            if name == "waitress":
                raise ImportError("no waitress")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", refuse_waitress)
        assert serve.main() == 1
        message = capsys.readouterr().err
        assert "pip install waitress" in message
        assert "app.py" in message

    def test_the_banner_says_whether_it_is_exposed(self) -> None:
        """A reader starting it should see which of the two they have.

        The string is asserted rather than the behaviour, because binding
        publicly inside a test is not something to do.
        """
        source = (ROOT / "serve.py").read_text(encoding="utf-8")
        assert "loopback only" in source
        assert "BOUND PUBLICLY" in source
        assert "No authentication" in source

    def test_it_states_what_it_does_not_fix(self) -> None:
        """A production-looking server invites the assumption that the rest is
        production too, so the docstring says plainly what is absent."""
        source = (ROOT / "serve.py").read_text(encoding="utf-8")
        for absent in ("authentication", "rate limiting", "HTTPS", "timeouts"):
            assert absent in source, absent


class TestTheLaunchers:
    """Shell scripts, so only their text can be checked here -- but the four
    things worth checking are all textual."""

    @pytest.mark.parametrize("name", ["run_app.ps1", "run_app.bat"])
    def test_they_no_longer_gate_on_py_vollib(self, name: str) -> None:
        """It was removed in Phase 1 and both launchers aborted without it."""
        text = (ROOT / name).read_text(encoding="utf-8")
        probe_lines = [
            line
            for line in text.splitlines()
            if "py_vollib" in line and not line.lstrip().startswith(("#", "rem"))
        ]
        assert not probe_lines, probe_lines

    @pytest.mark.parametrize("name", ["run_app.ps1", "run_app.bat"])
    def test_they_start_the_production_server(self, name: str) -> None:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "serve.py" in text

    @pytest.mark.parametrize("name", ["run_app.ps1", "run_app.bat"])
    def test_they_use_the_shared_probe(self, name: str) -> None:
        """One probe, two launchers -- and a file, because PowerShell mangled
        the inlined version."""
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "check_deps.py" in text

    @pytest.mark.parametrize("name", ["run_app.ps1", "run_app.bat"])
    def test_they_accept_an_interpreter_override(self, name: str) -> None:
        """So the launcher is not bound to one machine's conda layout."""
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "OTCPRICER_PYTHON" in text


class TestDeclaredDependencies:
    def test_pyarrow_is_declared_somewhere(self) -> None:
        """marketdata/wind.py reads parquet, and nothing declared the engine.

        It worked here only because conda happened to provide it. CI never
        noticed because the panel is not committed, so those tests skip -- but
        anyone who copied the data directory and installed from
        requirements.txt got an ImportError from inside pandas.
        """
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "pyarrow" in requirements
        assert "pyarrow" in pyproject

    def test_py_vollib_is_not_declared(self) -> None:
        """Pinned for four phases after the code stopped importing it."""
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        declarations = [
            line
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#") and "vollib" in line
        ]
        assert not declarations, declarations

    def test_nothing_imports_py_vollib(self) -> None:
        """The claim behind removing it, checked rather than trusted."""
        offenders = []
        for path in list(ROOT.glob("*.py")) + list((ROOT / "src").rglob("*.py")):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and "vollib" in stripped:
                    offenders.append(f"{path.name}: {stripped}")
        assert not offenders, offenders

    def test_the_numpy_cap_matches_what_numba_asks_for(self) -> None:
        """The <2.3 cap was stale by a minor version.

        It came from numba, which now requires numpy<2.4 -- and it sat on a
        core dependency for the sake of an optional extra, forcing every
        install down with it.
        """
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "numpy>=2.2,<2.4" in pyproject
        assert "numpy>=2.0,<2.3" not in pyproject


class TestTheDocumentedRouteTable:
    """The README lists the API, and the list had gone stale in both directions.

    It was missing four routes -- ``/api/snowball_quote`` (the one that solves
    for the par coupon, which is the point of the whole fifth phase),
    ``/api/snowball_report``, ``/api/carry`` and ``/api/export`` -- while listing
    ``/api/shark_fin`` and ``/api/irs`` twice each, duplicates left from before
    those endpoints grew their cross-checks.

    Neither direction is visible to anyone reading either file alone, which is
    what makes it the same failure mode as the engines silently substituting a
    coarser schedule: the artefact looks complete and is not. A reader
    following the table would conclude the service cannot quote.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def documented(cls) -> set[str]:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        return set(re.findall(r"\| `(/api/[^`]*)` \| (?:POST|GET)", readme))

    @pytest.fixture(scope="class")
    @classmethod
    def registered(cls) -> set[str]:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        # Read from the source rather than from app.url_map: importing the
        # application pulls in the whole library, and this file's other tests
        # deliberately run in subprocesses with an empty path for that reason.
        return {
            route
            for route in re.findall(r'@app\.route\("([^"]*)"', source)
            if route.startswith("/api/")
        }

    def test_every_route_is_documented(self, documented: set[str], registered: set[str]) -> None:
        assert registered, "no routes were found in app.py"
        missing = sorted(registered - documented)
        assert missing == [], f"registered but undocumented: {missing}"

    def test_nothing_documented_has_been_removed(
        self, documented: set[str], registered: set[str]
    ) -> None:
        """The other direction, and the worse one.

        A reader calling a route the README promises and the service no longer
        serves gets a 404 with no explanation. ``/api/cn_fdm`` is the live case
        to watch: it is documented with a warning and still registered, and the
        day it is finally removed this test is what will catch the row.
        """
        stale = sorted(documented - registered)
        assert stale == [], f"documented but not registered: {stale}"

    def test_no_route_is_listed_twice(self) -> None:
        """Duplicates are how the table drifted in the first place.

        The set comparisons above cannot see them -- both copies collapse to one
        entry -- so this counts the rows.
        """
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        rows = re.findall(r"\| `(/api/[^`]*)` \| (?:POST|GET)", readme)
        duplicates = sorted({route for route in rows if rows.count(route) > 1})
        assert duplicates == [], f"listed more than once: {duplicates}"


class TestTheDocumentedDependencies:
    """The README's Tech Stack table against the file that actually installs.

    Every numeric row disagreed: numpy documented at >=1.21 against a real
    floor of >=2.2, scipy at >=1.7 against >=1.14, pandas at >=1.3 against
    >=2.2, Flask at >=2.0 against >=3.0. And ``waitress``, ``pydantic``,
    ``pyarrow`` and ``openpyxl`` were not in the table at all -- the first
    three of those being the dependencies this phase added or discovered, so
    the table was documenting the project as it stood four phases ago.

    The direction that matters is **a floor documented lower than the real
    one**: someone installing numpy 1.21 on the strength of this table gets a
    project that does not import, and the table is why. A floor documented
    higher is merely conservative.
    """

    #: Rows whose version is deliberately not a requirements.txt pin.
    _NOT_INSTALLED = frozenset({"*(self-built)*", "**Plotly 2**", "QuantLib", "pytest"})

    @pytest.fixture(scope="class")
    @classmethod
    def documented(cls) -> dict[str, str]:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        table = readme[readme.index("## 🛠 Tech Stack") :]
        table = table[: table.index("\n---")]
        rows = re.findall(r"^\| (\S+) \| ([^|]+) \|", table, re.MULTILINE)
        # Strip the backticks *before* filtering: the table writes the package
        # names as `numpy`, `QuantLib` and so on, so a filter applied to the
        # raw capture matches none of them and QuantLib and pytest leak through
        # as if requirements.txt were expected to install them.
        named = ((name.strip("`"), version.strip()) for name, version in rows)
        return {
            name: version
            for name, version in named
            if name not in cls._NOT_INSTALLED and name != "Library"
        }

    @pytest.fixture(scope="class")
    @classmethod
    def required(cls) -> dict[str, str]:
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        found: dict[str, str] = {}
        for line in text.splitlines():
            line = line.split("#")[0].strip()
            match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", line)
            if match and match.group(2):
                found[match.group(1)] = match.group(2).replace(" ", "")
        return found

    def test_every_installed_dependency_is_in_the_table(
        self, documented: dict[str, str], required: dict[str, str]
    ) -> None:
        assert required, "requirements.txt did not parse"
        lowered = {name.lower() for name in documented}
        missing = sorted(n for n in required if n.lower() not in lowered)
        assert missing == [], f"installed but undocumented: {missing}"

    def test_the_table_lists_nothing_that_is_not_installed(
        self, documented: dict[str, str], required: dict[str, str]
    ) -> None:
        """A row for a package the project stopped needing is how the py_vollib
        claim survived four phases in requirements.txt."""
        lowered = {name.lower() for name in required}
        stale = sorted(n for n in documented if n.lower() not in lowered)
        assert stale == [], f"documented but not installed: {stale}"

    def test_no_documented_floor_is_below_the_real_one(
        self, documented: dict[str, str], required: dict[str, str]
    ) -> None:
        """The direction that breaks an install done on the table's authority."""
        canonical = {name.lower(): spec for name, spec in required.items()}
        for name, version in documented.items():
            spec = canonical[name.lower()]
            shown = re.search(r"≥\s*([0-9.]+)", version)
            actual = re.search(r">=\s*([0-9.]+)", spec)
            if not (shown and actual):
                continue
            as_tuple = lambda v: tuple(int(x) for x in v.rstrip(".").split("."))  # noqa: E731
            assert as_tuple(shown.group(1)) >= as_tuple(actual.group(1)), (
                f"{name}: the table says >={shown.group(1)} and the real floor "
                f"is >={actual.group(1)}"
            )

    def test_an_upper_bound_that_exists_is_documented(
        self, documented: dict[str, str], required: dict[str, str]
    ) -> None:
        """numpy carries a ``<2.4`` cap from numba, and a reader who installs
        2.4 on this table's word gets a version the project does not verify."""
        canonical = {name.lower(): spec for name, spec in required.items()}
        for name, version in documented.items():
            cap = re.search(r"<\s*([0-9.]+)", canonical[name.lower()])
            if cap:
                assert cap.group(1) in version, f"{name}: the <{cap.group(1)} cap is not shown"


class TestReproducibilityUnderConcurrency:
    """A seeded endpoint has to stay seeded once more than one thread exists.

    ``/api/mc_distribution`` carried ``np.random.seed(42)`` so that the same
    request returns the same histogram. Against the process-global generator it
    did not: seeding and drawing are two separate calls, so two concurrent
    requests can both seed before either draws, and the second continues the
    stream instead of starting it.

    Measured before the fix, eight threads against this route: **53 of 96
    requests matched the seeded result** -- 45% of callers got a histogram the
    seed was there to rule out. After switching to a per-request
    ``default_rng``, 96 of 96.

    This is a defect the project created for itself. Under the single-threaded
    Flask development server it could not happen; it became reachable the
    moment this phase put eight waitress threads in front of the application,
    which is exactly the class of bug a deployment change introduces and no
    pricing test can see.

    It was also the **last** global-``np.random`` site in the codebase -- every
    engine uses ``default_rng`` already. That is the only reason this was a
    reproducibility bug and not a pricing one: nothing else was downstream of
    the global state it was stamping on once per request.
    """

    BODY = {
        "option_type": "c",
        "S": 5500,
        "K": 5500,
        "T": 0.25,
        "r": 0.02,
        "sigma": 0.2,
        "q": 0.01,
    }

    @pytest.fixture(scope="class")
    @classmethod
    def client(cls):  # noqa: ANN206
        import app

        app.app.config.update(TESTING=True)
        return app.app.test_client()

    def _histogram(self, client) -> str:  # noqa: ANN001
        """A digest rather than the histogram.

        Fifty bins times eight threads is 1,798 lines of pytest output on a
        failure, which buries the one number that matters. The comparison only
        ever asks whether two runs are identical, so a digest answers it.
        """
        response = client.post("/api/mc_distribution", json=self.BODY)
        assert response.status_code == 200, response.data[:200]
        counts = response.get_json()["counts"]
        return hashlib.sha256(repr(counts).encode()).hexdigest()[:16]

    def test_it_is_reproducible_sequentially(self, client) -> None:  # noqa: ANN001
        """The property the seed was added for, and the one that always held --
        so a test that stopped here would have proved nothing."""
        assert self._histogram(client) == self._histogram(client)

    @pytest.mark.slow
    def test_it_is_reproducible_across_concurrent_requests(self, client) -> None:  # noqa: ANN001
        """The property that did not hold, with the threads the server runs.

        Eight, because that is ``ServerSettings.threads``. A barrier rather
        than a plain start, so the requests genuinely overlap instead of
        happening to serialise and proving nothing.
        """
        import threading

        baseline = self._histogram(client)
        workers = 8
        barrier = threading.Barrier(workers)
        results: list[str] = [""] * workers

        def run(slot: int) -> None:
            barrier.wait()
            results[slot] = self._histogram(client)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        matched = sum(1 for r in results if r == baseline)
        assert matched == workers, (
            f"{matched} of {workers} concurrent requests matched the seeded "
            f"result; the endpoint is drawing from shared generator state"
        )

    def test_the_route_does_not_touch_the_global_generator(self) -> None:
        """Said as a property of the source, because the behavioural test above
        is a race and a race can pass by luck.

        ``np.random.seed`` also mutates state for everything else in the
        process. Every engine here uses ``default_rng``, so this is the
        invariant that keeps it that way.
        """
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        for banned in ("np.random.seed(", "np.random.standard_normal("):
            assert banned not in source, (
                f"{banned} is back in app.py; use a local np.random.default_rng(seed) instead"
            )
        assert "np.random.default_rng(42)" in source
