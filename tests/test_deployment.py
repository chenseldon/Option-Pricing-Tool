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

import os
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
