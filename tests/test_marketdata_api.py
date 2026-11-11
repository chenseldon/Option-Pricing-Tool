"""Tests for the three market-data routes.

Upload, publish, read -- the sequence a desk runs before pricing anything.
The module-level mark and upload directories are redirected at a temporary
one for every test, so nothing here publishes a mark into the repository: a
file under ``data/marks/`` reads as a decision somebody made, and a test
should not leave one.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    import app

    monkeypatch.setattr(app, "MARK_DIR", tmp_path / "marks")
    monkeypatch.setattr(app, "UPLOAD_DIR", tmp_path / "uploads")
    app.app.config.update(TESTING=True)
    return app.app.test_client()


#: The committed 510300 chain, as the CSV the upload route documents.
@pytest.fixture(scope="module")
def chain_csv() -> bytes:
    from otcpricer.marketdata.sources import build_source

    chain = build_source("snapshot").fetch("510300")
    lines = ["code,last,previous_settlement,strike,expiry,days_to_expiry,as_of"]
    for sl in chain.slices:
        for quote in sl.quotes:
            lines.append(
                f"{quote.contract.raw},{quote.last},{quote.previous_settlement},"
                f"{quote.strike},{sl.expiry.isoformat()},{sl.days_to_expiry},"
                f"{chain.as_of.isoformat()}"
            )
    return ("\n".join(lines) + "\n").encode("utf-8")


class TestReadingAMark:
    def test_an_unmarked_underlying_is_not_an_error(self, client) -> None:  # noqa: ANN001
        """It is every underlying before the first publish of the day, so a
        404 would make an ordinary state look like a fault."""
        response = client.post("/api/marketdata/mark", json={"underlying": "510300"})
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["marked"] is False
        assert "No published mark" in payload["note"]

    def test_an_unsupported_underlying_is_refused(self, client) -> None:  # noqa: ANN001
        response = client.post("/api/marketdata/mark", json={"underlying": "AAPL"})
        assert response.status_code == 400

    def test_a_published_mark_reads_back_and_reproduces(self, client) -> None:  # noqa: ANN001
        """Reading re-fits the mark's own inputs and checks the fingerprint.

        That is the property the whole mechanism rests on: a surface that no
        longer matches its own fingerprint is a different surface wearing
        yesterday's blessing.
        """
        published = client.post(
            "/api/marketdata/publish",
            json={"underlying": "510300", "source": "snapshot", "r": 0.02},
        ).get_json()
        assert published["published"] is True

        payload = client.post("/api/marketdata/mark", json={"underlying": "510300"}).get_json()
        assert payload["marked"] is True
        assert payload["reproduces"] is True
        assert payload["mark"]["fingerprint"] == published["mark"]["fingerprint"]

    def test_a_fingerprint_mismatch_answers_409_rather_than_serving_it(
        self, client, tmp_path: Path
    ) -> None:  # noqa: ANN001
        client.post(
            "/api/marketdata/publish",
            json={"underlying": "510300", "source": "snapshot", "r": 0.02},
        )
        path = tmp_path / "marks" / "510300.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["fingerprint"] = "deadbeefdeadbeef"
        path.write_text(json.dumps(raw), encoding="utf-8")

        response = client.post("/api/marketdata/mark", json={"underlying": "510300"})
        assert response.status_code == 409
        assert response.get_json()["reproduces"] is False


class TestPublishing:
    def test_it_publishes_and_reports_every_gate(self, client) -> None:  # noqa: ANN001
        """All five, not only the failures. A reader deciding whether to
        override needs the whole picture."""
        payload = client.post(
            "/api/marketdata/publish",
            json={"underlying": "510300", "source": "snapshot", "r": 0.02},
        ).get_json()
        assert payload["published"] is True
        names = [g["name"] for g in payload["gates"]]
        assert names == ["fit_quality", "butterfly", "calendar", "arbitrage_free", "drift"]
        assert all(g["passed"] for g in payload["gates"])

    def test_a_failed_gate_answers_409_and_names_the_distance(self, client, tmp_path: Path) -> None:  # noqa: ANN001
        """The check that makes the gates worth having.

        The stored mark's at-the-money volatilities are moved half a
        volatility point, which is the one gate reachable without corrupting
        the fit itself.
        """
        client.post(
            "/api/marketdata/publish",
            json={"underlying": "510300", "source": "snapshot", "r": 0.02},
        )
        path = tmp_path / "marks" / "510300.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["atm_vols"] = [v + 0.5 for v in raw["atm_vols"]]
        path.write_text(json.dumps(raw), encoding="utf-8")

        response = client.post(
            "/api/marketdata/publish",
            json={"underlying": "510300", "source": "snapshot", "r": 0.02},
        )
        assert response.status_code == 409
        payload = response.get_json()
        assert payload["published"] is False
        assert "drift" in payload["error"]
        assert "force=True" in payload["error"], "the refusal does not say how to override"
        assert [g["name"] for g in payload["gates"] if not g["passed"]] == ["drift"]

    def test_force_publishes_through_a_failed_gate_and_records_it(
        self, client, tmp_path: Path
    ) -> None:  # noqa: ANN001
        client.post(
            "/api/marketdata/publish",
            json={"underlying": "510300", "source": "snapshot", "r": 0.02},
        )
        path = tmp_path / "marks" / "510300.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["atm_vols"] = [v + 0.5 for v in raw["atm_vols"]]
        path.write_text(json.dumps(raw), encoding="utf-8")

        payload = client.post(
            "/api/marketdata/publish",
            json={"underlying": "510300", "source": "snapshot", "r": 0.02, "force": True},
        ).get_json()
        assert payload["published"] is True
        assert payload["forced"] is True
        assert json.loads(path.read_text(encoding="utf-8"))["forced"] is True


class TestUploading:
    def test_a_chain_uploads_and_returns_a_token(self, client, chain_csv: bytes) -> None:  # noqa: ANN001
        payload = client.post(
            "/api/marketdata/upload",
            data={"file": (io.BytesIO(chain_csv), "chain.csv")},
            content_type="multipart/form-data",
        ).get_json()
        assert len(payload["token"]) == 16
        assert payload["rows"] == 126
        assert payload["as_of"] == "2026-10-27"

    def test_the_same_chain_twice_is_one_token(self, client, chain_csv: bytes) -> None:  # noqa: ANN001
        """Named by a hash of the contents, so re-uploading is idempotent
        rather than accumulating files."""
        first = client.post(
            "/api/marketdata/upload", data=chain_csv, content_type="text/csv"
        ).get_json()["token"]
        second = client.post(
            "/api/marketdata/upload", data=chain_csv, content_type="text/csv"
        ).get_json()["token"]
        assert first == second

    def test_the_callers_filename_is_never_used_as_a_path(
        self, client, chain_csv: bytes, tmp_path: Path
    ) -> None:  # noqa: ANN001
        """A client-chosen filename reaching a path join is the oldest defect
        in this category. The stored file is named by the content hash, and
        the name sent is echoed and otherwise ignored.
        """
        payload = client.post(
            "/api/marketdata/upload",
            data={"file": (io.BytesIO(chain_csv), "../../../etc/passwd")},
            content_type="multipart/form-data",
        ).get_json()
        stored = list((tmp_path / "uploads").iterdir())
        assert [p.name for p in stored] == [f"{payload['token']}.csv"]

    def test_a_published_mark_off_an_upload_reproduces_the_snapshot_surface(
        self, client, chain_csv: bytes
    ) -> None:  # noqa: ANN001
        """The strongest statement the mark mechanism can make.

        The same chain through two completely different ingestion paths -- the
        committed JSON snapshot and an uploaded CSV -- must bless the same
        surface. Measured: fingerprint d1f5bab96427e6ed from both.
        """
        from_snapshot = client.post(
            "/api/marketdata/publish",
            json={"underlying": "510300", "source": "snapshot", "r": 0.02},
        ).get_json()["mark"]["fingerprint"]

        token = client.post(
            "/api/marketdata/upload", data=chain_csv, content_type="text/csv"
        ).get_json()["token"]
        from_csv = client.post(
            "/api/marketdata/publish",
            json={"underlying": "510300", "csv_token": token, "r": 0.02},
        ).get_json()
        assert from_csv["published"] is True
        assert from_csv["mark"]["fingerprint"] == from_snapshot
        assert from_csv["mark"]["csv_token"] == token

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (b"", "no CSV content"),
            (b"a,b\n1,2\n", "missing required column"),
            (
                b"code,last,previous_settlement,strike,expiry,days_to_expiry\n",
                "header and no rows",
            ),
        ],
    )
    def test_the_refusals_say_what_is_wrong(self, client, body: bytes, expected: str) -> None:  # noqa: ANN001
        response = client.post("/api/marketdata/upload", data=body, content_type="text/csv")
        assert response.status_code == 400
        assert expected in response.get_json()["error"]

    def test_a_non_utf8_file_is_reported_rather_than_mangled(self, client) -> None:  # noqa: ANN001
        response = client.post(
            "/api/marketdata/upload", data=b"\xff\xfe\x00bad", content_type="text/csv"
        )
        assert response.status_code == 400
        assert "UTF-8" in response.get_json()["error"]

    def test_an_oversized_file_is_refused_before_being_parsed(self, client) -> None:  # noqa: ANN001
        """A full CFFEX board is a few thousand rows. The cap is far above
        anything real and far below anything that would exhaust memory."""
        import app

        response = client.post(
            "/api/marketdata/upload",
            data=b"x" * (app.MAX_UPLOAD_BYTES + 1),
            content_type="text/csv",
        )
        assert response.status_code == 400
        assert "exceeds" in response.get_json()["error"]


class TestTokenHandling:
    @pytest.mark.parametrize(
        "token",
        [
            "../../etc/passwd",
            "..",
            "/absolute",
            "short",
            "NOTHEXADECIMAL00",
            "0123456789abcdef0",
        ],
    )
    def test_anything_that_is_not_a_token_is_refused_by_shape(self, client, token: str) -> None:  # noqa: ANN001
        """Validated against the pattern rather than merely joined to a
        directory: a token is sixteen hex characters, so path separators and
        ``..`` cannot occur in a valid one, and refusing on the shape leaves
        nothing to reason about afterwards.
        """
        response = client.post(
            "/api/marketdata/publish",
            json={"underlying": "510300", "csv_token": token},
        )
        assert response.status_code == 400
        assert "16-character token" in response.get_json()["error"]

    def test_a_well_formed_but_unknown_token_says_to_upload_again(self, client) -> None:  # noqa: ANN001
        response = client.post(
            "/api/marketdata/publish",
            json={"underlying": "510300", "csv_token": "0123456789abcdef"},
        )
        assert response.status_code == 400
        assert "upload it again" in response.get_json()["error"]
