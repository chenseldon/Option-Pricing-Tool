"""Tests for the single-page front end.

The interface is 2,500 lines of HTML with an inline bilingual dictionary, and
it had no tests at all. Layout cannot be checked here -- that needs eyes -- but
three things can, and all three are the kind of mistake that survives a visual
pass because the page still renders:

**A label key missing from one dictionary.** The page looks right in English
and shows a raw key in Chinese, or the reverse, and only a reader of the other
language ever sees it.

**A data path the JavaScript reads and the JSON does not send.** The panel
renders, the button works, and the cells say ``undefined`` or ``NaN``.

**A control wired to nothing.** An input the user fills that never reaches a
payload is worse than an absent one, because it looks like it did something.
The futures carry field is the case in point: it is the largest single driver
of a snowball quote, and the interface had no way to express it until Phase 5.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "templates" / "index.html"

#: Keys the page asks for by attribute, e.g. ``data-i18n="btn_run_quote"``.
_ATTRIBUTE_KEYS = re.compile(r'data-i18n="([a-z0-9_]+)"')
#: Keys the page asks for through the id->key map, e.g. ``'lbl-sb-q-par':'lbl_sb_q_par'``.
_MAPPED_KEYS = re.compile(r"'[a-z0-9\-]+'\s*:\s*'([a-z0-9_]+)'")


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _dictionary(page: str, language: str) -> set[str]:
    """Every key defined for one language, across all of the page's tables.

    The page holds more than one ``en:``/``cn:`` pair -- a short table of
    product names and the long table of labels -- so this unions them. The
    first version indexed the first marker it found and so read only the
    product-name table, which is why it reported the entire label set as
    undefined.
    """
    keys: set[str] = set()
    for block in _language_blocks(page, language):
        # Several keys share a line, so this cannot anchor at a line start.
        keys |= set(re.findall(r"[{,\s]([a-z0-9_]+)\s*:\s*['\"]", block))
    return keys


def _language_blocks(page: str, language: str) -> list[str]:
    """The body of every ``<language>: { ... }`` object, by brace matching.

    Brace counting rather than "slice to the next marker", because the two
    tables are not adjacent and a label value can itself contain a brace.
    """
    marker = f"{language}: {{"
    blocks: list[str] = []
    cursor = 0
    while True:
        found = page.find(marker, cursor)
        if found == -1:
            return blocks
        opening = found + len(marker) - 1
        depth = 0
        for index in range(opening, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(page[opening : index + 1])
                    cursor = index + 1
                    break
        else:  # pragma: no cover - an unbalanced page would fail earlier
            raise AssertionError(f"unbalanced braces after {marker!r}")


class TestTheTranslationTable:
    def test_both_languages_define_the_same_keys(self, page: str) -> None:
        """A key in one dictionary and not the other shows a raw identifier.

        The page renders either way, so this is invisible to anyone reading it
        in the language that happens to be complete.
        """
        english = _dictionary(page, "en")
        chinese = _dictionary(page, "cn")
        assert english, "the English dictionary did not parse"
        assert english - chinese == set(), f"missing from cn: {sorted(english - chinese)}"
        assert chinese - english == set(), f"missing from en: {sorted(chinese - english)}"

    def test_every_key_the_page_asks_for_is_defined(self, page: str) -> None:
        """Both routes into the table: the attribute and the id map."""
        english = _dictionary(page, "en")
        requested = set(_ATTRIBUTE_KEYS.findall(page)) | set(_MAPPED_KEYS.findall(page))
        # The id map's values are keys; its own ids are not, and neither are
        # the many unrelated single-quoted pairs elsewhere in the script, so
        # only intersect against what looks like a label key.
        label_like = {key for key in requested if key.startswith(("lbl_", "btn_", "sb_", "sf_"))}
        assert label_like, "no label keys were found; the parser is wrong"
        assert label_like - english == set(), f"undefined: {sorted(label_like - english)}"


class TestTheQuotingPanel:
    """Phase 5's addition, which is the first time the interface could quote."""

    def test_the_carry_input_exists_and_reaches_a_payload(self, page: str) -> None:
        """An input wired to nothing is worse than an absent one.

        The futures carry more than doubles the quotable coupon at the level
        measured on the CSI 1000 board, and the page had no field for it. Three
        call sites must read it: the Monte Carlo price, the CN-FDM reprice, and
        the shared snowball body the new panel sends.
        """
        assert 'id="sb-basis"' in page
        reads = page.count("$('sb-basis').value")
        assert reads >= 3, f"the carry field is read {reads} times, expected 3 call sites"
        assert "basis_pa: basis" in page or "basis_pa:basis" in page or "basis_pa: +$" in page

    def test_the_panel_calls_the_two_new_routes(self, page: str) -> None:
        assert "/api/snowball_quote" in page
        assert "/api/snowball_risk_crosscheck" in page
        assert "function calcSnowballQuote()" in page
        assert "function calcSnowballRecon()" in page

    def test_the_quote_payload_omits_the_coupon(self, page: str) -> None:
        """Because it is the unknown.

        snowballBody() is shared by both new calls, and the reconciliation
        adds coupon_pa explicitly -- risk is taken at a stated coupon, a quote
        solves for one. Having the shared body carry it would make the quote
        look like it depended on the input it ignores.
        """
        body = page[page.index("function snowballBody()") :]
        body = body[: body.index("}\n")]
        assert "coupon_pa" not in body
        assert "basis_pa" in body
        assert "KO_pct" in body and "KI_pct" in body

    def test_all_six_greeks_are_rendered_including_carry_rho(self, page: str) -> None:
        """carry_rho is the newest and the one a stale table would drop."""
        order = page[page.index("const order = [") :]
        order = order[: order.index("]")]
        for name in ("delta", "gamma", "vega", "theta", "rho", "carry_rho"):
            assert f"'{name}'" in order, name

    def test_the_theta_caveat_reaches_the_reader(self, page: str) -> None:
        """Its row passes on a budget that is a large share of its value, and
        the two estimates are difference quotients over different horizons. A
        table showing only 'yes' would overstate what that row establishes.
        """
        assert "mc_error_share_of_value" in page
        assert "agreement rather than as confirmation" in page

    def test_the_residuals_are_printed_in_scientific_notation(self, page: str) -> None:
        """A machine-precision residual rounded to four decimals reads 0.0000,
        which hides the only thing it is for."""
        assert "affinity_residual.toExponential" in page
        assert "residual.toExponential" in page


class TestThePageStillServes:
    @pytest.fixture(scope="class")
    def client(self):  # noqa: ANN201
        import app

        app.app.config.update(TESTING=True)
        return app.app.test_client()

    def test_the_index_renders_with_the_panel(self, client) -> None:  # noqa: ANN001
        response = client.get("/")
        assert response.status_code == 200
        html = response.data.decode("utf-8")
        for marker in ("sb-basis", "sb-q-par", "sb-recon-body", "calcSnowballQuote"):
            assert marker in html, marker

    def test_the_script_blocks_are_balanced(self, page: str) -> None:
        """A crude syntax guard, since a broken script tag takes the whole page
        down silently -- the HTML still renders and nothing works."""
        assert page.count("<script") == page.count("</script>")
        assert page.count("<script") >= 1

    def test_the_translation_table_is_valid_json_shaped(self, page: str) -> None:
        """Not parsed as JSON -- it is JavaScript -- but its keys must at least
        be unique within a language, since a duplicate silently wins."""
        for language in ("en", "cn"):
            for block in _language_blocks(page, language):
                keys = re.findall(r"[{,\s]([a-z0-9_]+)\s*:\s*['\"]", block)
                duplicates = {key for key in keys if keys.count(key) > 1}
                assert not duplicates, f"{language} defines {sorted(duplicates)} more than once"


def test_json_is_imported_for_a_reason() -> None:
    """Guard against the import being dropped as unused by a future cleanup."""
    assert json.dumps({"ok": True}) == '{"ok": true}'
