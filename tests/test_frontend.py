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
#:
#: Upper case is allowed in a key and twelve of them use it -- ``lbl_sb_S``,
#: ``lbl_sf_H``, ``lbl_ir_T`` and the rest, named after the symbol they label.
#: These patterns matched ``[a-z0-9_]`` until the id-map tests below went
#: looking and found twelve keys that no test in this file could see. All
#: twelve were in fact defined in both languages, so nothing was broken -- but
#: the balance check had been passing over them, which means it would not have
#: caught one going missing from a dictionary either.
_ATTRIBUTE_KEYS = re.compile(r'data-i18n="([A-Za-z0-9_]+)"')
#: Keys the page asks for through the id->key map, e.g. ``'lbl-sb-q-par':'lbl_sb_q_par'``.
_MAPPED_KEYS = re.compile(r"'[a-z0-9\-]+'\s*:\s*'([A-Za-z0-9_]+)'")


def _id_map(page: str) -> dict[str, str]:
    """The id->key map, by brace matching.

    Not a page-wide regex for quoted pairs: the script is full of unrelated
    ones (``? 'pos' : 'neg'``, display values, chart options) and an earlier
    version of this parser reported fifteen of them as broken mappings.
    """
    opening = page.index("{", page.index("const idMap = {"))
    depth = 0
    for index in range(opening, len(page)):
        if page[index] == "{":
            depth += 1
        elif page[index] == "}":
            depth -= 1
            if depth == 0:
                block = page[opening : index + 1]
                break
    else:  # pragma: no cover - an unbalanced page would fail earlier
        raise AssertionError("unbalanced braces in idMap")
    return dict(re.findall(r"'([^']+)'\s*:\s*'([^']+)'", block))


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
        keys |= set(re.findall(r"[{,\s]([A-Za-z0-9_]+)\s*:\s*['\"]", block))
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
    @classmethod
    def client(cls):  # noqa: ANN201
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
                keys = re.findall(r"[{,\s]([A-Za-z0-9_]+)\s*:\s*['\"]", block)
                duplicates = {key for key in keys if keys.count(key) > 1}
                assert not duplicates, f"{language} defines {sorted(duplicates)} more than once"


def test_json_is_imported_for_a_reason() -> None:
    """Guard against the import being dropped as unused by a future cleanup."""
    assert json.dumps({"ok": True}) == '{"ok": true}'


class TestTheModelsTab:
    """Phase 6's addition: two phases of volatility work made visible.

    Everything here was reachable only through the API, which in practice meant
    the flat-volatility number was the only one anyone saw.
    """

    def test_the_tab_buttons_and_the_panel_keys_are_in_the_same_order(self, page: str) -> None:
        """The page's most fragile line, and a comment is not enough.

        ``switchMain`` highlights buttons with ``Object.keys(PANELS)[i]`` against
        the ``.mtab`` node list -- by **index**. So a tab inserted into one and
        not the other does not fail cleanly: the correct panel still shows,
        because that lookup is by key, while the highlight lands on a different
        button. It reads as a rendering glitch rather than as the structural
        mistake it is, which is exactly the kind of thing that survives a visual
        pass.
        """
        buttons = re.findall(r'class="mtab[^"]*"[^>]*onclick="switchMain\(.([a-z]+).\)', page)
        block = page[page.index("const PANELS={") :]
        keys = re.findall(r"([a-z]+):", block[: block.index("}")])

        assert buttons, "the tab buttons did not parse"
        assert buttons == keys, f"buttons {buttons} against PANELS {keys}"

    def test_every_panel_a_key_names_exists(self, page: str) -> None:
        """A key pointing at a missing id throws on the click and nothing moves."""
        ids = set(re.findall(r'id="([^"]+)"', page))
        block = page[page.index("const PANELS={") :]
        for panel in re.findall(r"'(panel-[a-z]+)'", block[: block.index("}")]):
            assert panel in ids, panel

    def test_the_surface_panel_calls_both_of_its_routes(self, page: str) -> None:
        """The fit for the diagnostics, the query for the grid it draws."""
        assert "/api/vol_surface/fit" in page
        assert "/api/vol_surface/query" in page
        assert "function calcSurface()" in page

    def test_it_draws_a_real_surface_rather_than_a_heat_map(self, page: str) -> None:
        """The reason the charting library was swapped at all.

        ECharts has no native 3D surface, so the panel drew a 2D heat map --
        the library's ceiling, not a choice. A heat map and a surface answer
        different questions: the shape of a volatility surface is the thing
        being judged, and a grid of coloured squares does not show a shape.
        """
        assert "type:'surface'" in page, "no 3D surface trace"
        assert "scene:" in page, "no 3D scene layout"

    def test_the_quotes_it_was_fitted_to_are_drawn_on_it(self, page: str) -> None:
        """The credibility of the whole picture.

        A fitted surface is smooth and plausible whether or not it passes
        through anything. The only way to see that it does is the market points
        on top of it, which is why the fit route grew an opt-in
        ``include_quotes`` for exactly this.
        """
        assert "include_quotes" in page
        assert "type:'scatter3d'" in page, "the quotes are not drawn in the scene"

    def test_the_surface_samples_around_its_own_forward(self, page: str) -> None:
        """SSVI is parameterised in k = log(K/F).

        A strike grid centred on spot samples the wrong part of the surface --
        the same mistake, in a different place, as the spot/forward mismatch
        Phase 2 took out of the engine. So the panel probes for the surface's
        reference forward first and builds the grid from that.

        Asserted on the page rather than inside a named function. The first
        version sliced from ``drawSurfaceHeatmap``, and the 3D rewrite renamed
        it -- the property was untouched and the test broke anyway, which is
        the test's fault and not the rewrite's.
        """
        assert "reference_forward" in page
        assert "F * Math.exp(k)" in page, "the strike grid is not built from the forward"

    def test_the_local_vol_panel_calls_both_of_its_routes(self, page: str) -> None:
        assert "/api/local_vol/surface" in page
        assert "/api/local_vol/validate" in page
        assert "function calcLocalVol()" in page
        assert "async function calcLocalVolRoundTrip()" in page

    def test_the_extrapolated_share_reaches_the_reader(self, page: str) -> None:
        """The largest single assumption behind any long-dated price.

        86.5% of a three-year horizon is beyond the last quoted expiry on the
        committed 510300 snapshot, and no amount of grid refinement touches it.
        It belongs in the metric row beside the fit quality, not in a footnote,
        so both panels have to read the field rather than only mention it in
        prose.
        """
        assert "extrapolated_fraction" in page
        assert "'lv-extrap'" in page and "'vs-extrap'" in page

    def test_the_two_volatilities_are_not_confused(self, page: str) -> None:
        """Local volatility is not implied volatility and the endpoint returns
        both at the same points for exactly that reason. The heatmap draws the
        local surface, so the tooltip has to name which is which and give the
        implied value beside it -- otherwise the panel is an invitation to the
        trap the endpoint was written to avoid.

        Asserted as a property rather than as the expressions that produce it.
        The first version pinned ``d.implied_vol[i][j]`` and
        ``local ${p.data[2]}%`` -- an ECharts tooltip formatter's exact text.
        Porting to Plotly changed both while leaving the tooltip identical to
        a reader, and a test that fails on a faithful port is testing the
        wrong thing.
        """
        body = page[page.index("function drawLocalVolHeatmap(") :]
        body = body[: body.index("\nasync function")]
        assert "implied_vol" in body, "the implied surface is not read at all"
        assert "hovertemplate" in body, "the heat map has no hover text"
        hover = body[body.index("hovertemplate") :]
        assert "local" in hover, "the hover does not say which volatility it draws"
        assert "implied" in hover, "the hover does not give the implied value beside it"

    def test_the_local_surface_is_drawn_beside_the_implied_one(self, page: str) -> None:
        """Two surfaces in one scene, which is the panel's whole argument.

        Reading local volatility as implied volatility is a standing trap, and
        the endpoint returns both at the same points for exactly that reason.
        A tooltip giving the other number is a sentence; two surfaces make the
        difference a shape.

        Measured on the committed 510300 snapshot over the drawn window: local
        volatility spans 13.4% to 89.4% while implied stays inside 18.7% to
        57.0%, and at the same (k, t) they differ by a mean of 14.67 and up to
        32.91 volatility points. Even at the money the gap is systematic --
        local 14.9-17.2% against implied 19.0-20.0%.
        """
        body = page[page.index("function redrawLocalVol()") :]
        body = body[: body.index("\nfunction stretchLocalVol")]
        assert body.count("type:'surface'") == 2, (
            "the implied surface is not drawn alongside the local one"
        )
        assert "lv-show-implied" in body, "the overlay cannot be turned off"

    def test_both_local_vol_slices_carry_both_volatilities(self, page: str) -> None:
        """For the same reason as the scene. A slice showing only the local
        curve would reintroduce the confusion the panel removes."""
        body = page[page.index("function lvPick(") :]
        assert body.count("d.local_vol") >= 2
        assert body.count("d.implied_vol") >= 2
        assert "'implied'" in body and "'local'" in body

    def test_the_round_trip_is_measured_against_the_fit_error(self, page: str) -> None:
        """Dupire makes the round trip exact in continuous time, so any residual
        is discretisation -- which makes the surface's own calibration error the
        only budget that means anything. A panel reporting the error without the
        budget would be reporting a number with no scale.
        """
        assert "calibration_rms_vol_points" in page
        assert "worst_error_vol_points" in page
        assert "share_of_budget" in page


class TestTheIdMap:
    """The second route into the dictionary, and the one with two failure modes.

    An undefined key renders a raw identifier, which is at least visible. An id
    that does not exist in the markup renders nothing at all: the label keeps
    whatever the template hard-coded, so the page looks correct in English and
    silently never translates. That one is invisible in both languages to
    anyone who does not switch.
    """

    def test_every_key_it_names_is_defined_in_both_languages(self, page: str) -> None:
        english, chinese = _dictionary(page, "en"), _dictionary(page, "cn")
        mapped = set(_id_map(page).values())
        assert mapped, "the id map did not parse"
        assert mapped - english == set(), f"undefined in en: {sorted(mapped - english)}"
        assert mapped - chinese == set(), f"undefined in cn: {sorted(mapped - chinese)}"

    def test_every_id_it_names_exists_in_the_markup(self, page: str) -> None:
        ids = set(re.findall(r'id="([^"]+)"', page))
        absent = sorted(i for i in _id_map(page) if i not in ids)
        assert absent == [], f"mapped but absent from the page: {absent}"


class TestTheHestonPanel:
    """The one model here with no second engine behind its headline number.

    Which makes the panel's honesty its actual function. Every test below is
    about something that would be easy and tempting to leave out, and whose
    absence the panel would survive looking perfectly good.
    """

    def test_it_calls_all_three_routes(self, page: str) -> None:
        assert "/api/heston/calibrate" in page
        assert "/api/heston/european" in page
        assert "/api/heston/snowball" in page
        for fn in ("calcHeston", "calcHestonEuropean", "calcHestonSnowball"):
            assert f"async function {fn}()" in page, fn

    def test_the_three_caveats_are_read_from_the_response(self, page: str) -> None:
        """Not asserted in prose -- read from the fields that measure them.

        A hard-coded sentence saying Heston fits worse would still be there
        after a future snapshot where it did not, and the panel would be
        asserting something it had stopped checking.
        """
        body = page[page.index("async function calcHeston()") :]
        body = body[: body.index("\nasync function calcHestonEuropean")]
        assert "d.heston_over_ssvi" in body, "the fit comparison is not read"
        assert "d.well_identified" in body, "identifiability is not read"
        assert "pr.satisfies_feller" in body, "Feller is not read"
        assert "d.parameter_spread" in body
        assert "d.feller_margin" in body or "pr.feller_margin" in body

    def test_the_caveats_render_above_the_numbers(self, page: str) -> None:
        """Order on the page, not merely presence.

        A panel that leads with a price and puts the caveats underneath reads
        as an endorsement with small print. The caveat block has to come before
        the metric cards in the markup.
        """
        panel = page[page.index('id="hs-results"') :]
        panel = panel[: panel.index('id="hs-params"')]
        assert panel.index('id="hs-caveats"') < panel.index('id="hs-vs"')

    def test_the_identifiability_warning_exists_at_all(self, page: str) -> None:
        """The subtlest of the three, and the one most likely to be dropped.

        Three of four starting points reach comparable fit quality with
        parameters spread 0.80 apart, so the five numbers shown are one member
        of a family that fits the quotes equally well. Printing them in a table
        without that sentence invites reading kappa and rho as estimated
        properties of the market, which they are not.
        """
        assert "not identified" in page
        assert "n_comparable_starts" in page
        assert "fits the quotes equally well" in page

    def test_the_single_engine_note_is_reproduced_not_paraphrased(self, page: str) -> None:
        """The response carries its own wording for exactly this reason.

        A UI that summarised it in its own words would be the place the caveat
        got quietly softened -- so the panel prints the field, and prints the
        boolean beside it.
        """
        body = page[page.index("async function calcHestonSnowball()") :]
        assert "d.single_engine_note" in body
        assert "d.cross_checked_by_a_second_engine" in body
        assert "Cross-checked by a second engine" in body

    def test_the_cos_cross_check_is_claimed_only_where_it_holds(self, page: str) -> None:
        """QuantLib backs the European, not the snowball.

        The distinction is the whole point: what is verified is the layer below
        the payoff, so the European section may cite it and the snowball
        section must not.
        """
        european = page[page.index('id="lbl-hs-eu-hint"') :]
        european = european[: european.index("</p>")]
        assert "QuantLib" in european and "5e-09" in european

        snowball = page[page.index('id="lbl-hs-sb-hint"') :]
        snowball = snowball[: snowball.index("</p>")]
        assert "QuantLib" not in snowball
        assert "without a second engine" in snowball

    def test_the_measured_fit_gap_is_quoted_with_both_figures(self, page: str) -> None:
        """1.10-1.12x on three expiries and 1.38-1.41x on six.

        The second pair matters more than the first: the gap widens with the
        number of expiries, because SSVI carries a parameter per expiry and
        Heston does not. Quoting only the three-expiry figure would understate
        it on any chain with real term structure.
        """
        assert "1.38" in page and "1.41" in page
        assert "one parameter per expiry" in page

    def test_qe_is_presented_as_required_rather_than_chosen(self, page: str) -> None:
        """Because Feller is violated, Euler would take the variance negative.

        Described as a refinement it sounds like a quality setting someone
        could turn down.
        """
        assert "a requirement, not a refinement" in page
        assert "Euler" in page


class TestTheCarryCoverageClaim:
    """The report panel leads with a percentage, and I inverted it.

    ``docs/data/futures_carry.md`` tabulates the fraction of each tenor's carry
    that the futures board **measures**: 48.5% at one year, 16.2% at three. The
    panel leads with the fraction that is *not* measured, and the first version
    printed 48% for the one-year note -- the measured number, relabelled. It
    read plausibly, it sat next to a correct 84% for the three-year note, and
    nothing connected the two files.

    One subtraction, in both directions, so the claim cannot drift from its
    source or be inverted again.
    """

    #: Fractions measured, from docs/data/futures_carry.md.
    _MEASURED = {1: 48.5, 3: 16.2}

    def test_the_documented_table_still_says_what_this_assumes(self) -> None:
        """If the measurement is re-run the table moves, and this test is the
        thing that should fail first -- before the panel quietly disagrees with
        it."""
        doc = (
            Path(__file__).resolve().parents[1] / "docs" / "data" / "futures_carry.md"
        ).read_text(encoding="utf-8")
        for tenor, measured in self._MEASURED.items():
            label = "1 year" if tenor == 1 else f"{tenor} years"
            row = re.search(rf"\| {re.escape(label)} \| \*?\*?([0-9.]+)%", doc)
            assert row, f"no row for {label} in futures_carry.md"
            assert float(row.group(1)) == measured, (
                f"{label}: the document says {row.group(1)}% measured, "
                f"this test assumes {measured}%"
            )

    def test_the_panel_states_the_unmeasured_share(self, page: str) -> None:
        hint = page[page.index('id="lbl-sb-rep-hint"') :]
        hint = hint[: hint.index("</p>")]

        one_year = re.search(r"([0-9]+)% of a one-year note", hint)
        three_year = re.search(r"([0-9]+)% of a three-year note", hint)
        assert one_year and three_year, f"the percentages did not parse from: {hint[:200]}"

        for stated, tenor in ((one_year, 1), (three_year, 3)):
            unmeasured = 100.0 - self._MEASURED[tenor]
            # Rounded to whole points on the page, so one point of slack and no
            # more -- two would readmit the 48/52 confusion this exists for.
            assert abs(float(stated.group(1)) - unmeasured) <= 1.0, (
                f"{tenor}y: the panel says {stated.group(1)}% unmeasured, "
                f"the table implies {unmeasured:.1f}%"
            )

    def test_it_is_not_the_measured_share_relabelled(self, page: str) -> None:
        """The actual mistake, named so the regression is unambiguous."""
        hint = page[page.index('id="lbl-sb-rep-hint"') :]
        hint = hint[: hint.index("</p>")]
        for tenor, measured in self._MEASURED.items():
            label = "one-year" if tenor == 1 else "three-year"
            assert f"{measured:.0f}% of a {label} note" not in hint, (
                f"the panel is printing the {label} *measured* share "
                f"({measured}%) as though it were the unmeasured one"
            )

    def test_both_languages_agree_on_the_number(self, page: str) -> None:
        """The Chinese string carries the figure separately, so it can drift
        alone -- and a reader of only one language would never see it."""
        english = re.findall(r"([0-9]+)% of a one-year note", page)
        chinese = re.findall(r"一年期雪球有 ([0-9]+)% 的 carry 测不到", page)
        assert english, "the English claim was not found"
        assert chinese, "the Chinese claim was not found"
        assert set(english) == set(chinese), f"en says {set(english)}, cn says {set(chinese)}"


class TestTheDataPathsThePanelsRead:
    """Every field a new panel reads, checked against a live response.

    This is the second failure mode in this module's docstring, and until now
    it was the one with no test behind it. A panel reading ``d.min_butterfly``
    when the response nests it under ``arbitrage`` renders perfectly: the
    button works, the request succeeds, and the cell says ``undefined``. The
    page cannot tell you, because to JavaScript a missing property is simply
    ``undefined`` rather than an error.

    The paths are listed explicitly rather than scraped out of the script.
    Scraping would drift into testing a regex -- template literals, optional
    chaining and computed indices all defeat it -- and the explicit list
    doubles as a statement of what each panel actually depends on.

    Marked slow because it calls the endpoints for real, including a Heston
    calibration. Calling them for real is the entire point: a mocked response
    would be a copy of the assumption under test.
    """

    #: route -> the dotted paths the panel reads from its response.
    PATHS: dict[str, tuple[str, ...]] = {
        "/api/vol_surface/fit": (
            "as_of",
            "source",
            "calibration.rms_vol_error",
            "calibration.max_vol_error",
            "calibration.n_points",
            "calibration.slices",
            "arbitrage.arbitrage_free",
            "arbitrage.min_butterfly",
            "arbitrage.min_calendar",
            "arbitrage.n_checked",
            "arbitrage.n_violations",
            "input.slices",
        ),
        "/api/local_vol/surface": (
            "as_of",
            "spot",
            "spots",
            "times",
            "moneyness",
            "local_vol",
            "implied_vol",
            "underlying",
            "extrapolated_fraction",
            "solver_moneyness_range",
            "forward_curve.maturities",
            "diagnostics.well_defined",
            "diagnostics.min_denominator",
            "diagnostics.clamped_fraction",
            "diagnostics.max_peclet",
            "diagnostics.min_vol",
            "diagnostics.max_vol",
            "diagnostics.n_evaluated",
            "diagnostics.n_undefined",
            "diagnostics.n_floored",
            "diagnostics.n_capped",
        ),
        "/api/local_vol/validate": (
            "worst_error_vol_points",
            "calibration_rms_vol_points",
            "share_of_budget",
            "passes",
            "rows",
        ),
        "/api/heston/calibrate": (
            "as_of",
            "n_points",
            "n_starts",
            "n_comparable_starts",
            "parameter_spread",
            "well_identified",
            "converged",
            "rms_vol_points",
            "ssvi_rms_vol_points",
            "heston_over_ssvi",
            "fits_worse_than_ssvi",
            "max_vol_error",
            "params.v0",
            "params.kappa",
            "params.theta",
            "params.sigma",
            "params.rho",
            "params.initial_vol",
            "params.long_run_vol",
            "params.satisfies_feller",
            "params.feller_margin",
        ),
        "/api/heston/european": (
            "price",
            "flag",
            "strike",
            "T",
            "spot",
            "method",
            "terms",
            "truncation",
            "params_source",
        ),
        "/api/heston/snowball": (
            "price",
            "ci95",
            "std_error",
            "paths",
            "steps_per_year",
            "vol_model",
            "ko_prob",
            "ki_prob",
            "ki_no_ko_prob",
            "expected_ko_time",
            "params_source",
            "single_engine_note",
            "cross_checked_by_a_second_engine",
        ),
    }

    #: Deliberately small where the endpoint allows it. The grids here are the
    #: smallest that still exercise every path, not the ones the panel sends.
    BODIES: dict[str, dict] = {
        "/api/vol_surface/fit": {"underlying": "510300", "source": "snapshot", "r": 0.02},
        "/api/local_vol/surface": {
            "underlying": "510300",
            "source": "snapshot",
            "r": 0.02,
            "horizon": 1.0,
            "n_spots": 5,
            "n_times": 3,
        },
        "/api/local_vol/validate": {
            "underlying": "510300",
            "source": "snapshot",
            "r": 0.02,
        },
        "/api/heston/calibrate": {"underlying": "510300", "source": "snapshot", "r": 0.02},
        "/api/heston/european": {
            "underlying": "510300",
            "source": "snapshot",
            "r": 0.02,
            "K": 4.1,
            "T": 0.5,
            "flag": "c",
        },
        "/api/heston/snowball": {
            "underlying": "510300",
            "source": "snapshot",
            "r": 0.02,
            "S0": 100.0,
            "T": 1.0,
            "paths": 5000,
        },
    }

    @pytest.fixture(scope="class")
    @classmethod
    def responses(cls) -> dict[str, dict]:
        import app

        app.app.config.update(TESTING=True)
        client = app.app.test_client()
        out: dict[str, dict] = {}
        for route, body in cls.BODIES.items():
            response = client.post(route, json=body)
            assert response.status_code == 200, (
                f"{route} answered {response.status_code}: "
                f"{response.data.decode('utf-8', 'replace')[:200]}"
            )
            out[route] = response.get_json()
        return out

    @pytest.mark.slow
    @pytest.mark.parametrize("route", sorted(PATHS))
    def test_every_field_the_panel_reads_is_sent(
        self, route: str, responses: dict[str, dict]
    ) -> None:
        payload = responses[route]
        missing = []
        for path in self.PATHS[route]:
            node = payload
            for part in path.split("."):
                if not isinstance(node, dict) or part not in node:
                    missing.append(path)
                    break
                node = node[part]
            else:
                # Present is not enough: a null reaches the page as a blank
                # cell, which is the same defect wearing a different hat.
                if node is None and path != "share_of_budget":
                    missing.append(f"{path} (sent as null)")
        assert missing == [], f"{route} does not send: {missing}"

    @pytest.mark.slow
    def test_the_slice_rows_carry_the_maturity_the_panel_sorts_on(
        self, responses: dict[str, dict]
    ) -> None:
        """Both volatility panels take the last quoted expiry from a row field.

        The surface panel reads it off ``calibration.slices`` and the local
        volatility panel off ``forward_curve.maturities``; a dashed line drawn
        at the wrong place is worse than none, because it labels extrapolated
        cells as quoted.
        """
        slices = responses["/api/vol_surface/fit"]["calibration"]["slices"]
        assert slices, "no fitted slices"
        for row in slices:
            assert "maturity" in row, row
            assert isinstance(row["maturity"], int | float)

        maturities = responses["/api/local_vol/surface"]["forward_curve"]["maturities"]
        assert maturities and all(isinstance(m, int | float) for m in maturities)

    @pytest.mark.slow
    def test_the_round_trip_rows_carry_every_column_the_table_prints(
        self, responses: dict[str, dict]
    ) -> None:
        rows = responses["/api/local_vol/validate"]["rows"]
        assert rows, "no round-trip rows"
        for column in (
            "maturity",
            "moneyness",
            "strike",
            "target_vol",
            "recovered_vol",
            "error_vol_points",
        ):
            assert column in rows[0], f"{column} missing from a round-trip row"

    @pytest.mark.slow
    def test_the_heston_snowball_still_declares_itself_unchecked(
        self, responses: dict[str, dict]
    ) -> None:
        """The panel prints this verbatim, so it has to keep arriving.

        If a second engine is ever built for it, this test fails -- and that is
        correct: the panel's red warning block would then be wrong, and
        somebody has to go and change it rather than have it quietly keep
        warning about something that is no longer true.
        """
        payload = responses["/api/heston/snowball"]
        assert payload["cross_checked_by_a_second_engine"] is False
        assert "no independent second engine" in payload["single_engine_note"]


class TestTheChartingLibraryClaim:
    """The page names its charting library in four places and loads one.

    Swapping ECharts for Plotly left **seven** stale claims behind: two
    "ECharts 5" badges in the About tab, a shields.io badge, a feature line, a
    project-structure comment and a footer in the README, and a row in the
    Tech Stack table. Every one of them rendered perfectly and every one was
    wrong. None of the existing tests could see it -- the dependency test
    deliberately excludes the charting library, because it comes from a CDN
    rather than from requirements.txt, which is exactly the hole the stale row
    sat in.

    So the library is asserted against the thing that actually decides it: the
    script tag.
    """

    #: Libraries this page could plausibly be using, by the substring that
    #: would appear in its CDN URL.
    _KNOWN = ("plotly", "echarts", "chart.js", "d3", "highcharts")

    @staticmethod
    def _loaded(page: str) -> str:
        """The charting library the page actually loads."""
        tags = re.findall(r'<script src="([^"]+)"', page)
        for url in tags:
            for name in TestTheChartingLibraryClaim._KNOWN:
                if name in url.lower():
                    return name
        raise AssertionError(f"no known charting library in the script tags: {tags}")

    def test_the_about_badges_name_the_library_that_is_loaded(self, page: str) -> None:
        """Two badges, in the English and Chinese copies of the tech stack."""
        loaded = self._loaded(page)
        badges = re.findall(r'<span class="badge">([^<]*)</span>', page)
        charting = [b for b in badges if any(k in b.lower() for k in self._KNOWN)]
        assert charting, f"no charting badge among {len(badges)} badges"
        for badge in charting:
            assert loaded in badge.lower(), (
                f"the About tab advertises {badge!r} and the page loads {loaded!r}"
            )

    def test_no_badge_names_a_library_that_is_not_loaded(self, page: str) -> None:
        """The failure that actually happened: the badge kept the old name."""
        loaded = self._loaded(page)
        stale = [k for k in self._KNOWN if k != loaded and f">{k}" in page.lower()]
        assert stale == [], f"the page still advertises {stale} but loads {loaded!r}"

    def test_the_readme_names_the_same_library(self, page: str) -> None:
        """Four mentions there, and they all went stale together."""
        loaded = self._loaded(page)
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
        wrong = sorted({k for k in self._KNOWN if k != loaded and k in readme.lower()})
        assert wrong == [], f"README still mentions {wrong} while the page loads {loaded!r}"


class TestTheSharedPlotTemplate:
    """PLOT() styles every chart, and it used to drop what it did not know.

    The first version named the keys it understood and silently discarded the
    rest, so the Monte Carlo chart's ``bargap:0`` never reached Plotly and its
    bars drew with gaps where the ECharts version had them touching. Nothing
    reported it: a layout key Plotly never receives is not an error, and a bar
    chart with gaps looks like a bar chart.

    Found by listing the keys every call site passes and intersecting with the
    keys the function reads -- which is the check below, kept rather than
    thrown away, because the forwarding is one deletion away from regressing.
    """

    _NAMED = ("x", "y", "xaxis", "yaxis", "title", "legend", "hovermode", "margin")

    @staticmethod
    def _plot_body(page: str) -> str:
        body = page[page.index("function PLOT(") :]
        return body[: body.index("\n}")]

    def test_it_forwards_keys_it_does_not_name(self, page: str) -> None:
        """The whole point: any valid Plotly layout key works without this
        function having to know about it."""
        body = self._plot_body(page)
        assert "...rest" in body, (
            "PLOT() no longer spreads its unrecognised keys into the layout, so "
            "any layout key it does not name is being dropped again"
        )

    def test_every_key_any_caller_passes_is_either_named_or_forwarded(self, page: str) -> None:
        """Belt and braces. With the rest spread this holds by construction;
        it fails loudly if the spread is removed and a caller is relying on it.
        """
        body = self._plot_body(page)
        forwards = "...rest" in body

        passed: set[str] = set()
        for match in re.finditer(r"PLOT\(\{", page):
            opening = match.end() - 1
            depth, index = 0, opening
            while index < len(page):
                if page[index] == "{":
                    depth += 1
                elif page[index] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                index += 1
            arg = page[opening : index + 1]
            for key in re.finditer(r"([a-zA-Z_]\w*)\s*:", arg):
                before = arg[: key.start()]
                # Top level of this object literal only.
                if before.count("{") - before.count("}") == 1 and before.count("[") == before.count(
                    "]"
                ):
                    passed.add(key.group(1))

        assert passed, "no PLOT() call sites parsed"
        if not forwards:
            dropped = sorted(passed - set(self._NAMED))
            assert dropped == [], f"PLOT() would drop {dropped}"


class TestTheMarketDataTab:
    """The page a desk opens first, and the first one that can write anything.

    Every other panel in this interface reads. This one uploads a file and
    publishes a mark, so most of what is worth testing here is about *not*
    writing when nobody asked.
    """

    def test_it_is_the_first_tab_and_the_default(self, page: str) -> None:
        """Both, and they have to agree.

        Step 1 of this phase deliberately left the default on Option Pricing
        while reordering, because the first tab then was Models, which shows
        nothing until a surface is fitted or Heston calibrated -- nineteen
        seconds the first time. This page fits in 0.3s, so landing on "mark
        the curves" is no longer landing on a blank panel to make a point
        about ordering.
        """
        buttons = re.findall(r'class="mtab[^"]*"[^>]*onclick="switchMain\(.([a-z]+).\)', page)
        assert buttons[0] == "market", f"the first tab is {buttons[0]!r}"
        highlighted = re.search(r'class="mtab on"[^>]*onclick="switchMain\(.([a-z]+).\)', page)
        opened = re.findall(r'class="tab-panel on" id="panel-([a-z]+)"', page)
        assert highlighted is not None
        assert highlighted.group(1) == "market"
        assert opened == ["market"], f"the open panel is {opened}"

    def test_it_calls_the_four_routes_it_needs(self, page: str) -> None:
        for route in (
            "/api/marketdata/upload",
            "/api/marketdata/publish",
            "/api/marketdata/mark",
            "/api/vol_surface/fit",
        ):
            assert route in page, route

    def test_the_gate_preview_cannot_publish(self, page: str) -> None:
        """The defect this nearly shipped with.

        The panel shows the gate results as soon as a surface is fitted, well
        before anyone decides to publish, and the only route that computes
        them is the publish route. Without ``dry_run`` that preview would have
        published a mark on **every fit** -- silently, since the response of a
        successful publish looks much like a gate report.
        """
        body = page[page.index("async function renderGates(") :]
        body = body[: body.index("\nfunction renderGateTable")]
        assert "dry_run: true" in body, (
            "the gate preview calls the publish route without dry_run, so it "
            "publishes a mark every time a surface is fitted"
        )

    def test_publishing_uses_the_request_the_gates_were_run_against(self, page: str) -> None:
        """Otherwise the gates on screen could describe a different surface
        from the one being blessed -- the controls may have been changed
        between the fit and the click."""
        assert "mdLastFit" in page
        body = page[page.index("async function publishMark()") :]
        body = body[: body.index("\nasync function readMark")]
        assert "mdLastFit.request" in body
        assert "Fit a surface first" in body, "publishing with no fit is not refused"

    def test_the_refusal_is_reproduced_rather_than_reworded(self, page: str) -> None:
        """The route's own message already names the gate, the distance and
        how to override. A panel that summarised it in its own words would be
        the place that softened it."""
        body = page[page.index("async function publishMark()") :]
        body = body[: body.index("\nasync function readMark")]
        assert "d.error" in body

    def test_a_mark_that_no_longer_reproduces_is_shown_as_such(self, page: str) -> None:
        """The whole mechanism rests on this case being visible. A surface
        that fails its own fingerprint check must not be presented as the
        published mark."""
        body = page[page.index("async function readMark()") :]
        assert "d.reproduces" in body
        assert "no longer reproduces" in body

    def test_the_screening_section_reads_both_counts(self, page: str) -> None:
        """A surface fitted to 42 of 126 quotes is a different claim from one
        fitted to all of them, so the denominator has to be read, not just the
        numerator."""
        body = page[page.index("function renderScreening(") :]
        body = body[: body.index("\nfunction renderForwards")]
        assert "n_rejected" in body, "the discarded count is not read"
        assert "n_points" in body
        assert "rejections" in body, "the per-reason breakdown is not read"

    def test_the_rates_section_says_a_flat_rate_is_a_flat_rate(self, page: str) -> None:
        """A term structure exists but needs the Wind panel, which this
        checkout may not have. Drawing a flat line and calling it a curve is
        exactly the kind of claim the rest of this interface removes.
        """
        body = page[page.index("function renderRatesNote(") :]
        body = body[: body.index("\nasync function renderGates")]
        assert "flat" in body.lower()
        assert "OTCPRICER_WIND_DIR" in body, "the reason there is no curve is not given"

    def test_the_wind_source_demands_its_date(self, page: str) -> None:
        """The panel holds 725 dates and build_source refuses without one, so
        the form has to ask rather than letting the request fail."""
        body = page[page.index("function marketRequest()") :]
        body = body[: body.index("\nasync function uploadMarketCsv")]
        assert "as_of" in body
        assert "725" in body, "the refusal does not say why a date is needed"

    def test_an_upload_token_is_required_before_a_csv_fit(self, page: str) -> None:
        body = page[page.index("function marketRequest()") :]
        body = body[: body.index("\nasync function uploadMarketCsv")]
        assert "mdCsvToken" in body
        assert "upload a CSV first" in body
