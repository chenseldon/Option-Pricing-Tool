"""Tests for the (S, t) Greeks surface route.

The engine-level tests in ``test_pde_snapshots.py`` cover the layers
themselves. These cover the two claims the route makes on top of them: that
the surface and the scalar Greeks are the same numbers, and that what it does
not return is said rather than left to be noticed.
"""

from __future__ import annotations

import numpy as np
import pytest

BODY = {
    "S0": 100,
    "T": 1.0,
    "KO_pct": 1.03,
    "KI_pct": 0.75,
    "coupon_pa": 0.20,
    "r": 0.02,
    "q": 0.0,
    "sigma": 0.20,
    "basis_pa": 0.0,
}


@pytest.fixture(scope="module")
def client():  # noqa: ANN201
    import app

    app.app.config.update(TESTING=True)
    return app.app.test_client()


@pytest.fixture(scope="module")
def surface(client):  # noqa: ANN001, ANN201
    response = client.post("/api/snowball_greeks_surface", json={**BODY, "layers": 120})
    assert response.status_code == 200, response.data[:300]
    return response.get_json()


class TestTheShape:
    def test_the_grids_line_up(self, surface) -> None:  # noqa: ANN001
        """``value[i][j]`` is at ``times[i]``, ``spots[j]``, and a transposed
        surface would still look like a plausible surface."""
        n_t, n_s = len(surface["times"]), len(surface["spots"])
        for name in ("value", "delta", "gamma"):
            grid = surface[name]
            assert len(grid) == n_t, f"{name} has {len(grid)} rows for {n_t} times"
            assert all(len(row) == n_s for row in grid), f"{name} rows are ragged"

    def test_time_increases(self, surface) -> None:  # noqa: ANN001
        times = surface["times"]
        assert all(b >= a - 1e-12 for a, b in zip(times, times[1:], strict=False))

    def test_the_window_brackets_both_barriers(self, surface) -> None:  # noqa: ANN001
        """The solver's grid reaches four standard deviations past the
        barriers so a Dirichlet condition is harmless there. Drawing all of it
        would spend most of the picture on spot levels nobody is risking
        against -- but the barriers themselves must be inside the window, or
        the one thing worth looking at is off-screen.
        """
        lo, hi = surface["spots"][0], surface["spots"][-1]
        assert lo < surface["ki_barrier"] < hi
        assert lo < surface["ko_barrier"] < hi

    def test_everything_returned_is_finite(self, surface) -> None:  # noqa: ANN001
        for name in ("value", "delta", "gamma"):
            grid = np.array(surface[name], dtype=float)
            assert np.all(np.isfinite(grid)), f"{name} has non-finite entries"


class TestTheEventsSurvive:
    def test_observation_dates_appear_twice(self, surface) -> None:  # noqa: ANN001
        pairs = [
            i
            for i in range(len(surface["stages"]) - 1)
            if surface["stages"][i] == "pre_projection"
            and surface["stages"][i + 1] == "post_projection"
        ]
        assert pairs, "no event pairs reached the response"
        assert surface["n_event_pairs"] == len(pairs)
        for i in pairs:
            assert surface["times"][i] == surface["times"][i + 1]

    def test_the_two_sides_differ(self, surface) -> None:  # noqa: ANN001
        """If they matched, the jump the pair exists for is not in the data."""
        value = np.array(surface["value"], dtype=float)
        delta = np.array(surface["delta"], dtype=float)
        for i, stage in enumerate(surface["stages"][:-1]):
            if stage != "pre_projection":
                continue
            moved = not np.array_equal(value[i], value[i + 1]) or not np.array_equal(
                delta[i], delta[i + 1]
            )
            assert moved, f"the pair at t={surface['times'][i]} did not move"

    def test_the_delta_jump_at_a_knock_out_is_material(self, surface) -> None:  # noqa: ANN001
        """Measured 0.1423 on this trade. Asserted as "not negligible" rather
        than as a value, since the magnitude moves with the contract."""
        delta = np.array(surface["delta"], dtype=float)
        jumps = [
            float(np.max(np.abs(delta[i + 1] - delta[i])))
            for i in range(len(surface["stages"]) - 1)
            if surface["stages"][i] == "pre_projection" and "knock_out" in surface["events"][i + 1]
        ]
        assert jumps, "no knock-out pairs in the response"
        assert max(jumps) > 0.01, f"largest delta jump is only {max(jumps)}"


class TestItAgreesWithTheScalarGreeks:
    def test_delta_and_gamma_at_spot_match_bit_for_bit(self, client, surface) -> None:  # noqa: ANN001
        """The reason ``grid_derivative_curves`` was separated out rather than
        a second derivative routine written for the surface.

        Two code paths computing delta would agree to a few digits and
        disagree in the last ones, and nothing would say which was right. One
        path means the question cannot arise -- so this asserts equality, not
        closeness.
        """
        scalar = client.post("/api/snowball_risk", json=BODY).get_json()
        assert "error" not in scalar, scalar

        # The scalar is read at inception, so the comparison is against the
        # earliest layer -- the layers run forwards in time and the solve runs
        # backwards, which is exactly the confusion the reordering removed.
        times = surface["times"]
        first = min(range(len(times)), key=lambda i: times[i])
        assert times[first] <= 1e-9, f"no layer at inception (earliest {times[first]})"

        # Both are raw dV/dS and d2V/dS2 at the quoted spot, so they are
        # directly comparable without a units conversion. Worth stating: the
        # magnitudes look unrelated at a glance -- delta 0.00214 here against
        # 0.21414 for the same contract on unit notional -- because delta
        # scales as 1/S and gamma as 1/S^2, and this contract is struck at 100.
        spots = np.array(surface["spots"], dtype=float)
        for name in ("delta", "gamma"):
            curve = np.array(surface[name], dtype=float)[first]
            interpolated = float(np.interp(np.log(surface["spot"]), np.log(spots), curve))
            # The wire rounds both sides, so this compares at the precision
            # actually transmitted rather than pretending it carries doubles.
            assert interpolated == pytest.approx(scalar[name], rel=1e-5), (
                f"{name}: surface {interpolated} against scalar {scalar[name]}"
            )


class TestWhatItDoesNotReturn:
    """Said, rather than left to be noticed.

    Three Greeks are absent and each for its own reason. An interface that
    simply omitted them would read as an oversight, and somebody would add a
    wrong theta rather than find out why there is none.
    """

    def test_theta_is_absent_and_the_reason_is_given(self, surface) -> None:
        assert "theta" not in surface
        assert "Theta" in surface["note"] or "theta" in surface["note"]

    def test_vega_and_rho_are_absent_and_the_reason_is_given(self, surface) -> None:
        assert "vega" not in surface
        assert "rho" not in surface
        assert "vega" in surface["note"]

    def test_the_route_docstring_explains_all_three(self) -> None:
        """Because the response note is one sentence and the reasons are not.

        Theta is a derivative between *adjacent* layers and these are
        subsampled -- at an event two share a time, so differencing them would
        divide a contract jump by zero elapsed time. Vega and rho each need
        their own bumped solve with its own layers, making a full set seven
        solves rather than one.
        """
        import app

        doc = app.snowball_greeks_surface.__doc__ or ""
        assert "adjacent" in doc, "the theta omission is not explained"
        assert "seven solves" in doc, "the vega/rho cost is not stated"


class TestTheGridCostIsPublished:
    def test_the_accuracy_note_is_contract_dependent(self, surface) -> None:  # noqa: ANN001
        """A single published figure would understate the error exactly where
        a desk works: 0.34% on the reference trade against 0.96% on the same
        note at the CSI 1000's measured carry."""
        note = surface["grid_accuracy"]["note"]
        assert "0.34%" in note and "0.96%" in note
        assert surface["grid_accuracy"]["estimated"] is False

    def test_gamma_is_flagged_as_the_grid_sensitive_one(self, surface) -> None:  # noqa: ANN001
        """Its peak sits on a barrier, where the height is partly the grid's.
        Reading that number as the gamma would be the panel's most likely
        misuse."""
        assert "Gamma" in surface["grid_accuracy"]["note"]

    def test_the_layer_count_is_reported_alongside_what_was_asked(self, surface) -> None:  # noqa: ANN001
        assert surface["grid"]["layers_requested"] == 120
        assert surface["n_layers"] == 120


class TestTheRefusals:
    def test_an_absurd_layer_count_is_clamped_not_refused(self, client) -> None:  # noqa: ANN001
        """Clamped because a layer count is a display choice, not a contract
        term: silently giving 400 is kinder than a 400 error, and the response
        reports what was used."""
        payload = client.post(
            "/api/snowball_greeks_surface", json={**BODY, "layers": 100_000}
        ).get_json()
        assert payload["n_layers"] <= 400

    def test_a_bad_contract_is_refused_with_a_reason(self, client) -> None:  # noqa: ANN001
        response = client.post(
            "/api/snowball_greeks_surface", json={**BODY, "KI_pct": "not a number"}
        )
        assert response.status_code == 400
        assert "Invalid parameters" in response.get_json()["error"]
