"""Tests for the solution layers the backward induction already computes.

The layers are new output, not new arithmetic -- the solver passed through
every one of them on its way to the answer and overwrote them. So the first
thing to establish is that asking for them changes nothing, and the rest is
about the one structure that makes them worth keeping: both sides of every
projection.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from otcpricer.core.market import EquityMarket
from otcpricer.engines.pde import PDEConfig, price_snowball_pde
from otcpricer.products.snowball import SnowballTerms

TERMS = SnowballTerms(S0=1.0, T=1.0, ko_pct=1.03, ki_pct=0.75, coupon_pa=0.20)
MARKET = EquityMarket(S=1.0, r=0.02, sigma=0.20, q=0.0)
GRID = PDEConfig(space_nodes=1600, steps_per_year=1008)


@pytest.fixture(scope="module")
def plain():  # noqa: ANN201
    return price_snowball_pde(TERMS, MARKET, GRID)


@pytest.fixture(scope="module")
def layered():  # noqa: ANN201
    return price_snowball_pde(TERMS, MARKET, replace(GRID, snapshot_layers=120))


class TestAskingForLayersChangesNothing:
    """The precondition for everything else here.

    These layers are a side output of a solve that already computed them, so a
    price that moved when they were requested would mean the capture had
    reached into the induction rather than observing it.
    """

    def test_the_price_is_bit_identical(self, plain, layered) -> None:  # noqa: ANN001
        assert plain.price == layered.price

    @pytest.mark.parametrize(
        "name", ["V_no_ki", "V_ki", "V_no_ki_next_step", "V_ki_next_step", "S_grid"]
    )
    def test_every_returned_array_is_bit_identical(self, plain, layered, name: str) -> None:  # noqa: ANN001
        assert np.array_equal(getattr(plain, name), getattr(layered, name))

    def test_none_are_kept_unless_asked_for(self, plain) -> None:  # noqa: ANN001
        """Zero by default, so the pricing path allocates nothing."""
        assert plain.snapshots is None
        assert PDEConfig().snapshot_layers == 0

    def test_a_negative_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="snapshot_layers must be non-negative"):
            PDEConfig(snapshot_layers=-1)


class TestTheBudgetIsHonoured:
    """It was not, at first, and silently.

    The first capture plan kept every event step unconditionally, on the
    reasoning that events are rare. Knock-ins are observed daily -- 252 a year
    against the knock-out's 12 -- so a request for 60 layers returned 504. A
    budget exceeded eightfold without complaint is not a budget, which is why
    this is tested at several sizes rather than one.
    """

    @pytest.mark.parametrize("want", [30, 60, 120, 250])
    def test_the_count_is_what_was_asked_for(self, want: int) -> None:
        result = price_snowball_pde(TERMS, MARKET, replace(GRID, snapshot_layers=want))
        assert result.snapshots is not None
        assert result.snapshots.n_layers == want

    def test_a_budget_below_the_knock_out_floor_exceeds_it_deliberately(self) -> None:
        """Rather than dropping a knock-out.

        Dropping a knock-out loses a feature that occurs nowhere else, while
        dropping a knock-in loses one of many near-identical samples of the
        same cliff. So the floor is twice the capturable knock-out count, and
        the docstring says so.

        **Capturable** is not the same as scheduled. The reference trade has
        nine knock-out observations and eight pairs appear, because the last
        one is at maturity -- and the maturity knock-out *is* the terminal
        condition, applied before the loop begins. There is no pre-projection
        side to sample: the induction starts there. Nothing is missing; the
        first version of this test simply counted the wrong thing.
        """
        result = price_snowball_pde(TERMS, MARKET, replace(GRID, snapshot_layers=4))
        assert result.snapshots is not None

        scheduled = TERMS.ko_observation_times()
        at_maturity = bool(scheduled.size and scheduled[-1] >= TERMS.T - 1e-12)
        capturable = int(scheduled.size) - (1 if at_maturity else 0)

        assert result.snapshots.n_layers >= 2 * capturable
        kept = sum(1 for e in result.snapshots.events if "knock_out" in e)
        assert kept == capturable, "a knock-out was dropped to fit the budget"

    def test_every_monthly_knock_out_is_also_a_knock_in_date(self) -> None:
        """Which is why the event labels compose rather than being exclusive.

        The knock-in is observed daily and the knock-out monthly, so every
        knock-out date is also a knock-in date and its layer is tagged
        ``knock_in+knock_out``. A label that had to pick one would under-report
        what happened at that instant.
        """
        result = price_snowball_pde(TERMS, MARKET, replace(GRID, snapshot_layers=120))
        assert result.snapshots is not None
        combined = [e for e in result.snapshots.events if "knock_out" in e]
        assert combined, "no knock-out layers at all"
        assert all(e == "knock_in+knock_out" for e in combined), combined


class TestTheLayersThemselves:
    def test_time_increases(self, layered) -> None:  # noqa: ANN001
        """The induction runs backwards and collects from maturity towards
        inception. Anything drawing a surface wants time increasing, and a
        consumer forced to know the solver's direction would be carrying the
        solver's implementation detail."""
        times = layered.snapshots.times
        assert np.all(np.diff(times) >= -1e-12)
        assert times[0] < times[-1]

    def test_each_layer_spans_the_whole_grid(self, layered) -> None:  # noqa: ANN001
        lay = layered.snapshots
        assert lay.V_ki.shape == (lay.n_layers, layered.S_grid.size)
        assert lay.V_no_ki.shape == lay.V_ki.shape

    def test_the_layers_span_the_note_s_life(self, layered) -> None:  # noqa: ANN001
        times = layered.snapshots.times
        assert times[0] <= 0.01, f"the first layer is at t={times[0]}"
        assert times[-1] >= TERMS.T - 0.01, f"the last layer is at t={times[-1]}"

    def test_every_layer_is_finite(self, layered) -> None:  # noqa: ANN001
        lay = layered.snapshots
        assert np.all(np.isfinite(lay.V_ki))
        assert np.all(np.isfinite(lay.V_no_ki))


class TestTheEventPairs:
    """The structure the layers exist for.

    An observation is a discontinuity in the contract, not in time. One sample
    at that instant would have to pick a side, and whichever it picked would
    make the jump invisible -- so each is sampled twice at the same time
    coordinate, before and after the projection.
    """

    @staticmethod
    def _pairs(lay) -> list[tuple[int, int]]:  # noqa: ANN001
        return [
            (i, i + 1)
            for i in range(lay.n_layers - 1)
            if lay.stages[i] == "pre_projection" and lay.stages[i + 1] == "post_projection"
        ]

    def test_there_are_pairs_at_all(self, layered) -> None:  # noqa: ANN001
        assert len(self._pairs(layered.snapshots)) > 0

    def test_each_pair_shares_its_time_coordinate(self, layered) -> None:  # noqa: ANN001
        """A repeated time is the event, not a duplicate to be cleaned up."""
        lay = layered.snapshots
        for i, j in self._pairs(lay):
            assert lay.times[i] == lay.times[j]

    def test_no_pair_is_identical(self, layered) -> None:  # noqa: ANN001
        """If the two sides matched, the projection was captured on one side
        twice and the whole mechanism is decoration."""
        lay = layered.snapshots
        for i, j in self._pairs(lay):
            moved = not np.array_equal(lay.V_no_ki[i], lay.V_no_ki[j]) or not np.array_equal(
                lay.V_ki[i], lay.V_ki[j]
            )
            assert moved, f"the pair at t={lay.times[i]} did not move"

    def test_the_post_side_carries_which_projection_fired(self, layered) -> None:  # noqa: ANN001
        lay = layered.snapshots
        for i, j in self._pairs(lay):
            assert lay.events[i] == ""
            assert lay.events[j] in {"knock_in", "knock_out", "knock_in+knock_out"}

    def test_the_knock_in_cliff_is_the_larger_jump(self, layered) -> None:  # noqa: ANN001
        """Which contradicts the assumption the first capture plan was built on.

        Knock-outs look like the dramatic event -- the note terminates and pays
        a coupon. But the knock-in projection replaces the un-knocked surface
        below the barrier with the knocked-in one, which is a short put and
        worth far less, so the cliff at the knock-in barrier is the largest
        feature on the surface.

        Measured on this trade: 0.2118 of notional against the knock-out's
        0.0935, 2.3 times. Asserted as an ordering rather than a value, since
        the magnitudes move with the contract and the ordering is the finding.
        """
        lay = layered.snapshots
        ki, ko = [], []
        for i, j in self._pairs(lay):
            jump = max(
                float(np.max(np.abs(lay.V_no_ki[j] - lay.V_no_ki[i]))),
                float(np.max(np.abs(lay.V_ki[j] - lay.V_ki[i]))),
            )
            (ko if "knock_out" in lay.events[j] else ki).append(jump)
        assert ki and ko, "both kinds of projection should be present"
        assert max(ki) > max(ko), (
            f"the knock-in cliff ({max(ki):.4f}) is no longer the larger jump "
            f"({max(ko):.4f}); the capture plan's reasoning rests on it"
        )


class TestSubsampling:
    @pytest.mark.parametrize("limit", [10, 20, 30, 60, 119, 120, 400])
    def test_no_pair_is_ever_split(self, layered, limit: int) -> None:  # noqa: ANN001
        """A sample on one side of a projection and not the other turns a jump
        into a slope, which is the one thing these layers must not do."""
        sub = layered.snapshots.subsample(limit)
        for i, stage in enumerate(sub.stages):
            if stage == "pre_projection":
                assert i + 1 < sub.n_layers, "a pre-projection layer ends the series"
                assert sub.stages[i + 1] == "post_projection"
            if stage == "post_projection":
                assert i > 0 and sub.stages[i - 1] == "pre_projection"

    def test_it_keeps_every_knock_out(self, layered) -> None:  # noqa: ANN001
        full = layered.snapshots
        n_ko = sum(1 for e in full.events if "knock_out" in e)
        for limit in (10, 30, 60):
            sub = full.subsample(limit)
            assert sum(1 for e in sub.events if "knock_out" in e) == n_ko

    def test_asking_for_more_than_there_is_returns_the_same_object(self, layered) -> None:  # noqa: ANN001
        lay = layered.snapshots
        assert lay.subsample(lay.n_layers + 10) is lay
        assert lay.subsample(0) is lay

    def test_time_still_increases_afterwards(self, layered) -> None:  # noqa: ANN001
        sub = layered.snapshots.subsample(40)
        assert np.all(np.diff(sub.times) >= -1e-12)


class TestTheCost:
    def test_it_is_memory_rather_than_time(self) -> None:
        """The layers are already computed; keeping them is a copy.

        Measured at 1600 x 1008: 0.115s with none and 0.111s with sixty, which
        is noise. A timing assertion on a shared machine should check the
        direction of a claim, not a threshold, and the claim here is that there
        is no arithmetic cost -- so this asserts the weaker, true thing.
        """
        import time

        start = time.perf_counter()
        price_snowball_pde(TERMS, MARKET, GRID)
        bare = time.perf_counter() - start

        start = time.perf_counter()
        result = price_snowball_pde(TERMS, MARKET, replace(GRID, snapshot_layers=120))
        kept = time.perf_counter() - start

        assert kept < bare * 2.0, f"{kept:.3f}s against {bare:.3f}s for the same solve"
        assert result.snapshots is not None

    def test_the_memory_scales_with_the_layer_count(self) -> None:
        small = price_snowball_pde(TERMS, MARKET, replace(GRID, snapshot_layers=60))
        large = price_snowball_pde(TERMS, MARKET, replace(GRID, snapshot_layers=240))
        assert small.snapshots is not None and large.snapshots is not None
        ratio = large.snapshots.V_ki.nbytes / small.snapshots.V_ki.nbytes
        assert 3.5 < ratio < 4.5, f"240 layers is {ratio:.2f}x the bytes of 60"
