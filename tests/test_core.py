"""Tests for the cross-cutting core layer: types, day counts, validation, config."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from otcpricer.core.config import get_settings
from otcpricer.core.daycount import DayCount, parse_tenor, year_fraction
from otcpricer.core.exceptions import OtcPricerError, ParameterError
from otcpricer.core.logging import configure_logging, get_logger
from otcpricer.core.types import BarrierKind, Greeks, KIStrikeMode, OptionType
from otcpricer.core.validation import (
    check_in_range,
    check_non_negative,
    check_positive,
    validate_snowball,
    validate_vanilla,
)


class TestOptionType:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("c", OptionType.CALL),
            ("C", OptionType.CALL),
            ("call", OptionType.CALL),
            ("CALL", OptionType.CALL),
            (" call ", OptionType.CALL),
            ("看涨", OptionType.CALL),
            ("p", OptionType.PUT),
            ("put", OptionType.PUT),
            ("看跌", OptionType.PUT),
            (OptionType.CALL, OptionType.CALL),
        ],
    )
    def test_parse_accepts_known_spellings(self, raw: str, expected: OptionType) -> None:
        assert OptionType.parse(raw) is expected

    @pytest.mark.parametrize("raw", ["x", "", "callput", "1", None])
    def test_parse_rejects_unknown(self, raw: object) -> None:
        with pytest.raises(ParameterError, match="not recognised"):
            OptionType.parse(raw)  # type: ignore[arg-type]

    def test_sign(self) -> None:
        assert OptionType.CALL.sign == 1
        assert OptionType.PUT.sign == -1

    def test_serialises_as_plain_string(self) -> None:
        """StrEnum keeps API responses as 'c'/'p' rather than 'OptionType.CALL'."""
        assert json.dumps({"t": OptionType.CALL}) == '{"t": "c"}'
        assert str(OptionType.PUT) == "p"
        assert OptionType.CALL == "c"


class TestBarrierKind:
    @pytest.mark.parametrize(
        ("kind", "is_up", "is_out"),
        [
            (BarrierKind.UP_AND_OUT, True, True),
            (BarrierKind.UP_AND_IN, True, False),
            (BarrierKind.DOWN_AND_OUT, False, True),
            (BarrierKind.DOWN_AND_IN, False, False),
        ],
    )
    def test_direction_and_knock_sense(self, kind: BarrierKind, is_up: bool, is_out: bool) -> None:
        assert kind.is_up is is_up
        assert kind.is_knock_out is is_out


class TestGreeks:
    def test_addition_aggregates_componentwise(self) -> None:
        a = Greeks(0.5, 0.001, 10.0, -1.2, 6.7)
        b = Greeks(-0.3, 0.002, 4.0, -0.5, 2.0)
        total = a + b
        assert total.delta == pytest.approx(0.2)
        assert total.gamma == pytest.approx(0.003)
        assert total.vega == pytest.approx(14.0)
        assert total.theta == pytest.approx(-1.7)
        assert total.rho == pytest.approx(8.7)

    def test_scaling_by_position_size(self) -> None:
        scaled = Greeks(0.5, 0.001, 10.0, -1.2, 6.7).scaled(100)
        assert scaled.delta == pytest.approx(50.0)
        assert scaled.vega == pytest.approx(1000.0)

    def test_short_position_flips_every_sign(self) -> None:
        g = Greeks(0.5, 0.001, 10.0, -1.2, 6.7)
        short = g.scaled(-1)
        assert short.delta == pytest.approx(-0.5)
        assert short.theta == pytest.approx(1.2)

    def test_is_immutable(self) -> None:
        g = Greeks(0.5, 0.001, 10.0, -1.2, 6.7)
        with pytest.raises((AttributeError, TypeError)):
            g.delta = 1.0  # type: ignore[misc]

    def test_as_dict_rounds_by_default_and_can_keep_precision(self) -> None:
        g = Greeks(1 / 3, 1 / 7, 1 / 11, -1 / 13, 1 / 17)
        assert g.as_dict()["delta"] == pytest.approx(0.333333, abs=1e-9)
        assert g.as_dict(None)["delta"] == pytest.approx(1 / 3, rel=1e-15)

    def test_addition_with_non_greeks_is_not_implemented(self) -> None:
        with pytest.raises(TypeError):
            _ = Greeks(0, 0, 0, 0, 0) + 1  # type: ignore[operator]


class TestDayCount:
    def test_thirty_360_gives_exact_quarter(self) -> None:
        """31 Jan to 30 Apr is exactly a quarter under 30/360 by construction."""
        assert year_fraction(
            date(2026, 1, 31), date(2026, 4, 30), DayCount.THIRTY_360
        ) == pytest.approx(0.25)

    def test_act_365f(self) -> None:
        assert year_fraction(
            date(2026, 1, 1), date(2027, 1, 1), DayCount.ACT_365F
        ) == pytest.approx(1.0)

    def test_act_360_exceeds_act_365(self) -> None:
        start, end = date(2026, 1, 1), date(2026, 7, 1)
        assert year_fraction(start, end, DayCount.ACT_360) > year_fraction(
            start, end, DayCount.ACT_365F
        )

    def test_act_act_handles_leap_year_exactly(self) -> None:
        """2024 is a leap year: a full year must still be exactly 1.0."""
        assert year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCount.ACT_ACT) == pytest.approx(
            1.0, abs=1e-15
        )

    def test_act_act_spanning_leap_and_non_leap(self) -> None:
        """Each calendar year contributes on its own basis, so this is ~2.0."""
        value = year_fraction(date(2024, 1, 1), date(2026, 1, 1), DayCount.ACT_ACT)
        assert value == pytest.approx(2.0, abs=1e-12)

    def test_act_365f_overstates_a_leap_year(self) -> None:
        """366 actual days over a 365 basis -- the convention's defining quirk."""
        value = year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCount.ACT_365F)
        assert value == pytest.approx(366 / 365)

    def test_reversed_dates_give_negative_fraction(self) -> None:
        """An expired trade, not an error -- callers decide what to do."""
        assert year_fraction(date(2026, 6, 1), date(2026, 1, 1)) < 0.0

    def test_enum_method_matches_function(self) -> None:
        start, end = date(2026, 3, 15), date(2027, 9, 30)
        for convention in DayCount:
            assert convention.year_fraction(start, end) == year_fraction(start, end, convention)


class TestParseTenor:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("3M", 0.25),
            ("1Y", 1.0),
            ("12M", 1.0),
            ("6m", 0.5),
            ("90D", 90 / 365),
            ("2W", 14 / 365),
            ("1y", 1.0),
            (" 3M ", 0.25),
            (0.25, 0.25),
            (1, 1.0),
        ],
    )
    def test_parses_known_forms(self, raw: str | float, expected: float) -> None:
        assert parse_tenor(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["3X", "M3", "", "three months", "3MM"])
    def test_rejects_unparseable(self, raw: str) -> None:
        with pytest.raises(ParameterError, match="cannot parse tenor"):
            parse_tenor(raw)

    def test_months_use_exact_twelfths_not_day_counts(self) -> None:
        """Observation schedules are built on month counts, so 1M is 1/12."""
        assert parse_tenor("1M") == pytest.approx(1 / 12)
        assert parse_tenor("1M") != pytest.approx(30 / 365)


class TestValidators:
    def test_check_positive(self) -> None:
        assert check_positive(1.5, "x") == 1.5
        for bad in (0.0, -1.0):
            with pytest.raises(ParameterError, match="greater than 0"):
                check_positive(bad, "x")

    def test_check_non_negative_allows_zero(self) -> None:
        assert check_non_negative(0.0, "x") == 0.0
        with pytest.raises(ParameterError):
            check_non_negative(-1e-9, "x")

    def test_check_in_range_boundaries(self) -> None:
        assert check_in_range(1.0, "x", 0.0, 1.0) == 1.0  # high inclusive by default
        with pytest.raises(ParameterError):
            check_in_range(0.0, "x", 0.0, 1.0)  # low exclusive by default
        assert check_in_range(0.0, "x", 0.0, 1.0, include_low=True) == 0.0

    def test_parameter_error_is_a_value_error(self) -> None:
        """Legacy callers catch ValueError; that must keep working."""
        assert issubclass(ParameterError, ValueError)
        assert issubclass(ParameterError, OtcPricerError)

    @pytest.mark.parametrize(
        "bad",
        [
            {"S": 0.0},
            {"S": -1.0},
            {"K": 0.0},
            {"T": 0.0},
            {"T": -1.0},
            {"sigma": 0.0},
            {"sigma": 6.0},
            {"q": -0.01},
        ],
    )
    def test_validate_vanilla_rejects(self, bad: dict[str, float]) -> None:
        args = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.02, "sigma": 0.2, "q": 0.0}
        args.update(bad)
        with pytest.raises(ParameterError):
            validate_vanilla(**args)

    def test_validate_vanilla_accepts_negative_rate(self) -> None:
        """Negative rates are real; a CNY desk also sees negative net carry."""
        validate_vanilla(S=100.0, K=100.0, T=1.0, r=-0.01, sigma=0.2, q=0.0)


class TestValidateSnowball:
    """The structural relations that make the product a snowball at all."""

    GOOD = {
        "S0": 5000.0,
        "T": 1.0,
        "r": 0.02,
        "sigma": 0.20,
        "q": 0.01,
        "ko_pct": 1.03,
        "ki_pct": 0.75,
        "coupon_pa": 0.20,
        "obs_freq": 12,
        "ki_obs_freq": 252,
        "lockup_months": 3,
    }

    def test_accepts_a_standard_structure(self) -> None:
        validate_snowball(**self.GOOD)

    def test_rejects_knock_in_above_knock_out(self) -> None:
        """Crossed barriers leave the payoff undefined."""
        with pytest.raises(ParameterError, match="strictly below"):
            validate_snowball(**{**self.GOOD, "ki_pct": 1.05})

    def test_rejects_equal_barriers(self) -> None:
        with pytest.raises(ParameterError, match="strictly below"):
            validate_snowball(**{**self.GOOD, "ki_pct": 1.03})

    def test_rejects_lockup_longer_than_tenor(self) -> None:
        """A lock-up swallowing the tenor silently turns it into a short put."""
        with pytest.raises(ParameterError, match="shorter than"):
            validate_snowball(**{**self.GOOD, "lockup_months": 18})

    def test_rejects_lockup_equal_to_tenor(self) -> None:
        with pytest.raises(ParameterError, match="shorter than"):
            validate_snowball(**{**self.GOOD, "lockup_months": 12})

    def test_accepts_zero_lockup(self) -> None:
        validate_snowball(**{**self.GOOD, "lockup_months": 0})

    @pytest.mark.parametrize("field", ["obs_freq", "ki_obs_freq"])
    def test_rejects_non_positive_observation_frequency(self, field: str) -> None:
        with pytest.raises(ParameterError, match=">= 1"):
            validate_snowball(**{**self.GOOD, field: 0})

    def test_rejects_negative_lockup(self) -> None:
        with pytest.raises(ParameterError, match=">= 0"):
            validate_snowball(**{**self.GOOD, "lockup_months": -1})


class TestKIStrikeMode:
    def test_both_conventions_available(self) -> None:
        """The legacy MC used KI and the legacy PDE used S0; both are now explicit."""
        assert KIStrikeMode.S0 == "s0"
        assert KIStrikeMode.KI == "ki"
        assert set(KIStrikeMode) == {KIStrikeMode.S0, KIStrikeMode.KI}


class TestSettings:
    def test_defaults_are_sane(self) -> None:
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.numerical.mc_paths >= 1000
        assert settings.numerical.rannacher_steps >= 0
        # Default must be offline so tests and CI never depend on the network.
        assert settings.data.source == "snapshot"

    def test_environment_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTCPRICER_MC_PATHS", "7777")
        monkeypatch.setenv("OTCPRICER_DATA_SOURCE", "akshare")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.numerical.mc_paths == 7777
        assert settings.data.source == "akshare"
        # Untouched fields keep their defaults.
        assert settings.numerical.mc_seed == 42
        get_settings.cache_clear()

    def test_settings_are_frozen(self) -> None:
        get_settings.cache_clear()
        settings = get_settings()
        with pytest.raises((TypeError, ValueError, AttributeError)):
            settings.numerical.mc_paths = 1  # type: ignore[misc]

    def test_invalid_override_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A path count below the floor must fail loudly, not silently clamp."""
        monkeypatch.setenv("OTCPRICER_MC_PATHS", "3")
        get_settings.cache_clear()
        with pytest.raises(ValidationError, match="mc_paths"):
            get_settings()
        get_settings.cache_clear()


class TestLogging:
    def test_json_format_emits_parseable_records(self, capsys: pytest.CaptureFixture) -> None:
        configure_logging("INFO", fmt="json")
        get_logger("test").info("priced", extra={"product": "snowball", "price": 0.0378})
        captured = capsys.readouterr().err.strip().splitlines()[-1]
        record = json.loads(captured)
        assert record["msg"] == "priced"
        assert record["level"] == "INFO"
        assert record["ctx"]["product"] == "snowball"
        assert record["ctx"]["price"] == pytest.approx(0.0378)

    def test_configure_is_idempotent(self, capsys: pytest.CaptureFixture) -> None:
        """The Flask reloader imports the app twice; lines must not double."""
        configure_logging("INFO", fmt="json")
        configure_logging("INFO", fmt="json")
        capsys.readouterr()
        get_logger("test").info("once")
        assert len(capsys.readouterr().err.strip().splitlines()) == 1

    def test_logger_namespacing(self) -> None:
        assert get_logger("mymod").name == "otcpricer.mymod"
        assert get_logger("otcpricer.models.bsm").name == "otcpricer.models.bsm"
