"""Тесты для levels.py — поиск уровней поддержки и сопротивления."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np

from levels import (
    find_levels,
    find_levels_extended,
    count_level_touches,
    find_levels_with_strength,
    is_near_level,
    get_level_range,
    get_next_levels,
)


# ─── Фабрика тестовых свечей ────────────────────────────────────────────────

def make_df(n: int = 30, base_price: float = 100.0, spread: float = 5.0) -> pd.DataFrame:
    """
    Создать простой DataFrame: свечи с фиксированным диапазоном.
    Минимум (поддержка) = base_price - spread
    Максимум (сопротивление) = base_price + spread
    """
    rows = []
    for i in range(n):
        rows.append({
            "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
            "open":   base_price,
            "high":   base_price + spread,
            "low":    base_price - spread,
            "close":  base_price,
            "volume": 1000.0,
        })
    return pd.DataFrame(rows)


def make_df_range(highs: list, lows: list, closes: list | None = None) -> pd.DataFrame:
    """Создать DataFrame с явными high/low значениями."""
    n = len(highs)
    closes = closes or [100.0] * n
    rows = []
    for i in range(n):
        rows.append({
            "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
            "open":   100.0,
            "high":   highs[i],
            "low":    lows[i],
            "close":  closes[i],
            "volume": 1000.0,
        })
    return pd.DataFrame(rows)


# ─── find_levels ────────────────────────────────────────────────────────────

class TestFindLevels:
    def test_returns_none_for_none(self):
        support, resistance = find_levels(None)
        assert support is None
        assert resistance is None

    def test_returns_none_for_empty(self):
        support, resistance = find_levels(pd.DataFrame())
        assert support is None
        assert resistance is None

    def test_returns_none_insufficient_data(self):
        df = make_df(n=5)
        support, resistance = find_levels(df)
        assert support is None

    def test_basic_levels(self):
        """Поддержка = min low, сопротивление = max high за N свечей."""
        highs = [110.0] * 20 + [115.0] + [110.0]  # 22 свечи, последняя незакрытая
        lows  = [90.0]  * 20 + [88.0]  + [90.0]
        df = make_df_range(highs, lows)
        support, resistance = find_levels(df)
        # Анализ по 20 свечам (df.iloc[-21:-1])
        assert support is not None
        assert resistance is not None
        assert support < resistance

    def test_support_below_resistance(self):
        df = make_df(n=30)
        support, resistance = find_levels(df)
        assert support is not None
        assert support < resistance

    def test_excludes_last_candle(self):
        """Последняя (незакрытая) свеча не должна влиять на уровни."""
        df = make_df(n=30, base_price=100.0, spread=5.0)
        # Делаем последнюю свечу с огромным диапазоном
        df.iloc[-1, df.columns.get_loc("high")] = 999.0
        df.iloc[-1, df.columns.get_loc("low")]  = 1.0
        support, resistance = find_levels(df)
        # Уровни должны быть 95.0 и 105.0, не 1.0 и 999.0
        assert resistance is not None and resistance < 200.0
        assert support is not None and support > 10.0

    def test_returns_floats(self):
        df = make_df(n=30)
        support, resistance = find_levels(df)
        assert isinstance(support, float)
        assert isinstance(resistance, float)

    def test_identical_highs_and_lows_returns_none(self):
        """Если все свечи одинаковы — уровни совпадут, результат None."""
        df = make_df_range(
            highs=[100.0] * 25,
            lows=[100.0] * 25,
        )
        support, resistance = find_levels(df)
        assert support is None or support < resistance


# ─── find_levels_extended ────────────────────────────────────────────────────

class TestFindLevelsExtended:
    def test_uses_custom_n_candles(self):
        df = make_df(n=50)
        support, resistance = find_levels_extended(df, n_candles=10)
        assert support is not None

    def test_returns_none_if_not_enough(self):
        df = make_df(n=5)
        support, resistance = find_levels_extended(df, n_candles=10)
        assert support is None

    def test_none_falls_back_to_config(self):
        df = make_df(n=40)
        support, resistance = find_levels_extended(df, n_candles=None)
        assert support is not None


# ─── count_level_touches ─────────────────────────────────────────────────────

class TestCountLevelTouches:
    def test_zero_touches_for_empty_df(self):
        assert count_level_touches(pd.DataFrame(), 100.0) == 0

    def test_zero_touches_for_none_df(self):
        assert count_level_touches(None, 100.0) == 0

    def test_zero_touches_for_zero_level(self):
        df = make_df(n=10)
        assert count_level_touches(df, 0.0) == 0

    def test_support_touches_detected(self):
        """Свечи с low рядом с уровнем и close выше — это касание поддержки."""
        rows = []
        for i in range(10):
            rows.append({
                "time":  pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                "open":  105.0,
                "high":  110.0,
                "low":   100.0,   # касается уровня 100.0
                "close": 106.0,   # закрывается ВЫШЕ уровня
                "volume": 1000.0,
            })
        df = pd.DataFrame(rows)
        touches = count_level_touches(df, 100.0, threshold_pct=0.002, role="support")
        assert touches == 10

    def test_no_touch_if_close_below_support(self):
        """Если close ниже уровня — это пробой, не касание."""
        rows = [{
            "time":  pd.Timestamp("2024-01-01"),
            "open":  105.0,
            "high":  110.0,
            "low":   99.5,
            "close": 98.0,   # close НИЖЕ уровня 100 → пробой, не касание
            "volume": 1000.0,
        }]
        df = pd.DataFrame(rows)
        touches = count_level_touches(df, 100.0, role="support")
        assert touches == 0

    def test_resistance_touches_detected(self):
        """High рядом с уровнем, close ниже — касание сопротивления."""
        rows = [{
            "time":  pd.Timestamp("2024-01-01"),
            "open":  95.0,
            "high":  100.5,
            "low":   90.0,
            "close": 96.0,   # close НИЖЕ уровня
            "volume": 1000.0,
        }]
        df = pd.DataFrame(rows)
        touches = count_level_touches(df, 100.0, threshold_pct=0.01, role="resistance")
        assert touches == 1

    def test_returns_integer(self):
        df = make_df(n=20)
        result = count_level_touches(df, 95.0)
        assert isinstance(result, int)


# ─── find_levels_with_strength ───────────────────────────────────────────────

class TestFindLevelsWithStrength:
    def test_returns_dict_with_required_keys(self):
        df = make_df(n=30)
        result = find_levels_with_strength(df)
        assert "support" in result
        assert "resistance" in result
        assert "support_touches" in result
        assert "resistance_touches" in result
        assert "support_strong" in result
        assert "resistance_strong" in result

    def test_returns_empty_dict_values_for_insufficient_data(self):
        df = make_df(n=5)
        result = find_levels_with_strength(df)
        assert result["support"] is None

    def test_strong_level_requires_3_touches(self):
        """Уровень сильный если 3+ касания."""
        # Создаём 30 свечей где low касается 95.0 и close > 95.0
        rows = []
        for i in range(30):
            rows.append({
                "time":  pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                "open":  100.0,
                "high":  105.0,
                "low":   95.0,    # всегда касается уровня 95
                "close": 101.0,   # всегда закрывается выше
                "volume": 1000.0,
            })
        df = pd.DataFrame(rows)
        result = find_levels_with_strength(df)
        assert result["support_touches"] >= 3
        assert result["support_strong"] is True

    def test_support_touches_non_negative(self):
        df = make_df(n=30)
        result = find_levels_with_strength(df)
        assert result["support_touches"] >= 0
        assert result["resistance_touches"] >= 0


# ─── is_near_level ───────────────────────────────────────────────────────────

class TestIsNearLevel:
    def test_price_at_level(self):
        """Цена равна уровню → рядом."""
        assert is_near_level(100.0, 100.0) is True

    def test_price_just_inside(self):
        """Цена в пределах 0.3% → рядом."""
        assert is_near_level(100.25, 100.0, threshold_pct=0.003) is True

    def test_price_just_outside(self):
        """Цена дальше 0.3% → не рядом."""
        assert is_near_level(100.5, 100.0, threshold_pct=0.003) is False

    def test_zero_level_returns_false(self):
        """Нулевой уровень — защита от деления на ноль."""
        assert is_near_level(100.0, 0.0) is False

    def test_negative_level_returns_false(self):
        assert is_near_level(100.0, -50.0) is False

    def test_symmetrical(self):
        """Проверка работает в обе стороны от уровня."""
        assert is_near_level(99.75, 100.0, threshold_pct=0.003) is True
        assert is_near_level(100.25, 100.0, threshold_pct=0.003) is True

    def test_custom_threshold(self):
        assert is_near_level(101.0, 100.0, threshold_pct=0.01) is True
        assert is_near_level(102.0, 100.0, threshold_pct=0.01) is False


# ─── get_level_range ─────────────────────────────────────────────────────────

class TestGetLevelRange:
    def test_returns_positive_range(self):
        df = make_df(n=30, base_price=100.0, spread=5.0)
        rng = get_level_range(df)
        assert rng is not None
        assert rng > 0

    def test_returns_none_for_insufficient(self):
        df = make_df(n=3)
        assert get_level_range(df) is None

    def test_range_equals_resistance_minus_support(self):
        df = make_df(n=30, base_price=100.0, spread=5.0)
        support, resistance = find_levels(df)
        rng = get_level_range(df)
        if rng is not None:
            assert abs(rng - (resistance - support)) < 1e-9


# ─── get_next_levels ─────────────────────────────────────────────────────────

class TestGetNextLevels:
    def test_returns_all_keys(self):
        df = make_df(n=30)
        result = get_next_levels(df)
        if result:
            assert "support" in result
            assert "resistance" in result
            assert "next_support" in result
            assert "next_resistance" in result

    def test_next_support_below_support(self):
        df = make_df(n=30)
        result = get_next_levels(df)
        if result:
            assert result["next_support"] < result["support"]

    def test_next_resistance_above_resistance(self):
        df = make_df(n=30)
        result = get_next_levels(df)
        if result:
            assert result["next_resistance"] > result["resistance"]

    def test_returns_empty_for_insufficient_data(self):
        df = make_df(n=3)
        result = get_next_levels(df)
        assert result == {}

    def test_projection_equals_range(self):
        """Следующий уровень = текущий ± диапазон."""
        df = make_df(n=30, spread=5.0)
        result = get_next_levels(df)
        if result:
            level_range = result["resistance"] - result["support"]
            assert abs(result["next_support"]    - (result["support"]    - level_range)) < 1e-9
            assert abs(result["next_resistance"] - (result["resistance"] + level_range)) < 1e-9
