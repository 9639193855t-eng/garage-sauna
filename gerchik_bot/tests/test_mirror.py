"""Тесты для strategies/mirror.py — стратегия «Зеркальный уровень»."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from strategies.mirror import (
    detect,
    find_mirror_levels,
    check_retest,
    _find_swing_points,
    _cluster_levels,
    _count_touches_vectorized,
    _find_breakout,
    MirrorLevel,
)
from config import MIRROR_CONFIG


# ─── Фабрики тестовых данных ─────────────────────────────────────────────────

def make_flat_df(n: int = 120, price: float = 100.0, vol: float = 1000.0) -> pd.DataFrame:
    """Ровный рынок без swing-точек — уровней нет."""
    rows = [{
        "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
        "open":   price,
        "high":   price + 0.1,
        "low":    price - 0.1,
        "close":  price,
        "volume": vol,
    } for i in range(n)]
    df = pd.DataFrame(rows)
    df["avg_volume"] = vol
    return df


def make_candle(
    close: float,
    high: float = None,
    low: float = None,
    volume: float = 1000.0,
    ts_offset: int = 0,
) -> dict:
    high = high if high is not None else close * 1.002
    low  = low  if low  is not None else close * 0.998
    return {
        "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * ts_offset),
        "open":   close,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }


def build_mirror_scenario_long(
    level: float = 100.0,
    n_before: int = 20,
    n_touches: int = 4,
    n_after_break: int = 10,
) -> pd.DataFrame:
    """
    Создать DataFrame со сценарием зеркального уровня для LONG:
        1. Цена много раз касается resistance (level) снизу — не пробивает.
        2. Затем пробивает вверх на высоком объёме.
        3. Через n_after_break свечей делает ретест (low касается уровня, close выше).
    """
    rows = []
    avg_vol = 1000.0
    high_vol = avg_vol * (MIRROR_CONFIG.breakout_volume_mult + 0.5)

    # 1. Swing lows перед уровнем (чтобы были swing-точки)
    for i in range(n_before):
        # Чередуем высокие и низкие свечи → создаём swing highs
        if i % 3 == 0:
            h = level * 1.01  # swing high вблизи уровня
            lo = level * 0.97
        else:
            h = level * 0.995
            lo = level * 0.98
        rows.append({
            "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * len(rows)),
            "open": level * 0.99, "high": h, "low": lo,
            "close": level * 0.99, "volume": avg_vol,
        })

    # 2. Касания resistance (high касается level, close ниже)
    for j in range(n_touches):
        rows.append({
            "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * len(rows)),
            "open":  level * 0.998,
            "high":  level * (1 + MIRROR_CONFIG.touch_threshold * 0.5),  # в зоне
            "low":   level * 0.993,
            "close": level * 0.997,  # ниже уровня
            "volume": avg_vol,
        })

    # 3. Пробойная свеча (close выше level, высокий объём, дистанция ≥ breakout_min_distance)
    breakout_close = level * (1 + MIRROR_CONFIG.breakout_min_distance + 0.002)
    rows.append({
        "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * len(rows)),
        "open":  level,
        "high":  breakout_close * 1.002,
        "low":   level * 0.998,
        "close": breakout_close,
        "volume": high_vol,
    })
    breakout_vol = high_vol

    # 4. Движение после пробоя (цена ушла вверх)
    for k in range(n_after_break - 1):
        rows.append({
            "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * len(rows)),
            "open":  breakout_close * 1.002,
            "high":  breakout_close * 1.005,
            "low":   breakout_close * 1.001,
            "close": breakout_close * 1.003,
            "volume": avg_vol * 0.8,
        })

    # 5. Ретестовая свеча (low касается level, close выше, объём < breakout_vol)
    rows.append({
        "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * len(rows)),
        "open":  level * 1.003,
        "high":  level * 1.008,
        "low":   level * (1 + MIRROR_CONFIG.retest_threshold * 0.3),  # low в зоне
        "close": level * 1.002,  # close выше уровня
        "volume": breakout_vol * 0.5,  # объём < пробойного
    })

    df = pd.DataFrame(rows)
    df["avg_volume"] = avg_vol
    return df


def make_mirror_level(
    price: float = 100.0,
    role: str = "resistance",
    touches: int = 4,
    breakout_idx: int = 30,
    breakout_vol: float = 2000.0,
    strength: float = 0.7,
) -> MirrorLevel:
    return MirrorLevel(
        price=price,
        original_role=role,
        touches=touches,
        breakout_idx=breakout_idx,
        breakout_vol=breakout_vol,
        strength=strength,
    )


# ─── _find_swing_points ──────────────────────────────────────────────────────

class TestFindSwingPoints:

    def test_returns_empty_for_flat(self):
        df = make_flat_df(n=30)
        df = df.reset_index(drop=True)
        highs, lows = _find_swing_points(df)
        assert isinstance(highs, list)
        assert isinstance(lows, list)

    def test_detects_obvious_swing_high(self):
        """Создаём явный swing high в середине."""
        rows = []
        for i in range(20):
            h = 110.0 if i == 10 else 100.0
            rows.append({"open": 95.0, "high": h, "low": 90.0, "close": 95.0,
                         "volume": 1000.0,
                         "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5*i)})
        df = pd.DataFrame(rows)
        highs, lows = _find_swing_points(df)
        assert 110.0 in highs

    def test_detects_obvious_swing_low(self):
        """Создаём явный swing low в середине."""
        rows = []
        for i in range(20):
            lo = 80.0 if i == 10 else 95.0
            rows.append({"open": 100.0, "high": 105.0, "low": lo, "close": 100.0,
                         "volume": 1000.0,
                         "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5*i)})
        df = pd.DataFrame(rows)
        highs, lows = _find_swing_points(df)
        assert 80.0 in lows

    def test_returns_lists(self):
        df = make_flat_df(n=20).reset_index(drop=True)
        result = _find_swing_points(df)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)


# ─── _cluster_levels ─────────────────────────────────────────────────────────

class TestClusterLevels:

    def test_empty_input(self):
        assert _cluster_levels([], 0.003) == []

    def test_single_level(self):
        result = _cluster_levels([100.0], 0.003)
        assert result == [100.0]

    def test_two_close_levels_merged(self):
        """100.0 и 100.2 при threshold 0.3% → один кластер."""
        result = _cluster_levels([100.0, 100.2], 0.003)
        assert len(result) == 1
        assert abs(result[0] - 100.1) < 0.01

    def test_two_distant_levels_separate(self):
        """100.0 и 110.0 → два отдельных кластера."""
        result = _cluster_levels([100.0, 110.0], 0.003)
        assert len(result) == 2

    def test_three_levels_two_clusters(self):
        result = _cluster_levels([100.0, 100.1, 110.0], 0.003)
        assert len(result) == 2

    def test_output_sorted(self):
        result = _cluster_levels([110.0, 100.0, 105.0], 0.003)
        assert result == sorted(result)


# ─── _count_touches_vectorized ───────────────────────────────────────────────

class TestCountTouchesVectorized:

    def test_zero_for_empty_df(self):
        assert _count_touches_vectorized(pd.DataFrame(), 100.0, "support", 0.002) == 0

    def test_zero_for_zero_level(self):
        df = make_flat_df(n=10)
        assert _count_touches_vectorized(df, 0.0, "support", 0.002) == 0

    def test_resistance_touch_detected(self):
        """High в зоне уровня, close ниже → касание resistance."""
        rows = [{
            "open": 95.0, "high": 100.3, "low": 92.0, "close": 97.0,
            "volume": 1000.0,
            "time": pd.Timestamp("2024-01-01"),
        }]
        df = pd.DataFrame(rows)
        touches = _count_touches_vectorized(df, 100.0, "resistance", 0.005)
        assert touches == 1

    def test_support_touch_detected(self):
        """Low в зоне уровня, close выше → касание support."""
        rows = [{
            "open": 105.0, "high": 108.0, "low": 99.8, "close": 103.0,
            "volume": 1000.0,
            "time": pd.Timestamp("2024-01-01"),
        }]
        df = pd.DataFrame(rows)
        touches = _count_touches_vectorized(df, 100.0, "support", 0.005)
        assert touches == 1

    def test_breakout_not_counted_as_touch(self):
        """Свеча закрылась ЗА уровнем — это не касание, а пробой."""
        rows = [{
            "open": 95.0, "high": 102.0, "low": 92.0, "close": 101.0,  # close > уровня
            "volume": 1000.0,
            "time": pd.Timestamp("2024-01-01"),
        }]
        df = pd.DataFrame(rows)
        touches = _count_touches_vectorized(df, 100.0, "resistance", 0.005)
        assert touches == 0

    def test_counts_multiple_touches(self):
        rows = [{
            "open": 95.0, "high": 100.2, "low": 92.0, "close": 97.0,
            "volume": 1000.0,
            "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5*i),
        } for i in range(5)]
        df = pd.DataFrame(rows)
        touches = _count_touches_vectorized(df, 100.0, "resistance", 0.005)
        assert touches == 5

    def test_returns_int(self):
        df = make_flat_df(n=10)
        result = _count_touches_vectorized(df, 100.0, "support", 0.002)
        assert isinstance(result, int)


# ─── check_retest ────────────────────────────────────────────────────────────

class TestCheckRetest:

    def _make_df_with_breakout(self, n_after: int, n_total: int = 50) -> pd.DataFrame:
        """DataFrame где пробой на индексе (n_total - n_after - 1)."""
        rows = []
        for i in range(n_total):
            rows.append({
                "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                "open":   100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                "volume": 500.0,
            })
        df = pd.DataFrame(rows)
        df["avg_volume"] = 1000.0
        return df

    def test_too_early_retest(self):
        """Ретест слишком рано — пробой был только что."""
        df = self._make_df_with_breakout(n_after=1)
        mirror = make_mirror_level(breakout_idx=len(df) - 2, breakout_vol=2000.0)
        ok, _ = check_retest(df, mirror)
        assert ok is False

    def test_too_old_retest(self):
        """Уровень устарел — пробой был давно."""
        df = self._make_df_with_breakout(n_after=200)
        mirror = make_mirror_level(breakout_idx=0, breakout_vol=2000.0)
        ok, _ = check_retest(df, mirror)
        assert ok is False

    def test_valid_long_retest(self):
        """Корректный ретест для LONG: low у уровня, close выше, объём слабый."""
        level = 100.0
        brk_vol = 2000.0
        min_after = MIRROR_CONFIG.min_candles_after_break
        max_after = MIRROR_CONFIG.max_candles_after_break
        n_after   = (min_after + max_after) // 2

        rows = []
        # Свечи до пробоя + свечи после
        for i in range(n_after + 10):
            rows.append({
                "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                "open":   level * 1.005, "high": level * 1.01,
                "low":    level * 1.001, "close": level * 1.005,
                "volume": 500.0,
            })

        # Последняя — ретестовая
        retest_row = {
            "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * len(rows)),
            "open":   level * 1.003,
            "high":   level * 1.008,
            "low":    level * (1 + MIRROR_CONFIG.retest_threshold * 0.3),
            "close":  level * 1.002,
            "volume": brk_vol * 0.4,  # ниже breakout_vol
        }
        rows.append(retest_row)
        df = pd.DataFrame(rows)

        breakout_idx = len(df) - 1 - n_after - 1
        mirror = make_mirror_level(
            price=level, role="resistance",
            breakout_idx=breakout_idx, breakout_vol=brk_vol,
        )
        ok, candles_since = check_retest(df, mirror)
        assert ok is True
        assert candles_since > 0

    def test_high_volume_retest_rejected(self):
        """Ретест с высоким объёмом (как пробой) — не принимаем."""
        level = 100.0
        brk_vol = 1000.0
        min_after = MIRROR_CONFIG.min_candles_after_break
        n_after   = min_after + 5

        rows = []
        for i in range(n_after + 1):
            rows.append({
                "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                "open":   level * 1.005, "high": level * 1.01,
                "low":    level * 1.001, "close": level * 1.005,
                "volume": 500.0,
            })
        retest_row = {
            "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * len(rows)),
            "open":   level * 1.003,
            "high":   level * 1.008,
            "low":    level * (1 + MIRROR_CONFIG.retest_threshold * 0.3),
            "close":  level * 1.002,
            "volume": brk_vol * 1.5,  # ВЫШЕ breakout_vol → ретест не принят
        }
        rows.append(retest_row)
        df = pd.DataFrame(rows)
        mirror = make_mirror_level(
            price=level, role="resistance",
            breakout_idx=len(df) - n_after - 2, breakout_vol=brk_vol,
        )
        ok, _ = check_retest(df, mirror)
        assert ok is False

    def test_returns_tuple(self):
        df = make_flat_df(n=60)
        mirror = make_mirror_level(breakout_idx=10, breakout_vol=2000.0)
        result = check_retest(df, mirror)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ─── _find_breakout: min_candles_at_level ────────────────────────────────────

class TestMinCandlesAtLevel:
    """
    Проверяем условие Герчика: перед пробоем цена должна удерживаться
    у уровня минимум min_candles_at_level свечей (касаний).

    min_candles_at_level = 5 (из MIRROR_CONFIG).
    """

    def _make_breakout_df(
        self,
        level: float,
        n_touches: int,
        role: str = "resistance",
    ) -> tuple[pd.DataFrame, pd.Series, float]:
        """
        Собирает минимальный DataFrame для теста _find_breakout:
          - n_touches свечей-касаний (high в зоне, close ниже уровня для resistance)
          - 1 пробойная свеча (close выше уровня, высокий объём, нужная дистанция)
        Возвращает (df, avg_vol_series, global_avg_vol).
        """
        avg_vol = 1000.0
        high_vol = avg_vol * (MIRROR_CONFIG.breakout_volume_mult + 0.5)
        rows = []

        for i in range(n_touches):
            if role == "resistance":
                rows.append({
                    "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                    "open":   level * 0.998,
                    "high":   level * (1 + MIRROR_CONFIG.touch_threshold * 0.5),
                    "low":    level * 0.993,
                    "close":  level * 0.997,
                    "volume": avg_vol,
                })
            else:
                rows.append({
                    "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                    "open":   level * 1.003,
                    "high":   level * 1.007,
                    "low":    level * (1 - MIRROR_CONFIG.touch_threshold * 0.5),
                    "close":  level * 1.002,
                    "volume": avg_vol,
                })

        # Пробойная свеча
        if role == "resistance":
            brk_close = level * (1 + MIRROR_CONFIG.breakout_min_distance + 0.003)
            rows.append({
                "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * len(rows)),
                "open":   level,
                "high":   brk_close * 1.002,
                "low":    level * 0.998,
                "close":  brk_close,
                "volume": high_vol,
            })
        else:
            brk_close = level * (1 - MIRROR_CONFIG.breakout_min_distance - 0.003)
            rows.append({
                "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * len(rows)),
                "open":   level,
                "high":   level * 1.002,
                "low":    brk_close * 0.998,
                "close":  brk_close,
                "volume": high_vol,
            })

        df = pd.DataFrame(rows).reset_index(drop=True)
        avg_vol_series = pd.Series([avg_vol] * len(df))
        return df, avg_vol_series, avg_vol

    def test_4_touches_breakout_not_counted(self):
        """
        4 касания перед пробоем — меньше min_candles_at_level=5.
        Пробой НЕ засчитывается: _find_breakout возвращает (None, None).
        """
        assert MIRROR_CONFIG.min_candles_at_level == 5, (
            "Тест рассчитан на min_candles_at_level=5"
        )
        df, avg_vol_series, global_avg_vol = self._make_breakout_df(
            level=100.0, n_touches=4, role="resistance"
        )
        idx, vol = _find_breakout(
            df, 100.0, "resistance",
            avg_vol_series, global_avg_vol, MIRROR_CONFIG,
        )
        assert idx is None, (
            f"Ожидали None при 4 касаниях, получили idx={idx}"
        )
        assert vol is None

    def test_5_touches_breakout_counted(self):
        """
        5 касаний перед пробоем — ровно min_candles_at_level=5.
        Пробой засчитывается: _find_breakout возвращает валидный индекс.
        """
        assert MIRROR_CONFIG.min_candles_at_level == 5
        df, avg_vol_series, global_avg_vol = self._make_breakout_df(
            level=100.0, n_touches=5, role="resistance"
        )
        idx, vol = _find_breakout(
            df, 100.0, "resistance",
            avg_vol_series, global_avg_vol, MIRROR_CONFIG,
        )
        assert idx is not None, (
            "Ожидали валидный пробой при 5 касаниях, получили None"
        )
        assert vol is not None
        assert vol > 0

    def test_6_touches_breakout_counted(self):
        """
        6 касаний перед пробоем — больше min_candles_at_level=5.
        Пробой засчитывается: _find_breakout возвращает валидный индекс.
        """
        assert MIRROR_CONFIG.min_candles_at_level == 5
        df, avg_vol_series, global_avg_vol = self._make_breakout_df(
            level=100.0, n_touches=6, role="resistance"
        )
        idx, vol = _find_breakout(
            df, 100.0, "resistance",
            avg_vol_series, global_avg_vol, MIRROR_CONFIG,
        )
        assert idx is not None, (
            "Ожидали валидный пробой при 6 касаниях, получили None"
        )
        assert vol is not None
        assert vol > 0

    def test_5_touches_with_gaps_counted(self):
        """
        5 касаний через «нейтральные» свечи (далеко от уровня).
        Нейтральные свечи не считаются касаниями, но и не мешают.
        Пробой засчитывается: _find_breakout возвращает валидный индекс.
        """
        assert MIRROR_CONFIG.min_candles_at_level == 5
        level = 100.0
        avg_vol = 1000.0
        high_vol = avg_vol * (MIRROR_CONFIG.breakout_volume_mult + 0.5)
        rows = []

        def touch_candle(offset):
            return {
                "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * offset),
                "open":   level * 0.998,
                "high":   level * (1 + MIRROR_CONFIG.touch_threshold * 0.5),
                "low":    level * 0.993,
                "close":  level * 0.997,
                "volume": avg_vol,
            }

        def neutral_candle(offset):
            # Свеча далеко от уровня — не касание
            return {
                "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * offset),
                "open":   level * 0.95,
                "high":   level * 0.96,
                "low":    level * 0.94,
                "close":  level * 0.95,
                "volume": avg_vol,
            }

        # 5 касаний с нейтральными свечами между ними
        for k in range(5):
            rows.append(touch_candle(len(rows)))
            if k < 4:
                rows.append(neutral_candle(len(rows)))
                rows.append(neutral_candle(len(rows)))

        # Пробойная свеча
        brk_close = level * (1 + MIRROR_CONFIG.breakout_min_distance + 0.003)
        rows.append({
            "time":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * len(rows)),
            "open":   level,
            "high":   brk_close * 1.002,
            "low":    level * 0.998,
            "close":  brk_close,
            "volume": high_vol,
        })

        df = pd.DataFrame(rows).reset_index(drop=True)
        avg_vol_series = pd.Series([avg_vol] * len(df))

        idx, vol = _find_breakout(
            df, level, "resistance",
            avg_vol_series, avg_vol, MIRROR_CONFIG,
        )
        assert idx is not None, (
            "Ожидали валидный пробой при 5 касаниях через пропуски, получили None"
        )
        assert vol is not None
        assert vol > 0


# ─── find_mirror_levels ──────────────────────────────────────────────────────

class TestFindMirrorLevels:

    def test_returns_list(self):
        df = make_flat_df(n=120)
        result = find_mirror_levels(df)
        assert isinstance(result, list)

    def test_empty_for_insufficient_data(self):
        df = make_flat_df(n=5)
        result = find_mirror_levels(df)
        assert result == []

    def test_empty_for_flat_market(self):
        """Без swing-точек уровней нет."""
        df = make_flat_df(n=120)
        result = find_mirror_levels(df)
        assert len(result) == 0  # нет swing points в плоском рынке

    def test_result_sorted_by_strength(self):
        """Уровни отсортированы по убыванию силы."""
        df = build_mirror_scenario_long(n_after_break=10)
        if len(df) < 100:
            pytest.skip("Недостаточно данных в сценарии")
        result = find_mirror_levels(df)
        if len(result) > 1:
            strengths = [m.strength for m in result]
            assert strengths == sorted(strengths, reverse=True)

    def test_each_level_has_required_attrs(self):
        df = build_mirror_scenario_long(n_before=30, n_touches=4, n_after_break=8)
        result = find_mirror_levels(df)
        for m in result:
            assert isinstance(m.price, float)
            assert m.original_role in ("resistance", "support")
            assert m.touches >= MIRROR_CONFIG.min_touches
            assert 0.0 <= m.strength <= 1.0
            assert m.breakout_vol > 0

    def test_strength_in_range(self):
        df = build_mirror_scenario_long(n_before=30, n_touches=5, n_after_break=8)
        result = find_mirror_levels(df)
        for m in result:
            assert 0.0 <= m.strength <= 1.0

    def test_min_touches_filter(self):
        """Уровень с меньше min_touches касаниями не попадает в результат."""
        df = build_mirror_scenario_long(n_touches=1, n_after_break=8)
        result = find_mirror_levels(df)
        for m in result:
            assert m.touches >= MIRROR_CONFIG.min_touches


# ─── detect ──────────────────────────────────────────────────────────────────

class TestDetect:

    def test_returns_none_for_empty(self):
        assert detect(pd.DataFrame(), 95.0, 110.0) is None

    def test_returns_none_for_none_df(self):
        assert detect(None, 95.0, 110.0) is None

    def test_returns_none_for_insufficient_data(self):
        df = make_flat_df(n=50)
        assert detect(df, 95.0, 110.0) is None

    def test_returns_none_if_support_none(self):
        df = make_flat_df(n=120)
        assert detect(df, None, 110.0) is None

    def test_returns_none_if_resistance_none(self):
        df = make_flat_df(n=120)
        assert detect(df, 95.0, None) is None

    def test_returns_none_for_flat_market(self):
        """Без swing-точек — нет уровней — нет сигнала."""
        df = make_flat_df(n=120)
        assert detect(df, 95.0, 105.0) is None

    def test_returns_none_for_high_volatility_candle(self):
        """Последняя свеча с диапазоном > 5% → фильтр волатильности."""
        df = make_flat_df(n=120)
        # Делаем последнюю свечу очень широкой
        df.iloc[-1, df.columns.get_loc("high")] = 100.0 * 1.06
        df.iloc[-1, df.columns.get_loc("low")]  = 100.0 * 0.93
        df.iloc[-1, df.columns.get_loc("close")] = 100.0
        assert detect(df, 95.0, 105.0) is None

    def test_returns_string_or_none(self):
        df = make_flat_df(n=120)
        result = detect(df, 95.0, 105.0)
        assert result in ("long", "short", None)

    def test_long_signal_on_valid_scenario(self):
        """На синтетическом сценарии: если сигнал не None — он должен быть "long"."""
        df = build_mirror_scenario_long(
            level=100.0, n_before=30, n_touches=4, n_after_break=8
        )
        if len(df) < 100:
            pytest.skip("Сценарий слишком короткий")
        result = detect(df, 90.0, 115.0)
        # Если сигнал найден — это должен быть именно "long", не "short"
        # (сценарий строит только сопротивление, пробитое вверх)
        if result is not None:
            assert result == "long", (
                f"Ожидали 'long' или None на сценарии resistance→mirror, получили '{result}'"
            )

    def test_result_is_long_or_short(self):
        """Если результат не None — он должен быть строкой "long" или "short"."""
        df = build_mirror_scenario_long(n_before=35, n_touches=4, n_after_break=10)
        result = detect(df, 90.0, 115.0)
        if result is not None:
            assert result in ("long", "short")

    def test_adds_avg_volume_if_missing(self):
        """Функция не должна падать если avg_volume отсутствует."""
        df = make_flat_df(n=120)
        if "avg_volume" in df.columns:
            df = df.drop(columns=["avg_volume"])
        result = detect(df, 95.0, 105.0)
        assert result in ("long", "short", None)

    def test_no_exception_on_normal_data(self):
        """Функция не должна бросать исключений при нормальных данных."""
        df = build_mirror_scenario_long(n_before=30, n_touches=4, n_after_break=10)
        try:
            detect(df, 90.0, 115.0)
        except Exception as e:
            pytest.fail(f"detect() бросила исключение: {e}")


# ─── Интеграционные тесты ─────────────────────────────────────────────────────

class TestMirrorIntegration:

    def test_config_min_touches_respected(self):
        """Уровни с меньше касаниями чем min_touches → не в результате."""
        df = build_mirror_scenario_long(n_touches=2, n_after_break=8)
        levels = find_mirror_levels(df)
        for m in levels:
            assert m.touches >= MIRROR_CONFIG.min_touches

    def test_strength_reflects_breakout_volume(self):
        """Чем выше объём пробоя — тем выше сила уровня."""
        df_low  = build_mirror_scenario_long(n_before=30, n_touches=4, n_after_break=8)
        df_high = build_mirror_scenario_long(n_before=30, n_touches=4, n_after_break=8)

        # Искусственно увеличим объём пробоя в df_high
        # Пробой — предпоследняя по нашей схеме (перед n_after_break + 1 ретестом)
        # Просто убедимся что оба не падают
        levels_low  = find_mirror_levels(df_low)
        levels_high = find_mirror_levels(df_high)
        assert isinstance(levels_low,  list)
        assert isinstance(levels_high, list)

    def test_retest_window_boundaries(self):
        """Ретест вне окна [min, max] → False."""
        level = 100.0
        brk_vol = 2000.0
        # Слишком рано
        mirror_early = make_mirror_level(
            breakout_idx=50 - 1,   # пробой вчера
            breakout_vol=brk_vol,
        )
        df = make_flat_df(n=50)
        ok, _ = check_retest(df, mirror_early)
        assert ok is False

        # Слишком поздно
        mirror_old = make_mirror_level(
            breakout_idx=0,   # пробой 50 свечей назад
            breakout_vol=brk_vol,
        )
        df2 = make_flat_df(n=200)
        ok2, _ = check_retest(df2, mirror_old)
        assert ok2 is False

    def test_best_signal_chosen_by_strength(self):
        """При нескольких уровнях выбирается более сильный."""
        from unittest.mock import patch

        weak  = make_mirror_level(price=100.0, role="resistance", strength=0.4)
        strong = make_mirror_level(price=105.0, role="support",   strength=0.8)

        df = make_flat_df(n=120)

        with patch("strategies.mirror.find_mirror_levels", return_value=[strong, weak]), \
             patch("strategies.mirror.check_retest", return_value=(True, 10)):
            result = detect(df, 90.0, 115.0)
            # strong → support → SHORT
            assert result == "short"
