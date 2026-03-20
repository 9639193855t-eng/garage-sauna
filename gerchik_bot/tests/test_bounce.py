"""Тесты для strategies/bounce.py — стратегия «Отбой от уровня» (пин-бар). Приоритет 4."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
from unittest.mock import patch

from strategies.bounce import detect
from config import BOUNCE_CONFIG


# ─── Константы ───────────────────────────────────────────────────────────────

SUPPORT    = 100.0
RESISTANCE = 110.0
AVG_VOL    = 1000.0

# volume_threshold = 1.5 (из конфига)
GOOD_VOL = AVG_VOL * BOUNCE_CONFIG.volume_threshold          # 1500 — ровно на пороге
HIGH_VOL = AVG_VOL * (BOUNCE_CONFIG.volume_threshold + 0.2)  # 1700 — точно выше
LOW_VOL  = AVG_VOL * (BOUNCE_CONFIG.volume_threshold - 0.2)  # 1300 — точно ниже


# ─── Фабрики свечей ──────────────────────────────────────────────────────────

def _make_prefix(n: int, avg_vol: float) -> list:
    """Нейтральные свечи (без нулевого диапазона)."""
    rows = []
    for i in range(n):
        rows.append({
            "time":       pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
            "open":       105.0,
            "high":       106.0,
            "low":        104.0,
            "close":      105.0,
            "volume":     avg_vol,
            "avg_volume": avg_vol,
        })
    return rows


def make_df_bounce_long(
    signal_close: float = 101.2,   # close > support=100, dist=1.2% ≤ 1.5% ✓
    signal_open:  float = 101.0,   # open близко к close → маленькое тело
    signal_low:   float = 99.7,    # 0.3% → касание support=100 ✓
    signal_high:  float = 101.5,   # верх
    signal_vol:   float = None,
    avg_vol:      float = AVG_VOL,
    n_prefix:     int   = 5,
) -> pd.DataFrame:
    """
    Собрать DataFrame для теста LONG пин-бара (молоток у support=100).

    Сигнальная свеча по умолчанию:
        low=99.7, high=101.5, open=101.0, close=101.2
        range = 101.5 - 99.7 = 1.8
        lower_wick = min(101.0, 101.2) - 99.7 = 1.3  → ratio = 1.3/1.8 = 0.722 > 0.6 ✓
        body = |101.2 - 101.0| = 0.2                  → ratio = 0.2/1.8 = 0.111 < 0.3 ✓
        mid = (101.5 + 99.7)/2 = 100.6 → close=101.2 > 100.6 ✓
        low_touch = |99.7 - 100| / 100 = 0.003 ≤ 0.003 ✓
        close_near = |101.2 - 100| / 100 = 0.012 ≤ 0.015 ✓
    """
    if signal_vol is None:
        signal_vol = HIGH_VOL

    rows = _make_prefix(n_prefix, avg_vol)
    t_signal = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * n_prefix)
    rows.append({
        "time":       t_signal,
        "open":       signal_open,
        "high":       signal_high,
        "low":        signal_low,
        "close":      signal_close,
        "volume":     signal_vol,
        "avg_volume": avg_vol,
    })
    return pd.DataFrame(rows)


def make_df_bounce_short(
    signal_close: float = 109.1,   # close < resistance=110, dist=0.82% ≤ 1.5% ✓
    signal_open:  float = 109.3,   # open близко к close → маленькое тело
    signal_low:   float = 108.8,   # нижний конец
    signal_high:  float = 110.3,   # 0.27% → касание resistance=110 ✓
    signal_vol:   float = None,
    avg_vol:      float = AVG_VOL,
    n_prefix:     int   = 5,
) -> pd.DataFrame:
    """
    Собрать DataFrame для теста SHORT пин-бара (висельник у resistance=110).

    Сигнальная свеча по умолчанию:
        high=110.3, low=108.8, open=109.3, close=109.1
        range = 110.3 - 108.8 = 1.5
        upper_wick = 110.3 - max(109.3, 109.1) = 1.0  → ratio = 1.0/1.5 = 0.667 > 0.6 ✓
        body = |109.3 - 109.1| = 0.2                   → ratio = 0.2/1.5 = 0.133 < 0.3 ✓
        mid = (110.3 + 108.8)/2 = 109.55 → close=109.1 < 109.55 ✓
        high_touch = |110.3 - 110| / 110 = 0.00273 ≤ 0.003 ✓
        close_near = |109.1 - 110| / 110 = 0.0082 ≤ 0.015 ✓
    """
    if signal_vol is None:
        signal_vol = HIGH_VOL

    rows = _make_prefix(n_prefix, avg_vol)
    t_signal = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * n_prefix)
    rows.append({
        "time":       t_signal,
        "open":       signal_open,
        "high":       signal_high,
        "low":        signal_low,
        "close":      signal_close,
        "volume":     signal_vol,
        "avg_volume": avg_vol,
    })
    return pd.DataFrame(rows)


# ─── TestEdgeCases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Защитные проверки в начале detect()."""

    def test_none_df_returns_none(self):
        assert detect(None, SUPPORT, RESISTANCE) is None

    def test_empty_df_returns_none(self):
        assert detect(pd.DataFrame(), SUPPORT, RESISTANCE) is None

    def test_single_row_df_returns_none(self):
        df = make_df_bounce_long(n_prefix=0)
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_support_none_returns_none(self):
        df = make_df_bounce_long()
        assert detect(df, None, RESISTANCE) is None

    def test_resistance_none_returns_none(self):
        df = make_df_bounce_long()
        assert detect(df, SUPPORT, None) is None

    def test_support_zero_returns_none(self):
        df = make_df_bounce_long()
        assert detect(df, 0.0, RESISTANCE) is None

    def test_resistance_zero_returns_none(self):
        df = make_df_bounce_long()
        assert detect(df, SUPPORT, 0.0) is None

    def test_resistance_equal_support_returns_none(self):
        df = make_df_bounce_long()
        assert detect(df, 100.0, 100.0) is None

    def test_resistance_less_than_support_returns_none(self):
        df = make_df_bounce_long()
        assert detect(df, 110.0, 100.0) is None

    def test_avg_volume_missing_returns_none(self):
        df = make_df_bounce_long()
        df = df.drop(columns=["avg_volume"])
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_avg_volume_nan_returns_none(self):
        df = make_df_bounce_long()
        df.iloc[-1, df.columns.get_loc("avg_volume")] = float("nan")
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_avg_volume_zero_returns_none(self):
        df = make_df_bounce_long()
        df.iloc[-1, df.columns.get_loc("avg_volume")] = 0.0
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_zero_candle_range_returns_none(self):
        """Свеча high == low → нулевой range."""
        rows = _make_prefix(5, AVG_VOL)
        rows.append({
            "time": pd.Timestamp("2024-01-01T01:00"),
            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "volume": HIGH_VOL, "avg_volume": AVG_VOL,
        })
        df = pd.DataFrame(rows)
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_zero_volume_returns_none(self):
        """Нулевой объём ниже порога volume_threshold → нет сигнала."""
        df = make_df_bounce_long(signal_vol=0.0)
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_result_is_long_or_short_or_none(self):
        df = make_df_bounce_long()
        result = detect(df, SUPPORT, RESISTANCE)
        assert result in ("long", "short", None)


# ─── TestDetectLong ───────────────────────────────────────────────────────────

class TestDetectLong:
    """Основной сигнал LONG и нарушение каждого условия."""

    def test_basic_long_signal(self):
        """Все условия выполнены → сигнал "long"."""
        df = make_df_bounce_long()
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_long_needs_low_touching_support(self):
        """Low НЕ касается support (слишком далеко) → нет сигнала."""
        # low=97.0 → |97-100|/100=3% > proximity=0.3%
        df = make_df_bounce_long(
            signal_low=97.0, signal_open=101.0, signal_close=101.2, signal_high=101.5
        )
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_long_needs_close_above_support(self):
        """Close НИЖЕ support → пробой вниз → нет сигнала."""
        df = make_df_bounce_long(
            signal_low=99.7, signal_open=99.9, signal_close=99.8, signal_high=100.5
        )
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_long_needs_close_in_upper_half(self):
        """Close в нижней половине диапазона → тело направлено вниз → нет LONG.

        low=99.7, high=101.5 → mid=100.6
        close=100.4 < 100.6 → нижняя половина → нет сигнала.
        """
        df = make_df_bounce_long(
            signal_low=99.7, signal_open=100.5, signal_close=100.4,
            signal_high=101.5
        )
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_long_no_signal_if_body_too_large(self):
        """Тело >= pin_bar_body_ratio=0.3 → НЕ пин-бар → нет сигнала.

        low=99.7, high=102.0 → range=2.3
        open=100.2, close=101.0 → body=0.8 → ratio=0.348 > 0.3 → тело велико.
        """
        df = make_df_bounce_long(
            signal_low=99.7, signal_open=100.2, signal_close=101.0,
            signal_high=102.0
        )
        # body=0.8, range=2.3, body_ratio=0.348 >= 0.3 → нет сигнала
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_long_no_signal_if_wick_too_small(self):
        """Нижний хвост <= pin_bar_wick_ratio=0.6 → хвост короткий → нет сигнала.

        low=99.7, high=102.0, open=100.0, close=101.5 → range=2.3
        lower_wick = min(100.0,101.5) - 99.7 = 100.0 - 99.7 = 0.3
        lower_wick_ratio = 0.3/2.3 = 0.13 < 0.6 → хвост слишком короткий.
        body = 1.5 → слишком велико, но хвост — основное нарушение.
        """
        df = make_df_bounce_long(
            signal_low=99.7, signal_open=100.0, signal_close=101.5,
            signal_high=102.0
        )
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_long_needs_sufficient_volume(self):
        """Объём ниже порога (volume_threshold * avg_volume) → нет сигнала."""
        df = make_df_bounce_long(signal_vol=LOW_VOL)
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_long_needs_close_near_support(self):
        """Close слишком далеко от support (> proximity * 5 = 1.5%) → нет сигнала."""
        # support=100, max_close_dist=0.015 → close должен быть <= 101.5
        # close=102.0 → |102-100|/100=0.02 > 0.015 → нет сигнала
        # Нужно сохранить молоток-форму: маленькое тело, длинный хвост вниз
        # low=99.7 (касание), high=103.0, open=101.9, close=102.0
        # range=3.3, lower_wick=101.9-99.7=2.2 → ratio=2.2/3.3=0.667>0.6 ✓
        # body=0.1 → ratio=0.1/3.3=0.030<0.3 ✓
        # mid=(103+99.7)/2=101.35 → close=102>101.35 ✓
        # close_near: |102-100|/100=0.02 > 0.015 → нет сигнала ✓
        df = make_df_bounce_long(
            signal_low=99.7, signal_open=101.9, signal_close=102.0,
            signal_high=103.0
        )
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_long_with_minimum_2_rows(self):
        """Минимально допустимый DataFrame: 2 строки."""
        df = make_df_bounce_long(n_prefix=1)
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_long_at_exact_proximity_boundary(self):
        """Low точно на границе proximity_threshold (0.3%) → касание засчитывается."""
        # proximity=0.003, support=100 → граница: low=100*(1-0.003)=99.7
        df = make_df_bounce_long(signal_low=99.7)
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_long_low_just_outside_proximity(self):
        """Low чуть за границей proximity → касания нет → нет сигнала."""
        # low=99.69 → |99.69-100|/100=0.0031 > 0.003
        df = make_df_bounce_long(
            signal_low=99.69, signal_open=101.0, signal_close=101.2,
            signal_high=101.5
        )
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_long_body_exactly_at_threshold_no_signal(self):
        """Тело ровно на pin_bar_body_ratio=0.3 → НЕ < 0.3 → нет сигнала.

        Строгое неравенство: нужно body_ratio < 0.3, ровно 0.3 не проходит.
        Используем патч порога = 0.3 и свечу с ratio=0.3.
        """
        # range=2.0, body=0.6 → ratio=0.3 ровно → не пройдёт (нужно строго <)
        # low=99.7, high=101.7 → range=2.0
        # lower_wick = open - low; нужно wick_ratio > 0.6: lower_wick > 1.2
        # open=101.0 → lower_wick = 1.3 → ratio=1.3/2.0=0.65 > 0.6 ✓
        # close=101.6 → body=0.6 → ratio=0.3 → нет сигнала
        # mid=(101.7+99.7)/2=100.7 → close=101.6 > 100.7 ✓
        # close_near=|101.6-100|/100=0.016 > 0.015 — нарушено, добавим высоту
        # low=99.7, high=101.4 → range=1.7, lower_wick=101.0-99.7=1.3 → ratio=1.3/1.7=0.765>0.6✓
        # close=101.3+0.51=... сложно. Проще: патч max_dist или использовать патч body_ratio=0.3
        # и свечу с ratio=0.3 ровно.
        # Используем: low=99.7, high=101.5, open=101.0, close=101.24
        # range=1.8, body=0.24 → ratio=0.24/1.8=0.133... Это < 0.3 → сигнал.
        # Нужно ровно 0.3: range=1.8, body=0.54 → open=101.0, close=101.54
        # lower_wick=101.0-99.7=1.3 → ratio=1.3/1.8=0.722>0.6✓
        # mid=100.6 → close=101.54>100.6✓
        # close_near=|101.54-100|/100=0.0154>0.015 → нарушено.
        # Используем патч close_dist_factor или swapped close.
        # Простейший способ: патч pin_bar_body_ratio = 0.31
        # свеча с body_ratio=0.3 → 0.3 < 0.31 → ДАСТ сигнал
        # Но нам надо показать ровно 0.3 НЕ даёт с порогом 0.3.
        # Берём: low=99.7, high=101.5(range=1.8)
        # body_ratio=0.3 → body=0.54 → open=100.7, close=101.24
        # lower_wick=min(100.7,101.24)-99.7=100.7-99.7=1.0 → ratio=1.0/1.8=0.556 < 0.6 → хвост мал
        # Нужно lower_wick_ratio > 0.6: lower_wick > 1.08 → min(open,close) > 100.78
        # open=100.8, close=101.34 → body=0.54 → ratio=0.3, lower_wick=100.8-99.7=1.1 → ratio=0.611✓
        # mid=100.6 → close=101.34>100.6✓; close_near=|101.34-100|/100=0.0134≤0.015✓
        # body_ratio=0.3 → строго < 0.3 не выполнено → нет сигнала ✓
        df = make_df_bounce_long(
            signal_low=99.7, signal_open=100.8, signal_close=101.34,
            signal_high=101.5
        )
        # body = 0.54, range = 1.8, body_ratio = 0.3 ровно → не < 0.3 → нет сигнала
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_long_body_just_below_threshold_gives_signal(self):
        """Тело чуть меньше pin_bar_body_ratio=0.3 → пин-бар → сигнал есть.

        body_ratio = 0.295 < 0.3 ✓
        Используем патч порога 0.31 чтобы гарантировать прохождение.
        """
        # low=99.7, high=101.5(range=1.8)
        # open=100.8, close=101.33 → body=0.53 → ratio=0.53/1.8=0.2944<0.3 ✓
        # lower_wick=100.8-99.7=1.1 → ratio=1.1/1.8=0.611>0.6 ✓
        # mid=100.6 → close=101.33>100.6 ✓
        # close_near=|101.33-100|/100=0.0133≤0.015 ✓
        with patch.object(BOUNCE_CONFIG, "pin_bar_body_ratio", 0.31):
            df = make_df_bounce_long(
                signal_low=99.7, signal_open=100.8, signal_close=101.33,
                signal_high=101.5
            )
            assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_long_volume_exactly_at_threshold(self):
        """Объём ровно на пороге (volume_threshold × avg_volume) → сигнал есть (>=)."""
        df = make_df_bounce_long(signal_vol=GOOD_VOL)
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_long_volume_just_below_threshold(self):
        """Объём чуть ниже порога → нет сигнала."""
        just_below = AVG_VOL * (BOUNCE_CONFIG.volume_threshold - 0.01)
        df = make_df_bounce_long(signal_vol=just_below)
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_long_uses_last_candle_only(self):
        """Предыдущие свечи — молотки, последняя — нет. Должен вернуть None."""
        rows = _make_prefix(5, AVG_VOL)
        for i in range(3):
            rows.append({
                "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * (5 + i)),
                "open": 101.0, "high": 101.5, "low": 99.7,
                "close": 101.2, "volume": HIGH_VOL, "avg_volume": AVG_VOL,
            })
        # Последняя свеча — нейтральная, далеко от уровня
        rows.append({
            "time": pd.Timestamp("2024-01-01T01:00"),
            "open": 105.0, "high": 106.0, "low": 104.0,
            "close": 105.5, "volume": HIGH_VOL, "avg_volume": AVG_VOL,
        })
        df = pd.DataFrame(rows)
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_long_not_triggered_as_short(self):
        """Сигнал LONG не должен ошибочно возвращать 'short'."""
        df = make_df_bounce_long()
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_long_signal_hammer_bearish_candle(self):
        """Медвежий молоток (open > close) тоже даёт LONG — направление свечи не важно.

        low=99.7, high=101.5, open=101.2, close=101.0 (open > close, медвежья)
        range=1.8
        lower_wick=min(101.2,101.0)-99.7=101.0-99.7=1.3 → ratio=1.3/1.8=0.722>0.6 ✓
        body=|101.2-101.0|=0.2 → ratio=0.2/1.8=0.111<0.3 ✓
        mid=(101.5+99.7)/2=100.6 → close=101.0>100.6 ✓
        """
        df = make_df_bounce_long(
            signal_low=99.7, signal_open=101.2, signal_close=101.0,
            signal_high=101.5
        )
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_long_wick_exactly_at_threshold_no_signal(self):
        """Нижний хвост ровно 0.6 (не > 0.6) → нет сигнала (строгое >).

        range=1.8, lower_wick_ratio=0.6 → lower_wick=1.08 → min(open,close)=100.78
        open=100.8, close=100.78 → body=0.02 → ratio=0.011<0.3 ✓
        lower_wick=min(100.8,100.78)-99.7=100.78-99.7=1.08 → ratio=1.08/1.8=0.6 ровно → не > 0.6
        Используем патч wick_ratio=0.6, свечу с ratio=0.6 ровно.
        """
        # Строгое неравенство: ровно 0.6 не проходит
        # low=99.7, high=101.5 → range=1.8
        # lower_wick=1.08 → min(open,close)=100.78
        # close=100.9, open=100.78 → body=0.12 → ratio=0.12/1.8=0.067<0.3 ✓
        # lower_wick=100.78-99.7=1.08 → ratio=1.08/1.8=0.6 → не > 0.6 → нет сигнала
        # mid=100.6 → close=100.9>100.6 ✓
        # close_near=|100.9-100|/100=0.009≤0.015 ✓
        df = make_df_bounce_long(
            signal_low=99.7, signal_open=100.78, signal_close=100.9,
            signal_high=101.5
        )
        # lower_wick=min(100.78,100.9)-99.7=100.78-99.7=1.08 → ratio=0.6 ровно → нет сигнала
        assert detect(df, SUPPORT, RESISTANCE) != "long"


# ─── TestDetectShort ──────────────────────────────────────────────────────────

class TestDetectShort:
    """Основной сигнал SHORT и нарушение каждого условия."""

    def test_basic_short_signal(self):
        """Все условия выполнены → сигнал "short"."""
        df = make_df_bounce_short()
        assert detect(df, SUPPORT, RESISTANCE) == "short"

    def test_short_needs_high_touching_resistance(self):
        """High НЕ касается resistance (слишком далеко) → нет сигнала."""
        # high=113.0 → |113-110|/110≈0.027 > proximity=0.003
        df = make_df_bounce_short(
            signal_high=113.0, signal_open=109.3, signal_close=109.1,
            signal_low=108.8
        )
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_short_needs_close_below_resistance(self):
        """Close ВЫШЕ resistance → не отбой → нет сигнала."""
        df = make_df_bounce_short(
            signal_high=110.3, signal_open=110.5, signal_close=111.0,
            signal_low=108.8
        )
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_short_needs_close_in_lower_half(self):
        """Close в верхней половине диапазона → тело направлено вверх → нет SHORT.

        high=110.3, low=108.8 → mid=109.55
        close=109.7 > 109.55 → верхняя половина → нет SHORT.
        """
        df = make_df_bounce_short(
            signal_high=110.3, signal_open=109.6, signal_close=109.7,
            signal_low=108.8
        )
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_short_no_signal_if_body_too_large(self):
        """Тело >= pin_bar_body_ratio=0.3 → НЕ пин-бар → нет сигнала.

        high=110.3, low=108.8 → range=1.5
        open=109.8, close=109.3 → body=0.5 → ratio=0.5/1.5=0.333 > 0.3 → тело велико.
        """
        df = make_df_bounce_short(
            signal_high=110.3, signal_open=109.8, signal_close=109.3,
            signal_low=108.8
        )
        # body=0.5, range=1.5, ratio=0.333 >= 0.3 → нет сигнала
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_short_no_signal_if_wick_too_small(self):
        """Верхний хвост <= pin_bar_wick_ratio=0.6 → хвост короткий → нет сигнала.

        high=110.3, low=108.8, open=110.0, close=109.1 → range=1.5
        upper_wick=110.3-max(110.0,109.1)=110.3-110.0=0.3 → ratio=0.3/1.5=0.2<0.6 → хвост мал.
        body=|110.0-109.1|=0.9 → ratio=0.9/1.5=0.6 ≥ 0.3 → но хвост — основное нарушение.
        """
        df = make_df_bounce_short(
            signal_high=110.3, signal_open=110.0, signal_close=109.1,
            signal_low=108.8
        )
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_short_needs_sufficient_volume(self):
        """Объём ниже порога → нет сигнала."""
        df = make_df_bounce_short(signal_vol=LOW_VOL)
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_short_needs_close_near_resistance(self):
        """Close слишком далеко от resistance (> 1.5%) → нет сигнала."""
        # resistance=110, max_close_dist=0.015 → close должен быть >= 108.35
        # close=107.0 → |107-110|/110=0.027 > 0.015 → нет
        # Строим корректный висельник но с close слишком далеко:
        # high=110.3 (касание), low=106.5, open=109.3, close=107.0
        # range=3.8, upper_wick=110.3-max(109.3,107.0)=110.3-109.3=1.0 → ratio=1.0/3.8=0.263 < 0.6 → не висельник
        # Нужно: close=107.0, open=107.1, high=110.3, low=106.5
        # upper_wick=110.3-107.1=3.2 → ratio=3.2/3.8=0.842>0.6 ✓
        # body=0.1 → ratio=0.1/3.8=0.026<0.3 ✓
        # mid=(110.3+106.5)/2=108.4 → close=107.0<108.4 ✓
        # close_near=|107.0-110|/110=0.027>0.015 → нет сигнала ✓
        df = make_df_bounce_short(
            signal_high=110.3, signal_open=107.1, signal_close=107.0,
            signal_low=106.5
        )
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_short_with_minimum_2_rows(self):
        """Минимально допустимый DataFrame: 2 строки."""
        df = make_df_bounce_short(n_prefix=1)
        assert detect(df, SUPPORT, RESISTANCE) == "short"

    def test_short_at_exact_proximity_boundary(self):
        """High точно на границе proximity_threshold → касание засчитывается."""
        # proximity=0.003, resistance=110 → граница: high=110*(1+0.003)=110.33
        # Строим корректный висельник с high=110.33
        # low=108.8, open=109.3, close=109.1 → range=110.33-108.8=1.53
        # upper_wick=110.33-109.3=1.03 → ratio=1.03/1.53=0.673>0.6 ✓
        # body=0.2 → ratio=0.2/1.53=0.131<0.3 ✓
        df = make_df_bounce_short(
            signal_high=110.33, signal_open=109.3, signal_close=109.1,
            signal_low=108.8
        )
        assert detect(df, SUPPORT, RESISTANCE) == "short"

    def test_short_high_just_outside_proximity(self):
        """High чуть за границей proximity → нет касания → нет сигнала."""
        # high=110.34 → |110.34-110|/110=0.00309 > 0.003
        df = make_df_bounce_short(
            signal_high=110.34, signal_open=109.3, signal_close=109.1,
            signal_low=108.8
        )
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_short_body_exactly_at_threshold_no_signal(self):
        """Тело ровно на pin_bar_body_ratio=0.3 → НЕ < 0.3 → нет сигнала.

        Из-за float64-погрешности сложно получить ratio ровно 0.3 без отклонений.
        Используем патч: pin_bar_body_ratio=0.15, свеча с body_ratio≈0.133 даёт сигнал
        при пороге 0.15, но патчим порог ровно на значение body_ratio (≈0.133) +эпсилон
        чтобы 0.133 < 0.133+ε → НЕ < body_ratio.

        Проще: свеча с body_ratio=0.333 > 0.3 → нет сигнала ровно по телу.
        Дополнительно проверяем через патч pin_bar_body_ratio=0.35:
        при 0.35 тело 0.333 < 0.35 → сигнал.
        При дефолтном 0.3 тело 0.333 >= 0.3 → нет сигнала.
        high=110.3, low=108.8, range=1.5
        open=109.8, close=109.3 → body=0.5 → ratio=0.333 >= 0.3 → нет сигнала.
        НО upper_wick=110.3-109.8=0.5 → ratio=0.333 < 0.6 → тоже нарушено.
        Хвост мал — это second reason. Но тест проверяет тело:
        патчим wick_ratio=0.3 чтобы wick прошёл, остаётся только body_ratio=0.333>=0.3.
        """
        # При default: body_ratio=0.333>=0.3 → нет сигнала (тело велико)
        df = make_df_bounce_short(
            signal_high=110.3, signal_open=109.8, signal_close=109.3,
            signal_low=108.8
        )
        assert detect(df, SUPPORT, RESISTANCE) != "short"
        # С мягким порогом тела (0.35) и патчем wick_ratio=0.2 (очень мягкий для хвоста):
        # body_ratio=0.333 < 0.35 → тело ОК; но upper_wick_ratio=0.333 > 0.2 → хвост ОК.
        # mid=109.55 → close=109.3 < 109.55 ✓; close_near=|109.3-110|/110=0.0064 ≤ 0.015 ✓
        with patch.object(BOUNCE_CONFIG, "pin_bar_body_ratio", 0.35):
            with patch.object(BOUNCE_CONFIG, "pin_bar_wick_ratio", 0.2):
                assert detect(df, SUPPORT, RESISTANCE) == "short"

    def test_short_body_just_below_threshold_gives_signal(self):
        """Тело чуть меньше pin_bar_body_ratio=0.3 → пин-бар → сигнал есть.

        Используем патч pin_bar_body_ratio=0.31 и свечу с ratio≈0.3.
        """
        # high=110.3, low=108.8, range=1.5
        # open=109.35, close=108.91 → body=0.44 → ratio=0.44/1.5=0.293<0.3 ✓
        # upper_wick=110.3-109.35=0.95 → ratio=0.95/1.5=0.633>0.6 ✓
        # mid=109.55 → close=108.91<109.55 ✓
        with patch.object(BOUNCE_CONFIG, "pin_bar_body_ratio", 0.31):
            df = make_df_bounce_short(
                signal_high=110.3, signal_open=109.35, signal_close=108.91,
                signal_low=108.8
            )
            assert detect(df, SUPPORT, RESISTANCE) == "short"

    def test_short_volume_exactly_at_threshold(self):
        """Объём ровно на пороге → сигнал есть (>=)."""
        df = make_df_bounce_short(signal_vol=GOOD_VOL)
        assert detect(df, SUPPORT, RESISTANCE) == "short"

    def test_short_volume_just_below_threshold(self):
        """Объём чуть ниже порога → нет сигнала."""
        just_below = AVG_VOL * (BOUNCE_CONFIG.volume_threshold - 0.01)
        df = make_df_bounce_short(signal_vol=just_below)
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_short_uses_last_candle_only(self):
        """Предыдущие свечи — висельники, последняя — нейтральная. Должен вернуть None."""
        rows = _make_prefix(5, AVG_VOL)
        for i in range(3):
            rows.append({
                "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * (5 + i)),
                "open": 109.3, "high": 110.3, "low": 108.8,
                "close": 109.1, "volume": HIGH_VOL, "avg_volume": AVG_VOL,
            })
        # Последняя — нейтральная
        rows.append({
            "time": pd.Timestamp("2024-01-01T01:00"),
            "open": 105.0, "high": 106.0, "low": 104.0,
            "close": 105.5, "volume": HIGH_VOL, "avg_volume": AVG_VOL,
        })
        df = pd.DataFrame(rows)
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_short_not_triggered_as_long(self):
        """Сигнал SHORT не должен ошибочно возвращать 'long'."""
        df = make_df_bounce_short()
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_short_signal_hanging_man_bullish_candle(self):
        """Бычий висельник (close > open) тоже даёт SHORT — направление свечи не важно.

        high=110.3, low=108.8, open=109.1, close=109.3 (close>open, бычья)
        range=1.5
        upper_wick=110.3-max(109.1,109.3)=110.3-109.3=1.0 → ratio=1.0/1.5=0.667>0.6 ✓
        body=|109.3-109.1|=0.2 → ratio=0.2/1.5=0.133<0.3 ✓
        mid=(110.3+108.8)/2=109.55 → close=109.3<109.55 ✓
        """
        df = make_df_bounce_short(
            signal_high=110.3, signal_open=109.1, signal_close=109.3,
            signal_low=108.8
        )
        assert detect(df, SUPPORT, RESISTANCE) == "short"

    def test_short_wick_exactly_at_threshold_no_signal(self):
        """Верхний хвост ровно 0.6 (не > 0.6) → нет сигнала (строгое >).

        range=1.5, upper_wick_ratio=0.6 → upper_wick=0.9 → max(open,close)=110.3-0.9=109.4
        open=109.2, close=109.0 → body=0.2 → ratio=0.2/1.5=0.133<0.3 ✓
        upper_wick=110.3-max(109.2,109.0)=110.3-109.2=1.1 → ratio=1.1/1.5=0.733>0.6... слишком много.
        Нужно upper_wick=0.9 ровно: max(open,close)=109.4 → open=109.4, close=109.2
        upper_wick=110.3-109.4=0.9 → ratio=0.9/1.5=0.6 ровно → не > 0.6 → нет сигнала
        body=0.2 → ratio=0.133<0.3 ✓
        mid=(110.3+108.8)/2=109.55 → close=109.2<109.55 ✓
        close_near=|109.2-110|/110=0.0073≤0.015 ✓
        """
        df = make_df_bounce_short(
            signal_high=110.3, signal_open=109.4, signal_close=109.2,
            signal_low=108.8
        )
        # upper_wick=110.3-109.4=0.9, range=1.5, ratio=0.6 ровно → не > 0.6 → нет сигнала
        assert detect(df, SUPPORT, RESISTANCE) != "short"


# ─── TestDetectConfig ─────────────────────────────────────────────────────────

class TestDetectConfig:
    """Параметры берутся из BOUNCE_CONFIG, не хардкодированы."""

    def test_proximity_threshold_from_config(self):
        """Изменение proximity_threshold меняет поведение."""
        # low=99.5 → |99.5-100|/100=0.005 > дефолтный proximity=0.003 → нет касания
        # Нужно сохранить молоток: high=101.5, open=101.0, close=101.2
        # lower_wick=min(101.0,101.2)-99.5=99.5? Нет: lower_wick=101.0-99.5=1.5
        # range=101.5-99.5=2.0 → lower_wick_ratio=1.5/2.0=0.75>0.6 ✓
        # body=0.2/2.0=0.1<0.3 ✓
        # mid=(101.5+99.5)/2=100.5 → close=101.2>100.5 ✓
        df = make_df_bounce_long(
            signal_low=99.5, signal_open=101.0, signal_close=101.2,
            signal_high=101.5
        )
        result_default = detect(df, SUPPORT, RESISTANCE)

        # С relaxed proximity=0.006 → 0.005 ≤ 0.006 → касание → сигнал
        with patch.object(BOUNCE_CONFIG, "proximity_threshold", 0.006):
            result_relaxed = detect(df, SUPPORT, RESISTANCE)

        assert result_default != "long"
        assert result_relaxed == "long"

    def test_volume_threshold_from_config(self):
        """Изменение volume_threshold меняет поведение."""
        # vol = AVG_VOL * 1.0 = 1000 < дефолтный threshold 1.5 → нет сигнала
        medium_vol = AVG_VOL * 1.0
        df = make_df_bounce_long(signal_vol=medium_vol)
        result_default = detect(df, SUPPORT, RESISTANCE)

        # Ослабим threshold до 0.9 → 1000 >= 900 → сигнал
        with patch.object(BOUNCE_CONFIG, "volume_threshold", 0.9):
            result_relaxed = detect(df, SUPPORT, RESISTANCE)

        assert result_default != "long"
        assert result_relaxed == "long"

    def test_pin_bar_body_ratio_from_config(self):
        """Изменение pin_bar_body_ratio меняет поведение.

        Свеча с body_ratio=0.111 (< 0.3) — по дефолту даёт сигнал.
        При строгом пороге 0.15: 0.111 < 0.15 → всё ещё даёт сигнал.
        При пороге 0.1: 0.111 >= 0.1 → нет сигнала (тело больше нового порога).
        """
        # Дефолтная фабрика: body_ratio=0.111 < pin_bar_body_ratio=0.3 → сигнал
        df = make_df_bounce_long()
        result_default = detect(df, SUPPORT, RESISTANCE)

        # Ужесточим порог до 0.1 → body_ratio=0.111 >= 0.1 → нет сигнала
        with patch.object(BOUNCE_CONFIG, "pin_bar_body_ratio", 0.1):
            result_strict = detect(df, SUPPORT, RESISTANCE)

        assert result_default == "long"
        assert result_strict != "long"

    def test_pin_bar_wick_ratio_from_config(self):
        """Изменение pin_bar_wick_ratio меняет поведение.

        Дефолтная фабрика: lower_wick_ratio=0.722 > 0.6 → сигнал.
        При пороге 0.8: 0.722 < 0.8 → нет сигнала (хвост не достаточно длинный).
        """
        df = make_df_bounce_long()
        result_default = detect(df, SUPPORT, RESISTANCE)

        # Ужесточим порог до 0.8 → lower_wick_ratio=0.722 ≤ 0.8 → нет сигнала
        with patch.object(BOUNCE_CONFIG, "pin_bar_wick_ratio", 0.8):
            result_strict = detect(df, SUPPORT, RESISTANCE)

        assert result_default == "long"
        assert result_strict != "long"

    def test_short_volume_threshold_from_config(self):
        """Изменение volume_threshold влияет на SHORT."""
        medium_vol = AVG_VOL * 1.0
        df = make_df_bounce_short(signal_vol=medium_vol)
        result_default = detect(df, SUPPORT, RESISTANCE)

        with patch.object(BOUNCE_CONFIG, "volume_threshold", 0.9):
            result_relaxed = detect(df, SUPPORT, RESISTANCE)

        assert result_default != "short"
        assert result_relaxed == "short"

    def test_short_pin_bar_body_ratio_from_config(self):
        """Изменение pin_bar_body_ratio меняет поведение для SHORT.

        Дефолтная фабрика short: body_ratio=0.133 < 0.3 → сигнал.
        При пороге 0.1: 0.133 >= 0.1 → нет сигнала.
        """
        df = make_df_bounce_short()
        result_default = detect(df, SUPPORT, RESISTANCE)

        with patch.object(BOUNCE_CONFIG, "pin_bar_body_ratio", 0.1):
            result_strict = detect(df, SUPPORT, RESISTANCE)

        assert result_default == "short"
        assert result_strict != "short"


# ─── TestBoundaryValues ───────────────────────────────────────────────────────

class TestBoundaryValues:
    """Граничные значения и дополнительные случаи."""

    def test_long_signal_with_many_rows(self):
        """Большой DataFrame тоже работает корректно."""
        df = make_df_bounce_long(n_prefix=100)
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_short_signal_with_many_rows(self):
        """Большой DataFrame для SHORT."""
        df = make_df_bounce_short(n_prefix=100)
        assert detect(df, SUPPORT, RESISTANCE) == "short"

    def test_both_signals_cannot_fire_simultaneously(self):
        """Функция возвращает ровно одно значение — или None."""
        df = make_df_bounce_long()
        result = detect(df, SUPPORT, RESISTANCE)
        assert result in ("long", "short", None)
        # LONG проверяется первым
        assert result != "short"

    def test_no_signal_when_price_far_from_both_levels(self):
        """Цена в середине диапазона — нет сигнала."""
        rows = _make_prefix(5, AVG_VOL)
        rows.append({
            "time": pd.Timestamp("2024-01-01T01:00"),
            "open": 105.0, "high": 105.5, "low": 104.5, "close": 105.2,
            "volume": HIGH_VOL, "avg_volume": AVG_VOL,
        })
        df = pd.DataFrame(rows)
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_negative_support_returns_none(self):
        """Отрицательный support — некорректные данные."""
        df = make_df_bounce_long()
        assert detect(df, -100.0, RESISTANCE) is None

    def test_negative_resistance_returns_none(self):
        """Отрицательный resistance — некорректные данные."""
        df = make_df_bounce_long()
        assert detect(df, SUPPORT, -110.0) is None

    def test_long_close_exactly_at_max_dist(self):
        """Close ровно на max_close_dist от support → сигнал есть (<=).

        max_close_dist = proximity * 5 = 0.003 * 5 = 0.015
        close = 100 * 1.015 = 101.5
        Нужен молоток: low=99.7, high=101.5(close=high), open≈lower
        range=101.5-99.7=1.8
        lower_wick=min(open,101.5)-99.7: нужно > 0.6*1.8=1.08
        open=100.78 → lower_wick=100.78-99.7=1.08 → ratio=0.6 ровно → нет сигнала.
        open=100.75 → lower_wick=100.75-99.7=1.05 → ratio=0.583 < 0.6 → тоже нет.
        Нужно lower_wick > 1.08 → open < 100.78. Возьмём open=100.7:
        lower_wick=100.7-99.7=1.0 → ratio=1.0/1.8=0.556 < 0.6 → нет.
        Расширим range: low=99.5(не касание с proximity=0.003).
        Попробуем: low=99.7, high=102.0 → range=2.3
        close=101.5 → close_near=|101.5-100|/100=0.015 ровно ≤ 0.015 ✓
        lower_wick нужно > 0.6*2.3=1.38 → min(open,close)=101.5: lower_wick=101.5-99.7=1.8>1.38 ✓
        Но close=high → upper_wick=0 ✓ (не влияет на LONG)
        body: open=101.4, close=101.5 → body=0.1 → ratio=0.1/2.3=0.043<0.3 ✓
        lower_wick=min(101.4,101.5)-99.7=101.4-99.7=1.7 → ratio=1.7/2.3=0.739>0.6 ✓
        mid=(102+99.7)/2=100.85 → close=101.5>100.85 ✓
        """
        df = make_df_bounce_long(
            signal_low=99.7, signal_open=101.4, signal_close=101.5,
            signal_high=102.0
        )
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_long_close_just_beyond_max_dist(self):
        """Close чуть дальше max_close_dist → нет сигнала.

        close=101.51 → |101.51-100|/100=0.01510 > 0.015
        Та же форма молотка, но close=101.51.
        """
        df = make_df_bounce_long(
            signal_low=99.7, signal_open=101.4, signal_close=101.51,
            signal_high=102.1
        )
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_returns_string_type_for_long(self):
        """Возвращаемый тип — строка 'long'."""
        df = make_df_bounce_long()
        result = detect(df, SUPPORT, RESISTANCE)
        assert isinstance(result, str)
        assert result == "long"

    def test_returns_string_type_for_short(self):
        """Возвращаемый тип — строка 'short'."""
        df = make_df_bounce_short()
        result = detect(df, SUPPORT, RESISTANCE)
        assert isinstance(result, str)
        assert result == "short"

    def test_long_with_avg_volume_negative_returns_none(self):
        """Отрицательный avg_volume → нет сигнала."""
        df = make_df_bounce_long()
        df.iloc[-1, df.columns.get_loc("avg_volume")] = -100.0
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_short_close_near_max_dist(self):
        """Close вблизи max_close_dist от resistance (чуть внутри) → сигнал есть.

        max_close_dist=0.015 → close=110*(1-0.014)=108.46
        Висельник: high=110.3, low=108.0, open=108.5, close=108.46
        range=110.3-108.0=2.3
        upper_wick=110.3-max(108.5,108.46)=110.3-108.5=1.8 → ratio=1.8/2.3=0.782>0.6 ✓
        body=0.04 → ratio=0.04/2.3=0.017<0.3 ✓
        mid=(110.3+108.0)/2=109.15 → close=108.46<109.15 ✓
        close_near=|108.46-110|/110=0.014≤0.015 ✓
        """
        df = make_df_bounce_short(
            signal_high=110.3, signal_open=108.5, signal_close=108.46,
            signal_low=108.0
        )
        assert detect(df, SUPPORT, RESISTANCE) == "short"
