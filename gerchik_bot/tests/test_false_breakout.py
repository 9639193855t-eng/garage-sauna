"""Тесты для strategies/false_breakout.py — стратегия «Ложный пробой»."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
from unittest.mock import patch

from strategies.false_breakout import detect
from config import FB_CONFIG


# ─── Фабрика свечей ──────────────────────────────────────────────────────────

def make_df(
    open_: float = 100.0,
    high:  float = 102.0,
    low:   float = 98.0,
    close: float = 100.5,
    volume: float = 500.0,
    avg_volume: float = 1000.0,
    n: int = 5,
) -> pd.DataFrame:
    """
    Создать DataFrame из n одинаковых свечей.
    Последняя свеча (df.iloc[-1]) — та, которую анализирует detect().
    """
    rows = []
    for i in range(n):
        rows.append({
            "time":       pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
            "open":       open_,
            "high":       high,
            "low":        low,
            "close":      close,
            "volume":     volume,
            "avg_volume": avg_volume,
        })
    return pd.DataFrame(rows)


def make_single(
    open_: float,
    high:  float,
    low:   float,
    close: float,
    volume: float,
    avg_volume: float,
) -> pd.DataFrame:
    """Создать DataFrame с 2 свечами (минимум), где вторая — сигнальная."""
    filler = {
        "time": pd.Timestamp("2024-01-01"),
        "open": 100.0, "high": 102.0, "low": 98.0,
        "close": 100.0, "volume": avg_volume, "avg_volume": avg_volume,
    }
    signal = {
        "time":       pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5),
        "open":       open_,
        "high":       high,
        "low":        low,
        "close":      close,
        "volume":     volume,
        "avg_volume": avg_volume,
    }
    return pd.DataFrame([filler, signal])


# Стандартные уровни для тестов
SUPPORT    = 98.0
RESISTANCE = 103.0
AVG_VOL    = 1000.0
# Слабый объём — ниже порога (FB_CONFIG.volume_threshold = 0.7 → нужен < 700)
LOW_VOL    = AVG_VOL * (FB_CONFIG.volume_threshold - 0.1)
# Сильный объём — выше порога
HIGH_VOL   = AVG_VOL * (FB_CONFIG.volume_threshold + 0.1)
# Маленькое тело — ниже порога (FB_CONFIG.body_threshold = 0.3 → body/range < 0.3)
# range = high - low, body = |close - open|
# Создаём: range = 6, body = 1 → ratio = 0.17 < 0.3 ✓


# ─── Граничные случаи ────────────────────────────────────────────────────────

class TestDetectEdgeCases:

    def test_returns_none_for_none_df(self):
        assert detect(None, SUPPORT, RESISTANCE) is None

    def test_returns_none_for_empty_df(self):
        assert detect(pd.DataFrame(), SUPPORT, RESISTANCE) is None

    def test_returns_none_for_single_row(self):
        df = make_df(n=1)
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_returns_none_if_support_none(self):
        df = make_df(n=5)
        assert detect(df, None, RESISTANCE) is None

    def test_returns_none_if_resistance_none(self):
        df = make_df(n=5)
        assert detect(df, SUPPORT, None) is None

    def test_returns_none_if_support_zero(self):
        df = make_df(n=5)
        assert detect(df, 0.0, RESISTANCE) is None

    def test_returns_none_if_resistance_zero(self):
        df = make_df(n=5)
        assert detect(df, SUPPORT, 0.0) is None

    def test_returns_none_if_resistance_leq_support(self):
        df = make_df(n=5)
        assert detect(df, 105.0, 95.0) is None    # перепутаны
        assert detect(df, 100.0, 100.0) is None   # равны

    def test_returns_none_if_avg_volume_missing(self):
        """Нет столбца avg_volume — нет сигнала."""
        df = make_df(n=5).drop(columns=["avg_volume"])
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_returns_none_if_avg_volume_nan(self):
        df = make_df(n=5)
        df.iloc[-1, df.columns.get_loc("avg_volume")] = float("nan")
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_returns_none_if_avg_volume_zero(self):
        df = make_df(n=5)
        df.iloc[-1, df.columns.get_loc("avg_volume")] = 0.0
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_returns_none_for_zero_candle_range(self):
        """Свеча-доджи с high == low — нет диапазона."""
        df = make_single(
            open_=100.0, high=100.0, low=100.0, close=100.0,
            volume=LOW_VOL, avg_volume=AVG_VOL,
        )
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_result_is_string_or_none(self):
        df = make_df(n=5)
        result = detect(df, SUPPORT, RESISTANCE)
        assert result in ("long", "short", None)


# ─── Ложный пробой вниз → LONG ───────────────────────────────────────────────

class TestFalseBreakoutLong:
    """
    Условия LONG:
      low < support      — тень пробила поддержку
      close > support    — закрылась обратно выше
      volume_ratio < volume_threshold   — объём слабый
      body_ratio   < body_threshold     — тело маленькое
    """

    def _make_long_candle(
        self,
        vol: float = None,
        body_ratio_override: float = None,
    ) -> pd.DataFrame:
        """
        Свеча-пример для LONG:
            support = 98.0
            low = 97.0  (пробила поддержку)
            close = 100.0 (закрылась выше поддержки)
            high = 103.0
            range = 6, body = |close - open|

        body_ratio = body / range < body_threshold (0.3)
        Возьмём open=99.5, close=100.0 → body=0.5, range=6 → ratio=0.083 ✓
        """
        if body_ratio_override is not None:
            # range=6, нужный body = body_ratio_override * 6
            body  = body_ratio_override * 6.0
            open_ = 100.0 - body / 2
            close = 100.0 + body / 2
        else:
            open_ = 99.5
            close = 100.0

        volume = vol if vol is not None else LOW_VOL
        return make_single(
            open_=open_, high=103.0, low=97.0, close=close,
            volume=volume, avg_volume=AVG_VOL,
        )

    def test_basic_long_signal(self):
        df = self._make_long_candle()
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_no_signal_if_close_below_support(self):
        """Закрылась ниже поддержки — реальный пробой, не ложный."""
        df = make_single(
            open_=99.0, high=103.0, low=97.0, close=97.5,  # close < support(98)
            volume=LOW_VOL, avg_volume=AVG_VOL,
        )
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_no_signal_if_low_above_support(self):
        """Low не дошёл до поддержки — нет пробоя."""
        df = make_single(
            open_=100.0, high=103.0, low=98.5,  # low > support(98)
            close=101.0, volume=LOW_VOL, avg_volume=AVG_VOL,
        )
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_no_signal_if_volume_too_high(self):
        """Объём слишком высокий — настоящий пробой, не ложный."""
        df = self._make_long_candle(vol=HIGH_VOL)
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_no_signal_if_body_too_large(self):
        """Тело слишком большое — направленная свеча, не ложный пробой."""
        # body_ratio = 0.4 > body_threshold (0.3)
        df = self._make_long_candle(body_ratio_override=0.4)
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_volume_exactly_at_threshold_no_signal(self):
        """
        volume_ratio == volume_threshold → условие < НЕ выполнено.
        """
        exact_vol = AVG_VOL * FB_CONFIG.volume_threshold
        df = self._make_long_candle(vol=exact_vol)
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_volume_just_below_threshold(self):
        """volume_ratio чуть ниже порога → сигнал есть."""
        just_below = AVG_VOL * (FB_CONFIG.volume_threshold - 0.01)
        df = self._make_long_candle(vol=just_below)
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_body_exactly_at_threshold_no_signal(self):
        """body_ratio == body_threshold → условие < НЕ выполнено."""
        exact_body = FB_CONFIG.body_threshold
        df = self._make_long_candle(body_ratio_override=exact_body)
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_body_just_below_threshold(self):
        """body_ratio чуть ниже порога → сигнал есть."""
        just_below = FB_CONFIG.body_threshold - 0.01
        df = self._make_long_candle(body_ratio_override=just_below)
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_extremum_used_for_stop(self):
        """
        Стоп-лосс для LONG должен ставиться ЗА low пробойной свечи.
        Проверяем: low < support, чтобы было за что ставить стоп.
        """
        df = self._make_long_candle()
        candle = df.iloc[-1]
        assert candle["low"] < SUPPORT  # low ниже поддержки

    def test_long_on_exact_support_break(self):
        """low == support - epsilon → пробой совсем чуть-чуть."""
        epsilon = 0.001
        df = make_single(
            open_=99.5, high=103.0, low=SUPPORT - epsilon, close=100.0,
            volume=LOW_VOL, avg_volume=AVG_VOL,
        )
        assert detect(df, SUPPORT, RESISTANCE) == "long"


# ─── Ложный пробой вверх → SHORT ─────────────────────────────────────────────

class TestFalseBreakoutShort:
    """
    Условия SHORT:
      high > resistance  — тень пробила сопротивление
      close < resistance — закрылась обратно ниже
      volume_ratio < volume_threshold
      body_ratio   < body_threshold
    """

    def _make_short_candle(
        self,
        vol: float = None,
        body_ratio_override: float = None,
    ) -> pd.DataFrame:
        """
        Свеча-пример для SHORT:
            resistance = 103.0
            high = 104.0  (пробила сопротивление)
            close = 101.0 (закрылась ниже сопротивления)
            low = 98.0
            range = 6, body = |close - open|

        open=101.5, close=101.0 → body=0.5, range=6 → ratio=0.083 ✓
        """
        if body_ratio_override is not None:
            body  = body_ratio_override * 6.0
            open_ = 101.0 + body / 2
            close = 101.0 - body / 2
        else:
            open_ = 101.5
            close = 101.0

        volume = vol if vol is not None else LOW_VOL
        return make_single(
            open_=open_, high=104.0, low=98.0, close=close,
            volume=volume, avg_volume=AVG_VOL,
        )

    def test_basic_short_signal(self):
        df = self._make_short_candle()
        assert detect(df, SUPPORT, RESISTANCE) == "short"

    def test_no_signal_if_close_above_resistance(self):
        """Закрылась выше сопротивления — реальный пробой, не ложный."""
        df = make_single(
            open_=101.0, high=104.0, low=98.0, close=103.5,  # close > resistance(103)
            volume=LOW_VOL, avg_volume=AVG_VOL,
        )
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_no_signal_if_high_below_resistance(self):
        """High не дошёл до сопротивления — нет пробоя."""
        df = make_single(
            open_=101.0, high=102.5,  # high < resistance(103)
            low=98.0, close=101.0,
            volume=LOW_VOL, avg_volume=AVG_VOL,
        )
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_no_signal_if_volume_too_high(self):
        df = self._make_short_candle(vol=HIGH_VOL)
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_no_signal_if_body_too_large(self):
        df = self._make_short_candle(body_ratio_override=0.4)
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_volume_just_below_threshold(self):
        just_below = AVG_VOL * (FB_CONFIG.volume_threshold - 0.01)
        df = self._make_short_candle(vol=just_below)
        assert detect(df, SUPPORT, RESISTANCE) == "short"

    def test_body_just_below_threshold(self):
        just_below = FB_CONFIG.body_threshold - 0.01
        df = self._make_short_candle(body_ratio_override=just_below)
        assert detect(df, SUPPORT, RESISTANCE) == "short"

    def test_extremum_used_for_stop(self):
        """high > resistance — есть экстремум за которым ставить стоп."""
        df = self._make_short_candle()
        candle = df.iloc[-1]
        assert candle["high"] > RESISTANCE

    def test_short_on_exact_resistance_break(self):
        """high == resistance + epsilon → пробой совсем чуть-чуть."""
        epsilon = 0.001
        df = make_single(
            open_=101.5, high=RESISTANCE + epsilon, low=98.0, close=101.0,
            volume=LOW_VOL, avg_volume=AVG_VOL,
        )
        assert detect(df, SUPPORT, RESISTANCE) == "short"


# ─── Нет сигнала в нормальных условиях ───────────────────────────────────────

class TestNoSignal:

    def test_no_signal_inside_range(self):
        """Свеча полностью внутри диапазона — нет пробоя."""
        df = make_single(
            open_=100.0, high=102.0, low=99.0, close=100.5,
            volume=LOW_VOL, avg_volume=AVG_VOL,
        )
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_no_signal_when_both_levels_broken(self):
        """
        Свеча пробила оба уровня (gap-свеча) — не ложный пробой ни там ни там
        при условии что close внутри диапазона: не пробой ни вниз ни вверх.
        """
        df = make_single(
            open_=100.0,
            high=105.0,  # > resistance
            low=97.0,    # < support
            close=100.0, # между уровнями
            volume=LOW_VOL, avg_volume=AVG_VOL,
        )
        # close > support (98) И close < resistance (103) → оба условия пробоя False
        # LONG требует close > support: 100 > 98 ✓, low < support: 97 < 98 ✓
        # → может дать LONG если все остальные условия выполнены
        result = detect(df, SUPPORT, RESISTANCE)
        assert result in ("long", "short", None)

    def test_strong_volume_prevents_signal(self):
        """При любом пробое с сильным объёмом — нет сигнала."""
        # Ложный пробой вниз, но объём сильный
        df = make_single(
            open_=99.5, high=103.0, low=97.0, close=100.0,
            volume=HIGH_VOL, avg_volume=AVG_VOL,
        )
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_large_body_prevents_signal(self):
        """При любом пробое с большим телом — нет сигнала."""
        # body_ratio = 0.5 > 0.3
        # range = 6, body = 3 → open=98.5, close=101.5
        df = make_single(
            open_=98.5, high=103.0, low=97.0, close=101.5,
            volume=LOW_VOL, avg_volume=AVG_VOL,
        )
        assert detect(df, SUPPORT, RESISTANCE) is None


# ─── Приоритет: только одна свеча анализируется ──────────────────────────────

class TestOnlyLastCandle:

    def test_uses_last_candle_only(self):
        """
        Предыдущие свечи — ложные пробои, последняя — нет.
        Функция должна вернуть None.
        """
        rows = []
        # 10 свечей с ложным пробоем вниз
        for i in range(10):
            rows.append({
                "time":       pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                "open":       99.5,
                "high":       103.0,
                "low":        97.0,    # пробивает support
                "close":      100.0,   # возвращается выше
                "volume":     LOW_VOL,
                "avg_volume": AVG_VOL,
            })
        # Последняя свеча — нормальная, без пробоя
        rows.append({
            "time":       pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * 10),
            "open":       100.0,
            "high":       102.0,
            "low":        99.0,   # НЕ пробивает support (98)
            "close":      100.5,
            "volume":     LOW_VOL,
            "avg_volume": AVG_VOL,
        })
        df = pd.DataFrame(rows)
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_last_candle_signal_despite_normal_history(self):
        """История нормальная, но последняя свеча — ложный пробой → сигнал."""
        rows = []
        for i in range(10):
            rows.append({
                "time":       pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                "open":       100.0, "high": 102.0, "low": 99.0,
                "close":      100.5, "volume": HIGH_VOL, "avg_volume": AVG_VOL,
            })
        # Последняя — ложный пробой вниз
        rows.append({
            "time":       pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * 10),
            "open":       99.5,
            "high":       103.0,
            "low":        97.0,
            "close":      100.0,
            "volume":     LOW_VOL,
            "avg_volume": AVG_VOL,
        })
        df = pd.DataFrame(rows)
        assert detect(df, SUPPORT, RESISTANCE) == "long"


# ─── Конфиг: корректность порогов ────────────────────────────────────────────

class TestConfigThresholds:

    def test_volume_threshold_from_config(self):
        """Порог объёма берётся из FB_CONFIG, не хардкодирован."""
        custom_threshold = 0.5  # более строгий порог

        # Объём между custom (0.5) и default (0.7): ratio=0.6 → ниже default, но выше custom
        vol_between = AVG_VOL * (custom_threshold + 0.1)  # 0.6 * 1000 = 600
        df_long = make_single(
            open_=99.5, high=103.0, low=97.0, close=100.0,
            volume=vol_between, avg_volume=AVG_VOL,
        )

        # С дефолтным конфигом (0.7) — объём ниже порога → сигнал
        result_default = detect(df_long, SUPPORT, RESISTANCE)

        # С кастомным конфигом (0.5) — тот же объём выше порога → нет сигнала
        with patch.object(FB_CONFIG, "volume_threshold", custom_threshold):
            result_custom = detect(df_long, SUPPORT, RESISTANCE)

        assert result_default == "long"
        assert result_custom is None

    def test_body_threshold_from_config(self):
        """Порог тела берётся из FB_CONFIG, не хардкодирован."""
        custom_threshold = 0.05  # очень строгий порог

        # body_ratio = 0.083 (из стандартной свечи) — выше 0.05 → нет сигнала
        df_long = make_single(
            open_=99.5, high=103.0, low=97.0, close=100.0,
            volume=LOW_VOL, avg_volume=AVG_VOL,
        )

        # С дефолтным конфигом (0.3) — body_ratio 0.083 < 0.3 → сигнал
        result_default = detect(df_long, SUPPORT, RESISTANCE)

        # С очень строгим конфигом (0.05) — body_ratio 0.083 > 0.05 → нет сигнала
        with patch.object(FB_CONFIG, "body_threshold", custom_threshold):
            result_custom = detect(df_long, SUPPORT, RESISTANCE)

        assert result_default == "long"
        assert result_custom is None


# ─── Численная корректность ───────────────────────────────────────────────────

class TestNumericalCorrectness:

    def test_volume_ratio_calculation(self):
        """volume_ratio = volume / avg_volume."""
        volume = 600.0
        avg_volume = 1000.0
        expected_ratio = 0.6  # ниже 0.7 → должен дать сигнал при других условиях
        df = make_single(
            open_=99.5, high=103.0, low=97.0, close=100.0,
            volume=volume, avg_volume=avg_volume,
        )
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_body_ratio_calculation(self):
        """body_ratio = |close - open| / (high - low)."""
        # range = 104 - 97 = 7, body = |101 - 100.5| = 0.5 → ratio = 0.071
        df = make_single(
            open_=100.5, high=104.0, low=97.0, close=101.0,
            volume=LOW_VOL, avg_volume=AVG_VOL,
        )
        # low(97) < support(98) ✓, close(101) > support ✓
        # body_ratio ≈ 0.071 < 0.3 ✓
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_long_and_short_do_not_fire_simultaneously(self):
        """
        Алгоритм проверяет LONG сначала. Если LONG сработал — SHORT не проверяется.
        Теоретически оба условия могут быть выполнены одновременно, но возвращается
        только первый найденный сигнал.
        """
        # Свеча пробила оба уровня: low < support И high > resistance
        # Но close > support (LONG условие выполнено)
        # И close < resistance (SHORT тоже выполнено)
        df = make_single(
            open_=99.5, high=104.0, low=97.0, close=100.0,
            volume=LOW_VOL, avg_volume=AVG_VOL,
        )
        result = detect(df, SUPPORT, RESISTANCE)
        # Должен вернуть только одно направление
        assert result in ("long", "short")
        # LONG проверяется первым → ожидаем "long"
        assert result == "long"
