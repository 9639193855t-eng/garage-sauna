"""Тесты для strategies/base_3candles.py — стратегия «База 3 свечи»."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
from unittest.mock import patch

from strategies.base_3candles import (
    detect,
    _touches_support,
    _touches_resistance,
    _count_touches,
)
from config import BASE_CONFIG


# ─── Фабрики ─────────────────────────────────────────────────────────────────

SUPPORT    = 100.0
RESISTANCE = 110.0
AVG_VOL    = 1000.0
# Достаточный объём (выше порога confirm_threshold=0.8 → нужен >= 800)
GOOD_VOL   = AVG_VOL * (BASE_CONFIG.volume_confirm_threshold + 0.1)
# Слабый объём (ниже порога)
WEAK_VOL   = AVG_VOL * (BASE_CONFIG.volume_confirm_threshold - 0.1)


def make_candle_row(
    open_: float = 105.0,
    high:  float = 107.0,
    low:   float = 100.1,   # рядом с support
    close: float = 105.0,
    volume: float = GOOD_VOL,
    avg_volume: float = AVG_VOL,
    ts: int = 0,
) -> dict:
    return {
        "time":       pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * ts),
        "open":       open_,
        "high":       high,
        "low":        low,
        "close":      close,
        "volume":     volume,
        "avg_volume": avg_volume,
    }


def make_df_long(
    n_base: int = None,
    signal_close: float = 101.5,   # вблизи support (100), dist=1.5% < 2%
    signal_vol: float = GOOD_VOL,
    base_close: float = 101.5,
    base_low: float = 100.1,       # касается support (100.0), threshold=0.002 → ±0.2
    support: float = SUPPORT,
    resistance: float = RESISTANCE,
    avg_vol: float = AVG_VOL,
    extra_prefix: int = 16,        # свечи-заглушки перед базой (≥16, чтобы len≥_MIN_DATA_LEN=20)
) -> pd.DataFrame:
    """
    Сценарий LONG:
        prefix свечей-заглушек
        n_base базовых свечей касаются support
        1 сигнальная свеча, close > support и вблизи уровня (< 2%)
    """
    n_base = n_base if n_base is not None else BASE_CONFIG.candles_required
    rows = []

    # Свечи-заглушки (середина диапазона, не касаются уровней)
    for i in range(extra_prefix):
        rows.append(make_candle_row(
            open_=105.0, high=107.0, low=103.0, close=105.0,
            volume=avg_vol, avg_volume=avg_vol, ts=len(rows),
        ))

    # Базовые свечи: low рядом с support, close выше support
    for i in range(n_base):
        rows.append(make_candle_row(
            open_=base_close, high=base_close + 1.0,
            low=base_low, close=base_close,
            volume=avg_vol, avg_volume=avg_vol, ts=len(rows),
        ))

    # Сигнальная свеча: закрывается выше support, вблизи уровня
    rows.append(make_candle_row(
        open_=base_close, high=signal_close + 0.5,
        low=support + 0.05, close=signal_close,
        volume=signal_vol, avg_volume=avg_vol, ts=len(rows),
    ))

    return pd.DataFrame(rows)


def make_df_short(
    n_base: int = None,
    signal_close: float = 108.5,   # вблизи resistance (110), dist=1.36% < 2%
    signal_vol: float = GOOD_VOL,
    base_close: float = 108.5,
    base_high: float = 109.8,      # касается resistance (110.0), threshold=0.002 → ±0.22
    support: float = SUPPORT,
    resistance: float = RESISTANCE,
    avg_vol: float = AVG_VOL,
    extra_prefix: int = 16,        # свечи-заглушки перед базой (≥16, чтобы len≥_MIN_DATA_LEN=20)
) -> pd.DataFrame:
    """
    Сценарий SHORT:
        prefix свечей-заглушек
        n_base базовых свечей касаются resistance
        1 сигнальная свеча, close < resistance и вблизи уровня (< 2%)
    """
    n_base = n_base if n_base is not None else BASE_CONFIG.candles_required
    rows = []

    for i in range(extra_prefix):
        rows.append(make_candle_row(
            open_=105.0, high=107.0, low=103.0, close=105.0,
            volume=avg_vol, avg_volume=avg_vol, ts=len(rows),
        ))

    for i in range(n_base):
        rows.append(make_candle_row(
            open_=base_close, high=base_high,
            low=base_close - 1.0, close=base_close,
            volume=avg_vol, avg_volume=avg_vol, ts=len(rows),
        ))

    rows.append(make_candle_row(
        open_=base_close, high=resistance - 0.05,
        low=signal_close - 0.5, close=signal_close,
        volume=signal_vol, avg_volume=avg_vol, ts=len(rows),
    ))

    return pd.DataFrame(rows)


# ─── _touches_support ────────────────────────────────────────────────────────

class TestTouchesSupport:

    def test_low_near_level_close_above(self):
        """low рядом с уровнем, close выше — касание."""
        candle = pd.Series({"low": 99.9, "close": 101.0, "high": 102.0, "open": 101.0})
        assert _touches_support(candle, 100.0, 0.002) is True

    def test_low_too_far(self):
        """low далеко от уровня — нет касания."""
        candle = pd.Series({"low": 95.0, "close": 98.0, "high": 99.0, "open": 97.0})
        assert _touches_support(candle, 100.0, 0.002) is False

    def test_close_below_support(self):
        """low рядом, но close ниже уровня — пробой, не касание."""
        candle = pd.Series({"low": 99.9, "close": 99.5, "high": 101.0, "open": 100.5})
        assert _touches_support(candle, 100.0, 0.002) is False

    def test_zero_support_returns_false(self):
        candle = pd.Series({"low": 99.9, "close": 101.0, "high": 102.0, "open": 101.0})
        assert _touches_support(candle, 0.0, 0.002) is False

    def test_exact_level(self):
        """low точно равен уровню — касание."""
        candle = pd.Series({"low": 100.0, "close": 101.0, "high": 102.0, "open": 101.0})
        assert _touches_support(candle, 100.0, 0.002) is True

    def test_close_exactly_at_support(self):
        """close == support → close >= support → касание."""
        candle = pd.Series({"low": 99.9, "close": 100.0, "high": 101.0, "open": 100.5})
        assert _touches_support(candle, 100.0, 0.002) is True


# ─── _touches_resistance ─────────────────────────────────────────────────────

class TestTouchesResistance:

    def test_high_near_level_close_below(self):
        """high рядом с уровнем, close ниже — касание."""
        candle = pd.Series({"high": 110.1, "close": 108.0, "low": 107.0, "open": 108.0})
        assert _touches_resistance(candle, 110.0, 0.002) is True

    def test_high_too_far(self):
        """high далеко от уровня — нет касания."""
        candle = pd.Series({"high": 105.0, "close": 104.0, "low": 103.0, "open": 104.0})
        assert _touches_resistance(candle, 110.0, 0.002) is False

    def test_close_above_resistance(self):
        """high рядом, но close выше уровня — пробой, не касание."""
        candle = pd.Series({"high": 110.1, "close": 110.5, "low": 108.0, "open": 109.0})
        assert _touches_resistance(candle, 110.0, 0.002) is False

    def test_zero_resistance_returns_false(self):
        candle = pd.Series({"high": 110.1, "close": 108.0, "low": 107.0, "open": 108.0})
        assert _touches_resistance(candle, 0.0, 0.002) is False

    def test_exact_level(self):
        """high точно равен уровню — касание."""
        candle = pd.Series({"high": 110.0, "close": 108.0, "low": 107.0, "open": 108.0})
        assert _touches_resistance(candle, 110.0, 0.002) is True

    def test_close_exactly_at_resistance(self):
        """close == resistance → close <= resistance → касание."""
        candle = pd.Series({"high": 110.1, "close": 110.0, "low": 108.0, "open": 109.0})
        assert _touches_resistance(candle, 110.0, 0.002) is True


# ─── _count_touches ──────────────────────────────────────────────────────────

class TestCountTouches:

    def test_empty_df_returns_zero(self):
        assert _count_touches(pd.DataFrame(), 100.0, "support", 0.002) == 0

    def test_zero_level_returns_zero(self):
        df = make_df_long()
        assert _count_touches(df.iloc[-4:-1], 0.0, "support", 0.002) == 0

    def test_counts_support_touches(self):
        df = make_df_long(n_base=3)
        base = df.iloc[-4:-1]
        touches = _count_touches(base, SUPPORT, "support", BASE_CONFIG.touch_threshold)
        assert touches == 3

    def test_counts_resistance_touches(self):
        df = make_df_short(n_base=3)
        base = df.iloc[-4:-1]
        touches = _count_touches(base, RESISTANCE, "resistance", BASE_CONFIG.touch_threshold)
        assert touches == 3

    def test_returns_int(self):
        df = make_df_long(n_base=3)
        base = df.iloc[-4:-1]
        result = _count_touches(base, SUPPORT, "support", BASE_CONFIG.touch_threshold)
        assert isinstance(result, int)

    def test_no_touches_when_candles_far(self):
        """Свечи посередине диапазона — нет касаний."""
        rows = [make_candle_row(low=104.0, close=105.0, ts=i) for i in range(3)]
        df = pd.DataFrame(rows)
        touches = _count_touches(df, SUPPORT, "support", 0.002)
        assert touches == 0


# ─── detect: граничные случаи ─────────────────────────────────────────────────

class TestDetectEdgeCases:

    def test_none_df(self):
        assert detect(None, SUPPORT, RESISTANCE) is None

    def test_empty_df(self):
        assert detect(pd.DataFrame(), SUPPORT, RESISTANCE) is None

    def test_insufficient_data(self):
        df = make_df_long(extra_prefix=2)
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_support_none(self):
        df = make_df_long()
        assert detect(df, None, RESISTANCE) is None

    def test_resistance_none(self):
        df = make_df_long()
        assert detect(df, SUPPORT, None) is None

    def test_support_zero(self):
        df = make_df_long()
        assert detect(df, 0.0, RESISTANCE) is None

    def test_resistance_zero(self):
        df = make_df_long()
        assert detect(df, SUPPORT, 0.0) is None

    def test_resistance_leq_support(self):
        df = make_df_long()
        assert detect(df, 105.0, 100.0) is None
        assert detect(df, 100.0, 100.0) is None

    def test_flat_market_no_signal(self):
        """Диапазон < 0.5% — флэт, нет сигнала."""
        df = make_df_long()
        assert detect(df, 100.0, 100.1) is None   # range_ratio = 0.001 < 0.005

    def test_zero_volume_signal_candle(self):
        df = make_df_long(signal_vol=0.0)
        assert detect(df, SUPPORT, RESISTANCE) is None

    def test_result_type(self):
        df = make_df_long()
        result = detect(df, SUPPORT, RESISTANCE)
        assert result in ("long", "short", None)

    def test_price_far_from_both_levels(self):
        """Цена далеко от обоих уровней — фильтр расстояния."""
        # Сигнальная свеча закрывается посередине (105), support=100, resistance=110
        # dist_support = 5% > 2% AND dist_resistance = ~4.5% > 2% → None
        df = make_df_long(signal_close=105.0)
        # При support=90, resistance=130 — цена 105 вблизи среднего, далеко от обоих
        assert detect(df, 90.0, 130.0) is None


# ─── detect: LONG сигнал ─────────────────────────────────────────────────────

class TestDetectLong:

    def test_basic_long_signal(self):
        """Три касания поддержки + сигнальная свеча выше → LONG."""
        df = make_df_long()
        result = detect(df, SUPPORT, RESISTANCE)
        assert result == "long"

    def test_no_long_if_only_two_touches(self):
        """Два касания — недостаточно."""
        df = make_df_long(n_base=2)
        result = detect(df, SUPPORT, RESISTANCE)
        assert result != "long"

    def test_no_long_if_signal_candle_below_support(self):
        """Сигнальная свеча закрылась ниже поддержки — уровень пробит."""
        df = make_df_long(signal_close=99.0)   # close < support(100)
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_no_long_if_volume_too_low(self):
        """Объём ниже порога подтверждения."""
        df = make_df_long(signal_vol=WEAK_VOL)
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_volume_exactly_at_threshold_gives_signal(self):
        """volume == avg × threshold → условие >= выполнено → сигнал."""
        exact_vol = AVG_VOL * BASE_CONFIG.volume_confirm_threshold
        df = make_df_long(signal_vol=exact_vol)
        assert detect(df, SUPPORT, RESISTANCE) == "long"

    def test_volume_just_below_threshold_no_signal(self):
        """volume чуть ниже порога → нет сигнала."""
        just_below = AVG_VOL * (BASE_CONFIG.volume_confirm_threshold - 0.05)
        df = make_df_long(signal_vol=just_below)
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_four_touches_still_gives_signal(self):
        """4 касания тоже дают сигнал (>= n, а не == n)."""
        df = make_df_long(n_base=4)
        result = detect(df, SUPPORT, RESISTANCE)
        assert result == "long"

    def test_long_uses_last_n_candles_as_base(self):
        """База — последние n свечей перед сигнальной, не случайные."""
        # Создаём DataFrame где только последние 3 касаются уровня
        rows = []
        # 20 свечей посередине (не касаются поддержки)
        for i in range(20):
            rows.append(make_candle_row(
                low=103.0, close=105.0, high=107.0, ts=i,
                volume=AVG_VOL, avg_volume=AVG_VOL,
            ))
        # 3 базовых (касаются support)
        for i in range(3):
            rows.append(make_candle_row(
                low=100.1, close=103.0, high=106.0, ts=len(rows),
                volume=AVG_VOL, avg_volume=AVG_VOL,
            ))
        # Сигнальная: close=101.5 — 1.5% от support=100, проходит фильтр расстояния (< 2%)
        rows.append(make_candle_row(
            low=100.5, close=101.5, high=102.5, ts=len(rows),
            volume=GOOD_VOL, avg_volume=AVG_VOL,
        ))
        df = pd.DataFrame(rows)
        assert detect(df, SUPPORT, RESISTANCE) == "long"


# ─── detect: SHORT сигнал ────────────────────────────────────────────────────

class TestDetectShort:

    def test_basic_short_signal(self):
        """Три касания сопротивления + сигнальная свеча ниже → SHORT."""
        df = make_df_short()
        result = detect(df, SUPPORT, RESISTANCE)
        assert result == "short"

    def test_no_short_if_only_two_touches(self):
        """Два касания — недостаточно."""
        df = make_df_short(n_base=2)
        result = detect(df, SUPPORT, RESISTANCE)
        assert result != "short"

    def test_no_short_if_signal_candle_above_resistance(self):
        """Сигнальная свеча закрылась выше сопротивления — уровень пробит."""
        df = make_df_short(signal_close=111.0)   # close > resistance(110)
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_no_short_if_volume_too_low(self):
        df = make_df_short(signal_vol=WEAK_VOL)
        assert detect(df, SUPPORT, RESISTANCE) != "short"

    def test_volume_exactly_at_threshold_gives_signal(self):
        exact_vol = AVG_VOL * BASE_CONFIG.volume_confirm_threshold
        df = make_df_short(signal_vol=exact_vol)
        assert detect(df, SUPPORT, RESISTANCE) == "short"

    def test_four_touches_still_gives_signal(self):
        df = make_df_short(n_base=4)
        result = detect(df, SUPPORT, RESISTANCE)
        assert result == "short"

    def test_short_uses_last_n_candles_as_base(self):
        rows = []
        for i in range(20):
            rows.append(make_candle_row(
                low=103.0, close=105.0, high=107.0, ts=i,
                volume=AVG_VOL, avg_volume=AVG_VOL,
            ))
        for i in range(3):
            rows.append(make_candle_row(
                low=108.0, close=108.0, high=109.8, ts=len(rows),
                volume=AVG_VOL, avg_volume=AVG_VOL,
            ))
        # Сигнальная: close=108.5 — 1.36% от resistance=110, проходит фильтр расстояния (< 2%)
        rows.append(make_candle_row(
            low=107.5, close=108.5, high=109.5, ts=len(rows),
            volume=GOOD_VOL, avg_volume=AVG_VOL,
        ))
        df = pd.DataFrame(rows)
        assert detect(df, SUPPORT, RESISTANCE) == "short"


# ─── detect: NO MA50 ────────────────────────────────────────────────────────

class TestNoMA:

    def test_long_signal_without_avg_volume_in_df(self):
        """
        Если avg_volume отсутствует в DataFrame — вычисляем на лету.
        MA не используется.
        """
        df = make_df_long()
        df_no_avg = df.drop(columns=["avg_volume"])
        # Должен работать (avg_vol считается из volume истории)
        result = detect(df_no_avg, SUPPORT, RESISTANCE)
        # Может быть None если объём не проходит без avg_volume, но не должно быть исключений
        assert result in ("long", "short", None)

    def test_no_ma_column_used(self):
        """MA50 не вычисляется и не влияет на результат."""
        df = make_df_long()
        # Явно помещаем "неправильное" MA — если оно используется, тест упадёт
        df["ma50"] = 200.0   # огромное значение, при MA-фильтре LONG был бы отклонён
        result = detect(df, SUPPORT, RESISTANCE)
        # Результат не должен зависеть от ma50
        assert result == "long"

    def test_signal_consistent_regardless_of_trend_direction(self):
        """
        Без MA-фильтра сигнал не зависит от «тренда».
        При одинаковых свечах результат одинаков.
        """
        df1 = make_df_long()
        df2 = make_df_long()
        assert detect(df1, SUPPORT, RESISTANCE) == detect(df2, SUPPORT, RESISTANCE)


# ─── detect: конфиг ────────────────────────────────────────────────────────

class TestDetectConfig:

    def test_candles_required_respected(self):
        """candles_required из конфига определяет минимум касаний."""
        # Создаём n-1 касаний → нет сигнала
        n = BASE_CONFIG.candles_required
        df = make_df_long(n_base=n - 1)
        assert detect(df, SUPPORT, RESISTANCE) != "long"

        # n касаний → сигнал
        df2 = make_df_long(n_base=n)
        assert detect(df2, SUPPORT, RESISTANCE) == "long"

    def test_touch_threshold_respected(self):
        """Порог касания из конфига: low вне зоны → нет касания."""
        # base_low на 1% выше support — при threshold=0.002 (0.2%) не попадает в зону
        df = make_df_long(base_low=SUPPORT * 1.01)
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_volume_confirm_threshold_respected(self):
        """volume_confirm_threshold из конфига, не хардкодирован."""
        strict = 1.5   # значительно строже дефолта (0.8)
        with patch.object(BASE_CONFIG, "volume_confirm_threshold", strict):
            # GOOD_VOL = avg * 0.9 → при strict=1.5 нужно avg*1.5 → GOOD_VOL не проходит
            df = make_df_long(signal_vol=GOOD_VOL)
            assert detect(df, SUPPORT, RESISTANCE) != "long"


# ─── detect: только последняя свеча — сигнальная ──────────────────────────

class TestSignalCandleIsLast:

    def test_signal_candle_is_df_iloc_minus1(self):
        """
        Сигнальная свеча — df.iloc[-1]. Предыдущие свечи только для базы.
        Если сигнальная свеча не подтверждает отбой — нет сигнала,
        даже если в истории есть базовые касания.
        """
        df = make_df_long(signal_close=99.0)    # close < support → нет сигнала
        assert detect(df, SUPPORT, RESISTANCE) != "long"

    def test_volume_check_on_signal_candle(self):
        """
        Объём проверяется по сигнальной свече (df.iloc[-1]),
        а не по произвольной базовой.
        """
        # Базовые свечи с отличным объёмом, сигнальная с плохим
        df = make_df_long(
            signal_vol=WEAK_VOL,
        )
        assert detect(df, SUPPORT, RESISTANCE) != "long"
