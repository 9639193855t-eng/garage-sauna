"""
strategies/base_3candles.py — Стратегия «База 3 свечи». Приоритет 3.

Логика Герчика:
    Три последовательные свечи подряд касаются уровня поддержки или
    сопротивления, не пробивая его. Рынок «тестирует» уровень и каждый раз
    отталкивается. После трёх касаний уровень подтверждён — он держит.
    Четвёртая свеча (сигнальная) начинает движение от уровня.

Признаки паттерна:
    1. Три последовательные свечи касаются уровня в пределах touch_threshold %.
    2. Каждая из трёх закрывается по правильную сторону от уровня
       (не пробивает его насквозь).
    3. Сигнальная свеча (текущая) закрылась в нужную сторону от уровня.
    4. Объём сигнальной свечи выше среднего × volume_confirm_threshold
       (подтверждает что за движением стоят крупные игроки).

Намеренно НЕ используется:
    - MA50 и другие индикаторы: Герчик не использует скользящие средние
      на 5-минутных таймфреймах, только ценовые уровни.

Интерфейс:
    detect(df, support, resistance) -> Optional[str]
    — Возвращает "long", "short" или None.
"""

import logging
from typing import Optional

import pandas as pd

from config import BASE_CONFIG

log = logging.getLogger(__name__)

# Минимальное количество свечей для работы стратегии
# (n касаний + 1 сигнальная + запас для avg_volume)
_MIN_DATA_LEN = 20

# Цена должна быть не дальше этой доли от уровня, иначе паттерн нерелевантен
_MAX_DIST_FROM_LEVEL = 0.02


# ─────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def _touches_support(candle: pd.Series, support: float, threshold: float) -> bool:
    """
    Свеча касается уровня поддержки?

    Касание: low попал в зону [support × (1 - threshold), support × (1 + threshold)]
    и свеча закрылась ВЫШЕ уровня (не пробила насквозь).

    Аргументы:
        candle:    Строка DataFrame (одна свеча).
        support:   Уровень поддержки.
        threshold: Допустимое отклонение (доля от цены).

    Возвращает:
        True если касание подтверждено.
    """
    if support <= 0:
        return False
    low_near  = abs(float(candle["low"]) - support) / support <= threshold
    close_ok  = float(candle["close"]) >= support
    return low_near and close_ok


def _touches_resistance(candle: pd.Series, resistance: float, threshold: float) -> bool:
    """
    Свеча касается уровня сопротивления?

    Касание: high попал в зону уровня, свеча закрылась НИЖЕ уровня.

    Аргументы:
        candle:     Строка DataFrame (одна свеча).
        resistance: Уровень сопротивления.
        threshold:  Допустимое отклонение (доля от цены).

    Возвращает:
        True если касание подтверждено.
    """
    if resistance <= 0:
        return False
    high_near = abs(float(candle["high"]) - resistance) / resistance <= threshold
    close_ok  = float(candle["close"]) <= resistance
    return high_near and close_ok


def _count_touches(
    candles: pd.DataFrame,
    level: float,
    role: str,
    threshold: float,
) -> int:
    """
    Посчитать количество касаний уровня в наборе свечей (векторизованно).

    Аргументы:
        candles:   DataFrame с несколькими свечами.
        level:     Ценовой уровень.
        role:      "support" или "resistance".
        threshold: Допустимое отклонение.

    Возвращает:
        Количество касаний (int).
    """
    if level <= 0 or candles is None or candles.empty:
        return 0

    if role == "support":
        near_mask  = (candles["low"] - level).abs() / level <= threshold
        close_mask = candles["close"] >= level
    else:
        near_mask  = (candles["high"] - level).abs() / level <= threshold
        close_mask = candles["close"] <= level

    return int((near_mask & close_mask).sum())


def _touches_are_consecutive(
    candles: pd.DataFrame,
    level: float,
    role: str,
    threshold: float,
    max_gap: int = 2,
) -> bool:
    """
    Проверить что касания уровня образуют кластер без больших разрывов.

    По Герчику "база" — это серия близких касаний: цена раз за разом
    возвращается к уровню, не давая ему "остыть". Разрыв > max_gap свечей
    между двумя касаниями означает что рынок ушёл от уровня и база сломалась.

    Аргументы:
        candles:   DataFrame (расширенное окно, 2×n свечей перед сигнальной).
        level:     Ценовой уровень.
        role:      "support" или "resistance".
        threshold: Допустимое отклонение.
        max_gap:   Максимальный разрыв (свечей без касания) между двумя касаниями.

    Возвращает:
        True если все последовательные пары касаний имеют зазор ≤ max_gap.
        True если касаний меньше 2 (нечего проверять).
    """
    if level <= 0 or candles is None or candles.empty:
        return False

    if role == "support":
        mask = (
            ((candles["low"] - level).abs() / level <= threshold)
            & (candles["close"] >= level)
        )
    else:
        mask = (
            ((candles["high"] - level).abs() / level <= threshold)
            & (candles["close"] <= level)
        )

    # Позиции касаний в локальной нумерации (0..len-1)
    positions = [i for i, v in enumerate(mask.values) if v]

    if len(positions) < 2:
        return True  # 0 или 1 касание — зазор проверять не нужно

    # Зазор = число свечей БЕЗ касания между двумя последовательными касаниями
    for i in range(1, len(positions)):
        gap = positions[i] - positions[i - 1] - 1   # -1 т.к. считаем промежуток
        if gap > max_gap:
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def detect(
    df: pd.DataFrame,
    support: Optional[float],
    resistance: Optional[float],
) -> Optional[str]:
    """
    Обнаружить сигнал стратегии «База 3 свечи». Приоритет 3.

    Проверяет последние BASE_CONFIG.candles_required + 1 свечей:
    три базовые + одна сигнальная (df.iloc[-1]).

    Аргументы:
        df:         DataFrame со свечами (open, high, low, close, volume, avg_volume).
                    Минимум _MIN_DATA_LEN строк.
        support:    Уровень поддержки из levels.find_levels().
        resistance: Уровень сопротивления из levels.find_levels().

    Возвращает:
        "long"  — три касания поддержки, сигнальная свеча уходит вверх.
        "short" — три касания сопротивления, сигнальная свеча уходит вниз.
        None    — паттерн не обнаружен.

    Структура свечей:
        df.iloc[-4]  ─┐
        df.iloc[-3]  ─┤ base_candles (3 касания уровня)
        df.iloc[-2]  ─┘
        df.iloc[-1]    signal_candle (сигнальная, подтверждает отбой)

    Пример:
        signal = detect(df, support=41000.0, resistance=44000.0)
        if signal == "long":
            params = calculate_trade_params("long", df.iloc[-1]["close"], support, balance)
    """
    # ── Базовые проверки ─────────────────────────────────────────────────────
    if df is None or len(df) < _MIN_DATA_LEN:
        log.debug("База3Свечи: недостаточно данных (%d)", len(df) if df is not None else 0)
        return None

    if support is None or resistance is None:
        log.debug("База3Свечи: уровни не определены")
        return None

    if support <= 0 or resistance <= 0 or resistance <= support:
        log.debug("База3Свечи: некорректные уровни support=%.4f resistance=%.4f",
                  support, resistance)
        return None

    # ── Фильтр флэта: диапазон должен быть значимым ─────────────────────────
    range_ratio = (resistance - support) / support
    if range_ratio < 0.005:
        log.debug("База3Свечи: флэт (диапазон %.3f%% < 0.5%%)", range_ratio * 100)
        return None

    # ── Получаем свечи ────────────────────────────────────────────────────────
    signal_candle = df.iloc[-1]

    # Проверяем объём сигнальной свечи
    sig_vol = float(signal_candle.get("volume", 0) or 0)
    if sig_vol <= 0:
        log.debug("База3Свечи: нулевой объём сигнальной свечи")
        return None

    # avg_volume: берём из DataFrame или считаем на месте
    avg_vol_raw = signal_candle.get("avg_volume", None)
    if avg_vol_raw is None or pd.isna(avg_vol_raw) or float(avg_vol_raw) <= 0:
        # Считаем среднее по предыдущим свечам (без последней)
        avg_vol = float(df["volume"].iloc[:-1].mean())
    else:
        avg_vol = float(avg_vol_raw)

    if avg_vol <= 0:
        log.debug("База3Свечи: avg_volume недоступен")
        return None

    # ── Фильтр расстояния: цена должна быть вблизи одного из уровней ────────
    sig_close = float(signal_candle["close"])
    dist_support    = abs(sig_close - support)    / support
    dist_resistance = abs(sig_close - resistance) / resistance

    if dist_support > _MAX_DIST_FROM_LEVEL and dist_resistance > _MAX_DIST_FROM_LEVEL:
        log.debug(
            "База3Свечи: цена %.4f далеко от уровней (dist_sup=%.2f%% dist_res=%.2f%%)",
            sig_close, dist_support * 100, dist_resistance * 100,
        )
        return None

    # ── Базовые свечи ────────────────────────────────────────────────────────
    n = BASE_CONFIG.candles_required
    # Расширенное окно: 2×n свечей до сигнальной — ищем кластер касаний
    extended_n = 2 * n
    if len(df) < extended_n + 1:
        return None

    base_candles = df.iloc[-(extended_n + 1):-1]   # последние 2n свечей до сигнальной

    # ── LONG: касания поддержки (кластером) ──────────────────────────────────
    long_touches = _count_touches(
        base_candles, support, "support", BASE_CONFIG.touch_threshold
    )

    if long_touches >= n:
        # Проверяем что касания идут кластером — зазор не более 2 свечей
        consecutive_ok = _touches_are_consecutive(
            base_candles, support, "support", BASE_CONFIG.touch_threshold
        )

        # Сигнальная свеча закрылась ВЫШЕ поддержки (отбой подтверждён)
        signal_close_ok = sig_close > support

        # Объём сигнальной свечи >= avg × volume_confirm_threshold
        volume_ok = sig_vol >= avg_vol * BASE_CONFIG.volume_confirm_threshold

        if signal_close_ok and volume_ok and consecutive_ok:
            log.info(
                "[БАЗА 3 СВЕЧИ] LONG | support=%.4f | касаний=%d | "
                "vol=%.0f (avg=%.0f, порог=%.0f)",
                support, long_touches, sig_vol, avg_vol,
                avg_vol * BASE_CONFIG.volume_confirm_threshold,
            )
            return "long"
        else:
            log.debug(
                "База3Свечи: LONG касаний=%d, signal_close_ok=%s volume_ok=%s consecutive_ok=%s",
                long_touches, signal_close_ok, volume_ok, consecutive_ok,
            )

    # ── SHORT: касания сопротивления (кластером) ─────────────────────────────
    short_touches = _count_touches(
        base_candles, resistance, "resistance", BASE_CONFIG.touch_threshold
    )

    if short_touches >= n:
        # Проверяем что касания идут кластером
        consecutive_ok = _touches_are_consecutive(
            base_candles, resistance, "resistance", BASE_CONFIG.touch_threshold
        )

        # Сигнальная свеча закрылась НИЖЕ сопротивления (отбой подтверждён)
        signal_close_ok = sig_close < resistance

        # Объём сигнальной свечи >= avg × volume_confirm_threshold
        volume_ok = sig_vol >= avg_vol * BASE_CONFIG.volume_confirm_threshold

        if signal_close_ok and volume_ok and consecutive_ok:
            log.info(
                "[БАЗА 3 СВЕЧИ] SHORT | resistance=%.4f | касаний=%d | "
                "vol=%.0f (avg=%.0f, порог=%.0f)",
                resistance, short_touches, sig_vol, avg_vol,
                avg_vol * BASE_CONFIG.volume_confirm_threshold,
            )
            return "short"
        else:
            log.debug(
                "База3Свечи: SHORT касаний=%d, signal_close_ok=%s volume_ok=%s consecutive_ok=%s",
                short_touches, signal_close_ok, volume_ok, consecutive_ok,
            )

    log.debug("База3Свечи: паттерн не обнаружен")
    return None
