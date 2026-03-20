"""
strategies/bounce.py — Стратегия «Отбой от уровня». Приоритет 4.

Логика Герчика:
    Рынок подходит к значимому уровню поддержки или сопротивления, касается его
    тенью свечи и сразу отталкивается — тело свечи уходит прочь от уровня.
    Один сильный отбой, без предварительного накопления (в отличие от «Базы 3 свечи»).

Признаки пин-бара (LONG — молоток у поддержки):
    Пин-бар = свеча с ДЛИННЫМ хвостом и МАЛЕНЬКИМ телом.
    1. Нижняя тень касается support: low в зоне ±proximity_threshold от support.
    2. Свеча закрылась ВЫШЕ support (close > support).
    3. Тело направлено ВВЕРХ: close в верхней половине диапазона
       (close >= (high + low) / 2).
    4. Тело МАЛЕНЬКОЕ: body_size / candle_range < pin_bar_body_ratio.
    5. Нижняя тень (к уровню) ДЛИННАЯ:
       lower_wick / candle_range > pin_bar_wick_ratio.
    6. Объём подтверждает: volume >= avg_volume * volume_threshold.
    7. Цена закрытия не слишком далеко от уровня:
       |close - support| / support <= proximity_threshold * 5.

Признаки пин-бара (SHORT — висельник у сопротивления):
    Пин-бар = свеча с ДЛИННЫМ хвостом и МАЛЕНЬКИМ телом.
    1. Верхняя тень касается resistance: high в зоне ±proximity_threshold.
    2. Свеча закрылась НИЖЕ resistance (close < resistance).
    3. Тело направлено ВНИЗ: close в нижней половине диапазона
       (close <= (high + low) / 2).
    4. Тело МАЛЕНЬКОЕ: body_size / candle_range < pin_bar_body_ratio.
    5. Верхняя тень (к уровню) ДЛИННАЯ:
       upper_wick / candle_range > pin_bar_wick_ratio.
    6. Объём: volume >= avg_volume * volume_threshold.
    7. |close - resistance| / resistance <= proximity_threshold * 5.

Анализируется только последняя свеча df.iloc[-1].

Интерфейс:
    detect(df, support, resistance) -> Optional[str]
    — Возвращает "long", "short" или None.
"""

import logging
from typing import Optional

import pandas as pd

from config import BOUNCE_CONFIG

log = logging.getLogger(__name__)

# Максимальное расстояние close от уровня — 5 ширин зоны proximity
_CLOSE_DIST_FACTOR = 5


def detect(
    df: pd.DataFrame,
    support: Optional[float],
    resistance: Optional[float],
) -> Optional[str]:
    """
    Обнаружить сигнал стратегии «Отбой от уровня». Приоритет 4.

    Анализирует ПОСЛЕДНЮЮ закрытую свечу (df.iloc[-1]).

    Аргументы:
        df:         DataFrame со свечами (open, high, low, close, volume, avg_volume).
                    Минимум 2 строки.
        support:    Уровень поддержки.
        resistance: Уровень сопротивления.

    Возвращает:
        "long"  — отбой от поддержки вверх.
        "short" — отбой от сопротивления вниз.
        None    — сигнала нет.
    """
    # ── Базовые защитные проверки ─────────────────────────────────────────────
    if df is None or len(df) < 2:
        log.debug("Отбой: недостаточно данных")
        return None

    if support is None or resistance is None:
        log.debug("Отбой: уровни не определены")
        return None

    if support <= 0 or resistance <= 0 or resistance <= support:
        log.debug(
            "Отбой: некорректные уровни support=%.4f resistance=%.4f",
            support, resistance,
        )
        return None

    candle = df.iloc[-1]

    # ── Проверяем avg_volume ──────────────────────────────────────────────────
    avg_volume = candle.get("avg_volume", None)
    if avg_volume is None or pd.isna(avg_volume) or float(avg_volume) <= 0:
        log.debug("Отбой: avg_volume недоступен или равен нулю")
        return None

    avg_volume = float(avg_volume)

    # ── Геометрия свечи ───────────────────────────────────────────────────────
    c_open   = float(candle["open"])
    c_close  = float(candle["close"])
    c_high   = float(candle["high"])
    c_low    = float(candle["low"])
    c_volume = float(candle["volume"])

    candle_range = c_high - c_low
    if candle_range <= 0:
        log.debug("Отбой: нулевой диапазон свечи (high == low)")
        return None

    body_size   = abs(c_close - c_open)
    body_ratio  = body_size / candle_range
    lower_wick  = min(c_open, c_close) - c_low    # тень снизу
    upper_wick  = c_high - max(c_open, c_close)   # тень сверху
    mid_price   = (c_high + c_low) / 2.0

    vol_ok = c_volume >= avg_volume * BOUNCE_CONFIG.volume_threshold
    body_ok = body_ratio < BOUNCE_CONFIG.pin_bar_body_ratio

    prox = BOUNCE_CONFIG.proximity_threshold
    max_close_dist = prox * _CLOSE_DIST_FACTOR

    # ── Отбой от ПОДДЕРЖКИ → сигнал LONG ─────────────────────────────────────
    low_touches_support  = abs(c_low - support) / support <= prox
    close_above_support  = c_close > support
    close_upper_half     = c_close >= mid_price
    lower_wick_ok        = (lower_wick / candle_range) > BOUNCE_CONFIG.pin_bar_wick_ratio
    close_near_support   = abs(c_close - support) / support <= max_close_dist

    if (low_touches_support
            and close_above_support
            and close_upper_half
            and body_ok
            and lower_wick_ok
            and vol_ok
            and close_near_support):

        log.info(
            "[ОТБОЙ] LONG | support=%.4f | low=%.4f | close=%.4f | "
            "body_ratio=%.2f | lower_wick/range=%.2f | vol=%.0f (avg×thr=%.0f)",
            support, c_low, c_close,
            body_ratio, lower_wick / candle_range,
            c_volume, avg_volume * BOUNCE_CONFIG.volume_threshold,
        )
        return "long"

    # ── Отбой от СОПРОТИВЛЕНИЯ → сигнал SHORT ────────────────────────────────
    high_touches_res   = abs(c_high - resistance) / resistance <= prox
    close_below_res    = c_close < resistance
    close_lower_half   = c_close <= mid_price
    upper_wick_ok      = (upper_wick / candle_range) > BOUNCE_CONFIG.pin_bar_wick_ratio
    close_near_res     = abs(c_close - resistance) / resistance <= max_close_dist

    if (high_touches_res
            and close_below_res
            and close_lower_half
            and body_ok
            and upper_wick_ok
            and vol_ok
            and close_near_res):

        log.info(
            "[ОТБОЙ] SHORT | resistance=%.4f | high=%.4f | close=%.4f | "
            "body_ratio=%.2f | upper_wick/range=%.2f | vol=%.0f (avg×thr=%.0f)",
            resistance, c_high, c_close,
            body_ratio, upper_wick / candle_range,
            c_volume, avg_volume * BOUNCE_CONFIG.volume_threshold,
        )
        return "short"

    log.debug("Отбой: паттерн не обнаружен")
    return None
