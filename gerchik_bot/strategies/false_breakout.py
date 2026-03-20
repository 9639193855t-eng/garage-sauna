"""
strategies/false_breakout.py — Стратегия «Ложный пробой». Приоритет 2.

Логика Герчика:
    Рынок пробивает значимый уровень поддержки или сопротивления, но не
    «уходит» за него — свеча закрывается обратно по другую сторону уровня.
    Это ловушка для слабых игроков: те, кто поставил на пробой, теряют деньги,
    а рынок разворачивается в обратную сторону.

Признаки ложного пробоя:
    1. Тень свечи пробила уровень (low < support ИЛИ high > resistance).
    2. Свеча закрылась по правильную сторону (close > support ИЛИ close < resistance).
    3. Объём СЛАБЫЙ — ниже avg_volume × volume_threshold.
       Сильный объём = реальный пробой, слабый = ложный.
    4. Тело свечи МАЛЕНЬКОЕ относительно диапазона high–low.
       Большое тело = направленное движение, маленькое = нерешительность.
    5. Объём сигнальной свечи УБЫВАЮЩИЙ — меньше объёма предыдущей на 10%.
       Убывающий объём подтверждает угасание импульса пробоя.

Интерфейс:
    detect(df, support, resistance) -> Optional[str]
    — Возвращает "long", "short" или None.
"""

import logging
from typing import Optional

import pandas as pd

from config import FB_CONFIG

log = logging.getLogger(__name__)


def detect(
    df: pd.DataFrame,
    support: Optional[float],
    resistance: Optional[float],
) -> Optional[str]:
    """
    Обнаружить сигнал стратегии «Ложный пробой». Приоритет 2.

    Анализирует ПОСЛЕДНЮЮ закрытую свечу (df.iloc[-1]).
    Требует наличия avg_volume в DataFrame (рассчитывается в exchange.get_candles).

    Аргументы:
        df:         DataFrame со свечами (open, high, low, close, volume, avg_volume).
                    Минимум 2 строки.
        support:    Уровень поддержки из levels.find_levels().
        resistance: Уровень сопротивления из levels.find_levels().

    Возвращает:
        "long"  — ложный пробой вниз (пробили поддержку, вернулись выше → покупаем).
        "short" — ложный пробой вверх (пробили сопротивление, вернулись ниже → продаём).
        None    — сигнала нет.

    Алгоритм:
        Ложный пробой ВНИЗ → LONG:
            candle.low  < support     (пробили снизу)
            candle.close > support    (закрылись обратно выше)
            volume_ratio < volume_threshold  (объём слабый)
            body_ratio   < body_threshold    (тело маленькое)

        Ложный пробой ВВЕРХ → SHORT:
            candle.high  > resistance  (пробили сверху)
            candle.close < resistance  (закрылись обратно ниже)
            volume_ratio < volume_threshold
            body_ratio   < body_threshold

    Пример:
        signal = detect(df, support=41000.0, resistance=44000.0)
        if signal == "long":
            params = calculate_trade_params("long", df.iloc[-1]["close"], df.iloc[-1]["low"], balance)
    """
    # ── Базовые проверки ─────────────────────────────────────────────────────
    if df is None or len(df) < 2:
        log.debug("ЛожнойПробой: недостаточно данных")
        return None

    if support is None or resistance is None:
        log.debug("ЛожнойПробой: уровни не определены")
        return None

    if support <= 0 or resistance <= 0 or resistance <= support:
        log.debug(f"ЛожнойПробой: некорректные уровни support={support} resistance={resistance}")
        return None

    candle = df.iloc[-1]

    # ── Проверяем avg_volume ─────────────────────────────────────────────────
    avg_volume = candle.get("avg_volume", None)
    if avg_volume is None or pd.isna(avg_volume) or avg_volume <= 0:
        log.debug("ЛожнойПробой: avg_volume недоступен или равен нулю")
        return None

    # ── Геометрия свечи ──────────────────────────────────────────────────────
    c_open   = float(candle["open"])
    c_close  = float(candle["close"])
    c_high   = float(candle["high"])
    c_low    = float(candle["low"])
    c_volume = float(candle["volume"])

    candle_range = c_high - c_low
    if candle_range <= 0:
        log.debug("ЛожнойПробой: нулевой диапазон свечи")
        return None

    body_size    = abs(c_close - c_open)
    volume_ratio = c_volume / float(avg_volume)
    body_ratio   = body_size / candle_range

    # Объём убывает: сигнальная свеча слабее предыдущей на 10%
    # Подтверждает угасание импульса ложного пробоя
    prev_volume      = float(df.iloc[-2]["volume"])
    volume_decreasing = c_volume < prev_volume * 0.9

    # ── Ложный пробой ВНИЗ → сигнал LONG ────────────────────────────────────
    # Тень пробила поддержку, но свеча закрылась выше неё
    if (c_low    < support
            and c_close  > support
            and volume_ratio      < FB_CONFIG.volume_threshold
            and body_ratio        < FB_CONFIG.body_threshold
            and volume_decreasing):

        log.info(
            f"[ЛОЖНЫЙ ПРОБОЙ] LONG | "
            f"поддержка={support:.4f} | low={c_low:.4f} | close={c_close:.4f} | "
            f"volume_ratio={volume_ratio:.2f} | body_ratio={body_ratio:.2f} | "
            f"vol_prev={prev_volume:.0f}→{c_volume:.0f}"
        )
        return "long"

    # ── Ложный пробой ВВЕРХ → сигнал SHORT ──────────────────────────────────
    # Тень пробила сопротивление, но свеча закрылась ниже него
    if (c_high   > resistance
            and c_close  < resistance
            and volume_ratio      < FB_CONFIG.volume_threshold
            and body_ratio        < FB_CONFIG.body_threshold
            and volume_decreasing):

        log.info(
            f"[ЛОЖНЫЙ ПРОБОЙ] SHORT | "
            f"сопротивление={resistance:.4f} | high={c_high:.4f} | close={c_close:.4f} | "
            f"volume_ratio={volume_ratio:.2f} | body_ratio={body_ratio:.2f} | "
            f"vol_prev={prev_volume:.0f}→{c_volume:.0f}"
        )
        return "short"

    return None
