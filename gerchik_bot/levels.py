"""
levels.py — поиск уровней поддержки и сопротивления по методике Герчика.

Герчик использует явные уровни — реальные максимумы и минимумы цены,
от которых рынок разворачивался. Это не теоретические расчёты,
а конкретные ценовые зоны, которые рынок «помнит».

Функции:
    find_levels(df)               → базовые уровень поддержки и сопротивления
    find_levels_with_strength(df) → уровни с оценкой силы (сколько раз держались)
    is_near_level(price, level)   → цена рядом с уровнем?
    get_level_range(df)           → диапазон между уровнями
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from config import CONFIG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# БАЗОВЫЕ УРОВНИ (High / Low за N свечей)
# ─────────────────────────────────────────────────────────────────────────────

def find_levels(df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    """
    Найти ключевые уровни поддержки и сопротивления.

    Метод: максимум High и минимум Low за последние N закрытых свечей.
    Текущая (незакрытая) свеча намеренно исключается — df.iloc[-1] живая.

    Аргументы:
        df: DataFrame со столбцами open, high, low, close, volume.
            Последняя строка — текущая незакрытая свеча (или последняя закрытая).

    Возвращает:
        (support, resistance) — оба float, или (None, None) при нехватке данных.

    Пример:
        support, resistance = find_levels(df)
        if support:
            print(f"Поддержка: {support:.4f}  Сопротивление: {resistance:.4f}")
    """
    if df is None or len(df) < CONFIG.level_candles + 1:
        log.debug(f"find_levels: недостаточно данных ({len(df) if df is not None else 0} свечей, нужно {CONFIG.level_candles + 1})")
        return None, None

    # Берём N закрытых свечей, исключая последнюю (текущую)
    window = df.iloc[-(CONFIG.level_candles + 1):-1]

    resistance = float(window["high"].max())   # Верхний уровень — сопротивление
    support    = float(window["low"].min())    # Нижний уровень — поддержка

    # Защита: уровни должны быть разными (не совпадать)
    if resistance <= support:
        log.warning(f"find_levels: resistance ({resistance}) <= support ({support}) — данные некорректны")
        return None, None

    log.debug(f"find_levels: поддержка={support:.6f}  сопротивление={resistance:.6f}")
    return support, resistance


def find_levels_extended(
    df: pd.DataFrame,
    n_candles: int | None = None,
) -> tuple[Optional[float], Optional[float]]:
    """
    Уровни с настраиваемой глубиной анализа.
    Позволяет задать любое количество свечей, не зависящее от CONFIG.

    Аргументы:
        df:       DataFrame с данными.
        n_candles: количество свечей для анализа (если None — берётся CONFIG.level_candles).

    Возвращает:
        (support, resistance) или (None, None).
    """
    n = n_candles or CONFIG.level_candles

    if df is None or len(df) < n + 1:
        return None, None

    window = df.iloc[-(n + 1):-1]
    resistance = float(window["high"].max())
    support    = float(window["low"].min())

    if resistance <= support:
        return None, None

    return support, resistance


# ─────────────────────────────────────────────────────────────────────────────
# СИЛА УРОВНЯ (сколько раз цена от него отбивалась)
# ─────────────────────────────────────────────────────────────────────────────

def count_level_touches(
    df: pd.DataFrame,
    level: float,
    threshold_pct: float = 0.002,
    role: str = "support",
) -> int:
    """
    Подсчитать сколько свечей касались уровня (без его пробоя).

    Касание поддержки: low свечи попало в зону level ± threshold_pct,
                       но close остался выше уровня.
    Касание сопротивления: high свечи попало в зону, close остался ниже.

    Аргументы:
        df:            DataFrame со свечами.
        level:         Ценовой уровень.
        threshold_pct: Допустимое отклонение для «касания» (0.2% по умолчанию).
        role:          "support" или "resistance".

    Возвращает:
        Количество касаний (int).
    """
    if df is None or df.empty or level <= 0:
        return 0

    zone_low  = level * (1 - threshold_pct)
    zone_high = level * (1 + threshold_pct)
    touches   = 0

    for _, candle in df.iterrows():
        if role == "support":
            # Хвост свечи достал до уровня, но закрылась выше
            if candle["low"] <= zone_high and candle["close"] > level:
                touches += 1
        elif role == "resistance":
            # Хвост достал до уровня, но закрылась ниже
            if candle["high"] >= zone_low and candle["close"] < level:
                touches += 1

    return touches


def find_levels_with_strength(
    df: pd.DataFrame,
    threshold_pct: float = 0.002,
) -> dict:
    """
    Найти уровни и оценить их силу (количество касаний).

    Возвращает словарь:
        {
            "support":            float,  # цена поддержки
            "resistance":         float,  # цена сопротивления
            "support_touches":    int,    # сколько раз держала
            "resistance_touches": int,    # сколько раз держало
            "support_strong":     bool,   # касаний >= 3 → сильный уровень
            "resistance_strong":  bool,
        }

    Пример использования в стратегии:
        info = find_levels_with_strength(df)
        if info["support_strong"]:
            # уровень поддержки проверен рынком 3+ раза → можно торговать
    """
    result = {
        "support":            None,
        "resistance":         None,
        "support_touches":    0,
        "resistance_touches": 0,
        "support_strong":     False,
        "resistance_strong":  False,
    }

    support, resistance = find_levels(df)
    if support is None:
        return result

    result["support"]    = support
    result["resistance"] = resistance

    # Считаем касания по всем доступным свечам (кроме последней)
    analysis_df = df.iloc[:-1]
    result["support_touches"]    = count_level_touches(analysis_df, support,    threshold_pct, "support")
    result["resistance_touches"] = count_level_touches(analysis_df, resistance, threshold_pct, "resistance")

    # Уровень считается «сильным» если цена касалась его 3+ раза
    result["support_strong"]    = result["support_touches"]    >= 3
    result["resistance_strong"] = result["resistance_touches"] >= 3

    log.debug(
        f"Уровни | поддержка={support:.6f} (касаний: {result['support_touches']}) | "
        f"сопротивление={resistance:.6f} (касаний: {result['resistance_touches']})"
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def is_near_level(
    price: float,
    level: float,
    threshold_pct: float = 0.003,
) -> bool:
    """
    Проверить: цена находится рядом с уровнем?

    Используется в стратегиях «Отбой» и «Зеркальный уровень»
    чтобы убедиться что цена достаточно близко к уровню для входа.

    Аргументы:
        price:         текущая цена (или close свечи).
        level:         ценовой уровень.
        threshold_pct: допустимое расстояние (0.3% по умолчанию).

    Пример:
        if is_near_level(current_price, support, threshold_pct=0.003):
            # цена в пределах 0.3% от поддержки → можно рассматривать вход
    """
    if level <= 0:
        return False
    return abs(price - level) / level <= threshold_pct


def get_level_range(df: pd.DataFrame) -> Optional[float]:
    """
    Вернуть абсолютный диапазон между уровнями сопротивления и поддержки.

    Нужен в стратегии «Отбой» для оценки следующих уровней
    и расчёта целей тейк-профита.

    Возвращает:
        float — диапазон в ценовых единицах, или None при ошибке.
    """
    support, resistance = find_levels(df)
    if support is None:
        return None
    return resistance - support


def get_next_levels(df: pd.DataFrame) -> dict:
    """
    Вернуть текущие уровни и следующие за ними (проекция диапазона).

    Используется для расчёта «зеркальных» целей тейк-профита когда
    цена может дойти до следующего уровня.

    Возвращает:
        {
            "support":          float,  # текущая поддержка
            "resistance":       float,  # текущее сопротивление
            "next_support":     float,  # следующий уровень ниже поддержки
            "next_resistance":  float,  # следующий уровень выше сопротивления
        }
    """
    support, resistance = find_levels(df)
    if support is None:
        return {}

    level_range    = resistance - support
    next_support   = support    - level_range
    next_resistance = resistance + level_range

    return {
        "support":         support,
        "resistance":      resistance,
        "next_support":    next_support,
        "next_resistance": next_resistance,
    }
