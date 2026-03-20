"""
strategies/mirror.py — Стратегия «Зеркальный уровень». Приоритет 1 (высший).

Логика Герчика:
    Рынок «помнит» уровни, от которых разворачивался. Когда уровень пробит
    с объёмом — он меняет роль: бывшее сопротивление становится поддержкой,
    бывшая поддержка становится сопротивлением. При возврате к этому уровню
    (ретест) рынок отталкивается в направлении пробоя.

Алгоритм:
    1. Найти swing-highs и swing-lows в истории (реальные рыночные экстремумы).
    2. Кластеризовать близкие уровни в один.
    3. Для каждого уровня найти: N касаний → пробой на объёме → ретест текущей свечи.
    4. Если ретест подтверждён — сигнал на вход:
       - бывшее сопротивление пробито вверх → ретест сверху → LONG
       - бывшая поддержка пробита вниз → ретест снизу → SHORT

Интерфейс:
    detect(df, support, resistance) -> Optional[str]
    — Возвращает "long", "short" или None.
    — support, resistance передаются из levels.py для согласованности уровней.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from config import MIRROR_CONFIG

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# СТРУКТУРА ДАННЫХ УРОВНЯ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MirrorLevel:
    """Описание одного зеркального уровня."""
    price:         float   # Ценовой уровень
    original_role: str     # Исходная роль: "resistance" или "support"
    touches:       int     # Количество касаний до пробоя
    breakout_idx:  int     # Индекс пробойной свечи в df
    breakout_vol:  float   # Объём пробойной свечи
    strength:      float   # Сила уровня [0..1]


# ─────────────────────────────────────────────────────────────────────────────
# ПОИСК ЗЕРКАЛЬНЫХ УРОВНЕЙ
# ─────────────────────────────────────────────────────────────────────────────

def _find_swing_points(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    Найти swing-highs и swing-lows — реальные рыночные экстремумы.

    Swing high: свеча higher чем 2 предыдущих и 2 следующих.
    Swing low:  свеча lower  чем 2 предыдущих и 2 следующих.

    Аргументы:
        df: DataFrame со свечами (reset_index уже применён).

    Возвращает:
        (swing_highs, swing_lows) — списки ценовых значений.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    swing_highs = []
    swing_lows  = []

    for i in range(2, n - 2):
        h = highs[i]
        if h > highs[i - 2] and h > highs[i - 1] and h > highs[i + 1] and h > highs[i + 2]:
            swing_highs.append(h)

        lo = lows[i]
        if lo < lows[i - 2] and lo < lows[i - 1] and lo < lows[i + 1] and lo < lows[i + 2]:
            swing_lows.append(lo)

    return swing_highs, swing_lows


def _cluster_levels(raw_levels: list[float], threshold: float) -> list[float]:
    """
    Объединить близкие уровни в кластеры (среднее по кластеру).

    Аргументы:
        raw_levels: Список цен (swing points).
        threshold:  Максимальное отклонение для объединения (доля от цены).

    Возвращает:
        Список усреднённых уровней по кластерам.
    """
    if not raw_levels:
        return []

    sorted_lvls    = sorted(raw_levels)
    clusters       = []
    current_cluster = [sorted_lvls[0]]

    for price in sorted_lvls[1:]:
        ref = current_cluster[0]
        if ref > 0 and abs(price - ref) / ref <= threshold:
            current_cluster.append(price)
        else:
            clusters.append(float(np.mean(current_cluster)))
            current_cluster = [price]

    clusters.append(float(np.mean(current_cluster)))
    return clusters


def _count_touches_vectorized(
    df: pd.DataFrame,
    level: float,
    role: str,
    threshold: float,
) -> int:
    """
    Подсчитать касания уровня векторизованно (без iterrows).

    Касание поддержки (role="support"):
        low свечи попало в зону уровня ± threshold, close > уровня.
    Касание сопротивления (role="resistance"):
        high свечи попало в зону уровня ± threshold, close < уровня.

    Аргументы:
        df:        DataFrame со свечами.
        level:     Ценовой уровень.
        role:      "support" или "resistance".
        threshold: Допустимое отклонение (доля от цены).

    Возвращает:
        Количество касаний (int).
    """
    if level <= 0 or df is None or df.empty:
        return 0

    zone_low  = level * (1 - threshold)
    zone_high = level * (1 + threshold)

    if role == "resistance":
        mask = (df["high"] >= zone_low) & (df["high"] <= zone_high) & (df["close"] < level)
    else:
        mask = (df["low"] >= zone_low) & (df["low"] <= zone_high) & (df["close"] > level)

    return int(mask.sum())


def _find_breakout(
    df: pd.DataFrame,
    level: float,
    role: str,
    avg_vol_series: pd.Series,
    global_avg_vol: float,
    config,
) -> tuple[Optional[int], Optional[float]]:
    """
    Найти пробойную свечу для данного уровня.

    Ищем первую свечу после накопления касаний, которая:
    - закрылась за уровнем
    - с объёмом выше avg_vol × breakout_volume_mult
    - с минимальным отрывом breakout_min_distance от уровня

    По Герчику: перед пробоем цена должна удерживаться у уровня минимум
    config.min_candles_at_level свечей. Считаем количество касаний (свечей,
    которые попали в зону ±touch_threshold) — их должно быть не менее
    min_candles_at_level перед пробойной свечой.

    Аргументы:
        df:             DataFrame со свечами (reset_index применён).
        level:          Ценовой уровень.
        role:           "resistance" или "support" (исходная роль).
        avg_vol_series: Серия средних объёмов (по индексу df).
        global_avg_vol: Глобальный средний объём (fallback при NaN).
        config:         MIRROR_CONFIG.

    Возвращает:
        (breakout_idx, breakout_vol) или (None, None).
    """
    zone_threshold = config.touch_threshold

    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    volumes = df["volume"].values
    n       = len(df)

    # Предварительно вычисляем маску касаний векторизованно.
    # Касание resistance: high попал в зону ±touch_threshold, close ниже уровня.
    # Касание support:    low  попал в зону ±touch_threshold, close выше уровня.
    if role == "resistance":
        touch_mask = (np.abs(highs - level) / level <= zone_threshold) & (closes < level)
    else:
        touch_mask = (np.abs(lows - level) / level <= zone_threshold) & (closes > level)

    # Накопленное количество касаний ДО каждой свечи (не включая её).
    # cum_touches[i] = число касаний среди свечей [0..i-1].
    cum_touches = np.concatenate([[0], np.cumsum(touch_mask)[:-1]])

    for i in range(n):
        # Считаем количество касаний ДО текущей свечи (не включая её)
        touches_before = int(cum_touches[i])

        # По Герчику: перед пробоем цена должна касаться уровня минимум
        # min_candles_at_level раз. Это единственный источник истины.
        if touches_before < config.min_candles_at_level:
            continue

        cur_avg = float(avg_vol_series.iloc[i])
        if pd.isna(cur_avg) or cur_avg <= 0:
            cur_avg = global_avg_vol

        vol_ok = volumes[i] > cur_avg * config.breakout_volume_mult

        if role == "resistance":
            dist_ok = level > 0 and (closes[i] - level) / level >= config.breakout_min_distance
            if closes[i] > level and vol_ok and dist_ok:
                return i, float(volumes[i])
        else:
            dist_ok = level > 0 and (level - closes[i]) / level >= config.breakout_min_distance
            if closes[i] < level and vol_ok and dist_ok:
                return i, float(volumes[i])

    return None, None


def find_mirror_levels(df: pd.DataFrame) -> list[MirrorLevel]:
    """
    Найти все зеркальные уровни в DataFrame.

    Полный алгоритм:
        1. Берём последние level_lookback свечей.
        2. Находим swing-highs / swing-lows.
        3. Кластеризуем близкие уровни.
        4. Для каждого кластера: считаем касания и ищем пробой.
        5. Если касаний >= min_touches и пробой найден → MirrorLevel.
        6. Сортируем по силе (strength) — самые сильные первые.

    Аргументы:
        df: DataFrame со свечами (open, high, low, close, volume).
            Желательно наличие avg_volume.

    Возвращает:
        Список MirrorLevel, отсортированный по силе (убывание).
    """
    config = MIRROR_CONFIG

    if df is None or len(df) < config.min_candles_at_level + 10:
        return []

    # Рабочее окно: последние level_lookback свечей
    if len(df) > config.level_lookback:
        global_offset = len(df) - config.level_lookback
        work_df = df.iloc[-config.level_lookback:].reset_index(drop=True)
    else:
        global_offset = 0
        work_df = df.reset_index(drop=True)

    # Средний объём (векторизованно, без iterrows)
    if "avg_volume" in work_df.columns:
        avg_vol_series = work_df["avg_volume"].fillna(work_df["volume"].mean())
    else:
        avg_vol_series = work_df["volume"].rolling(20, min_periods=1).mean()

    global_avg_vol = float(avg_vol_series.mean())
    if pd.isna(global_avg_vol) or global_avg_vol <= 0:
        global_avg_vol = float(work_df["volume"].mean())
    if global_avg_vol <= 0:
        return []

    # Swing-точки → кластеры уровней
    swing_highs, swing_lows = _find_swing_points(work_df)
    resistance_levels = _cluster_levels(swing_highs, config.level_cluster_threshold)
    support_levels    = _cluster_levels(swing_lows,  config.level_cluster_threshold)

    mirror_levels: list[MirrorLevel] = []

    def _process(level_price: float, original_role: str) -> None:
        if level_price <= 0:
            return

        # 1. Считаем касания
        touches = _count_touches_vectorized(
            work_df, level_price, original_role, config.touch_threshold
        )
        if touches < config.min_touches:
            return

        # 2. Ищем пробой
        brk_idx, brk_vol = _find_breakout(
            work_df, level_price, original_role,
            avg_vol_series, global_avg_vol, config,
        )
        if brk_idx is None:
            return

        # 3. Рассчитываем силу уровня
        candles_since = (len(df) - 1) - (brk_idx + global_offset)
        strength = (
            min(touches / 10, 1.0) * 0.4
            + min(brk_vol / global_avg_vol, 2.0) / 2.0 * 0.3
            + max(0.0, 1.0 - candles_since / config.max_candles_after_break) * 0.3
        )
        strength = min(1.0, max(0.0, strength))

        log.debug(
            f"Зеркало: {level_price:.4f} [{original_role}] | "
            f"касаний={touches} | сила={strength:.2f} | пробой {candles_since} свечей назад"
        )

        mirror_levels.append(MirrorLevel(
            price         = level_price,
            original_role = original_role,
            touches       = touches,
            breakout_idx  = brk_idx + global_offset,
            breakout_vol  = brk_vol,
            strength      = strength,
        ))

    for lvl in resistance_levels:
        _process(lvl, "resistance")

    for lvl in support_levels:
        _process(lvl, "support")

    log.debug(f"Найдено зеркальных уровней: {len(mirror_levels)}")
    mirror_levels.sort(key=lambda m: m.strength, reverse=True)
    return mirror_levels


# ─────────────────────────────────────────────────────────────────────────────
# ПРОВЕРКА РЕТЕСТА
# ─────────────────────────────────────────────────────────────────────────────

def check_retest(df: pd.DataFrame, mirror: MirrorLevel) -> tuple[bool, int]:
    """
    Проверить: последняя свеча делает ретест зеркального уровня?

    Правила ретеста:
        - Прошло min_candles_after_break свечей (рынок «улетел» от уровня).
        - Не прошло max_candles_after_break (уровень ещё актуален).
        - Свеча вернулась к уровню в пределах retest_threshold.
        - Свеча закрылась по правильную сторону от уровня.
        - Объём ретеста < объёма пробоя (слабая реакция продавцов / покупателей).

    Аргументы:
        df:     DataFrame со свечами.
        mirror: Объект MirrorLevel.

    Возвращает:
        (True, candles_since) или (False, 0).
    """
    config         = MIRROR_CONFIG
    candles_since  = len(df) - 1 - mirror.breakout_idx

    if candles_since < config.min_candles_after_break:
        log.debug(f"Ретест {mirror.price:.4f}: слишком рано ({candles_since} свечей)")
        return False, 0

    if candles_since > config.max_candles_after_break:
        log.debug(f"Ретест {mirror.price:.4f}: уровень устарел ({candles_since} свечей)")
        return False, 0

    candle     = df.iloc[-1]
    level      = mirror.price
    threshold  = config.retest_threshold
    close_ok   = 0.001   # допуск закрытия у уровня (0.1%)

    if mirror.original_role == "resistance":
        # Пробой вверх → ретест сверху → ищем LONG
        # Low свечи дотянулся до уровня, close >= уровня (закрылся выше или у уровня)
        low_near  = abs(candle["low"] - level) / level <= threshold
        close_above = candle["close"] >= level * (1 - close_ok)
        vol_weak  = candle["volume"] < mirror.breakout_vol
        if low_near and close_above and vol_weak:
            return True, candles_since

    else:
        # Пробой вниз → ретест снизу → ищем SHORT
        # High свечи дотянулся до уровня, close <= уровня
        high_near  = abs(candle["high"] - level) / level <= threshold
        close_below = candle["close"] <= level * (1 + close_ok)
        vol_weak   = candle["volume"] < mirror.breakout_vol
        if high_near and close_below and vol_weak:
            return True, candles_since

    return False, 0


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def detect(
    df: pd.DataFrame,
    support: Optional[float],
    resistance: Optional[float],
) -> Optional[str]:
    """
    Обнаружить сигнал стратегии «Зеркальный уровень». Приоритет 1.

    Функция ищет зеркальные уровни (бывшие поддержки/сопротивления,
    которые были пробиты и теперь делаются ретест). Если ретест подтверждён —
    возвращает направление входа.

    Аргументы:
        df:         DataFrame со свечами (≥100 штук).
        support:    Текущая поддержка из levels.py (используется для контекста).
        resistance: Текущее сопротивление из levels.py.

    Возвращает:
        "long"  — если бывшее сопротивление держит как поддержка (ретест снизу).
        "short" — если бывшая поддержка держит как сопротивление (ретест сверху).
        None    — сигнала нет.

    Пример:
        signal = detect(df, support=41000.0, resistance=44000.0)
        if signal:
            params = calculate_trade_params(signal, df.iloc[-1]["close"], ...)
    """
    config = MIRROR_CONFIG

    # ── Проверка входных данных ──────────────────────────────────────────────
    if df is None or len(df) < 100:
        log.debug("Зеркало: недостаточно данных (нужно ≥100 свечей)")
        return None

    # Нет данных об уровнях от levels.py → не торгуем
    if support is None or resistance is None:
        log.debug("Зеркало: уровни не определены")
        return None

    # ── Фильтр: не торгуем при экстремальной волатильности свечи ────────────
    candle = df.iloc[-1]
    close  = float(candle["close"])
    if close <= 0:
        return None

    candle_range = float(candle["high"]) - float(candle["low"])
    if candle_range / close > 0.05:
        log.debug(f"Зеркало: волатильность свечи {candle_range/close*100:.1f}% > 5%, пропускаем")
        return None

    # ── Подготовка avg_volume если нет ──────────────────────────────────────
    if "avg_volume" not in df.columns:
        df = df.copy()
        df["avg_volume"] = df["volume"].rolling(20, min_periods=1).mean()

    # ── Поиск зеркальных уровней ─────────────────────────────────────────────
    mirror_levels = find_mirror_levels(df)

    if not mirror_levels:
        log.debug("Зеркало: уровней не найдено")
        return None

    # ── Проверка ретестов ─────────────────────────────────────────────────────
    long_candidates:  list[MirrorLevel] = []
    short_candidates: list[MirrorLevel] = []

    for mirror in mirror_levels:
        # Пропускаем слабые уровни
        if mirror.strength < 0.3:
            continue

        is_retest, candles_since = check_retest(df, mirror)
        if not is_retest:
            continue

        if mirror.original_role == "resistance":
            # Пробой вверх → теперь поддержка → входим LONG при ретесте
            long_candidates.append(mirror)
            log.info(
                f"[ЗЕРКАЛО] LONG кандидат | уровень={mirror.price:.4f} | "
                f"была: сопротивление | сила={mirror.strength:.2f} | "
                f"касаний={mirror.touches} | пробой {candles_since} свечей назад"
            )
        else:
            # Пробой вниз → теперь сопротивление → входим SHORT при ретесте
            short_candidates.append(mirror)
            log.info(
                f"[ЗЕРКАЛО] SHORT кандидат | уровень={mirror.price:.4f} | "
                f"была: поддержка | сила={mirror.strength:.2f} | "
                f"касаний={mirror.touches}"
            )

    if not long_candidates and not short_candidates:
        log.debug("Зеркало: нет ретестов")
        return None

    # ── Выбор лучшего сигнала (при конфликте — по силе) ─────────────────────
    best_long  = max(long_candidates,  key=lambda m: m.strength) if long_candidates  else None
    best_short = max(short_candidates, key=lambda m: m.strength) if short_candidates else None

    if best_long and best_short:
        # Оба направления — берём более сильный уровень
        if best_long.strength >= best_short.strength:
            log.info(f"[ЗЕРКАЛО] Выбрано LONG (сила {best_long.strength:.2f} >= {best_short.strength:.2f})")
            return "long"
        else:
            log.info(f"[ЗЕРКАЛО] Выбрано SHORT (сила {best_short.strength:.2f} > {best_long.strength:.2f})")
            return "short"

    if best_long:
        log.info(f"[ЗЕРКАЛО] Сигнал LONG | уровень={best_long.price:.4f}")
        return "long"

    if best_short:
        log.info(f"[ЗЕРКАЛО] Сигнал SHORT | уровень={best_short.price:.4f}")
        return "short"

    return None
