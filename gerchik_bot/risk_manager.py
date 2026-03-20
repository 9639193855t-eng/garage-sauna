"""
risk_manager.py — управление рисками по методике Герчика.

Всё что касается денег и защиты капитала — здесь:
    calculate_trade_params   → стоп, тейк, размер позиции
    update_trailing_stop     → двигаем трейлинг-стоп вслед за ценой
    check_exit               → проверяем нужно ли закрывать позицию
    calculate_pnl            → считаем прибыль/убыток по позиции

Правила Герчика которые реализованы:
    - Стоп ЗА экстремум пробойной свечи + буфер
    - Тейк = риск × 3 (соотношение 1:3 минимум)
    - Риск на сделку = 1% от депозита
    - Размер позиции рассчитывается ЧЕРЕЗ риск (не фиксированный лот)
    - Трейлинг-стоп активируется только в прибыли, движется только вперёд
"""

import logging
from typing import Optional

import pandas as pd

from config import CONFIG

# Буфер за свинг-экстремум для динамического трейлинга (0.1%)
_SWING_TRAIL_BUFFER = 0.001
# Количество свечей для поиска свинг-экстремума
_SWING_CANDLES = 3

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# РАСЧЁТ ПАРАМЕТРОВ СДЕЛКИ
# ─────────────────────────────────────────────────────────────────────────────

def calculate_trade_params(
    direction: str,
    entry_price: float,
    candle_extremum: float,
    account_balance: float,
    price_precision: int = 2,
    qty_precision: int = 3,
) -> dict:
    """
    Рассчитать стоп, тейк и размер позиции для новой сделки.

    Алгоритм Герчика:
        1. Стоп ставится ЗА экстремум пробойной свечи + небольшой буфер.
           Для LONG: стоп = low свечи - буфер
           Для SHORT: стоп = high свечи + буфер
        2. Тейк = entry ± (риск × reward_ratio).
           По умолчанию reward_ratio = 3.0 (правило 1:3).
        3. Размер позиции рассчитывается так, чтобы при срабатывании стопа
           мы потеряли ровно risk_per_trade % от баланса.
           С учётом плеча: quantity = (баланс × риск%) / (risk_points × leverage).
        4. Максимальный размер позиции ограничен max_position_size_pct
           от (баланс × плечо) — защита от перегруза маржи.

    Аргументы:
        direction:       "long" или "short".
        entry_price:     Ожидаемая цена входа (обычно текущая цена).
        candle_extremum: Экстремум пробойной свечи:
                         - для LONG: low свечи (стоп ставится ниже)
                         - для SHORT: high свечи (стоп ставится выше)
        account_balance: Полный баланс счёта в USDT (totalWalletBalance).
        price_precision: Число знаков после запятой для цен.
        qty_precision:   Число знаков после запятой для объёма.

    Возвращает:
        Словарь с ключами:
            entry    (float) — цена входа
            stop     (float) — цена стоп-лосса
            take     (float) — цена тейк-профита
            quantity (float) — объём в контрактах
            risk     (float) — размер риска в ценовых пунктах
            risk_usd (float) — размер риска в USDT
            ratio    (float) — реальное соотношение прибыль/риск
        Пустой словарь {} если параметры некорректны.

    Пример:
        params = calculate_trade_params("long", 41000.0, 40800.0, 1000.0, 1, 3)
        # → {"entry": 41000, "stop": 40718.0, "take": 42686.0, ...}
    """
    if entry_price <= 0 or candle_extremum <= 0 or account_balance <= 0:
        log.warning("calculate_trade_params: некорректные входные данные")
        return {}

    if direction not in ("long", "short"):
        log.warning(f"calculate_trade_params: неизвестное направление {direction!r}")
        return {}

    # 1. Стоп-лосс
    buffer = entry_price * CONFIG.stop_buffer_pct

    if direction == "long":
        stop_price  = round(candle_extremum - buffer, price_precision)
        risk_points = entry_price - stop_price
        take_price  = round(entry_price + risk_points * CONFIG.reward_ratio, price_precision)
    else:
        stop_price  = round(candle_extremum + buffer, price_precision)
        risk_points = stop_price - entry_price
        take_price  = round(entry_price - risk_points * CONFIG.reward_ratio, price_precision)

    if risk_points <= 0:
        log.warning(
            f"calculate_trade_params: risk_points={risk_points:.6f} <= 0 "
            f"(entry={entry_price}, extremum={candle_extremum}, direction={direction})"
        )
        return {}

    # 2. Размер позиции
    # Потеря при стопе = quantity × risk_points × leverage (для USDT-M фьючерсов)
    # Хотим: потеря = account_balance × risk_per_trade
    # → quantity = (account_balance × risk_per_trade) / (risk_points × leverage)
    risk_usd = account_balance * CONFIG.risk_per_trade
    quantity  = risk_usd / (risk_points * CONFIG.leverage)
    quantity  = round(quantity, qty_precision)
    quantity  = max(quantity, 10 ** -qty_precision)  # минимальный лот

    # 3. Ограничение по размеру: не более max_position_size_pct от (баланс × плечо)
    max_notional = account_balance * CONFIG.leverage * CONFIG.max_position_size_pct
    max_qty      = round(max_notional / entry_price, qty_precision)
    if quantity > max_qty:
        log.info(
            f"calculate_trade_params: размер урезан {quantity} → {max_qty} "
            f"(лимит {max_notional:.0f} USDT)"
        )
        quantity = max(max_qty, 10 ** -qty_precision)

    # 4. Реальное соотношение прибыль/риск
    take_points = abs(take_price - entry_price)
    ratio       = round(take_points / risk_points, 2) if risk_points > 0 else 0.0

    log.info(
        f"Параметры сделки [{direction.upper()}]: "
        f"вход={entry_price:.{price_precision}f}  "
        f"стоп={stop_price:.{price_precision}f}  "
        f"тейк={take_price:.{price_precision}f}  "
        f"объём={quantity}  риск={risk_usd:.2f} USDT  соотношение 1:{ratio}"
    )

    return {
        "entry":    entry_price,
        "stop":     stop_price,
        "take":     take_price,
        "quantity": quantity,
        "risk":     round(risk_points, price_precision),
        "risk_usd": round(risk_usd, 2),
        "ratio":    ratio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ТРЕЙЛИНГ-СТОП
# ─────────────────────────────────────────────────────────────────────────────

def update_trailing_stop(
    pos: dict,
    current_price: float,
    price_precision: int = 2,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Обновить трейлинг-стоп для открытой позиции.

    Два режима в зависимости от наличия DataFrame со свечами:

    Динамический (df передан):
        Трейлинг ставится за свинг-экстремум последних _SWING_CANDLES свечей.
        LONG:  new_stop = min(last 3 low)  × (1 - _SWING_TRAIL_BUFFER)
        SHORT: new_stop = max(last 3 high) × (1 + _SWING_TRAIL_BUFFER)
        Соответствует методике Герчика — стоп за последний свинг.

    Процентный fallback (df не передан или данных мало):
        trailing_stop = best_price × (1 ± trailing_stop_pct)

    В обоих режимах:
        1. Следим за лучшей достигнутой ценой (best_price).
        2. Активируем трейлинг когда прибыль >= trailing_activation_pct.
        3. Трейлинг двигается только в выгодную сторону — никогда назад.

    Аргументы:
        pos:             Словарь позиции (direction, entry, best_price, ...).
        current_price:   Текущая рыночная цена.
        price_precision: Знаков после запятой для округления.
        df:              DataFrame со свечами (опционально). Если передан —
                         используется динамический свинг-трейлинг.

    Возвращает:
        Обновлённый словарь позиции (мутирует pos на месте и возвращает его).

    Пример:
        pos = update_trailing_stop(pos, current_price=42500.0, price_precision=2, df=df)
        if pos["trailing_active"]:
            print(f"Трейлинг на уровне: {pos['trailing_stop']}")
    """
    if not CONFIG.trailing_stop_enabled:
        return pos

    direction = pos["direction"]
    entry     = pos["entry"]

    # Обновляем лучшую цену
    if direction == "long":
        if current_price > pos.get("best_price", entry):
            pos["best_price"] = current_price
    else:
        if current_price < pos.get("best_price", entry):
            pos["best_price"] = current_price

    best = pos.get("best_price", entry)

    # Проверяем активацию
    if not pos.get("trailing_active", False):
        profit_pct = (best - entry) / entry if direction == "long" else (entry - best) / entry
        if profit_pct >= CONFIG.trailing_activation_pct:
            pos["trailing_active"] = True
            log.info(
                f"Трейлинг активирован | {direction.upper()} | "
                f"вход={entry} лучшая_цена={best} прибыль={profit_pct * 100:.2f}%"
            )

    # Двигаем трейлинг (только если активен)
    if pos.get("trailing_active", False):
        # ── Выбираем режим расчёта нового уровня ─────────────────────────────
        use_swing = (
            df is not None
            and not df.empty
            and len(df) >= _SWING_CANDLES + 1
        )

        if use_swing:
            # Динамический трейлинг: за свинг-экстремум последних N свечей
            # Берём N закрытых свечей, исключая сигнальную (последнюю)
            recent = df.iloc[-(_SWING_CANDLES + 1):-1]
            if direction == "long":
                swing_low  = float(recent["low"].min())
                new_level  = round(swing_low * (1 - _SWING_TRAIL_BUFFER), price_precision)
            else:
                swing_high = float(recent["high"].max())
                new_level  = round(swing_high * (1 + _SWING_TRAIL_BUFFER), price_precision)
        else:
            # Процентный fallback
            if direction == "long":
                new_level = round(best * (1 - CONFIG.trailing_stop_pct), price_precision)
            else:
                new_level = round(best * (1 + CONFIG.trailing_stop_pct), price_precision)

        if direction == "long":
            # Двигаем только вверх (никогда назад)
            if pos.get("trailing_stop") is None or new_level > pos["trailing_stop"]:
                old = pos.get("trailing_stop")
                pos["trailing_stop"] = new_level
                mode = "свинг" if use_swing else "%"
                if old is not None:
                    log.info(
                        f"Трейлинг↑ [{mode}] | {old:.{price_precision}f} → {new_level:.{price_precision}f}"
                    )
        else:
            # Двигаем только вниз (никогда назад)
            if pos.get("trailing_stop") is None or new_level < pos["trailing_stop"]:
                old = pos.get("trailing_stop")
                pos["trailing_stop"] = new_level
                mode = "свинг" if use_swing else "%"
                if old is not None:
                    log.info(
                        f"Трейлинг↓ [{mode}] | {old:.{price_precision}f} → {new_level:.{price_precision}f}"
                    )

    return pos


# ─────────────────────────────────────────────────────────────────────────────
# ПРОВЕРКА УСЛОВИЙ ВЫХОДА
# ─────────────────────────────────────────────────────────────────────────────

def check_exit(pos: dict, current_price: float) -> Optional[str]:
    """
    Проверить: нужно ли закрыть позицию?

    Проверяет стоп-лосс (или трейлинг-стоп если активен) и тейк-профит.
    Приоритет: трейлинг-стоп > фиксированный стоп > тейк-профит.

    Аргументы:
        pos:           Словарь позиции с ключами:
                       - direction      (str): "long" или "short"
                       - stop           (float|None): фиксированный стоп
                       - take           (float|None): тейк-профит
                       - trailing_stop  (float|None): уровень трейлинга
                       - trailing_active (bool): активирован ли трейлинг
        current_price: Текущая рыночная цена.

    Возвращает:
        "СТОП-ЛОСС"     — если сработал стоп или трейлинг
        "ТРЕЙЛИНГ-СТОП" — если сработал трейлинг (уточнённая причина)
        "ТЕЙК-ПРОФИТ"   — если достигнута цель
        None            — позиция ещё жива

    Пример:
        reason = check_exit(pos, current_price=40500.0)
        if reason:
            close_position(client, symbol, pos["direction"], pos["quantity"])
    """
    direction = pos.get("direction")
    take      = pos.get("take")

    # Определяем эффективный стоп
    trailing_active = pos.get("trailing_active", False)
    trailing_stop   = pos.get("trailing_stop")
    fixed_stop      = pos.get("stop")

    if trailing_active and trailing_stop is not None:
        effective_stop  = trailing_stop
        stop_reason     = "ТРЕЙЛИНГ-СТОП"
    else:
        effective_stop  = fixed_stop
        stop_reason     = "СТОП-ЛОСС"

    if direction == "long":
        # Стоп: цена упала ниже уровня защиты
        if effective_stop is not None and current_price <= effective_stop:
            log.info(
                f"check_exit: {stop_reason} LONG | "
                f"цена={current_price} <= стоп={effective_stop}"
            )
            return stop_reason
        # Тейк: цена достигла цели
        if take is not None and current_price >= take:
            log.info(f"check_exit: ТЕЙК-ПРОФИТ LONG | цена={current_price} >= тейк={take}")
            return "ТЕЙК-ПРОФИТ"

    elif direction == "short":
        # Стоп: цена выросла выше уровня защиты
        if effective_stop is not None and current_price >= effective_stop:
            log.info(
                f"check_exit: {stop_reason} SHORT | "
                f"цена={current_price} >= стоп={effective_stop}"
            )
            return stop_reason
        # Тейк: цена упала до цели
        if take is not None and current_price <= take:
            log.info(f"check_exit: ТЕЙК-ПРОФИТ SHORT | цена={current_price} <= тейк={take}")
            return "ТЕЙК-ПРОФИТ"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# РАСЧЁТ P&L
# ─────────────────────────────────────────────────────────────────────────────

def calculate_pnl(
    direction: str,
    entry_price: float,
    exit_price: float,
    quantity: float,
) -> dict:
    """
    Рассчитать прибыль/убыток по закрытой позиции.

    На USDT-M фьючерсах:
        PnL = (exit - entry) × quantity × leverage   (для LONG)
        PnL = (entry - exit) × quantity × leverage   (для SHORT)

    Аргументы:
        direction:   "long" или "short".
        entry_price: Цена входа.
        exit_price:  Цена выхода.
        quantity:    Объём позиции в контрактах.

    Возвращает:
        Словарь:
            pnl_usd    (float) — прибыль/убыток в USDT с учётом плеча
            pnl_pct    (float) — прибыль/убыток в % от маржи на позицию
            pnl_points (float) — движение цены в пунктах
            is_profit  (bool)  — True если прибыльная сделка

    Пример:
        result = calculate_pnl("long", 41000.0, 43000.0, 0.001)
        # → {"pnl_usd": 10.0, "pnl_pct": 4.88, "pnl_points": 2000.0, "is_profit": True}
    """
    if direction == "long":
        pnl_points = exit_price - entry_price
    else:
        pnl_points = entry_price - exit_price

    pnl_usd = pnl_points * quantity * CONFIG.leverage

    # % от маржи (маржа = quantity × entry / leverage)
    margin  = (quantity * entry_price) / CONFIG.leverage
    pnl_pct = (pnl_usd / margin * 100) if margin > 0 else 0.0

    return {
        "pnl_usd":    round(pnl_usd, 4),
        "pnl_pct":    round(pnl_pct, 2),
        "pnl_points": round(pnl_points, 8),
        "is_profit":  pnl_usd > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def is_params_valid(params: dict) -> bool:
    """
    Проверить что словарь параметров сделки заполнен корректно.

    Используется перед отправкой ордера на биржу.

    Аргументы:
        params: Словарь который вернул calculate_trade_params.

    Возвращает:
        True если все необходимые поля присутствуют и корректны.

    Пример:
        if not is_params_valid(params):
            log.warning("Некорректные параметры, пропускаем сигнал")
            continue
    """
    required = ("entry", "stop", "take", "quantity", "risk")
    for key in required:
        if key not in params:
            log.warning(f"is_params_valid: отсутствует поле {key!r}")
            return False

    if params.get("risk", 0) <= 0:
        log.warning(f"is_params_valid: risk={params.get('risk')} <= 0")
        return False

    if params.get("quantity", 0) <= 0:
        log.warning(f"is_params_valid: quantity={params.get('quantity')} <= 0")
        return False

    return True


def make_position(
    symbol: str,
    direction: str,
    params: dict,
    strategy: str = "unknown",
    order_id: Optional[str] = None,
) -> dict:
    """
    Создать стандартный словарь позиции для хранения в position_manager.

    Все поля трейлинга инициализируются в исходное состояние.

    Аргументы:
        symbol:    Тикер (например "BTCUSDT").
        direction: "long" или "short".
        params:    Словарь из calculate_trade_params.
        strategy:  Название стратегии которая дала сигнал.
        order_id:  ID рыночного ордера входа.

    Возвращает:
        Полный словарь позиции готовый для хранения.
    """
    from datetime import datetime
    return {
        "symbol":          symbol,
        "order_id":        order_id,
        "direction":       direction,
        "entry":           params["entry"],
        "stop":            params["stop"],
        "take":            params["take"],
        "quantity":        params["quantity"],
        "risk_usd":        params.get("risk_usd", 0.0),
        "opened_at":       datetime.now(),
        "strategy":        strategy,
        # Трейлинг-стоп (инициализация)
        "trailing_active": False,
        "trailing_stop":   None,
        "best_price":      params["entry"],
    }
