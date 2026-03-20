"""
position_manager.py — Управление открытыми позициями.

Хранит позиции в памяти (symbol → dict). Один символ — одна позиция.
Проверяет условия выхода (SL, TP, трейлинг) и закрывает через exchange.py.

Интерфейс:
    pm = PositionManager()
    pm.add(pos)                           → добавить позицию
    pm.remove(symbol)                     → удалить позицию
    pm.get(symbol)                        → получить позицию
    pm.is_open(symbol)                    → есть ли позиция
    pm.count()                            → сколько открыто
    pm.all()                              → список всех позиций
    pm.check_exits(client, cache)         → проверить и закрыть сработавшие
    pm.restore_from_exchange(client, cache) → восстановить после перезапуска
"""

import json
import logging
import os as _os
from pathlib import Path
from typing import Optional

import exchange
from risk_manager import update_trailing_stop, check_exit, calculate_pnl
from config import CONFIG
from stats_tracker import StatsTracker
from trade_history import close_trade

_STATS_FILE = _os.path.join(_os.path.dirname(__file__), "logs", "trades.json")
_POSITIONS_FILE = Path("logs/positions.json")

log = logging.getLogger(__name__)


class PositionManager:
    """
    Менеджер открытых позиций.

    Хранит позиции в памяти: {symbol: pos_dict}.
    # Основные операции синхронны; stats thread-safe через внутренний Lock

    Аргументы:
        tg: Опциональный TelegramNotifier. Если задан — отправляет
            уведомление при каждом закрытии позиции.
    """

    def __init__(self, tg=None) -> None:
        self._positions: dict[str, dict] = {}
        self.tg = tg
        self.stats = StatsTracker(_STATS_FILE)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def _save_positions(self) -> None:
        """Сохранить текущие позиции в файл для восстановления после перезапуска."""
        try:
            _POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for symbol, pos in self._positions.items():
                row = {}
                for k, v in pos.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        row[k] = v
                    elif hasattr(v, 'isoformat'):  # datetime
                        row[k] = v.isoformat()
                data[symbol] = row
            tmp = _POSITIONS_FILE.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp.replace(_POSITIONS_FILE)
        except Exception as e:
            log.warning("[PM] Не удалось сохранить позиции: %s", e)

    def add(self, pos: dict) -> None:
        """Добавить позицию. Перезаписывает если символ уже есть."""
        symbol = pos["symbol"]
        self._positions[symbol] = pos
        self._save_positions()
        log.info(
            "[PM] ОТКРЫТА: %s %s | вход=%.4f стоп=%.4f тейк=%.4f | стратегия=%s",
            symbol, pos["direction"].upper(),
            pos["entry"], pos.get("stop", 0), pos.get("take", 0),
            pos.get("strategy", "?"),
        )

    def remove(self, symbol: str) -> Optional[dict]:
        """Удалить и вернуть позицию. None если не было."""
        pos = self._positions.pop(symbol, None)
        if pos:
            log.info("[PM] УДАЛЕНА: %s", symbol)
            self._save_positions()
        return pos

    def get(self, symbol: str) -> Optional[dict]:
        """Получить позицию по символу."""
        return self._positions.get(symbol)

    def all(self) -> list[dict]:
        """Список всех открытых позиций (копия)."""
        return list(self._positions.values())

    def count(self) -> int:
        """Количество открытых позиций."""
        return len(self._positions)

    def is_open(self, symbol: str) -> bool:
        """Есть ли открытая позиция по символу."""
        return symbol in self._positions

    # ── ПРОВЕРКА ВЫХОДОВ ─────────────────────────────────────────────────────

    def check_exits(self, client, precision_cache: dict) -> list[str]:
        """
        Проверить условия выхода для всех открытых позиций.

        Для каждой позиции:
            1. Получает текущую цену с биржи.
            2. Обновляет трейлинг-стоп (двигает вслед за лучшей ценой).
            3. Проверяет условие выхода: SL / TP / трейлинг.
            4. Если условие сработало — закрывает рыночно и удаляет из менеджера.
            5. Если задан tg — отправляет Telegram-уведомление о закрытии.

        Аргументы:
            client:          Клиент Binance.
            precision_cache: Кэш точностей {symbol: (price_prec, qty_prec)}.

        Возвращает:
            Список символов закрытых в этом вызове позиций.
        """
        closed: list[str] = []

        for symbol, pos in list(self._positions.items()):
            try:
                # 1. Текущая цена
                current_price = exchange.get_current_price(client, symbol)
                if current_price is None:
                    log.warning("[PM] %s: не удалось получить цену, пропускаем", symbol)
                    continue

                # 2. Обновляем трейлинг (свинг-based если удалось загрузить свечи)
                price_prec, _ = exchange.get_precision(client, symbol, cache=precision_cache)
                df = exchange.get_candles(client, symbol)
                update_trailing_stop(
                    pos, current_price, price_prec,
                    df=df if df is not None and not df.empty else None,
                )

                # 3. Проверяем выход
                reason = check_exit(pos, current_price)
                if reason is None:
                    continue

                # 4. Закрываем позицию на бирже
                log.info(
                    "[PM] %s: выход [%s] | %s | цена=%.4f",
                    symbol, reason, pos["direction"].upper(), current_price,
                )
                success = exchange.close_position(
                    client, symbol, pos["direction"], pos["quantity"]
                )

                if success:
                    pnl = calculate_pnl(
                        pos["direction"], pos["entry"], current_price, pos["quantity"]
                    )
                    log.info(
                        "[PM] %s: ЗАКРЫТА [%s] | PnL=%+.2f USDT (%+.2f%%) | "
                        "вход=%.4f выход=%.4f",
                        symbol, reason,
                        pnl["pnl_usd"], pnl["pnl_pct"],
                        pos["entry"], current_price,
                    )

                    # Записываем сделку в статистику
                    try:
                        self.stats.record(pos, reason, current_price, pnl)
                    except Exception as stats_exc:
                        log.warning(
                            "[PM] %s: ошибка записи статистики: %s",
                            symbol, stats_exc,
                        )

                    # 5. Telegram-уведомление о закрытии
                    if self.tg:
                        try:
                            self.tg.notify_close(pos, reason, current_price, pnl)
                        except Exception as tg_exc:
                            log.warning(
                                "[PM] %s: ошибка Telegram-уведомления о закрытии: %s",
                                symbol, tg_exc,
                            )

                    # 6. Сохраняем закрытие в историю сделок
                    trade_id = pos.get("trade_id")
                    if trade_id:
                        try:
                            close_trade(trade_id, reason, pnl, pos)
                        except Exception as hist_exc:
                            log.warning(
                                "[PM] %s: ошибка записи закрытия в историю: %s",
                                symbol, hist_exc,
                            )

                    self.remove(symbol)
                    closed.append(symbol)
                else:
                    # Позиция может уже не существовать на бирже (закрыта вручную или по
                    # другому ордеру). Синхронизируемся: если биржа её не видит — убираем.
                    real_pos = exchange.get_open_position(client, symbol)
                    if real_pos is None or float(real_pos.get("positionAmt", 0)) == 0:
                        log.info(
                            "[PM] %s: позиции нет на бирже, удаляем из памяти", symbol
                        )
                        try:
                            pnl = calculate_pnl(pos["direction"], pos["entry"], current_price, pos["quantity"])
                            self.stats.record(pos, reason, current_price, pnl)
                            log.info(
                                "[PM] %s: записана в статистику (sync) | PnL=%+.2f USDT (%+.2f%%)",
                                symbol, pnl["pnl_usd"], pnl["pnl_pct"],
                            )
                        except Exception as exc:
                            log.warning("[PM] %s: не удалось записать в статистику: %s", symbol, exc)
                        self.remove(symbol)
                        closed.append(symbol)
                    else:
                        log.error("[PM] %s: НЕ УДАЛОСЬ закрыть позицию!", symbol)

            except Exception as e:
                log.error("[PM] check_exits [%s]: неожиданная ошибка — %s", symbol, e, exc_info=True)

        return closed

    # ── ВОССТАНОВЛЕНИЕ ПОСЛЕ ПЕРЕЗАПУСКА ─────────────────────────────────────

    def restore_from_exchange(self, client, precision_cache: dict) -> None:
        """
        Восстановить позиции из биржи при перезапуске.

        Запрашивает открытые позиции, для каждой ищет SL/TP из открытых ордеров.
        Трейлинг не восстанавливается (начинает с нуля после перезапуска).

        Используй в начале main() чтобы бот «увидел» позиции открытые до краша.
        """
        log.info("[PM] Восстановление позиций с биржи...")

        # Загружаем сохранённые данные позиций (стратегия, trade_id и др.)
        saved: dict = {}
        if _POSITIONS_FILE.exists():
            try:
                with open(_POSITIONS_FILE) as f:
                    saved = json.load(f)
            except Exception:
                saved = {}

        open_positions = exchange.get_all_open_positions(client)

        if not open_positions:
            log.info("[PM] Открытых позиций на бирже нет")
            return

        restored = 0
        for p in open_positions:
            symbol = p.get("symbol", "")
            amt    = float(p.get("positionAmt", 0))
            entry  = float(p.get("entryPrice", 0))

            if amt == 0 or entry <= 0 or not symbol:
                continue

            direction = "long" if amt > 0 else "short"
            quantity  = abs(amt)

            # Ищем SL и TP из открытых ордеров
            stop: Optional[float] = None
            take: Optional[float] = None

            for order in exchange.get_open_orders(client, symbol):
                otype = order.get("type", "")
                sp    = float(order.get("stopPrice", 0) or 0)
                if sp <= 0:
                    continue
                if otype == "STOP_MARKET":
                    stop = sp
                elif otype == "TAKE_PROFIT_MARKET":
                    take = sp

            pos = {
                "symbol":          symbol,
                "order_id":        None,
                "direction":       direction,
                "entry":           entry,
                "stop":            stop,
                "take":            take,
                "quantity":        quantity,
                "risk_usd":        0.0,
                "opened_at":       None,
                "strategy":        saved.get(symbol, {}).get("strategy", "restored"),
                # Трейлинг инициализируется заново
                "trailing_active": False,
                "trailing_stop":   None,
                "best_price":      entry,
            }

            self._positions[symbol] = pos

            # Восстанавливаем сохранённые поля (стратегия, trade_id и др.)
            if symbol in saved:
                for key in ("strategy", "trade_id", "opened_at", "risk_usd", "best_price"):
                    if key in saved[symbol] and saved[symbol][key] is not None:
                        pos[key] = saved[symbol][key]
                # trailing_stop восстанавливаем только если трейлинг был активен;
                # pos["stop"] не перезаписываем трейлинг-стопом — биржевой стоп приоритетнее
                if saved[symbol].get("trailing_active") and saved[symbol].get("trailing_stop"):
                    pos["trailing_stop"] = saved[symbol]["trailing_stop"]
                    pos["trailing_active"] = True
                # Восстанавливаем stop/take только если биржа не вернула ордера
                if pos.get("stop") is None and saved[symbol].get("stop"):
                    pos["stop"] = saved[symbol]["stop"]
                if pos.get("take") is None and saved[symbol].get("take"):
                    pos["take"] = saved[symbol]["take"]

            restored += 1
            log.info(
                "[PM] Восстановлена: %s %s | объём=%.4f вход=%.4f стоп=%s тейк=%s | стратегия=%s",
                symbol, direction.upper(), quantity, entry,
                f"{stop:.4f}" if stop is not None else "нет",
                f"{take:.4f}" if take is not None else "нет",
                pos.get("strategy", "?"),
            )

            # Если SL или TP отсутствуют — пересчитываем по текущей цене
            if stop is None or take is None:
                self._recalc_sl_tp(pos, client, precision_cache)

        log.info("[PM] Восстановлено позиций: %d", restored)
        if restored > 0:
            self._save_positions()  # сохраняем только реально восстановленные (очищает устаревшие)

    def _recalc_sl_tp(self, pos: dict, client, precision_cache: dict) -> None:
        """
        Пересчитать SL/TP для восстановленной позиции у которой они неизвестны.
        Использует текущую цену как точку отсчёта.
        SL = текущая цена ± stop_buffer (небольшой буфер от текущей цены).
        TP = вход + (вход - SL) * reward_ratio
        """
        symbol = pos["symbol"]
        direction = pos["direction"]
        entry = pos["entry"]

        try:
            current_price = exchange.get_current_price(client, symbol)
            if current_price is None:
                return

            price_prec, qty_prec = exchange.get_precision(client, symbol, cache=precision_cache)

            # SL ставим за текущий экстремум: для LONG — ниже текущей цены на 1.5%, для SHORT — выше на 1.5%
            buffer = 0.015  # 1.5% отступ
            if direction == "long":
                stop = round(current_price * (1 - buffer), price_prec)
                take = round(entry + (entry - stop) * CONFIG.reward_ratio, price_prec)
            else:
                stop = round(current_price * (1 + buffer), price_prec)
                take = round(entry - (stop - entry) * CONFIG.reward_ratio, price_prec)

            # Проверяем что стоп не хуже входа (позиция уже в минусе)
            # Для LONG: если current_price < entry — стоп уже пройден, не трогаем
            if direction == "long" and stop >= entry:
                return
            if direction == "short" and stop <= entry:
                return

            pos["stop"] = stop
            pos["take"] = take
            log.info(
                "[PM] %s: SL/TP пересчитаны | %s | стоп=%.4f тейк=%.4f",
                symbol, direction.upper(), stop, take,
            )
        except Exception as e:
            log.warning("[PM] %s: не удалось пересчитать SL/TP: %s", symbol, e)
