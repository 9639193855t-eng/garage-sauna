"""
stats_tracker.py — Учёт закрытых сделок.

Хранит историю в JSON-файле logs/trades.json.
Структура одной записи:
{
    "symbol": "BTCUSDT",
    "direction": "long",
    "entry": 50000.0,
    "exit": 51500.0,
    "quantity": 0.001,
    "pnl_usd": 15.0,
    "pnl_pct": 3.0,     # % от входа без учёта плеча
    "strategy": "mirror",
    "reason": "tp",      # sl / tp / trailing
    "closed_at": "2026-03-17T15:30:00"   # UTC ISO
}
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class StatsTracker:
    """
    Трекер закрытых сделок.

    Загружает и сохраняет историю в JSON-файл.
    Предоставляет агрегированную статистику за сегодня и за всё время.
    """

    def __init__(self, filepath: str = "logs/trades.json") -> None:
        self._lock = threading.Lock()
        self._filepath = filepath
        self._trades: list[dict] = []
        self._load()

    # ── Загрузка / сохранение ─────────────────────────────────────────────────

    def _load(self) -> None:
        """Загрузить историю из JSON-файла. Если файл не существует — начать с пустого списка."""
        if not os.path.exists(self._filepath):
            self._trades = []
            return
        try:
            with open(self._filepath, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._trades = data
            else:
                log.warning("[STATS] Неожиданный формат файла %s, сброс", self._filepath)
                self._trades = []
        except Exception as exc:
            log.warning("[STATS] Не удалось загрузить %s: %s, начинаем с нуля", self._filepath, exc)
            self._trades = []

    def _save(self) -> None:
        """Сохранить текущую историю в JSON-файл (атомарно через временный файл)."""
        try:
            dirpath = os.path.dirname(self._filepath)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            tmp_path = self._filepath + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._trades, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._filepath)
        except Exception as exc:
            log.warning("[STATS] Не удалось сохранить %s: %s", self._filepath, exc)

    # ── Запись сделки ─────────────────────────────────────────────────────────

    def record(self, pos: dict, reason: str, exit_price: float, pnl: dict) -> None:
        """
        Записать закрытую сделку.

        Аргументы:
            pos:        Словарь позиции (symbol, direction, entry, quantity, strategy).
            reason:     Причина закрытия: "sl", "tp" или "trailing".
            exit_price: Цена закрытия.
            pnl:        Словарь {'pnl_usd': float, 'pnl_pct': float} из calculate_pnl.
        """
        trade = {
            "symbol":     pos.get("symbol", ""),
            "direction":  pos.get("direction", ""),
            "entry":      pos.get("entry", 0.0),
            "exit":       exit_price,
            "quantity":   pos.get("quantity", 0.0),
            "pnl_usd":    pnl.get("pnl_usd", 0.0),
            "pnl_pct":    pnl.get("pnl_pct", 0.0),
            "strategy":   pos.get("strategy", ""),
            "reason":     reason,
            "closed_at":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self._lock:
            self._trades.append(trade)
            self._save()
        log.info(
            "[STATS] Записана сделка: %s %s [%s] PnL=%+.2f$ (%+.2f%%)",
            trade["symbol"], trade["direction"].upper(), reason,
            trade["pnl_usd"], trade["pnl_pct"],
        )

    # ── Агрегация ─────────────────────────────────────────────────────────────

    def today_trades(self) -> list[dict]:
        """Вернуть список сделок за сегодня (UTC-дата)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            return [t for t in self._trades if t.get("closed_at", "").startswith(today)]

    def today_pnl_usd(self) -> float:
        """Суммарный PnL в USDT за сегодня."""
        with self._lock:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            return sum(
                t.get("pnl_usd", 0.0)
                for t in self._trades
                if t.get("closed_at", "").startswith(today)
            )

    def today_pnl_pct(self) -> float:
        """Средний PnL % на сделку за сегодня."""
        with self._lock:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            trades = [t for t in self._trades if t.get("closed_at", "").startswith(today)]
        if not trades:
            return 0.0
        return sum(t["pnl_pct"] for t in trades) / len(trades)

    def alltime_pnl_usd(self) -> float:
        """Суммарный PnL в USDT за всё время."""
        with self._lock:
            return sum(t.get("pnl_usd", 0.0) for t in self._trades)

    def alltime_pnl_pct(self) -> float:
        """Средний PnL % на сделку за всё время."""
        with self._lock:
            if not self._trades:
                return 0.0
            return sum(t["pnl_pct"] for t in self._trades) / len(self._trades)

    def total_count(self) -> int:
        """Общее количество записанных сделок."""
        with self._lock:
            return len(self._trades)
