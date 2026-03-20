"""
trade_history.py — Сохранение полной истории сделок в trades_history.json.

Каждая запись создаётся при открытии позиции (с сигнальной свечой, уровнями, 5 свечами до).
При закрытии дополняется: PnL, причина, 5 свечей после.
"""
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

log = logging.getLogger(__name__)

HISTORY_FILE = Path("logs/trades_history.json")


def _candle_dict(row) -> dict:
    """Преобразовать строку DataFrame в dict свечи."""
    return {
        "open":   float(row["open"]),
        "high":   float(row["high"]),
        "low":    float(row["low"]),
        "close":  float(row["close"]),
        "volume": float(row["volume"]),
    }


def _load() -> list:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save(records: list) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(HISTORY_FILE)  # атомарная замена


def open_trade(
    pos: dict,
    df: pd.DataFrame,
    support: float,
    resistance: float,
    conditions: dict,
) -> str:
    """
    Создать запись при открытии позиции.

    Аргументы:
        pos:        словарь позиции из PositionManager
        df:         DataFrame свечей в момент сигнала
        support:    уровень поддержки
        resistance: уровень сопротивления
        conditions: словарь условий стратегии (volume_ratio, body_ratio и т.д.)

    Возвращает:
        trade_id (str) — UUID записи, сохраняется в pos["trade_id"]
    """
    trade_id = str(uuid.uuid4())

    signal_candle = _candle_dict(df.iloc[-1])
    candles_before = [_candle_dict(df.iloc[i]) for i in range(max(0, len(df) - 6), len(df) - 1)]
    avg_vol = float(df.iloc[-1].get("avg_volume", 0) or 0)

    record = {
        "trade_id":          trade_id,
        "timestamp_open":    datetime.now().isoformat(timespec="seconds"),
        "timestamp_close":   None,
        "symbol":            pos["symbol"],
        "direction":         pos["direction"],
        "strategy":          pos.get("strategy", "unknown"),
        "signal": {
            "support":    support,
            "resistance": resistance,
            "conditions": conditions,
        },
        "signal_candle":        signal_candle,
        "candles_before":       candles_before,
        "candles_after":        [],
        "avg_volume_at_signal": avg_vol,
        "trade": {
            "entry":              pos.get("entry", 0.0),
            "stop":               pos.get("stop"),
            "take":               pos.get("take"),
            "quantity":           pos.get("quantity", 0.0),
            "leverage":           5,
            "risk_usd":           pos.get("risk_usd", 0.0),
            "pnl_usd":            None,
            "pnl_pct":            None,
            "result":             None,
            "close_reason":       None,
            "trailing_activated": False,
            "best_price":         None,
        },
    }

    records = _load()
    records.append(record)
    _save(records)
    log.info("[HISTORY] Открыта сделка %s | %s %s", trade_id[:8], pos["symbol"], pos["direction"].upper())
    return trade_id


def close_trade(
    trade_id: str,
    reason: str,
    pnl: dict,
    pos: dict,
    df_after: Optional[pd.DataFrame] = None,
) -> None:
    """
    Дополнить запись при закрытии позиции.

    Аргументы:
        trade_id:  UUID из pos["trade_id"]
        reason:    "tp" / "sl" / "trailing"
        pnl:       словарь {"pnl_usd": ..., "pnl_pct": ...}
        pos:       словарь позиции (для best_price, trailing_active)
        df_after:  свечи после закрытия (5 штук, опционально)
    """
    reason_map = {
        "tp":       "take_profit",
        "sl":       "stop_loss",
        "trailing": "trailing",
    }
    pnl_usd = pnl.get("pnl_usd", 0.0)

    if pnl_usd > 0 and reason == "trailing":
        result = "trailing_win"
    elif pnl_usd > 0:
        result = "win"
    else:
        result = "loss"

    candles_after = []
    if df_after is not None and not df_after.empty:
        candles_after = [_candle_dict(df_after.iloc[i]) for i in range(min(5, len(df_after)))]

    records = _load()
    for rec in records:
        if rec.get("trade_id") == trade_id:
            rec["timestamp_close"] = datetime.now().isoformat(timespec="seconds")
            rec["candles_after"]   = candles_after
            rec["trade"]["pnl_usd"]            = round(pnl_usd, 4)
            rec["trade"]["pnl_pct"]            = round(pnl.get("pnl_pct", 0.0), 4)
            rec["trade"]["result"]             = result
            rec["trade"]["close_reason"]       = reason_map.get(reason, reason)
            rec["trade"]["trailing_activated"] = pos.get("trailing_active", False)
            rec["trade"]["best_price"]         = pos.get("best_price")
            break
    else:
        log.warning("[HISTORY] trade_id %s не найден для закрытия", trade_id[:8])

    _save(records)
    log.info("[HISTORY] Закрыта сделка %s | %s | PnL=%+.2f", trade_id[:8], result, pnl_usd)
