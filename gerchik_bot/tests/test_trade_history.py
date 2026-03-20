"""
tests/test_trade_history.py — Тесты модуля trade_history.py.
"""
import json
import logging
import uuid
from pathlib import Path

import pandas as pd
import pytest

import trade_history as th


def _make_df(n: int = 10) -> pd.DataFrame:
    """Создать простой DataFrame с n свечами."""
    return pd.DataFrame({
        "open":   [100.0 + i for i in range(n)],
        "high":   [105.0 + i for i in range(n)],
        "low":    [95.0 + i for i in range(n)],
        "close":  [102.0 + i for i in range(n)],
        "volume": [1000.0 + i * 10 for i in range(n)],
    })


def _make_pos(symbol: str = "BTCUSDT", direction: str = "long") -> dict:
    return {
        "symbol":    symbol,
        "direction": direction,
        "strategy":  "bounce",
        "entry":     50000.0,
        "stop":      49000.0,
        "take":      53000.0,
        "quantity":  0.01,
        "risk_usd":  10.0,
    }


@pytest.fixture(autouse=True)
def patch_history_file(tmp_path, monkeypatch):
    """Перенаправить HISTORY_FILE во временную директорию для каждого теста."""
    fake_file = tmp_path / "logs" / "trades_history.json"
    monkeypatch.setattr(th, "HISTORY_FILE", Path(fake_file))
    return fake_file


# ─────────────────────────────────────────────────────────────────────────────
# Тесты open_trade
# ─────────────────────────────────────────────────────────────────────────────

def test_open_trade_creates_record():
    """open_trade создаёт файл с одной записью."""
    pos = _make_pos()
    df  = _make_df()
    th.open_trade(pos, df, support=48000.0, resistance=52000.0, conditions={})

    records = th._load()
    assert len(records) == 1
    rec = records[0]
    assert rec["symbol"] == "BTCUSDT"
    assert rec["direction"] == "long"
    assert rec["strategy"] == "bounce"
    assert rec["trade"]["entry"] == 50000.0
    assert rec["timestamp_close"] is None


def test_open_trade_returns_uuid():
    """open_trade возвращает валидный UUID."""
    pos = _make_pos()
    df  = _make_df()
    trade_id = th.open_trade(pos, df, support=48000.0, resistance=52000.0, conditions={})

    assert isinstance(trade_id, str)
    # Проверяем что это валидный UUID v4
    parsed = uuid.UUID(trade_id, version=4)
    assert str(parsed) == trade_id


def test_candles_before_max_5():
    """candles_before содержит не более 5 свечей."""
    pos = _make_pos()
    df  = _make_df(20)
    th.open_trade(pos, df, support=48000.0, resistance=52000.0, conditions={})

    records = th._load()
    assert len(records[0]["candles_before"]) <= 5


def test_candles_before_with_small_df():
    """С маленьким df (3 свечи) candles_before содержит 2 свечи (все кроме последней)."""
    pos = _make_pos()
    df  = _make_df(3)
    th.open_trade(pos, df, support=48000.0, resistance=52000.0, conditions={})

    records = th._load()
    assert len(records[0]["candles_before"]) == 2


def test_file_persists_multiple_trades():
    """Несколько вызовов open_trade добавляют записи, не перезаписывают файл."""
    pos1 = _make_pos("BTCUSDT", "long")
    pos2 = _make_pos("ETHUSDT", "short")
    df   = _make_df()

    th.open_trade(pos1, df, 48000.0, 52000.0, {})
    th.open_trade(pos2, df, 3000.0, 3500.0, {})

    records = th._load()
    assert len(records) == 2
    symbols = {r["symbol"] for r in records}
    assert symbols == {"BTCUSDT", "ETHUSDT"}


# ─────────────────────────────────────────────────────────────────────────────
# Тесты close_trade
# ─────────────────────────────────────────────────────────────────────────────

def test_close_trade_fills_pnl():
    """close_trade заполняет pnl_usd и pnl_pct."""
    pos = _make_pos()
    df  = _make_df()
    trade_id = th.open_trade(pos, df, 48000.0, 52000.0, {})

    pnl = {"pnl_usd": 30.0, "pnl_pct": 0.06}
    th.close_trade(trade_id, "tp", pnl, pos)

    records = th._load()
    rec = records[0]
    assert rec["trade"]["pnl_usd"] == 30.0
    assert rec["trade"]["pnl_pct"] == 0.06
    assert rec["timestamp_close"] is not None


def test_close_trade_result_win():
    """result=win при pnl_usd > 0 и reason != trailing."""
    pos = _make_pos()
    df  = _make_df()
    trade_id = th.open_trade(pos, df, 48000.0, 52000.0, {})

    th.close_trade(trade_id, "tp", {"pnl_usd": 20.0, "pnl_pct": 0.04}, pos)

    records = th._load()
    assert records[0]["trade"]["result"] == "win"
    assert records[0]["trade"]["close_reason"] == "take_profit"


def test_close_trade_result_loss():
    """result=loss при pnl_usd < 0."""
    pos = _make_pos()
    df  = _make_df()
    trade_id = th.open_trade(pos, df, 48000.0, 52000.0, {})

    th.close_trade(trade_id, "sl", {"pnl_usd": -10.0, "pnl_pct": -0.02}, pos)

    records = th._load()
    assert records[0]["trade"]["result"] == "loss"
    assert records[0]["trade"]["close_reason"] == "stop_loss"


def test_close_trade_result_trailing_win():
    """result=trailing_win при pnl_usd > 0 и reason == trailing."""
    pos = _make_pos()
    df  = _make_df()
    trade_id = th.open_trade(pos, df, 48000.0, 52000.0, {})

    th.close_trade(trade_id, "trailing", {"pnl_usd": 15.0, "pnl_pct": 0.03}, pos)

    records = th._load()
    assert records[0]["trade"]["result"] == "trailing_win"
    assert records[0]["trade"]["close_reason"] == "trailing"


def test_close_trade_unknown_id_warns(caplog):
    """close_trade с несуществующим trade_id не падает, только логирует warning."""
    fake_id = str(uuid.uuid4())
    pos = _make_pos()

    with caplog.at_level(logging.WARNING, logger="trade_history"):
        th.close_trade(fake_id, "sl", {"pnl_usd": -5.0, "pnl_pct": -0.01}, pos)

    # Проверяем что было залогировано предупреждение
    assert any("не найден" in m for m in caplog.messages)


def test_candles_after_max_5():
    """candles_after содержит не более 5 свечей."""
    pos = _make_pos()
    df  = _make_df()
    trade_id = th.open_trade(pos, df, 48000.0, 52000.0, {})

    df_after = _make_df(10)
    th.close_trade(trade_id, "tp", {"pnl_usd": 10.0, "pnl_pct": 0.02}, pos, df_after=df_after)

    records = th._load()
    assert len(records[0]["candles_after"]) == 5


def test_candles_after_none():
    """Если df_after=None — candles_after пустой список."""
    pos = _make_pos()
    df  = _make_df()
    trade_id = th.open_trade(pos, df, 48000.0, 52000.0, {})

    th.close_trade(trade_id, "tp", {"pnl_usd": 10.0, "pnl_pct": 0.02}, pos, df_after=None)

    records = th._load()
    assert records[0]["candles_after"] == []
