"""
tests/test_positions_persistence.py — Тесты сохранения/загрузки позиций из файла.

Проверяет что:
  - PositionManager сохраняет позиции в positions.json при add/remove
  - При restore_from_exchange восстанавливаются стратегия, trade_id и другие поля
  - Биржевые stop/take приоритетнее сохранённых
  - Отсутствие файла не ломает restore
  - opened_at (datetime) сериализуется и восстанавливается как строка
"""
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import position_manager as pm_module
from position_manager import PositionManager


# ── Вспомогательные функции ────────────────────────────────────────────────

def _make_pos(symbol: str = "BTCUSDT", strategy: str = "bounce") -> dict:
    return {
        "symbol":          symbol,
        "order_id":        "123",
        "direction":       "long",
        "entry":           50000.0,
        "stop":            49000.0,
        "take":            53000.0,
        "quantity":        0.01,
        "risk_usd":        10.0,
        "opened_at":       "2026-01-01T00:00:00",
        "strategy":        strategy,
        "trade_id":        "trade-abc",
        "trailing_active": False,
        "trailing_stop":   None,
        "best_price":      50000.0,
    }


@pytest.fixture(autouse=True)
def patch_positions_file(tmp_path, monkeypatch):
    """Патчим _POSITIONS_FILE на временный путь для каждого теста."""
    tmp_file = tmp_path / "positions.json"
    monkeypatch.setattr(pm_module, "_POSITIONS_FILE", tmp_file)
    yield tmp_file


# ── Тесты ─────────────────────────────────────────────────────────────────

class TestSavePositions:
    def test_save_creates_file(self, patch_positions_file):
        """add() создаёт файл positions.json."""
        mgr = PositionManager()
        mgr.add(_make_pos("BTCUSDT"))
        assert patch_positions_file.exists(), "Файл positions.json не создан после add()"

    def test_save_contains_strategy(self, patch_positions_file):
        """Стратегия сохраняется в файл."""
        mgr = PositionManager()
        mgr.add(_make_pos("BTCUSDT", strategy="mirror"))
        data = json.loads(patch_positions_file.read_text())
        assert data["BTCUSDT"]["strategy"] == "mirror"

    def test_remove_updates_file(self, patch_positions_file):
        """remove() убирает символ из файла."""
        mgr = PositionManager()
        mgr.add(_make_pos("BTCUSDT"))
        mgr.add(_make_pos("ETHUSDT"))
        mgr.remove("BTCUSDT")
        data = json.loads(patch_positions_file.read_text())
        assert "BTCUSDT" not in data
        assert "ETHUSDT" in data

    def test_multiple_positions_saved(self, patch_positions_file):
        """Несколько позиций корректно сохраняются."""
        mgr = PositionManager()
        mgr.add(_make_pos("BTCUSDT", strategy="bounce"))
        mgr.add(_make_pos("ETHUSDT", strategy="false_breakout"))
        data = json.loads(patch_positions_file.read_text())
        assert len(data) == 2
        assert data["ETHUSDT"]["strategy"] == "false_breakout"


class TestRestorePositions:
    def _make_exchange_position(self, symbol: str = "BTCUSDT", amt: float = 0.01) -> dict:
        return {
            "symbol":       symbol,
            "positionAmt":  str(amt),
            "entryPrice":   "50000.0",
        }

    def _make_client(self, positions, orders=None):
        """Создать мок клиента Binance."""
        client = MagicMock()
        with patch.object(pm_module.exchange, "get_all_open_positions", return_value=positions), \
             patch.object(pm_module.exchange, "get_open_orders", return_value=orders or []), \
             patch.object(pm_module.exchange, "get_current_price", return_value=50000.0), \
             patch.object(pm_module.exchange, "get_precision", return_value=(2, 4)):
            yield client

    def test_restore_loads_strategy(self, patch_positions_file):
        """После save + restore стратегия восстанавливается."""
        mgr = PositionManager()
        mgr.add(_make_pos("BTCUSDT", strategy="mirror"))

        # Создаём новый менеджер и восстанавливаем
        mgr2 = PositionManager()
        with patch.object(pm_module.exchange, "get_all_open_positions",
                          return_value=[self._make_exchange_position("BTCUSDT")]), \
             patch.object(pm_module.exchange, "get_open_orders", return_value=[]), \
             patch.object(pm_module.exchange, "get_current_price", return_value=50000.0), \
             patch.object(pm_module.exchange, "get_precision", return_value=(2, 4)):
            mgr2.restore_from_exchange(MagicMock(), {})

        pos = mgr2.get("BTCUSDT")
        assert pos is not None
        assert pos["strategy"] == "mirror", f"Ожидалась стратегия 'mirror', получили '{pos.get('strategy')}'"

    def test_restore_loads_trade_id(self, patch_positions_file):
        """trade_id восстанавливается из файла."""
        mgr = PositionManager()
        pos = _make_pos("BTCUSDT")
        pos["trade_id"] = "trade-xyz-999"
        mgr.add(pos)

        mgr2 = PositionManager()
        with patch.object(pm_module.exchange, "get_all_open_positions",
                          return_value=[self._make_exchange_position("BTCUSDT")]), \
             patch.object(pm_module.exchange, "get_open_orders", return_value=[]), \
             patch.object(pm_module.exchange, "get_current_price", return_value=50000.0), \
             patch.object(pm_module.exchange, "get_precision", return_value=(2, 4)):
            mgr2.restore_from_exchange(MagicMock(), {})

        pos2 = mgr2.get("BTCUSDT")
        assert pos2 is not None
        assert pos2["trade_id"] == "trade-xyz-999"

    def test_restore_prefers_exchange_stop_take(self, patch_positions_file):
        """Если биржа вернула stop/take, они приоритетнее сохранённых."""
        mgr = PositionManager()
        pos = _make_pos("BTCUSDT")
        pos["stop"] = 48000.0
        pos["take"] = 54000.0
        mgr.add(pos)

        # Биржа возвращает другой стоп
        exchange_orders = [
            {"type": "STOP_MARKET", "stopPrice": "47000.0"},
            {"type": "TAKE_PROFIT_MARKET", "stopPrice": "55000.0"},
        ]

        mgr2 = PositionManager()
        with patch.object(pm_module.exchange, "get_all_open_positions",
                          return_value=[self._make_exchange_position("BTCUSDT")]), \
             patch.object(pm_module.exchange, "get_open_orders", return_value=exchange_orders), \
             patch.object(pm_module.exchange, "get_current_price", return_value=50000.0), \
             patch.object(pm_module.exchange, "get_precision", return_value=(2, 4)):
            mgr2.restore_from_exchange(MagicMock(), {})

        pos2 = mgr2.get("BTCUSDT")
        assert pos2 is not None
        # Биржевые значения должны победить
        assert pos2["stop"] == pytest.approx(47000.0), f"stop должен быть 47000 (биржевой), а не {pos2['stop']}"
        assert pos2["take"] == pytest.approx(55000.0), f"take должен быть 55000 (биржевой), а не {pos2['take']}"

    def test_missing_file_doesnt_crash(self, patch_positions_file):
        """Если файла positions.json нет, restore работает нормально."""
        # Убеждаемся что файла нет
        assert not patch_positions_file.exists()

        mgr = PositionManager()
        with patch.object(pm_module.exchange, "get_all_open_positions",
                          return_value=[self._make_exchange_position("BTCUSDT")]), \
             patch.object(pm_module.exchange, "get_open_orders", return_value=[]), \
             patch.object(pm_module.exchange, "get_current_price", return_value=50000.0), \
             patch.object(pm_module.exchange, "get_precision", return_value=(2, 4)):
            mgr.restore_from_exchange(MagicMock(), {})

        pos = mgr.get("BTCUSDT")
        assert pos is not None
        assert pos["strategy"] == "restored"

    def test_restore_no_positions_on_exchange(self, patch_positions_file):
        """Если биржа вернула пустой список, restore не падает."""
        mgr = PositionManager()
        with patch.object(pm_module.exchange, "get_all_open_positions", return_value=[]):
            mgr.restore_from_exchange(MagicMock(), {})

        assert mgr.count() == 0

    def test_restore_uses_saved_stop_when_exchange_has_none(self, patch_positions_file):
        """Если биржа не вернула стоп, берём сохранённый (до вызова _recalc_sl_tp)."""
        mgr = PositionManager()
        pos = _make_pos("BTCUSDT")
        pos["stop"] = 48500.0
        pos["take"] = 53500.0
        mgr.add(pos)

        mgr2 = PositionManager()
        # Биржа не возвращает ордера (stop/take = None).
        # Патчим _recalc_sl_tp чтобы он не перезаписал stop/take из файла.
        with patch.object(pm_module.exchange, "get_all_open_positions",
                          return_value=[self._make_exchange_position("BTCUSDT")]), \
             patch.object(pm_module.exchange, "get_open_orders", return_value=[]), \
             patch.object(pm_module.exchange, "get_current_price", return_value=50000.0), \
             patch.object(pm_module.exchange, "get_precision", return_value=(2, 4)), \
             patch.object(mgr2, "_recalc_sl_tp"):
            mgr2.restore_from_exchange(MagicMock(), {})

        pos2 = mgr2.get("BTCUSDT")
        assert pos2 is not None
        assert pos2["stop"] == pytest.approx(48500.0), f"stop должен быть из файла 48500, а не {pos2['stop']}"
        assert pos2["take"] == pytest.approx(53500.0), f"take должен быть из файла 53500, а не {pos2['take']}"

    def test_opened_at_survives_restart(self, patch_positions_file):
        """opened_at в виде datetime сохраняется как ISO-строка и восстанавливается."""
        mgr = PositionManager()
        pos = _make_pos("BTCUSDT")
        # Устанавливаем opened_at как объект datetime (реальный сценарий)
        dt = datetime(2026, 3, 18, 12, 30, 0)
        pos["opened_at"] = dt
        mgr.add(pos)

        # Проверяем что в файле хранится строка, а не падает сериализация
        data = json.loads(patch_positions_file.read_text())
        assert "opened_at" in data["BTCUSDT"], "opened_at должен сохраняться в файл"
        assert data["BTCUSDT"]["opened_at"] == "2026-03-18T12:30:00", \
            f"opened_at должен быть ISO-строкой, получили: {data['BTCUSDT']['opened_at']}"

        # Восстанавливаем через новый менеджер — поле должно вернуться как строка
        mgr2 = PositionManager()
        with patch.object(pm_module.exchange, "get_all_open_positions",
                          return_value=[self._make_exchange_position("BTCUSDT")]), \
             patch.object(pm_module.exchange, "get_open_orders", return_value=[]), \
             patch.object(pm_module.exchange, "get_current_price", return_value=50000.0), \
             patch.object(pm_module.exchange, "get_precision", return_value=(2, 4)):
            mgr2.restore_from_exchange(MagicMock(), {})

        pos2 = mgr2.get("BTCUSDT")
        assert pos2 is not None
        assert pos2["opened_at"] == "2026-03-18T12:30:00", \
            f"opened_at должен восстановиться как строка, получили: {pos2['opened_at']}"
