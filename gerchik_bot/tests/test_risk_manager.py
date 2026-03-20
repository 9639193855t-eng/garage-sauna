"""Тесты для risk_manager.py — расчёт риска, трейлинг и проверка выхода."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch

from risk_manager import (
    calculate_trade_params,
    update_trailing_stop,
    check_exit,
    calculate_pnl,
    is_params_valid,
    make_position,
)
from config import CONFIG


# ─── Вспомогательная функция ─────────────────────────────────────────────────

def make_pos(
    direction: str = "long",
    entry: float = 100.0,
    stop: float = 95.0,
    take: float = 115.0,
    quantity: float = 1.0,
    best_price: float = 100.0,
    trailing_active: bool = False,
    trailing_stop: float = None,
) -> dict:
    return {
        "direction":      direction,
        "entry":          entry,
        "stop":           stop,
        "take":           take,
        "quantity":       quantity,
        "best_price":     best_price,
        "trailing_active": trailing_active,
        "trailing_stop":  trailing_stop,
    }


# ─── calculate_trade_params ───────────────────────────────────────────────────

class TestCalculateTradeParams:

    def test_long_basic(self):
        """LONG: стоп ниже экстремума, тейк выше входа."""
        params = calculate_trade_params("long", 100.0, 97.0, 1000.0, 2, 3)
        assert params
        assert params["stop"] < 97.0           # стоп ниже экстремума (с буфером)
        assert params["take"] > 100.0          # тейк выше входа
        assert params["quantity"] > 0

    def test_short_basic(self):
        """SHORT: стоп выше экстремума, тейк ниже входа."""
        params = calculate_trade_params("short", 100.0, 103.0, 1000.0, 2, 3)
        assert params
        assert params["stop"] > 103.0          # стоп выше экстремума
        assert params["take"] < 100.0          # тейк ниже входа
        assert params["quantity"] > 0

    def test_risk_ratio_is_3(self):
        """Соотношение прибыль/риск должно быть 1:3."""
        params = calculate_trade_params("long", 100.0, 97.0, 1000.0, 4, 4)
        assert params
        risk   = params["entry"] - params["stop"]
        profit = params["take"]  - params["entry"]
        assert abs(profit / risk - CONFIG.reward_ratio) < 0.01

    def test_short_risk_ratio(self):
        params = calculate_trade_params("short", 100.0, 103.0, 1000.0, 4, 4)
        assert params
        risk   = params["stop"] - params["entry"]
        profit = params["entry"] - params["take"]
        assert abs(profit / risk - CONFIG.reward_ratio) < 0.01

    def test_risk_usd_is_1pct_of_balance(self):
        """Риск в USDT = 1% от баланса."""
        balance = 1000.0
        params  = calculate_trade_params("long", 100.0, 97.0, balance, 2, 3)
        expected_risk_usd = balance * CONFIG.risk_per_trade
        assert abs(params["risk_usd"] - expected_risk_usd) < 0.01

    def test_leverage_in_quantity_formula(self):
        """
        При бо́льшем плече размер позиции должен быть МЕНЬШЕ
        (т.к. каждый контракт несёт больший риск).
        """
        p1 = calculate_trade_params("long", 100.0, 97.0, 1000.0, 2, 6)
        with patch.object(CONFIG, "leverage", CONFIG.leverage * 2):
            p2 = calculate_trade_params("long", 100.0, 97.0, 1000.0, 2, 6)
        # При 2x плече quantity должно быть ~вдвое меньше
        assert p1["quantity"] > p2["quantity"]

    def test_max_position_size_cap(self):
        """Позиция не должна превышать max_position_size_pct × баланс × плечо."""
        balance      = 1000.0
        entry        = 100.0
        max_notional = balance * CONFIG.leverage * CONFIG.max_position_size_pct
        max_qty      = max_notional / entry

        # Берём extremum очень близко к entry чтобы quantity вышел большим
        params = calculate_trade_params("long", entry, entry * 0.9999, balance, 4, 4)
        if params:
            actual_notional = params["quantity"] * entry
            assert actual_notional <= max_notional * 1.01  # допуск 1% на округление

    def test_returns_empty_for_invalid_direction(self):
        params = calculate_trade_params("sideways", 100.0, 97.0, 1000.0)
        assert params == {}

    def test_returns_empty_for_zero_entry(self):
        params = calculate_trade_params("long", 0.0, 97.0, 1000.0)
        assert params == {}

    def test_returns_empty_for_zero_balance(self):
        params = calculate_trade_params("long", 100.0, 97.0, 0.0)
        assert params == {}

    def test_returns_empty_when_extremum_past_entry(self):
        """
        Если экстремум уже за входом (цена ушла) — риск отрицательный.
        Например для LONG: extremum > entry → стоп ВЫШЕ входа → риск < 0.
        """
        params = calculate_trade_params("long", 100.0, 105.0, 1000.0)
        assert params == {}

    def test_price_precision_applied(self):
        """Цены должны быть округлены до price_precision знаков."""
        params = calculate_trade_params("long", 41000.123456, 40800.0, 1000.0, 1, 3)
        if params:
            # Знаков после запятой должно быть <= 1
            stop_decimals = len(str(params["stop"]).split(".")[-1]) if "." in str(params["stop"]) else 0
            assert stop_decimals <= 1

    def test_qty_precision_applied(self):
        """Объём должен быть округлён до qty_precision знаков."""
        params = calculate_trade_params("long", 100.0, 97.0, 1000.0, 2, 2)
        if params:
            qty_str = str(params["quantity"])
            decimals = len(qty_str.split(".")[-1]) if "." in qty_str else 0
            assert decimals <= 2

    def test_returns_all_required_keys(self):
        params = calculate_trade_params("long", 100.0, 97.0, 1000.0)
        for key in ("entry", "stop", "take", "quantity", "risk", "risk_usd", "ratio"):
            assert key in params, f"Ключ {key!r} отсутствует"

    def test_ratio_approximately_reward_ratio(self):
        params = calculate_trade_params("short", 100.0, 103.0, 1000.0, 4, 4)
        assert params
        assert abs(params["ratio"] - CONFIG.reward_ratio) < 0.05


# ─── update_trailing_stop ─────────────────────────────────────────────────────

class TestUpdateTrailingStop:

    def test_not_activated_at_entry(self):
        """Трейлинг не активируется сразу при входе."""
        pos = make_pos("long", entry=100.0, best_price=100.0)
        update_trailing_stop(pos, current_price=100.0)
        assert not pos["trailing_active"]

    def test_long_activates_when_profit_reached(self):
        """LONG: трейлинг активируется когда цена выросла на activation_pct."""
        pos = make_pos("long", entry=100.0, best_price=100.0)
        activation_price = 100.0 * (1 + CONFIG.trailing_activation_pct + 0.001)
        update_trailing_stop(pos, current_price=activation_price)
        assert pos["trailing_active"]

    def test_short_activates_when_profit_reached(self):
        """SHORT: трейлинг активируется когда цена упала на activation_pct."""
        pos = make_pos("short", entry=100.0, best_price=100.0)
        activation_price = 100.0 * (1 - CONFIG.trailing_activation_pct - 0.001)
        update_trailing_stop(pos, current_price=activation_price)
        assert pos["trailing_active"]

    def test_long_trailing_below_best_price(self):
        """LONG: трейлинг-стоп должен быть ниже лучшей цены."""
        pos = make_pos("long", entry=100.0, best_price=110.0, trailing_active=True)
        update_trailing_stop(pos, current_price=110.0)
        assert pos["trailing_stop"] is not None
        assert pos["trailing_stop"] < 110.0

    def test_short_trailing_above_best_price(self):
        """SHORT: трейлинг-стоп должен быть выше лучшей цены."""
        pos = make_pos("short", entry=100.0, best_price=90.0, trailing_active=True)
        update_trailing_stop(pos, current_price=90.0)
        assert pos["trailing_stop"] is not None
        assert pos["trailing_stop"] > 90.0

    def test_long_trailing_moves_up_with_price(self):
        """LONG: трейлинг поднимается вместе с ценой."""
        pos = make_pos("long", entry=100.0, best_price=110.0, trailing_active=True, trailing_stop=108.0)
        update_trailing_stop(pos, current_price=115.0)  # новый максимум
        assert pos["trailing_stop"] > 108.0

    def test_long_trailing_does_not_move_down(self):
        """LONG: трейлинг НЕ опускается если цена откатила."""
        pos = make_pos("long", entry=100.0, best_price=115.0, trailing_active=True, trailing_stop=112.0)
        update_trailing_stop(pos, current_price=110.0)  # откат
        assert pos["trailing_stop"] >= 112.0

    def test_short_trailing_moves_down_with_price(self):
        """SHORT: трейлинг опускается вместе с падением цены."""
        pos = make_pos("short", entry=100.0, best_price=90.0, trailing_active=True, trailing_stop=93.0)
        update_trailing_stop(pos, current_price=85.0)  # новый минимум
        assert pos["trailing_stop"] < 93.0

    def test_short_trailing_does_not_move_up(self):
        """SHORT: трейлинг НЕ поднимается если цена выросла."""
        pos = make_pos("short", entry=100.0, best_price=85.0, trailing_active=True, trailing_stop=87.0)
        update_trailing_stop(pos, current_price=90.0)  # рост
        assert pos["trailing_stop"] <= 87.0

    def test_best_price_updated_for_long(self):
        """best_price обновляется на новых максимумах для LONG."""
        pos = make_pos("long", entry=100.0, best_price=105.0)
        update_trailing_stop(pos, current_price=110.0)
        assert pos["best_price"] == 110.0

    def test_best_price_not_updated_on_pullback_long(self):
        """best_price не снижается на откате для LONG."""
        pos = make_pos("long", entry=100.0, best_price=110.0)
        update_trailing_stop(pos, current_price=105.0)
        assert pos["best_price"] == 110.0

    def test_disabled_if_config_off(self):
        """Если трейлинг выключен в конфиге — ничего не меняется."""
        pos = make_pos("long", entry=100.0, best_price=115.0)
        with patch.object(CONFIG, "trailing_stop_enabled", False):
            update_trailing_stop(pos, current_price=115.0)
        assert not pos.get("trailing_active")
        assert pos.get("trailing_stop") is None

    def test_precision_applied(self):
        """Уровень трейлинга округляется до price_precision знаков."""
        pos = make_pos("long", entry=100.0, best_price=110.0, trailing_active=True)
        update_trailing_stop(pos, current_price=110.0, price_precision=2)
        ts = pos.get("trailing_stop")
        if ts is not None:
            decimals = len(str(ts).split(".")[-1]) if "." in str(ts) else 0
            assert decimals <= 2


# ─── check_exit ───────────────────────────────────────────────────────────────

class TestCheckExit:

    def test_no_exit_in_middle(self):
        """Позиция жива — нет выхода."""
        pos = make_pos("long", stop=90.0, take=115.0)
        assert check_exit(pos, current_price=100.0) is None

    # ── LONG ──

    def test_long_stop_triggered(self):
        pos = make_pos("long", stop=90.0, take=115.0)
        result = check_exit(pos, current_price=89.0)
        assert result == "СТОП-ЛОСС"

    def test_long_stop_at_exactly_stop_price(self):
        pos = make_pos("long", stop=90.0, take=115.0)
        result = check_exit(pos, current_price=90.0)
        assert result == "СТОП-ЛОСС"

    def test_long_take_triggered(self):
        pos = make_pos("long", stop=90.0, take=115.0)
        result = check_exit(pos, current_price=115.0)
        assert result == "ТЕЙК-ПРОФИТ"

    def test_long_take_above_price(self):
        pos = make_pos("long", stop=90.0, take=115.0)
        result = check_exit(pos, current_price=116.0)
        assert result == "ТЕЙК-ПРОФИТ"

    # ── SHORT ──

    def test_short_stop_triggered(self):
        pos = make_pos("short", stop=110.0, take=85.0)
        result = check_exit(pos, current_price=111.0)
        assert result == "СТОП-ЛОСС"

    def test_short_stop_at_exactly_stop_price(self):
        pos = make_pos("short", stop=110.0, take=85.0)
        result = check_exit(pos, current_price=110.0)
        assert result == "СТОП-ЛОСС"

    def test_short_take_triggered(self):
        pos = make_pos("short", stop=110.0, take=85.0)
        result = check_exit(pos, current_price=85.0)
        assert result == "ТЕЙК-ПРОФИТ"

    def test_short_take_below_price(self):
        pos = make_pos("short", stop=110.0, take=85.0)
        result = check_exit(pos, current_price=84.0)
        assert result == "ТЕЙК-ПРОФИТ"

    # ── Трейлинг-стоп ──

    def test_long_trailing_stop_used_when_active(self):
        """Если трейлинг активен — используем его, а не фиксированный стоп."""
        pos = make_pos("long", stop=90.0, take=115.0,
                       trailing_active=True, trailing_stop=105.0)
        result = check_exit(pos, current_price=104.0)
        assert result == "ТРЕЙЛИНГ-СТОП"

    def test_long_fixed_stop_when_trailing_not_active(self):
        """Если трейлинг НЕ активен — используем фиксированный стоп."""
        pos = make_pos("long", stop=90.0, take=115.0,
                       trailing_active=False, trailing_stop=105.0)
        result = check_exit(pos, current_price=95.0)
        assert result is None  # цена выше фиксированного стопа (90)

    def test_short_trailing_stop_used_when_active(self):
        pos = make_pos("short", stop=110.0, take=85.0,
                       trailing_active=True, trailing_stop=95.0)
        result = check_exit(pos, current_price=96.0)
        assert result == "ТРЕЙЛИНГ-СТОП"

    def test_no_exit_if_stop_is_none(self):
        """Если стоп не задан (восстановление после рестарта) — нет выхода по стопу."""
        pos = make_pos("long", stop=None, take=115.0)
        pos["stop"] = None
        result = check_exit(pos, current_price=50.0)
        # Нет выхода по стопу, только по тейку
        assert result is None or result == "ТЕЙК-ПРОФИТ"

    def test_no_exit_if_take_is_none(self):
        pos = make_pos("long", stop=90.0, take=None)
        pos["take"] = None
        result = check_exit(pos, current_price=200.0)
        assert result is None  # нет тейка — не закрываем


# ─── calculate_pnl ───────────────────────────────────────────────────────────

class TestCalculatePnl:

    def test_long_profit(self):
        """LONG: цена выросла — прибыль."""
        result = calculate_pnl("long", 100.0, 110.0, 1.0)
        assert result["is_profit"] is True
        assert result["pnl_usd"] > 0

    def test_long_loss(self):
        """LONG: цена упала — убыток."""
        result = calculate_pnl("long", 100.0, 90.0, 1.0)
        assert result["is_profit"] is False
        assert result["pnl_usd"] < 0

    def test_short_profit(self):
        """SHORT: цена упала — прибыль."""
        result = calculate_pnl("short", 100.0, 90.0, 1.0)
        assert result["is_profit"] is True
        assert result["pnl_usd"] > 0

    def test_short_loss(self):
        """SHORT: цена выросла — убыток."""
        result = calculate_pnl("short", 100.0, 110.0, 1.0)
        assert result["is_profit"] is False
        assert result["pnl_usd"] < 0

    def test_leverage_multiplies_pnl(self):
        """Плечо умножает PnL."""
        result = calculate_pnl("long", 100.0, 110.0, 1.0)
        expected_pnl = 10.0 * 1.0 * CONFIG.leverage
        assert abs(result["pnl_usd"] - expected_pnl) < 0.001

    def test_pnl_points_long(self):
        result = calculate_pnl("long", 100.0, 115.0, 1.0)
        assert result["pnl_points"] == pytest.approx(15.0)

    def test_pnl_points_short(self):
        result = calculate_pnl("short", 100.0, 85.0, 1.0)
        assert result["pnl_points"] == pytest.approx(15.0)

    def test_pnl_zero_at_entry(self):
        result = calculate_pnl("long", 100.0, 100.0, 1.0)
        assert result["pnl_usd"] == 0.0
        assert result["pnl_points"] == 0.0

    def test_returns_all_keys(self):
        result = calculate_pnl("long", 100.0, 110.0, 1.0)
        for key in ("pnl_usd", "pnl_pct", "pnl_points", "is_profit"):
            assert key in result

    def test_pnl_pct_at_tp(self):
        """При тейке 1:3 и риске 1% → прибыль примерно 3% × leverage."""
        result = calculate_pnl("long", 100.0, 115.0, 0.1)
        assert result["pnl_pct"] > 0


# ─── is_params_valid ─────────────────────────────────────────────────────────

class TestIsParamsValid:

    def test_valid_params(self):
        params = {"entry": 100.0, "stop": 95.0, "take": 115.0, "quantity": 0.1, "risk": 5.0}
        assert is_params_valid(params) is True

    def test_missing_key(self):
        params = {"entry": 100.0, "stop": 95.0, "take": 115.0, "quantity": 0.1}
        # "risk" отсутствует
        assert is_params_valid(params) is False

    def test_zero_risk(self):
        params = {"entry": 100.0, "stop": 95.0, "take": 115.0, "quantity": 0.1, "risk": 0.0}
        assert is_params_valid(params) is False

    def test_negative_risk(self):
        params = {"entry": 100.0, "stop": 95.0, "take": 115.0, "quantity": 0.1, "risk": -5.0}
        assert is_params_valid(params) is False

    def test_zero_quantity(self):
        params = {"entry": 100.0, "stop": 95.0, "take": 115.0, "quantity": 0.0, "risk": 5.0}
        assert is_params_valid(params) is False

    def test_empty_dict(self):
        assert is_params_valid({}) is False


# ─── make_position ───────────────────────────────────────────────────────────

class TestMakePosition:

    def _params(self) -> dict:
        return {"entry": 100.0, "stop": 95.0, "take": 115.0, "quantity": 0.1, "risk": 5.0, "risk_usd": 10.0}

    def test_returns_dict_with_required_keys(self):
        pos = make_position("BTCUSDT", "long", self._params(), "ложный_пробой", "111")
        for key in ("symbol", "direction", "entry", "stop", "take", "quantity",
                    "opened_at", "strategy", "trailing_active", "trailing_stop", "best_price"):
            assert key in pos, f"Ключ {key!r} отсутствует"

    def test_trailing_not_active_by_default(self):
        pos = make_position("BTCUSDT", "long", self._params())
        assert pos["trailing_active"] is False
        assert pos["trailing_stop"] is None

    def test_best_price_equals_entry(self):
        pos = make_position("BTCUSDT", "long", self._params())
        assert pos["best_price"] == pos["entry"]

    def test_symbol_and_direction_stored(self):
        pos = make_position("ETHUSDT", "short", self._params(), order_id="999")
        assert pos["symbol"] == "ETHUSDT"
        assert pos["direction"] == "short"
        assert pos["order_id"] == "999"

    def test_strategy_stored(self):
        pos = make_position("BTCUSDT", "long", self._params(), strategy="зеркальный_уровень")
        assert pos["strategy"] == "зеркальный_уровень"
