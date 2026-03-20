"""
Тесты для exchange.py — все тесты используют моки Binance API.
Реальных сетевых запросов нет.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import time
from unittest.mock import MagicMock, patch, call
import pandas as pd

from binance.exceptions import BinanceAPIException

import exchange as ex


# ─── Вспомогательные фабрики ────────────────────────────────────────────────

def make_client() -> MagicMock:
    """Создать мок-клиент Binance."""
    return MagicMock()


def make_kline(
    ts_ms: int = 1_700_000_000_000,
    open_: float = 100.0,
    high: float = 110.0,
    low: float = 90.0,
    close: float = 105.0,
    volume: float = 1000.0,
) -> list:
    """Создать одну свечу в формате Binance klines."""
    return [ts_ms, str(open_), str(high), str(low), str(close), str(volume),
            ts_ms + 299999, "100000", 500, "50000", "5000", "0"]


def make_api_error(code: int = -1121, msg: str = "Invalid symbol") -> BinanceAPIException:
    resp = MagicMock()
    resp.status_code = 400
    resp.text = f'{{"code": {code}, "msg": "{msg}"}}'
    return BinanceAPIException(resp, 400, resp.text)


# ─── create_client ───────────────────────────────────────────────────────────

class TestCreateClient:
    def test_raises_if_no_api_key(self):
        with patch.object(ex, "API_KEY", ""), patch.object(ex, "API_SECRET", ""):
            with pytest.raises(ValueError, match="API ключи не найдены"):
                ex.create_client()

    def test_raises_if_no_api_secret(self):
        with patch.object(ex, "API_KEY", "key"), patch.object(ex, "API_SECRET", ""):
            with pytest.raises(ValueError):
                ex.create_client()

    def test_creates_client_with_keys(self):
        with patch.object(ex, "API_KEY", "testkey"), \
             patch.object(ex, "API_SECRET", "testsecret"), \
             patch("exchange.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = ex.create_client()
            assert client is not None
            mock_cls.assert_called_once()


# ─── get_candles ─────────────────────────────────────────────────────────────

class TestGetCandles:
    def test_returns_dataframe(self):
        client = make_client()
        # 35 свечей: 34 закрытых + 1 незакрытая (будет вырезана)
        client.futures_klines.return_value = [make_kline(ts_ms=1_700_000_000_000 + i * 300_000) for i in range(35)]
        df = ex.get_candles(client, "BTCUSDT", count=30)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_excludes_last_candle(self):
        """Последняя (незакрытая) свеча должна быть вырезана."""
        client = make_client()
        n = 15
        client.futures_klines.return_value = [make_kline(ts_ms=i * 300_000) for i in range(n)]
        df = ex.get_candles(client, "BTCUSDT", count=n)
        # После вырезания последней и обрезки до count — должно быть n-1
        assert len(df) == n - 1

    def test_has_required_columns(self):
        client = make_client()
        client.futures_klines.return_value = [make_kline() for _ in range(30)]
        df = ex.get_candles(client, "BTCUSDT")
        for col in ("time", "open", "high", "low", "close", "volume"):
            assert col in df.columns, f"Столбец {col!r} отсутствует"

    def test_has_avg_volume_column(self):
        client = make_client()
        client.futures_klines.return_value = [make_kline() for _ in range(30)]
        df = ex.get_candles(client, "BTCUSDT")
        assert "avg_volume" in df.columns

    def test_returns_empty_on_api_error(self):
        client = make_client()
        client.futures_klines.side_effect = make_api_error()
        df = ex.get_candles(client, "BTCUSDT")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_returns_empty_on_empty_response(self):
        client = make_client()
        client.futures_klines.return_value = []
        df = ex.get_candles(client, "BTCUSDT")
        assert df.empty

    def test_numeric_columns_are_float(self):
        client = make_client()
        client.futures_klines.return_value = [make_kline() for _ in range(25)]
        df = ex.get_candles(client, "BTCUSDT")
        for col in ("open", "high", "low", "close", "volume"):
            assert df[col].dtype in (float, "float64"), f"{col} не float"

    def test_uses_config_interval_by_default(self):
        client = make_client()
        client.futures_klines.return_value = [make_kline() for _ in range(25)]
        ex.get_candles(client, "ETHUSDT")
        call_kwargs = client.futures_klines.call_args[1]
        assert "interval" in call_kwargs


# ─── get_balance ─────────────────────────────────────────────────────────────

class TestGetBalance:
    def test_returns_available_balance(self):
        client = make_client()
        client.futures_account_balance.return_value = [
            {"asset": "BTC",  "availableBalance": "0.5"},
            {"asset": "USDT", "availableBalance": "1234.56"},
        ]
        balance = ex.get_balance(client)
        assert balance == pytest.approx(1234.56)

    def test_returns_zero_if_usdt_not_found(self):
        client = make_client()
        client.futures_account_balance.return_value = [
            {"asset": "BTC", "availableBalance": "0.5"}
        ]
        balance = ex.get_balance(client)
        assert balance == 0.0

    def test_returns_none_on_api_error(self):
        client = make_client()
        client.futures_account_balance.side_effect = make_api_error()
        balance = ex.get_balance(client)
        assert balance is None

    def test_returns_none_on_unexpected_error(self):
        client = make_client()
        client.futures_account_balance.side_effect = RuntimeError("Network error")
        balance = ex.get_balance(client)
        assert balance is None

    def test_returns_float(self):
        client = make_client()
        client.futures_account_balance.return_value = [
            {"asset": "USDT", "availableBalance": "500.0"}
        ]
        result = ex.get_balance(client)
        assert isinstance(result, float)


# ─── get_balance_detailed ────────────────────────────────────────────────────

class TestGetBalanceDetailed:
    def test_returns_dict_with_all_keys(self):
        client = make_client()
        client.futures_account_balance.return_value = [
            {"asset": "USDT", "balance": "1000.0", "availableBalance": "700.0"}
        ]
        result = ex.get_balance_detailed(client)
        assert result is not None
        assert "total" in result
        assert "free" in result
        assert "frozen" in result

    def test_frozen_equals_total_minus_free(self):
        client = make_client()
        client.futures_account_balance.return_value = [
            {"asset": "USDT", "balance": "1000.0", "availableBalance": "700.0"}
        ]
        result = ex.get_balance_detailed(client)
        assert result["total"] == pytest.approx(1000.0)
        assert result["free"]  == pytest.approx(700.0)
        assert result["frozen"] == pytest.approx(300.0)

    def test_returns_none_if_usdt_missing(self):
        client = make_client()
        client.futures_account_balance.return_value = [
            {"asset": "BNB", "balance": "10.0", "availableBalance": "10.0"}
        ]
        result = ex.get_balance_detailed(client)
        assert result is None

    def test_returns_none_on_api_error(self):
        client = make_client()
        client.futures_account_balance.side_effect = make_api_error()
        result = ex.get_balance_detailed(client)
        assert result is None


# ─── get_current_price ───────────────────────────────────────────────────────

class TestGetCurrentPrice:
    def test_returns_price(self):
        client = make_client()
        client.futures_symbol_ticker.return_value = {"price": "42000.50"}
        price = ex.get_current_price(client, "BTCUSDT")
        assert price == pytest.approx(42000.50)

    def test_returns_float(self):
        client = make_client()
        client.futures_symbol_ticker.return_value = {"price": "100.0"}
        price = ex.get_current_price(client, "BTCUSDT")
        assert isinstance(price, float)

    def test_returns_none_on_api_error(self):
        client = make_client()
        client.futures_symbol_ticker.side_effect = make_api_error()
        price = ex.get_current_price(client, "BTCUSDT")
        assert price is None

    def test_passes_correct_symbol(self):
        client = make_client()
        client.futures_symbol_ticker.return_value = {"price": "1.0"}
        ex.get_current_price(client, "DOGEUSDT")
        client.futures_symbol_ticker.assert_called_once_with(symbol="DOGEUSDT")


# ─── set_leverage ────────────────────────────────────────────────────────────

class TestSetLeverage:
    def test_returns_true_on_success(self):
        client = make_client()
        client.futures_change_leverage.return_value = {"leverage": 5}
        result = ex.set_leverage(client, "BTCUSDT", leverage=5)
        assert result is True

    def test_returns_false_on_api_error(self):
        client = make_client()
        client.futures_change_leverage.side_effect = make_api_error()
        result = ex.set_leverage(client, "BTCUSDT", leverage=5)
        assert result is False

    def test_uses_config_leverage_by_default(self):
        client = make_client()
        client.futures_change_leverage.return_value = {}
        ex.set_leverage(client, "BTCUSDT")
        call_kwargs = client.futures_change_leverage.call_args[1]
        from config import CONFIG
        assert call_kwargs["leverage"] == CONFIG.leverage


# ─── get_precision ───────────────────────────────────────────────────────────

class TestGetPrecision:
    def _make_exchange_info(self, symbol: str, tick_size: str, step_size: str) -> dict:
        return {
            "symbols": [{
                "symbol": symbol,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": tick_size},
                    {"filterType": "LOT_SIZE",     "stepSize": step_size},
                ]
            }]
        }

    def test_btcusdt_precision(self):
        """BTCUSDT: tickSize=0.10, stepSize=0.001 → price=1, qty=3."""
        client = make_client()
        client.futures_exchange_info.return_value = self._make_exchange_info(
            "BTCUSDT", "0.10", "0.001"
        )
        price_prec, qty_prec = ex.get_precision(client, "BTCUSDT")
        assert price_prec == 1
        assert qty_prec == 3

    def test_dogeusdt_precision(self):
        """DOGEUSDT: tickSize=0.00001 → price=5."""
        client = make_client()
        client.futures_exchange_info.return_value = self._make_exchange_info(
            "DOGEUSDT", "0.00001", "1"
        )
        price_prec, qty_prec = ex.get_precision(client, "DOGEUSDT")
        assert price_prec == 5
        assert qty_prec == 0

    def test_uses_cache(self):
        """При наличии кэша биржа не опрашивается."""
        client = make_client()
        cache = {"BTCUSDT": (2, 3)}
        price_prec, qty_prec = ex.get_precision(client, "BTCUSDT", cache=cache)
        client.futures_exchange_info.assert_not_called()
        assert price_prec == 2
        assert qty_prec == 3

    def test_saves_to_cache(self):
        client = make_client()
        client.futures_exchange_info.return_value = self._make_exchange_info(
            "SOLUSDT", "0.01", "0.1"
        )
        cache = {}
        ex.get_precision(client, "SOLUSDT", cache=cache)
        assert "SOLUSDT" in cache

    def test_returns_default_on_error(self):
        client = make_client()
        client.futures_exchange_info.side_effect = RuntimeError("timeout")
        price_prec, qty_prec = ex.get_precision(client, "BTCUSDT")
        assert price_prec == 2
        assert qty_prec == 3

    def test_returns_default_if_symbol_not_found(self):
        client = make_client()
        client.futures_exchange_info.return_value = {"symbols": []}
        price_prec, qty_prec = ex.get_precision(client, "UNKNOWNUSDT")
        assert price_prec == 2
        assert qty_prec == 3


# ─── cancel_all_orders ───────────────────────────────────────────────────────

class TestCancelAllOrders:
    def test_returns_true_on_success(self):
        client = make_client()
        result = ex.cancel_all_orders(client, "BTCUSDT")
        assert result is True
        client.futures_cancel_all_open_orders.assert_called_once_with(symbol="BTCUSDT")

    def test_returns_false_on_api_error(self):
        client = make_client()
        client.futures_cancel_all_open_orders.side_effect = make_api_error()
        result = ex.cancel_all_orders(client, "BTCUSDT")
        assert result is False


# ─── get_open_position ───────────────────────────────────────────────────────

class TestGetOpenPosition:
    def test_returns_position_if_exists(self):
        client = make_client()
        client.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0.001", "entryPrice": "42000"},
        ]
        pos = ex.get_open_position(client, "BTCUSDT")
        assert pos is not None
        assert pos["positionAmt"] == "0.001"

    def test_returns_none_if_no_position(self):
        client = make_client()
        client.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0.0"},
        ]
        pos = ex.get_open_position(client, "BTCUSDT")
        assert pos is None

    def test_returns_none_on_api_error(self):
        client = make_client()
        client.futures_position_information.side_effect = make_api_error()
        pos = ex.get_open_position(client, "BTCUSDT")
        assert pos is None

    def test_returns_none_for_empty_list(self):
        client = make_client()
        client.futures_position_information.return_value = []
        pos = ex.get_open_position(client, "BTCUSDT")
        assert pos is None


# ─── get_all_open_positions ──────────────────────────────────────────────────

class TestGetAllOpenPositions:
    def test_filters_zero_positions(self):
        client = make_client()
        client.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0.001"},
            {"symbol": "ETHUSDT", "positionAmt": "0.0"},
            {"symbol": "SOLUSDT", "positionAmt": "-1.5"},
        ]
        positions = ex.get_all_open_positions(client)
        assert len(positions) == 2

    def test_returns_empty_on_error(self):
        client = make_client()
        client.futures_position_information.side_effect = make_api_error()
        positions = ex.get_all_open_positions(client)
        assert positions == []


# ─── close_position ──────────────────────────────────────────────────────────

class TestClosePosition:
    def test_long_sells(self):
        """Закрытие LONG → ордер SELL."""
        client = make_client()
        client.futures_create_order.return_value = {"orderId": "1"}
        ex.close_position(client, "BTCUSDT", "long", 0.001)
        call_kwargs = client.futures_create_order.call_args[1]
        assert call_kwargs["side"] == "SELL"

    def test_short_buys(self):
        """Закрытие SHORT → ордер BUY."""
        client = make_client()
        client.futures_create_order.return_value = {"orderId": "1"}
        ex.close_position(client, "BTCUSDT", "short", 0.001)
        call_kwargs = client.futures_create_order.call_args[1]
        assert call_kwargs["side"] == "BUY"

    def test_uses_reduce_only(self):
        client = make_client()
        client.futures_create_order.return_value = {"orderId": "1"}
        ex.close_position(client, "BTCUSDT", "long", 0.001)
        call_kwargs = client.futures_create_order.call_args[1]
        assert call_kwargs.get("reduceOnly") is True

    def test_returns_true_on_success(self):
        client = make_client()
        client.futures_create_order.return_value = {"orderId": "1"}
        result = ex.close_position(client, "BTCUSDT", "long", 0.001)
        assert result is True

    def test_returns_false_on_api_error(self):
        client = make_client()
        client.futures_cancel_all_open_orders.return_value = {}
        client.futures_create_order.side_effect = make_api_error()
        result = ex.close_position(client, "BTCUSDT", "long", 0.001)
        assert result is False

    def test_cancels_orders_before_close(self):
        """Сначала отменяем ордера, потом закрываем."""
        client = make_client()
        client.futures_create_order.return_value = {"orderId": "1"}
        ex.close_position(client, "BTCUSDT", "long", 0.001)
        client.futures_cancel_all_open_orders.assert_called_once()


# ─── place_order ─────────────────────────────────────────────────────────────

class TestPlaceOrder:
    def _make_params(self) -> dict:
        return {
            "quantity": 0.001,
            "stop":     39000.0,
            "take":     45000.0,
            "entry":    41000.0,
        }

    def _setup_client(self, entry_price: float = 41000.0) -> MagicMock:
        client = make_client()
        client.futures_create_order.return_value = {"orderId": "999"}
        client.futures_symbol_ticker.return_value = {"price": str(entry_price)}
        return client

    def test_returns_order_id_on_success(self):
        client = self._setup_client()
        order_id = ex.place_order(client, "BTCUSDT", "long", self._make_params())
        assert order_id == "999"

    def test_long_uses_buy_side(self):
        client = self._setup_client()
        ex.place_order(client, "BTCUSDT", "long", self._make_params())
        first_call = client.futures_create_order.call_args_list[0]
        assert first_call[1]["side"] == "BUY"

    def test_short_uses_sell_side(self):
        params = {"quantity": 0.001, "stop": 43000.0, "take": 37000.0, "entry": 41000.0}
        client = self._setup_client(41000.0)
        ex.place_order(client, "BTCUSDT", "short", params)
        first_call = client.futures_create_order.call_args_list[0]
        assert first_call[1]["side"] == "SELL"

    def test_places_stop_market(self):
        """После входа должен быть выставлен STOP_MARKET."""
        client = self._setup_client()
        ex.place_order(client, "BTCUSDT", "long", self._make_params())
        order_types = [c[1].get("type") for c in client.futures_create_order.call_args_list]
        assert "STOP_MARKET" in order_types

    def test_places_take_profit_market(self):
        """После входа должен быть выставлен TAKE_PROFIT_MARKET."""
        client = self._setup_client()
        ex.place_order(client, "BTCUSDT", "long", self._make_params())
        order_types = [c[1].get("type") for c in client.futures_create_order.call_args_list]
        assert "TAKE_PROFIT_MARKET" in order_types

    def test_returns_none_if_sl_fails(self):
        """Если SL не выставился — позиция должна закрыться, вернуть None."""
        client = make_client()
        client.futures_symbol_ticker.return_value = {"price": "41000.0"}

        def create_order(**kwargs):
            if kwargs.get("type") == "STOP_MARKET":
                raise make_api_error(-2021, "Order would immediately trigger")
            return {"orderId": "999"}

        client.futures_create_order.side_effect = create_order
        result = ex.place_order(client, "BTCUSDT", "long", self._make_params())
        assert result is None

    def test_returns_none_if_entry_fails(self):
        """Ошибка при рыночном входе → None."""
        client = make_client()
        client.futures_cancel_all_open_orders.return_value = {}
        client.futures_symbol_ticker.return_value = {"price": "41000.0"}
        client.futures_create_order.side_effect = make_api_error(-2019, "Margin insufficient")
        result = ex.place_order(client, "BTCUSDT", "long", self._make_params())
        assert result is None
