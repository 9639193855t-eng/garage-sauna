"""Тесты для config.py — проверяем корректность всех констант."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from config import (
    CONFIG, FB_CONFIG, BASE_CONFIG, BOUNCE_CONFIG, MIRROR_CONFIG,
    TradingConfig, INSTRUMENTS, USE_TESTNET,
)


class TestTradingConfig:
    def test_risk_per_trade_valid(self):
        """Риск должен быть от 0 до 5% на сделку."""
        assert 0 < CONFIG.risk_per_trade <= 0.05

    def test_reward_ratio_gerchik(self):
        """Герчик: минимум 1:3."""
        assert CONFIG.reward_ratio >= 3.0

    def test_leverage_in_range(self):
        """Плечо от 1 до 20."""
        assert 1 <= CONFIG.leverage <= 20

    def test_max_positions_positive(self):
        assert CONFIG.max_positions > 0

    def test_max_position_size_pct_valid(self):
        """Размер позиции не должен быть больше 50% от счёта."""
        assert 0 < CONFIG.max_position_size_pct <= 0.5

    def test_stop_buffer_pct_valid(self):
        """Буфер стопа должен быть разумным (0.1%–2%)."""
        assert 0.001 <= CONFIG.stop_buffer_pct <= 0.02

    def test_trailing_stop_pct_larger_than_activation(self):
        """Отступ трейлинга должен быть больше порога активации."""
        assert CONFIG.trailing_stop_pct > CONFIG.trailing_activation_pct

    def test_log_level_valid(self):
        assert CONFIG.log_level in ("DEBUG", "INFO", "WARNING", "ERROR")

    def test_entry_interval_not_empty(self):
        assert CONFIG.entry_interval

    def test_level_candles_minimum(self):
        """Для уровней нужно минимум 10 свечей."""
        assert CONFIG.level_candles >= 10

    def test_default_values_not_zero(self):
        """Ни одна числовая настройка не должна быть нулём."""
        assert CONFIG.risk_per_trade != 0
        assert CONFIG.reward_ratio != 0
        assert CONFIG.leverage != 0
        assert CONFIG.max_positions != 0


class TestFalseBreakoutConfig:
    def test_volume_threshold_range(self):
        """Объём пробоя должен быть меньше 100% от среднего."""
        assert 0 < FB_CONFIG.volume_threshold < 1.0

    def test_body_threshold_range(self):
        """Тело свечи от 0 до 100%."""
        assert 0 < FB_CONFIG.body_threshold < 1.0


class TestBase3CandlesConfig:
    def test_touch_threshold_small(self):
        """Порог касания должен быть небольшим (меньше 1%)."""
        assert BASE_CONFIG.touch_threshold < 0.01

    def test_candles_required_minimum(self):
        assert BASE_CONFIG.candles_required >= 3

    def test_reward_ratio(self):
        assert BASE_CONFIG.reward_ratio >= 3.0

    def test_ma_period_positive(self):
        assert BASE_CONFIG.ma_period > 0


class TestBounceConfig:
    def test_volume_threshold_above_average(self):
        """Сигнальная свеча должна требовать объём выше среднего."""
        assert BOUNCE_CONFIG.volume_threshold >= 1.0

    def test_proximity_threshold_small(self):
        assert BOUNCE_CONFIG.proximity_threshold < 0.01

    def test_pin_bar_wick_larger_than_body(self):
        """Хвост пин-бара должен быть больше тела."""
        assert BOUNCE_CONFIG.pin_bar_wick_ratio > BOUNCE_CONFIG.pin_bar_body_ratio

    def test_reward_ratio(self):
        assert BOUNCE_CONFIG.reward_ratio >= 3.0


class TestMirrorLevelConfig:
    def test_min_touches_minimum(self):
        """Минимум 3 касания — правило Герчика."""
        assert MIRROR_CONFIG.min_touches >= 3

    def test_breakout_volume_mult_above_one(self):
        """Пробой должен быть на повышенном объёме."""
        assert MIRROR_CONFIG.breakout_volume_mult > 1.0

    def test_max_candles_after_break_reasonable(self):
        """Ретест должен приходить в разумный срок."""
        assert MIRROR_CONFIG.max_candles_after_break <= 100

    def test_min_less_than_max_candles(self):
        assert MIRROR_CONFIG.min_candles_after_break < MIRROR_CONFIG.max_candles_after_break

    def test_reward_ratio(self):
        assert MIRROR_CONFIG.reward_ratio >= 3.0

    def test_level_lookback_sufficient(self):
        """Глубина анализа должна быть достаточной для поиска уровней."""
        assert MIRROR_CONFIG.level_lookback >= 50


class TestInstruments:
    def test_instruments_not_empty(self):
        assert len(INSTRUMENTS) > 0

    def test_all_end_with_usdt(self):
        """Все символы должны быть USDT-парами."""
        for key, val in INSTRUMENTS.items():
            assert key.endswith("USDT"), f"{key} не USDT-пара"
            assert val.endswith("USDT"), f"{val} не USDT-пара"

    def test_btcusdt_present(self):
        assert "BTCUSDT" in INSTRUMENTS

    def test_ethusdt_present(self):
        assert "ETHUSDT" in INSTRUMENTS

    def test_key_equals_value(self):
        """Ключ и значение должны совпадать."""
        for key, val in INSTRUMENTS.items():
            assert key == val, f"Ключ {key!r} != значение {val!r}"


class TestTestnetFlag:
    def test_testnet_is_bool(self):
        assert isinstance(USE_TESTNET, bool)
