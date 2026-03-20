"""
exchange.py — вся работа с Binance Futures API.

Этот модуль — единственное место где используется python-binance.
Все остальные модули (стратегии, риск-менеджер) работают с чистыми
данными (DataFrame, float) и не знают про Binance напрямую.

Функции:
    create_client()                → подключение к Binance (testnet / mainnet)
    get_candles(client, ...)       → свечи в виде DataFrame
    get_balance(client)            → свободный баланс USDT
    get_balance_detailed(client)   → всего / свободно / заморожено
    get_current_price(client, ...) → текущая цена инструмента
    set_leverage(client, ...)      → установить плечо
    get_precision(client, ...)     → точность цены и количества
    place_order(client, ...)       → открыть позицию с SL и TP
    close_position(client, ...)    → закрыть позицию рыночным ордером
    get_open_position(client, ...) → текущая открытая позиция по символу
    cancel_all_orders(client, ...) → отменить все ордера по символу
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

from config import CONFIG, API_KEY, API_SECRET, USE_TESTNET, REQUEST_TIMEOUT

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ПОДКЛЮЧЕНИЕ
# ─────────────────────────────────────────────────────────────────────────────

def create_client() -> Client:
    """
    Создать и вернуть клиент Binance Futures.

    Автоматически выбирает testnet или mainnet на основе настройки USE_TESTNET.
    Testnet: testnet.binancefuture.com — виртуальные деньги, безопасно для тестов.
    Mainnet: binance.com — реальные деньги!

    Возвращает:
        Готовый клиент Client из python-binance.

    Пример:
        client = create_client()
    """
    if not API_KEY or not API_SECRET:
        raise ValueError(
            "API ключи не найдены. Добавь BINANCE_API_KEY и BINANCE_API_SECRET в .env файл."
        )

    client = Client(
        API_KEY,
        API_SECRET,
        testnet=USE_TESTNET,
        requests_params={"timeout": REQUEST_TIMEOUT},
    )

    mode = "TESTNET (виртуальные деньги)" if USE_TESTNET else "MAINNET (реальные деньги!)"
    log.info(f"Подключение к Binance Futures: {mode}")

    return client


# ─────────────────────────────────────────────────────────────────────────────
# СВЕЧИ
# ─────────────────────────────────────────────────────────────────────────────

def get_candles(
    client: Client,
    symbol: str,
    interval: str | None = None,
    count: int | None = None,
) -> pd.DataFrame:
    """
    Получить свечи фьючерса и вернуть DataFrame.

    Незакрытая (текущая) свеча автоматически исключается —
    работаем только с завершёнными свечами.
    Автоматически добавляет столбец avg_volume (средний объём за N свечей).

    Аргументы:
        client:   подключённый клиент Binance.
        symbol:   тикер, например "BTCUSDT".
        interval: таймфрейм ("1m", "5m", "1h" и т.д.). По умолчанию из CONFIG.
        count:    сколько свечей загрузить. По умолчанию CONFIG.level_candles + 10.

    Возвращает:
        DataFrame со столбцами: time, open, high, low, close, volume, avg_volume.
        Пустой DataFrame при ошибке.

    Пример:
        df = get_candles(client, "BTCUSDT")
        print(df.tail(3))
    """
    interval = interval or CONFIG.entry_interval
    count    = count    or (CONFIG.level_candles + 10)

    try:
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=count + 5,   # +5 чтобы после обрезки осталось достаточно
        )
    except BinanceAPIException as e:
        log.error(f"get_candles [{symbol}]: ошибка API — {e}")
        return pd.DataFrame()
    except Exception as e:
        log.error(f"get_candles [{symbol}]: неожиданная ошибка — {e}")
        return pd.DataFrame()

    if not klines:
        log.warning(f"get_candles [{symbol}]: пустой ответ от биржи")
        return pd.DataFrame()

    rows = []
    for k in klines:
        rows.append({
            "time":   datetime.fromtimestamp(k[0] / 1000),
            "open":   float(k[1]),
            "high":   float(k[2]),
            "low":    float(k[3]),
            "close":  float(k[4]),
            "volume": float(k[5]),
        })

    df = pd.DataFrame(rows)

    # Убираем последнюю незакрытую свечу
    df = df.iloc[:-1].reset_index(drop=True)

    # Средний объём за N свечей (используется в стратегиях для оценки активности)
    df["avg_volume"] = df["volume"].rolling(CONFIG.level_candles).mean().shift(1)

    log.debug(f"get_candles [{symbol}]: загружено {len(df)} свечей")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# БАЛАНС
# ─────────────────────────────────────────────────────────────────────────────

def get_balance(client: Client) -> Optional[float]:
    """
    Получить свободный баланс USDT (доступный для новых сделок).

    Возвращает availableBalance — это деньги, которые не заблокированы
    под открытые позиции и доступны для открытия новых.

    Аргументы:
        client: подключённый клиент Binance.

    Возвращает:
        float — свободный баланс в USDT, или None при ошибке.

    Пример:
        balance = get_balance(client)
        if balance is not None:
            print(f"Доступно: {balance:.2f} USDT")
    """
    try:
        balances = client.futures_account_balance()
        for b in balances:
            if b["asset"] == "USDT":
                return float(b["availableBalance"])
        return 0.0
    except BinanceAPIException as e:
        log.error(f"get_balance: ошибка API — {e}")
        return None
    except Exception as e:
        log.error(f"get_balance: неожиданная ошибка — {e}")
        return None


def get_balance_detailed(client: Client) -> Optional[dict]:
    """
    Получить детальный баланс USDT: всего, свободно и заморожено.

    Заморожено = маржа, заблокированная под открытые позиции.
    Формула: frozen = total - free.

    Аргументы:
        client: подключённый клиент Binance.

    Возвращает:
        dict с ключами: total, free, frozen — все в USDT.
        None при ошибке.

    Пример:
        info = get_balance_detailed(client)
        print(f"Всего: {info['total']:.2f}  Свободно: {info['free']:.2f}  Заморожено: {info['frozen']:.2f}")
    """
    try:
        balances = client.futures_account_balance()
        for b in balances:
            if b["asset"] == "USDT":
                total  = float(b["balance"])
                free   = float(b["availableBalance"])
                frozen = round(total - free, 4)
                return {"total": total, "free": free, "frozen": frozen}
        return None
    except BinanceAPIException as e:
        log.error(f"get_balance_detailed: ошибка API — {e}")
        return None
    except Exception as e:
        log.error(f"get_balance_detailed: неожиданная ошибка — {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ТЕКУЩАЯ ЦЕНА
# ─────────────────────────────────────────────────────────────────────────────

def get_current_price(client: Client, symbol: str) -> Optional[float]:
    """
    Получить текущую рыночную цену инструмента.

    Используется перед выставлением SL/TP чтобы убедиться,
    что цена не ушла за стоп с момента входа.

    Аргументы:
        client: подключённый клиент Binance.
        symbol: тикер, например "ETHUSDT".

    Возвращает:
        float — последняя цена, или None при ошибке.

    Пример:
        price = get_current_price(client, "BTCUSDT")
        print(f"BTCUSDT сейчас: {price}")
    """
    try:
        ticker = client.futures_symbol_ticker(symbol=symbol)
        return float(ticker["price"])
    except BinanceAPIException as e:
        log.error(f"get_current_price [{symbol}]: ошибка API — {e}")
        return None
    except Exception as e:
        log.error(f"get_current_price [{symbol}]: неожиданная ошибка — {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# НАСТРОЙКИ ИНСТРУМЕНТА
# ─────────────────────────────────────────────────────────────────────────────

def set_leverage(client: Client, symbol: str, leverage: int | None = None) -> bool:
    """
    Установить кредитное плечо для инструмента.

    Аргументы:
        client:   подключённый клиент Binance.
        symbol:   тикер, например "BTCUSDT".
        leverage: плечо (если None — берётся из CONFIG.leverage).

    Возвращает:
        True при успехе, False при ошибке.

    Пример:
        set_leverage(client, "BTCUSDT", leverage=5)
    """
    lev = leverage or CONFIG.leverage
    try:
        client.futures_change_leverage(symbol=symbol, leverage=lev)
        log.info(f"set_leverage [{symbol}]: плечо установлено {lev}x")
        return True
    except BinanceAPIException as e:
        log.warning(f"set_leverage [{symbol}]: ошибка API — {e}")
        return False
    except Exception as e:
        log.warning(f"set_leverage [{symbol}]: неожиданная ошибка — {e}")
        return False


def get_precision(
    client: Client,
    symbol: str,
    cache: dict | None = None,
) -> tuple[int, int]:
    """
    Получить точность цены и количества для инструмента.

    Binance требует чтобы цены и объёмы ордеров были округлены
    до строго определённого количества знаков (иначе ошибка API).

    Аргументы:
        client: подключённый клиент Binance.
        symbol: тикер, например "SOLUSDT".
        cache:  словарь-кэш {symbol: (price_prec, qty_prec)}.
                Если символ есть в кэше — запрос к бирже не делается.

    Возвращает:
        (price_precision, qty_precision) — количество знаков после запятой.
        По умолчанию (2, 3) если что-то пошло не так.

    Пример:
        price_prec, qty_prec = get_precision(client, "DOGEUSDT", cache=precision_cache)
        stop = round(0.12345678, price_prec)
    """
    DEFAULT = (2, 3)

    if cache and symbol in cache:
        return cache[symbol]

    try:
        info = client.futures_exchange_info()
        for s in info.get("symbols", []):
            if s["symbol"] != symbol:
                continue

            price_prec = 2
            qty_prec   = 3

            for f in s.get("filters", []):
                if f["filterType"] == "PRICE_FILTER":
                    tick = f["tickSize"].rstrip("0")
                    price_prec = len(tick.split(".")[-1]) if "." in tick else 0
                elif f["filterType"] == "LOT_SIZE":
                    step = f["stepSize"].rstrip("0")
                    qty_prec = len(step.split(".")[-1]) if "." in step else 0

            if cache is not None:
                cache[symbol] = (price_prec, qty_prec)

            log.debug(f"get_precision [{symbol}]: price={price_prec}, qty={qty_prec}")
            return price_prec, qty_prec

        log.warning(f"get_precision [{symbol}]: символ не найден в exchange_info, используем {DEFAULT}")
        return DEFAULT

    except Exception as e:
        log.warning(f"get_precision [{symbol}]: ошибка — {e}, используем {DEFAULT}")
        return DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# ВЫСТАВЛЕНИЕ ОРДЕРОВ
# ─────────────────────────────────────────────────────────────────────────────

def place_order(
    client: Client,
    symbol: str,
    direction: str,
    params: dict,
) -> Optional[str]:
    """
    Открыть позицию рыночным ордером и сразу выставить SL и TP.

    Алгоритм:
        1. Отменяем все старые ордера по символу (чтобы не мешали).
        2. Открываем позицию рыночным ордером (MARKET).
        3. Проверяем текущую цену — если она ушла за SL, корректируем.
        4. Выставляем STOP_MARKET (SL) с closePosition=True.
        5. Выставляем TAKE_PROFIT_MARKET (TP) с closePosition=True.
        6. Если SL не удалось выставить — закрываем позицию рыночно.

    Аргументы:
        client:    подключённый клиент Binance.
        symbol:    тикер, например "BTCUSDT".
        direction: "long" (покупка) или "short" (продажа).
        params:    словарь с ключами:
                   - quantity  (float) — объём в контрактах
                   - stop      (float) — цена стоп-лосса
                   - take      (float) — цена тейк-профита
                   - entry     (float) — цена входа (для логов)

    Возвращает:
        order_id (str) при успехе, None при ошибке.

    Пример:
        order_id = place_order(client, "ETHUSDT", "long", {
            "quantity": 0.1,
            "stop":     1800.0,
            "take":     2100.0,
            "entry":    1900.0,
        })
    """
    side       = "BUY"  if direction == "long"  else "SELL"
    close_side = "SELL" if direction == "long"  else "BUY"
    quantity   = params["quantity"]
    stop_price = params["stop"]
    take_price = params["take"]

    order_id = None

    # 1. Отменяем все старые ордера
    cancel_all_orders(client, symbol)

    try:
        # 2. Рыночный вход
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity,
        )
        order_id = str(order["orderId"])
        log.info(
            f"place_order [{symbol}]: ордер выставлен | "
            f"ID={order_id} | {direction.upper()} | объём={quantity}"
        )

        # 3. Проверяем текущую цену — за время исполнения цена могла уйти
        current_px = get_current_price(client, symbol)
        if current_px is not None:
            price_prec = len(str(stop_price).split(".")[-1]) if "." in str(stop_price) else 0

            if direction == "long":
                # SL должен быть НИЖЕ текущей цены
                if current_px <= stop_price * 1.001:
                    stop_price = round(current_px * 0.995, price_prec)
                    params["stop"] = stop_price
                    log.warning(f"place_order [{symbol}]: SL скорректирован до {stop_price} (цена {current_px})")
                # Если цена уже у TP — закрываем сразу
                if current_px >= take_price * 0.999:
                    log.warning(f"place_order [{symbol}]: цена {current_px} уже у TP {take_price} — закрываем рыночно")
                    close_position(client, symbol, direction, quantity)
                    return order_id

            else:  # short
                # SL должен быть ВЫШЕ текущей цены
                if current_px >= stop_price * 0.999:
                    stop_price = round(current_px * 1.005, price_prec)
                    params["stop"] = stop_price
                    log.warning(f"place_order [{symbol}]: SL скорректирован до {stop_price} (цена {current_px})")
                # Если цена уже у TP — закрываем сразу
                if current_px <= take_price * 1.001:
                    log.warning(f"place_order [{symbol}]: цена {current_px} уже у TP {take_price} — закрываем рыночно")
                    close_position(client, symbol, direction, quantity)
                    return order_id

        # 4. Стоп-лосс (STOP_MARKET закрывает всю позицию)
        try:
            client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=stop_price,
                closePosition=True,
            )
            log.info(f"place_order [{symbol}]: SL выставлен на {stop_price}")
        except BinanceAPIException as e:
            log.error(f"place_order [{symbol}]: не удалось выставить SL — {e}")
            log.warning(f"place_order [{symbol}]: закрываем позицию рыночно (SL не выставлен)")
            close_position(client, symbol, direction, quantity)
            return None

        # 5. Тейк-профит (TAKE_PROFIT_MARKET закрывает всю позицию)
        try:
            client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=take_price,
                closePosition=True,
            )
            log.info(f"place_order [{symbol}]: TP выставлен на {take_price}")
        except BinanceAPIException as e:
            log.error(f"place_order [{symbol}]: не удалось выставить TP — {e}")
            close_position(client, symbol, direction, quantity)
            return None

        return order_id

    except BinanceAPIException as e:
        log.error(f"place_order [{symbol}]: ошибка API при входе — {e}")
        return None
    except Exception as e:
        log.error(f"place_order [{symbol}]: неожиданная ошибка — {e}")
        return None


def close_position(
    client: Client,
    symbol: str,
    direction: str,
    quantity: float,
) -> bool:
    """
    Закрыть позицию рыночным ордером (reduceOnly).

    Используется при аварийном закрытии когда SL/TP не удалось выставить,
    или при ручном закрытии через команду Telegram.

    Аргументы:
        client:    подключённый клиент Binance.
        symbol:    тикер.
        direction: "long" или "short" — направление открытой позиции.
        quantity:  объём для закрытия (должен совпадать с размером позиции).

    Возвращает:
        True при успехе, False при ошибке.

    Пример:
        close_position(client, "BTCUSDT", "long", 0.001)
    """
    # Для закрытия long → продаём, для short → покупаем
    close_side = "SELL" if direction == "long" else "BUY"

    try:
        # Сначала отменяем все ордера (SL/TP) по символу
        cancel_all_orders(client, symbol)

        client.futures_create_order(
            symbol=symbol,
            side=close_side,
            type="MARKET",
            quantity=quantity,
            reduceOnly=True,
        )
        log.info(f"close_position [{symbol}]: позиция закрыта рыночно | {direction.upper()} | объём={quantity}")
        return True

    except BinanceAPIException as e:
        log.error(f"close_position [{symbol}]: ошибка API — {e}")
        return False
    except Exception as e:
        log.error(f"close_position [{symbol}]: неожиданная ошибка — {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ИНФОРМАЦИЯ О ПОЗИЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def get_open_position(client: Client, symbol: str) -> Optional[dict]:
    """
    Получить данные текущей открытой позиции по символу.

    Если позиция не открыта — возвращает None.

    Аргументы:
        client: подключённый клиент Binance.
        symbol: тикер.

    Возвращает:
        Словарь с полями биржи (positionAmt, entryPrice, unRealizedProfit и др.)
        или None если позиции нет.

    Пример:
        pos = get_open_position(client, "SOLUSDT")
        if pos:
            print(f"Объём: {pos['positionAmt']}  Вход: {pos['entryPrice']}")
    """
    try:
        positions = client.futures_position_information(symbol=symbol)
        for p in positions:
            if float(p.get("positionAmt", 0)) != 0:
                return p
        return None
    except BinanceAPIException as e:
        log.error(f"get_open_position [{symbol}]: ошибка API — {e}")
        return None
    except Exception as e:
        log.error(f"get_open_position [{symbol}]: неожиданная ошибка — {e}")
        return None


def get_all_open_positions(client: Client) -> list[dict]:
    """
    Получить список всех открытых позиций на аккаунте.

    Возвращает:
        Список словарей с данными позиций (только ненулевые).
        Пустой список при ошибке или отсутствии позиций.

    Пример:
        positions = get_all_open_positions(client)
        print(f"Открытых позиций: {len(positions)}")
    """
    try:
        all_pos = client.futures_position_information()
        return [p for p in all_pos if float(p.get("positionAmt", 0)) != 0]
    except BinanceAPIException as e:
        log.error(f"get_all_open_positions: ошибка API — {e}")
        return []
    except Exception as e:
        log.error(f"get_all_open_positions: неожиданная ошибка — {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# УПРАВЛЕНИЕ ОРДЕРАМИ
# ─────────────────────────────────────────────────────────────────────────────

def cancel_all_orders(client: Client, symbol: str) -> bool:
    """
    Отменить все активные ордера (SL, TP, лимитные) по символу.

    Используется перед открытием новой позиции и при аварийном закрытии.

    Аргументы:
        client: подключённый клиент Binance.
        symbol: тикер.

    Возвращает:
        True при успехе, False при ошибке.
    """
    try:
        client.futures_cancel_all_open_orders(symbol=symbol)
        log.debug(f"cancel_all_orders [{symbol}]: все ордера отменены")
        return True
    except BinanceAPIException as e:
        log.warning(f"cancel_all_orders [{symbol}]: ошибка API — {e}")
        return False
    except Exception as e:
        log.warning(f"cancel_all_orders [{symbol}]: неожиданная ошибка — {e}")
        return False


def get_open_orders(client: Client, symbol: str) -> list[dict]:
    """
    Получить список активных ордеров (SL, TP и т.д.) по символу.

    Используется при восстановлении после перезапуска — чтобы найти
    SL/TP ордера уже открытых позиций и восстановить параметры защиты.

    Аргументы:
        client: подключённый клиент Binance.
        symbol: тикер.

    Возвращает:
        Список ордеров (словарей) или пустой список при ошибке.

    Пример:
        orders = get_open_orders(client, "BTCUSDT")
        for o in orders:
            if o["type"] == "STOP_MARKET":
                stop_price = float(o["stopPrice"])
    """
    try:
        return client.futures_get_open_orders(symbol=symbol)
    except BinanceAPIException as e:
        log.warning(f"get_open_orders [{symbol}]: ошибка API — {e}")
        return []
    except Exception as e:
        log.warning(f"get_open_orders [{symbol}]: неожиданная ошибка — {e}")
        return []
