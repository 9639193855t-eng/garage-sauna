"""
main.py — Главный цикл торгового бота Герчика.

Архитектура:
    • Каждую минуту:   check_exits()  — проверяет SL / TP / трейлинг-стоп
    • Каждые 5 минут:  scan_signals() — сканирует закрытие свечи по всем инструментам

Каскад стратегий (первый сигнал побеждает, по приоритету):
    1. mirror         — Зеркальный уровень     (высший приоритет)
    2. false_breakout — Ложный пробой
    3. base_3candles  — База 3 свечи
    4. bounce         — Отбой от уровня         (низший приоритет)

Запуск:
    python3 main.py

Остановка:
    Ctrl+C — бот завершит текущую итерацию и остановится.
    Открытые позиции НЕ закрываются при остановке (SL/TP уже стоят на бирже).
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from config import CONFIG, INSTRUMENTS
from exchange import (
    create_client,
    get_balance,
    get_balance_detailed,
    get_candles,
    get_current_price,
    get_precision,
    place_order,
    set_leverage,
)
from levels import find_levels
from position_manager import PositionManager
from risk_manager import calculate_trade_params, is_params_valid, make_position
from strategies.bounce         import detect as bounce_detect
from strategies.base_3candles  import detect as b3_detect
from strategies.false_breakout import detect as fb_detect
from strategies.mirror         import detect as mirror_detect
from telegram_notifier import create_notifier, TelegramNotifier
from trade_history import open_trade, close_trade, HISTORY_FILE

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# КАСКАД СТРАТЕГИЙ — порядок строго по приоритету, менять нельзя
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = [
    ("mirror",         mirror_detect),
    ("false_breakout", fb_detect),
    ("base_3candles",  b3_detect),
    ("bounce",         bounce_detect),
]

# Интервал сканирования сигналов — совпадает с таймфреймом свечи (5 мин)
SIGNAL_SCAN_INTERVAL = 5 * 60   # секунд


# ─────────────────────────────────────────────────────────────────────────────
# ЛОГИРОВАНИЕ
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    """Настроить логирование: файл + вывод в консоль."""
    level = getattr(logging, CONFIG.log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if CONFIG.log_file:
        log_dir = os.path.dirname(CONFIG.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(CONFIG.log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ОБРАБОТЧИК TELEGRAM-КОМАНД
# ─────────────────────────────────────────────────────────────────────────────

def _handle_tg_command(cmd: str, tg, pm: PositionManager, client) -> None:
    """Обработать команду из Telegram."""
    # Нормализуем кнопки → команды
    mapping = {
        "📊 статус":      "/status",
        "💰 баланс":      "/balance",
        "📋 позиции":     "/positions",
        "⏸ пауза":       "/pause",
        "▶️ продолжить":  "/resume",
        "🏓 пинг":        "/ping",
        "⚙️ о боте":      "/about",
        "📈 статистика":  "/stats",
        "📥 экспорт":     "/export",
    }
    cmd_lower = cmd.lower()
    cmd = mapping.get(cmd_lower, cmd.lower())

    if cmd in ("/status", "/start"):
        balance   = get_balance(client) or 0.0
        positions = pm.all()
        lines     = [
            "🤖 <b>Статус бота</b>",
            f"💰 Баланс: {balance:.2f} USDT",
            f"📊 Позиций: {pm.count()}/{CONFIG.max_positions}",
        ]
        if positions:
            for p in positions:
                lines.append(
                    f"  • {p['symbol']} {p['direction'].upper()} @ {p['entry']:.4f}"
                )
        lines.append("⏸ Пауза: " + ("да" if tg.paused else "нет"))
        tg.send("\n".join(lines))

    elif cmd == "/balance":
        info = get_balance_detailed(client)
        if info is None:
            tg.send("⚠️ Ошибка получения баланса")
        else:
            free   = info['free']
            frozen = info['frozen']
            total  = info['total']
            tg.send(
                "💰 <b>Баланс USDT</b>\n\n"
                "<code>"
                f"Свободно:   {free:>10.2f} USDT\n"
                f"В позициях: {frozen:>10.2f} USDT\n"
                f"Всего:      {total:>10.2f} USDT"
                "</code>"
            )

    elif cmd == "/positions":
        positions = pm.all()
        if not positions:
            tg.send("📋 Открытых позиций нет")
        else:
            lines = [f"📋 <b>Открытые позиции ({len(positions)}):</b>"]
            for p in positions:
                stop_str = f"{p['stop']:.4f}" if p.get('stop') is not None else "—"
                take_str = f"{p['take']:.4f}" if p.get('take') is not None else "—"
                strategy = p.get('strategy', '?')
                strategy_display = {
                    'mirror': 'Зеркало',
                    'false_breakout': 'Лож. пробой',
                    'base_3candles': 'База 3 свечи',
                    'bounce': 'Отбой',
                    'restored': 'Восстановлена',
                }.get(strategy, strategy)
                lines.append(
                    f"• <b>{p['symbol']}</b> {p['direction'].upper()}\n"
                    f"  Вход: {p['entry']:.4f} | Стоп: {stop_str} | Тейк: {take_str}\n"
                    f"  Стратегия: {strategy_display}"
                )
            tg.send("\n".join(lines))

    elif cmd == "/pause":
        tg.paused = True
        tg.send("⏸ Сканирование <b>приостановлено</b>")

    elif cmd == "/resume":
        tg.paused = False
        tg.send("▶️ Сканирование <b>возобновлено</b>")

    elif cmd == "/ping":
        tg.send("🏓 Pong! Бот работает.")

    elif cmd == "/about":
        tg.send(
            "⚙️ <b>Бот Герчика — как он работает</b>\n"
            "\n"
            "<b>Режим работы:</b>\n"
            "• Binance Futures TESTNET (виртуальные деньги)\n"
            f"• Плечо: {CONFIG.leverage}x | Риск на сделку: {CONFIG.risk_per_trade * 100:.0f}% депозита\n"
            f"• Лимит позиций: {CONFIG.max_positions} одновременно\n"
            f"• Инструменты: {len(INSTRUMENTS)} пар (BTC, ETH, SOL и др.)\n"
            "\n"
            "<b>Принцип работы:</b>\n"
            "Каждые 5 минут бот сканирует все инструменты и ищет сигналы по 4 стратегиям. "
            "Каждую минуту проверяет открытые позиции на срабатывание стопа, тейка или трейлинга.\n"
            "\n"
            "<b>Стратегии (каскад, приоритет 1→4):</b>\n"
            "\n"
            "1️⃣ <b>Зеркало уровней</b>\n"
            "Цена отбивается от уровня, который раньше был поддержкой, а теперь стал сопротивлением (или наоборот). "
            "Классический принцип Герчика: уровень меняет роль.\n"
            "\n"
            "2️⃣ <b>Ложный пробой</b>\n"
            "Цена пробивает уровень, но не закрепляется — разворачивается обратно. "
            "Торгуем против толпы, которая попалась на ложный пробой.\n"
            "\n"
            "3️⃣ <b>База из 3 свечей</b>\n"
            "Несколько свечей подряд касаются одного уровня — формируется «база». "
            "Это накопление у уровня перед сильным движением.\n"
            "\n"
            "4️⃣ <b>Отбой (пин-бар)</b>\n"
            "Свеча с длинным хвостом (&gt;60% диапазона) и маленьким телом (&lt;30%) у уровня. "
            "Молоток снизу = LONG, висельник сверху = SHORT.\n"
            "\n"
            "<b>Управление сделкой:</b>\n"
            "• Стоп-лосс и тейк-профит выставляются сразу при открытии\n"
            f"• Соотношение риск/прибыль: 1:{CONFIG.reward_ratio:.0f}\n"
            f"• Трейлинг-стоп активируется при прибыли {CONFIG.trailing_activation_pct * 100:.0f}%+, "
            f"подтягивается по свингам последних 3 свечей\n"
        )

    elif cmd == "/stats":
        try:
            stats = pm.stats
            trades = stats.today_trades()
            lines = ["📈 <b>Статистика</b>"]

            # --- Сегодня ---
            today_usd = stats.today_pnl_usd()
            today_pct = stats.today_pnl_pct()
            sign = "+" if today_usd > 0 else ""
            lines.append(f"\n<b>Сегодня ({len(trades)} сделок):</b>")
            if trades:
                for t in trades:
                    icon = "✅" if t["pnl_usd"] >= 0 else "🔴"
                    s = "+" if t["pnl_usd"] > 0 else ""
                    lines.append(
                        f"{icon} {t['symbol']} {t['direction'].upper()} "
                        f"[{t['reason'].upper()}] "
                        f"{s}{t['pnl_usd']:.2f}$ ({s}{t['pnl_pct']:.2f}%)"
                    )
            lines.append(f"Итого за день: <b>{sign}{today_usd:.2f}$ ({sign}{today_pct:.2f}%)</b>")

            # --- Всё время ---
            all_usd = stats.alltime_pnl_usd()
            all_pct = stats.alltime_pnl_pct()
            sign2 = "+" if all_usd > 0 else ""
            all_trades_count = stats.total_count()
            lines.append(f"\n<b>За всё время ({all_trades_count} сделок):</b>")
            lines.append(f"Итого: <b>{sign2}{all_usd:.2f}$ ({sign2}{all_pct:.2f}%)</b>")

            tg.send("\n".join(lines))
        except Exception as exc:
            log.error("[TG] Ошибка команды /stats: %s", exc)
            tg.send("⚠️ Ошибка получения статистики")

    elif cmd == "/export":
        if HISTORY_FILE.exists():
            url = f"{tg.base_url}/sendDocument"
            with open(HISTORY_FILE, "rb") as f:
                requests.post(url, data={"chat_id": tg.chat_id}, files={"document": f})
        else:
            tg.send("📥 История сделок пуста")

    elif cmd in ("/help", "/команды"):
        tg.send(
            "📋 <b>Команды:</b>\n"
            "/status — статус и позиции\n"
            "/balance — баланс\n"
            "/positions — детальные позиции\n"
            "/pause — пауза сканирования\n"
            "/resume — возобновить\n"
            "/ping — проверка связи\n"
            "/stats — статистика сделок\n"
            "/export — скачать историю сделок"
        )


# ─────────────────────────────────────────────────────────────────────────────
# СКАНИРОВАНИЕ ОДНОГО ИНСТРУМЕНТА
# ─────────────────────────────────────────────────────────────────────────────

def scan_symbol(
    client,
    symbol: str,
    pm: PositionManager,
    precision_cache: dict,
    tg: Optional["TelegramNotifier"] = None,
) -> bool:
    """
    Проверить один инструмент на сигнал по всем стратегиям.

    Алгоритм:
        1. Пропускаем если уже есть открытая позиция по символу.
        2. Пропускаем если достигнут лимит max_positions.
        3. Загружаем свечи, находим уровни.
        4. Прогоняем каскад стратегий — берём первый сигнал.
        5. Считаем параметры сделки и открываем ордер на бирже.
        6. Сохраняем позицию в PositionManager.
        7. Отправляем Telegram-уведомление об открытии (если tg задан).

    Аргументы:
        client:          Клиент Binance.
        symbol:          Тикер, например "BTCUSDT".
        pm:              Менеджер позиций.
        precision_cache: Кэш точностей цены и объёма.
        tg:              Опциональный TelegramNotifier.

    Возвращает:
        True если позиция была открыта в этом вызове.
    """
    # Уже торгуем этот символ
    if pm.is_open(symbol):
        return False

    # Лимит позиций исчерпан
    if pm.count() >= CONFIG.max_positions:
        log.debug("[SCAN] %s: лимит позиций (%d) достигнут", symbol, CONFIG.max_positions)
        return False

    # ── Свечи и уровни ───────────────────────────────────────────────────────
    # Один HTTP-запрос: загружаем длинный df для mirror (и короткого среза)
    df_long = get_candles(client, symbol, count=CONFIG.mirror_candles)
    if df_long is None or df_long.empty:
        log.warning("[SCAN] %s: нет свечей", symbol)
        return False

    # Короткий df для остальных стратегий — срез из df_long (тот же источник, нет race condition)
    df = df_long.iloc[-(CONFIG.level_candles + 10):].reset_index(drop=True)

    support, resistance = find_levels(df)
    if support is None or resistance is None:
        log.debug("[SCAN] %s: уровни не определены", symbol)
        return False

    log.debug("[SCAN] %s: support=%.4f resistance=%.4f", symbol, support, resistance)

    # ── Каскад стратегий — первый сигнал побеждает ───────────────────────────
    signal: Optional[str] = None
    strategy_name: str    = ""

    for name, detect_fn in STRATEGIES:
        try:
            # Стратегия mirror требует длинной истории — передаём df_long
            data = df_long if name == "mirror" else df
            result = detect_fn(data, support, resistance)
            if result in ("long", "short"):
                signal        = result
                strategy_name = name
                log.info("[SCAN] %s: СИГНАЛ [%s] %s", symbol, name, result.upper())
                break
        except Exception as e:
            log.error("[SCAN] %s: ошибка стратегии [%s] — %s", symbol, name, e)

    if signal is None:
        return False

    # ── Параметры сделки ─────────────────────────────────────────────────────
    balance = get_balance(client)
    if balance is None or balance <= 0:
        log.warning("[SCAN] %s: баланс недоступен, пропускаем", symbol)
        return False

    price_prec, qty_prec = get_precision(client, symbol, cache=precision_cache)

    # Текущая рыночная цена — используем как цену входа (MARKET ордер)
    entry_price = get_current_price(client, symbol)
    if entry_price is None:
        log.warning("[SCAN] %s: не удалось получить цену для расчёта параметров", symbol)
        return False

    # Экстремум последней свечи для стопа (Герчик: стоп ЗА экстремум)
    signal_candle = df.iloc[-1]
    extremum = float(signal_candle["low"] if signal == "long" else signal_candle["high"])

    params = calculate_trade_params(
        direction       = signal,
        entry_price     = entry_price,
        candle_extremum = extremum,
        account_balance = balance,
        price_precision = price_prec,
        qty_precision   = qty_prec,
    )

    if not is_params_valid(params):
        log.warning("[SCAN] %s: параметры сделки некорректны, пропускаем", symbol)
        return False

    # ── Устанавливаем плечо и открываем позицию ──────────────────────────────
    set_leverage(client, symbol)

    order_id = place_order(client, symbol, signal, params)
    if order_id is None:
        log.error("[SCAN] %s: не удалось разместить ордер", symbol)
        return False

    pos = make_position(
        symbol     = symbol,
        direction  = signal,
        params     = params,
        strategy   = strategy_name,
        order_id   = order_id,
    )
    pm.add(pos)

    # Сохраняем в историю сделок
    try:
        trade_id = open_trade(pos, df, support, resistance, conditions={})
        pos["trade_id"] = trade_id
    except Exception as hist_exc:
        log.warning("[SCAN] %s: ошибка записи в историю сделок: %s", symbol, hist_exc)

    # Telegram-уведомление об открытии позиции
    if tg:
        try:
            tg.notify_open(pos)
        except Exception as tg_exc:
            log.warning("[SCAN] %s: ошибка Telegram-уведомления об открытии: %s", symbol, tg_exc)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# СКАНИРОВАНИЕ ВСЕХ ИНСТРУМЕНТОВ
# ─────────────────────────────────────────────────────────────────────────────

def scan_all_symbols(
    client,
    pm: PositionManager,
    precision_cache: dict,
    tg: Optional["TelegramNotifier"] = None,
) -> int:
    """
    Сканировать все инструменты из INSTRUMENTS на предмет сигналов.

    Аргументы:
        client:          Клиент Binance.
        pm:              Менеджер позиций.
        precision_cache: Кэш точностей.
        tg:              Опциональный TelegramNotifier.

    Возвращает:
        Количество открытых в этом вызове позиций.
    """
    # Если сканирование на паузе — пропускаем
    if tg and tg.paused:
        log.info("[ЦИКЛ] Сканирование на паузе (Telegram)")
        return 0

    balance = get_balance(client)
    log.info(
        "[ЦИКЛ] Сканирование %d инструментов | позиций=%d/%d | баланс=%.1f USDT",
        len(INSTRUMENTS),
        pm.count(),
        CONFIG.max_positions,
        balance or 0.0,
    )

    opened = 0
    for symbol in INSTRUMENTS:
        try:
            if pm.count() >= CONFIG.max_positions:
                log.info("[ЦИКЛ] Лимит позиций достигнут, сканирование остановлено")
                break
            if scan_symbol(client, symbol, pm, precision_cache, tg=tg):
                opened += 1
                # Небольшая пауза между ордерами — не перегружаем API
                time.sleep(1.0)
        except Exception as e:
            log.error("[ЦИКЛ] %s: необработанная ошибка при сканировании — %s", symbol, e, exc_info=True)

    if opened:
        log.info("[ЦИКЛ] Открыто новых позиций: %d", opened)
    else:
        log.info("[ЦИКЛ] Новых сигналов нет")

    return opened


# ─────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ВРЕМЕНИ
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _seconds_to_next_minute() -> float:
    """Секунд до начала следующей UTC-минуты."""
    now = _utcnow()
    return 60.0 - now.second - now.microsecond / 1_000_000


def _is_candle_close_minute() -> bool:
    """
    Сейчас — первые 10 секунд минуты, кратной 5?

    Используется чтобы сканировать сигналы ровно на закрытии 5-минутной свечи.
    Запасные 10 секунд нужны на случай если основной цикл проснулся чуть позже.
    """
    now = _utcnow()
    return now.minute % 5 == 0 and now.second < 10


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНЫЙ ЦИКЛ
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Точка входа бота.

    Цикл:
        1. Каждую минуту: проверяем выходы по всем позициям.
        2. На закрытии 5-минутной свечи: сканируем сигналы по всем инструментам.
        3. Спим до начала следующей минуты.

    Остановка через Ctrl+C — позиции НЕ закрываются, SL/TP уже стоят на бирже.
    """
    setup_logging()

    # Инициализируем Telegram-нотификатор (None если токен не задан в .env)
    tg = create_notifier(os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"))

    log.info("=" * 64)
    log.info("  Бот Герчика — Binance Futures%s",
             " TESTNET" if __import__("config").USE_TESTNET else " MAINNET (!)")
    log.info("  Инструментов: %d", len(INSTRUMENTS))
    log.info(
        "  Лимит: %d позиций | Плечо: %dx | Риск: %.1f%% на сделку",
        CONFIG.max_positions, CONFIG.leverage, CONFIG.risk_per_trade * 100,
    )
    log.info(
        "  Трейлинг: %s | Активация: %.1f%% | Отступ: %.1f%%",
        "вкл" if CONFIG.trailing_stop_enabled else "выкл",
        CONFIG.trailing_activation_pct * 100,
        CONFIG.trailing_stop_pct * 100,
    )
    log.info("  Telegram: %s", "включён" if tg else "отключён (токен не задан)")
    log.info("=" * 64)

    client          = create_client()
    pm              = PositionManager(tg=tg)
    precision_cache: dict = {}

    # Восстанавливаем позиции открытые до перезапуска
    pm.restore_from_exchange(client, precision_cache)

    # Запускаем Telegram-polling в отдельном фоновом потоке
    if tg:
        tg.start_polling_thread(lambda cmd: _handle_tg_command(cmd, tg, pm, client))

    # Уведомление о запуске через Telegram
    balance = get_balance(client) or 0.0
    if tg:
        tg.notify_start(balance, len(INSTRUMENTS), __import__("config").USE_TESTNET)

    # Флаг: было ли уже сканирование на текущей 5-минутке
    # Инициализируем в прошлое чтобы сразу запустить при старте
    last_scan_minute: Optional[int] = None

    log.info(
        "Главный цикл запущен. Выходы — каждую минуту, сигналы — каждые 5 минут."
    )

    while True:
        loop_start = time.monotonic()

        try:
            now_utc = _utcnow()

            # ── Watchdog: перезапускаем polling если поток умер ───────────────
            if tg and not tg.is_polling_alive():
                log.warning("[ЦИКЛ] Telegram polling-поток мёртв — перезапускаем")
                tg.start_polling_thread(
                    lambda cmd: _handle_tg_command(cmd, tg, pm, client)
                )

            # ── 1. Каждую минуту: проверяем выходы ──────────────────────────
            if pm.count() > 0:
                closed = pm.check_exits(client, precision_cache)
                if closed:
                    log.info(
                        "[ЦИКЛ] Закрыто позиций: %d — %s",
                        len(closed), closed,
                    )

            # ── 2. На закрытии 5-минутной свечи: сканируем сигналы ──────────
            # Используем номер 5-минутного периода в сутках как ключ,
            # чтобы на одной 5-минутке не запустить сканирование дважды
            five_min_period = (now_utc.hour * 60 + now_utc.minute) // 5

            if five_min_period != last_scan_minute and _is_candle_close_minute():
                last_scan_minute = five_min_period
                scan_all_symbols(client, pm, precision_cache, tg=tg)

            # Первый запуск: если ещё ни разу не сканировали — запустим сразу,
            # не дожидаясь ровной 5-минутки
            elif last_scan_minute is None:
                last_scan_minute = five_min_period
                scan_all_symbols(client, pm, precision_cache, tg=tg)

        except KeyboardInterrupt:
            log.info("Бот остановлен (Ctrl+C). Открытые позиции не закрыты — SL/TP на бирже.")
            if tg:
                tg.stop_polling_thread()
            break

        except Exception as e:
            log.error("[ЦИКЛ] Критическая ошибка: %s", e, exc_info=True)
            # Не прерываем цикл — ждём следующей итерации

        # ── Спим до начала следующей минуты ─────────────────────────────────
        elapsed   = time.monotonic() - loop_start
        sleep_sec = max(1.0, _seconds_to_next_minute() - elapsed)
        log.debug("[ЦИКЛ] Сон %.1f сек до следующей минуты", sleep_sec)
        try:
            time.sleep(sleep_sec)
        except KeyboardInterrupt:
            log.info("Бот остановлен (Ctrl+C). Открытые позиции не закрыты — SL/TP на бирже.")
            if tg:
                tg.stop_polling_thread()
            break


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
