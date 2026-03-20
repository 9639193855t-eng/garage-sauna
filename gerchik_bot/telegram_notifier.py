"""
telegram_notifier.py — Telegram-уведомления для бота Герчика.

Отправляет уведомления через Bot API (requests.post).
Поддерживает polling команд.

Команды:
    /status   — позиции + баланс одной строкой
    /positions — детальный список позиций
    /balance  — текущий баланс
    /ping     — проверка связи
    /pause    — приостановить сканирование
    /resume   — возобновить сканирование
    /stats    — статистика закрытых сделок
    /help     — список команд

Клавиатура (2×4):
    ["📊 Статус",     "💰 Баланс"]
    ["📋 Позиции",    "⏸ Пауза"]
    ["▶️ Продолжить", "🏓 Пинг"]
    ["📈 Статистика", "⚙️ О боте"]
"""

import json
import logging
import os
import threading
import time

import requests

log = logging.getLogger(__name__)

# Раскладка клавиатуры — 5 рядов
_KEYBOARD = [
    ["📊 Статус",       "💰 Баланс"],
    ["📋 Позиции",      "⏸ Пауза"],
    ["▶️ Продолжить",   "🏓 Пинг"],
    ["📈 Статистика",   "⚙️ О боте"],
    ["📥 Экспорт"],
]

# Подписи причин закрытия
_REASON_LABEL = {
    "sl":       ("🛑", "SL"),
    "tp":       ("✅", "TP"),
    "trailing": ("📍", "Trailing"),
}


class TelegramNotifier:
    """
    Синхронный нотификатор для Telegram Bot API.

    Отправляет HTML-сообщения и поддерживает long-polling команд
    с таймаутом 1 секунда (неблокирующий).

    Атрибуты:
        paused: если True — бот сканирование остановлено командой /pause.
    """

    def __init__(self, token: str, chat_id: str) -> None:
        self.token           = token
        self.chat_id         = str(chat_id)
        self.base_url        = f"https://api.telegram.org/bot{token}"
        self._offset         = 0
        self._lock           = threading.Lock()
        self._paused         = False  # флаг паузы сканирования (защищён _lock)
        self._stop_event     = threading.Event()
        self._polling_thread: threading.Thread | None = None

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        with self._lock:
            self._paused = value

    # ─────────────────────────────────────────────────────────────────────
    # Базовые методы отправки
    # ─────────────────────────────────────────────────────────────────────

    def send(self, text: str) -> None:
        """
        Отправить HTML-сообщение в чат.

        Ошибки поглощаются — бот не должен падать из-за Telegram.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id":    self.chat_id,
                    "text":       text,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            if not resp.ok:
                log.warning(
                    "[TG] Ошибка отправки сообщения: %s — %s",
                    resp.status_code, resp.text[:200],
                )
        except Exception as exc:
            log.warning("[TG] Исключение при отправке сообщения: %s", exc)

    def send_keyboard(self, text: str) -> None:
        """
        Отправить сообщение с постоянной ReplyKeyboard.

        Клавиатура остаётся видна под полем ввода (persistent=True).
        Ошибки поглощаются.
        """
        keyboard = {
            "keyboard":          _KEYBOARD,
            "resize_keyboard":   True,
            "persistent":        True,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id":      self.chat_id,
                    "text":         text,
                    "parse_mode":   "HTML",
                    "reply_markup": keyboard,
                },
                timeout=10,
            )
            if not resp.ok:
                log.warning(
                    "[TG] Ошибка отправки клавиатуры: %s — %s",
                    resp.status_code, resp.text[:200],
                )
        except Exception as exc:
            log.warning("[TG] Исключение при отправке клавиатуры: %s", exc)

    # ─────────────────────────────────────────────────────────────────────
    # Polling
    # ─────────────────────────────────────────────────────────────────────

    def poll(self, handler) -> None:
        """
        Получить входящие обновления и передать каждый текст в handler.

        Использует long-polling с timeout=1 (неблокирующий вариант).
        Фильтрует сообщения только из нашего chat_id.
        Обновляет self._offset чтобы не обрабатывать одно сообщение дважды.

        Ошибки поглощаются — бот не должен падать из-за проблем с polling.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/getUpdates",
                params={
                    "offset":  self._offset,
                    "timeout": 5,
                    "limit":   10,
                },
                timeout=20,   # было 10, HTTP-таймаут должен быть long-polling(5) + буфер(15)
            )
            if not resp.ok:
                log.debug("[TG] polling: статус %s", resp.status_code)
                return

            data = resp.json()
            updates = data.get("result", [])

            for update in updates:
                update_id = update.get("update_id", 0)
                # Сдвигаем offset до следующего необработанного update_id
                self._offset = update_id + 1

                message = update.get("message") or update.get("edited_message")
                if not message:
                    continue

                # Фильтруем по chat_id — игнорируем чужие чаты
                msg_chat_id = str(message.get("chat", {}).get("id", ""))
                if msg_chat_id != self.chat_id:
                    log.debug("[TG] polling: игнорируем чужой chat_id=%s", msg_chat_id)
                    continue

                text = message.get("text", "")
                if not text:
                    continue

                log.debug("[TG] polling: получена команда %r", text)
                try:
                    handler(text.strip())
                except Exception as handler_exc:
                    log.warning("[TG] Ошибка в обработчике команды %r: %s", text, handler_exc)

        except Exception as exc:
            log.debug("[TG] Исключение при polling: %s", exc)

    def start_polling_thread(self, handler) -> None:
        """
        Запустить фоновый daemon-поток для непрерывного polling Telegram-команд.

        Поток крутится в цикле с паузой ~1-2 секунды между запросами
        (long-polling timeout=1 + ожидание ~1 сек через Event.wait).
        Умирает автоматически вместе с основным процессом (daemon=True).

        Аргументы:
            handler: Callable[[str], None] — обработчик полученного текста.
        """
        if self._polling_thread and self._polling_thread.is_alive():
            return
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.is_set():
                try:
                    self.poll(handler)
                except Exception as exc:
                    log.error("[TG] Ошибка в polling-потоке: %s — возобновляем через 1 сек", exc)
                self._stop_event.wait(1.0)  # пауза ~1 сек между запросами

        self._polling_thread = threading.Thread(target=_loop, daemon=True, name="tg-polling")
        self._polling_thread.start()
        log.info("[TG] Polling-поток запущен")

    def is_polling_alive(self) -> bool:
        """Проверить что polling-поток жив."""
        return (self._polling_thread is not None
                and self._polling_thread.is_alive())

    def stop_polling_thread(self) -> None:
        """
        Остановить фоновый поток polling.

        Устанавливает событие остановки и ждёт завершения потока (не более 6 сек).
        Таймаут 6 сек > HTTP-timeout=5 сек — гарантирует корректное завершение.
        """
        self._stop_event.set()
        if self._polling_thread:
            self._polling_thread.join(timeout=6)
            self._polling_thread = None
        log.info("[TG] Polling-поток остановлен")

    # ─────────────────────────────────────────────────────────────────────
    # Форматированные уведомления
    # ─────────────────────────────────────────────────────────────────────

    def notify_start(self, balance: float, instruments: int, testnet: bool) -> None:
        """
        Уведомление о запуске бота.

        Отправляется вместе с клавиатурой чтобы пользователь сразу
        видел доступные кнопки управления.
        """
        network = "⚡️ TESTNET" if testnet else "⚡️ MAINNET"
        text = (
            f"🤖 <b>Бот Герчика запущен</b>\n"
            f"{network}\n"
            f"💰 Баланс: {balance:.2f} USDT\n"
            f"📊 Инструментов: {instruments}"
        )
        self.send_keyboard(text)

    def notify_open(self, pos: dict) -> None:
        """
        Уведомление об открытии позиции.

        pos должен содержать: symbol, direction, entry, stop, take,
        quantity, strategy.
        """
        direction = pos.get("direction", "long").upper()
        emoji     = "📈" if direction == "LONG" else "📉"
        text = (
            f"{emoji} <b>ОТКРЫТА [{direction}] {pos['symbol']}</b>\n"
            f"⚙️ Стратегия: {pos.get('strategy', '?')}\n"
            f"💵 Вход: {pos.get('entry', 0):.4f}\n"
            f"🛑 Стоп: {pos.get('stop', 0):.4f}\n"
            f"🎯 Тейк: {pos.get('take', 0):.4f}\n"
            f"📦 Объём: {pos.get('quantity', 0)}"
        )
        self.send(text)

    def notify_close(self, pos: dict, reason: str, current_price: float, pnl: dict) -> None:
        """
        Уведомление о закрытии позиции.

        Аргументы:
            pos:           Словарь позиции.
            reason:        "sl", "tp" или "trailing".
            current_price: Цена закрытия.
            pnl:           {'pnl_usd': float, 'pnl_pct': float}.
        """
        direction  = pos.get("direction", "long").upper()
        entry      = pos.get("entry", 0)
        pnl_usd    = pnl.get("pnl_usd", 0.0)
        pnl_pct    = pnl.get("pnl_pct", 0.0)

        reason_emoji, reason_label = _REASON_LABEL.get(reason, ("❓", reason.upper()))

        # Знак для PnL
        sign = "+" if pnl_usd >= 0 else ""

        text = (
            f"{reason_emoji} <b>ЗАКРЫТА [{direction}] {pos['symbol']}</b>\n"
            f"📋 Причина: {reason_label}\n"
            f"💵 Вход: {entry:.4f} → Выход: {current_price:.4f}\n"
            f"💰 PnL: {sign}{pnl_usd:.2f} USDT ({sign}{pnl_pct:.2f}%)"
        )
        self.send(text)


# ─────────────────────────────────────────────────────────────────────────────
# Фабрика
# ─────────────────────────────────────────────────────────────────────────────

def create_notifier(token: str | None, chat_id: str | None) -> TelegramNotifier | None:
    """Создать нотификатор или вернуть None если токен не задан."""
    if not token or not chat_id:
        return None
    return TelegramNotifier(token, chat_id)
