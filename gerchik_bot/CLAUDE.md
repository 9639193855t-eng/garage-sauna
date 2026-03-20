# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Начало каждой сессии

1. Прочитать память проекта: `/Users/urox/.claude/projects/-Users-urox-openclaw-sandbox/memory/`
2. Проверить статус бота: `ps aux | grep "python3 main.py" | grep -v grep`
3. Проверить последние логи: `tail -20 logs/bot.log`

---

## Рабочий процесс (ОБЯЗАТЕЛЬНО)

Для КАЖДОЙ задачи (фикс, фича, рефакторинг) использовать 5-агентную схему:

```
Оркестратор (главный агент)
├── Агент 1 → пишет/изменяет код
├── Агент 2 → code review, ищет баги
├── Агент 3 → исправляет найденное Агентом 2
├── Агент 4 → финальный code review
└── Агент 5 → исправляет найденное Агентом 4
```

Завершать:
1. `python3 -m pytest tests/ -q` — все тесты зелёные
2. Перезапуск бота: `pkill -f "main.py"; sleep 2; cd /Users/urox/openclaw_sandbox/gerchik_bot && python3 main.py > logs/bot.log 2>&1 &`
3. Через 5 сек: `tail -10 logs/bot.log` — убедиться что запустился без ошибок

---

## Команды

```bash
# Запуск бота (фон)
cd gerchik_bot && python3 main.py > logs/bot.log 2>&1 &

# Остановка
pkill -f "main.py"

# Логи
tail -f gerchik_bot/logs/bot.log

# Все тесты
cd gerchik_bot && python3 -m pytest tests/ -v

# Один тест-файл
python3 -m pytest tests/test_bounce.py -v

# Один тест
python3 -m pytest tests/test_bounce.py::TestDetectLong::test_basic_long_signal -v
```

---

## Архитектура

### Поток данных

```
main.py (цикл)
  ├─ Каждую минуту:    pm.check_exits(client)
  └─ Каждые 5 минут:  scan_all_symbols(client, pm)
       └─ для каждого символа:
            get_candles() → DataFrame
            find_levels(df) → (support, resistance)
            cascade стратегий → сигнал или None
            calculate_trade_params() → стоп/тейк/объём
            place_order() → открыть позицию на бирже
            pm.add(pos) → сохранить в PositionManager
```

Telegram polling работает в отдельном **daemon-потоке** (`start_polling_thread`), отвечает за 1–2 сек.

### Каскад стратегий (приоритет 1→4)

Первый найденный сигнал побеждает, остальные не проверяются.

| # | Стратегия | Файл | Логика |
|---|-----------|------|--------|
| 1 | Зеркальный уровень | `strategies/mirror.py` | Ретест пробитого уровня (S↔R поменялись ролями) |
| 2 | Ложный пробой | `strategies/false_breakout.py` | Пробой без закрепления + слабый объём |
| 3 | База 3 свечи | `strategies/base_3candles.py` | 3 последовательных касания уровня без пробоя |
| 4 | Отбой (пин-бар) | `strategies/bounce.py` | Молоток/висельник у уровня (хвост >60%, тело <30%) |

Все 4 стратегии имеют одинаковую сигнатуру:
```python
detect(df: pd.DataFrame, support: float, resistance: float) -> Optional[str]
# Возвращает: "long", "short" или None
```

### Ключевые инварианты

- **Пин-бар Герчика**: `lower_wick / range > 0.6` И `body / range < 0.3` (не наоборот!)
- **MA20 не используется** — Герчик работает с чистой ценой на 5-минутках
- **Один символ = одна позиция** максимум одновременно
- **`df or None` — запрещено** для DataFrame. Только `df if df is not None and not df.empty else None`

### Конфигурация (`config.py`)

Глобальные синглтоны: `CONFIG`, `FB_CONFIG`, `BASE_CONFIG`, `BOUNCE_CONFIG`, `MIRROR_CONFIG`.
Параметры грузятся из `.env` (через `python-dotenv`). Ключи: `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_TESTNET`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

Бот настроен на **Binance Futures Testnet** (`BINANCE_TESTNET=true`): виртуальные деньги, реальная биржа.

### Управление рисками

- Риск на сделку: 1% от баланса
- Плечо: 5x
- Тейк-профит = стоп × 3 (соотношение 1:3)
- Трейлинг-стоп: активируется при +1% прибыли, двигается по свингам последних 3 свечей

### PositionManager

Хранит позиции в памяти (dict `symbol → pos`). При перезапуске восстанавливает их с биржи (`restore_from_exchange`). Если `close_position` вернула ошибку `-2022` (ReduceOnly rejected), позиция синхронизируется с биржей — если её там нет, удаляется из памяти.

### Telegram

Команды: `/status`, `/positions`, `/balance`, `/pause`, `/resume`, `/ping`, `/about`, `/help`.
Кнопки-шорткаты нормализуются в команды через `mapping` в `_handle_tg_command()`.
Флаг `tg.paused` защищён `threading.Lock` (property с getter/setter).
