"""
Telegram-бот кафедры аэрогидромеханики КФУ (aiogram 3).

Запуск:
    python -m bot.bot

Требования:
    - TELEGRAM_BOT_TOKEN в .env
    - Запущенный Qdrant (для RAG-поиска)
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

# Загружаем .env из корня проекта
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Добавляем корень проекта в sys.path (чтобы import rag работал)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.handlers import router, get_pipeline


def create_bot() -> Bot:
    """Создаёт экземпляр бота."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN не задан!\n"
            "Добавьте его в .env файл:\n"
            "TELEGRAM_BOT_TOKEN=ваш_токен_от_BotFather"
        )

    return Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


async def main():
    """Запуск бота."""
    print("=" * 60)
    print("🤖 Telegram-бот кафедры аэрогидромеханики КФУ")
    print("=" * 60)

    bot = create_bot()
    dp = Dispatcher()

    # Подключаем роутер с хэндлерами
    dp.include_router(router)

    # Предварительная инициализация pipeline
    print("\n📦 Предварительная загрузка RAG Pipeline...")
    get_pipeline()
    print()

    # Запускаем polling
    print("🚀 Бот запущен! Ожидаю сообщений...")
    print("   Остановка: Ctrl+C\n")

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Бот остановлен.")
