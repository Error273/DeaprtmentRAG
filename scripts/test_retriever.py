"""
Скрипт для ручного тестирования гибридного ретривера (BM25 + semantic).

Запуск:
    python scripts/test_retriever.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import Retriever


def main():
    retriever = Retriever()

    print("=" * 60)
    print("  Тестирование гибридного ретривера (BM25 + semantic)")
    print("  Введите запрос (или 'q' для выхода)")
    print("  Префиксы фильтров: main: / news: / people:")
    print("  Пример: 'people: Гульнара Талгатовна'")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not user_input or user_input.lower() == "q":
            print("Выход.")
            break

        # Парсим фильтр по категории
        category = None
        query = user_input
        for prefix in ("main:", "news:", "people:"):
            if user_input.lower().startswith(prefix):
                category = prefix.rstrip(":")
                query = user_input[len(prefix):].strip()
                break

        if not query:
            print("Пустой запрос, попробуйте ещё раз.")
            continue

        # Поиск
        filter_info = f" [фильтр: {category}]" if category else ""
        print(f"\nИщу: '{query}'{filter_info}")
        print("-" * 60)

        results = retriever.search(query, top_k=5, category=category)

        if not results:
            print("Ничего не найдено.")
            continue

        for i, doc in enumerate(results, 1):
            print(f"\n  #{i} score={doc.score:.3f} [{doc.match_type:8s}]"
                  f" bm25={doc.bm25_norm:.2f} sem={doc.sem_norm:.2f}")
            print(f"  Заголовок: {doc.title}")
            print(f"  URL: {doc.source_url}")
            print(f"  Полный текст ({len(doc.full_text)} симв.):")
            # Первые 300 символов
            preview = doc.full_text[:300]
            for line in preview.split("\n"):
                print(f"    {line}")
            if len(doc.full_text) > 300:
                print(f"    ... (ещё {len(doc.full_text) - 300} симв.)")


if __name__ == "__main__":
    main()
