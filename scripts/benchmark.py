"""
Бенчмарк ретривера: Top@K метрика.

Читает benchmark_questions.txt, прогоняет каждый вопрос через ретривер
и проверяет, содержится ли ground-truth URL в top-K результатах.

Использование:
    python -m scripts.benchmark              # Top@5 (по умолчанию)
    python -m scripts.benchmark --top_k 3    # Top@3
    python -m scripts.benchmark --verbose     # подробный вывод
"""

import argparse
import sys
import time
from pathlib import Path

# UTF-8 для Windows
sys.stdout.reconfigure(encoding="utf-8")

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import Retriever


def parse_benchmark(filepath: Path) -> list[dict]:
    """
    Парсит файл с бенчмарк-вопросами.
    Формат каждой строки: <вопрос> <url>
    URL — последнее слово в строке (начинается с http).
    """
    questions = []
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # URL — последний токен, начинается с http
            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2 or not parts[1].startswith("http"):
                print(f"⚠ Строка {line_num}: не удалось распарсить — пропускаю")
                continue

            question, expected_url = parts
            questions.append({
                "id": line_num,
                "question": question.strip(),
                "expected_url": expected_url.strip(),
            })

    return questions


def run_benchmark(
    retriever: Retriever,
    questions: list[dict],
    top_k: int = 5,
    verbose: bool = False,
) -> dict:
    """
    Прогоняет бенчмарк и возвращает результаты.

    Returns:
        dict с полями: total, hits, misses, accuracy, details
    """
    hits = 0
    misses = 0
    details = []

    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK: Top@{top_k}")
    print(f"  Всего вопросов: {len(questions)}")
    print(f"{'=' * 70}\n")

    for i, q in enumerate(questions, 1):
        question = q["question"]
        expected_url = q["expected_url"]

        start_time = time.time()
        results = retriever.search(query=question, top_k=top_k)
        elapsed = time.time() - start_time

        # Собираем URL из результатов
        retrieved_urls = [doc.source_url for doc in results]

        # Проверяем, есть ли ground-truth URL в top-K
        found = expected_url in retrieved_urls
        if found:
            hits += 1
            status = "✅ HIT"
            rank = retrieved_urls.index(expected_url) + 1
        else:
            misses += 1
            status = "❌ MISS"
            rank = None

        details.append({
            "id": q["id"],
            "question": question,
            "expected_url": expected_url,
            "found": found,
            "rank": rank,
            "retrieved_urls": retrieved_urls,
            "elapsed": elapsed,
        })

        # Вывод
        rank_str = f" (rank={rank})" if rank else ""
        print(f"  [{i:2d}/{len(questions)}] {status}{rank_str}  ({elapsed:.2f}s)")
        print(f"           Q: {question}")
        if verbose:
            print(f"           Expected: {expected_url}")
            for j, doc in enumerate(results, 1):
                match_icon = "→" if doc.source_url == expected_url else " "
                print(
                    f"           {match_icon} {j}. [{doc.match_type:8s}] "
                    f"score={doc.score:.4f} "
                    f"(bm25={doc.bm25_norm:.3f}, sem={doc.sem_norm:.3f}) "
                    f"{doc.source_url}"
                )
        print()

    total = len(questions)
    accuracy = hits / total if total > 0 else 0.0

    # Итоги
    print(f"{'=' * 70}")
    print(f"  РЕЗУЛЬТАТЫ")
    print(f"{'=' * 70}")
    print(f"  Top@{top_k} Accuracy: {accuracy:.1%} ({hits}/{total})")
    print(f"  Hits:   {hits}")
    print(f"  Misses: {misses}")

    # Статистика по рангам
    ranks = [d["rank"] for d in details if d["rank"] is not None]
    if ranks:
        avg_rank = sum(ranks) / len(ranks)
        print(f"  Средний ранг (среди hits): {avg_rank:.2f}")
        for k in range(1, top_k + 1):
            count = sum(1 for r in ranks if r <= k)
            print(f"    Top@{k}: {count}/{total} ({count/total:.1%})")

    # Список промахов
    missed = [d for d in details if not d["found"]]
    if missed:
        print(f"\n  Промахи ({len(missed)}):")
        for d in missed:
            print(f"    #{d['id']}: {d['question']}")
            print(f"           Expected: {d['expected_url']}")
            print(f"           Получено:")
            for j, url in enumerate(d["retrieved_urls"], 1):
                print(f"             {j}. {url}")

    print(f"{'=' * 70}\n")

    return {
        "total": total,
        "hits": hits,
        "misses": misses,
        "accuracy": accuracy,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark ретривера (Top@K)")
    parser.add_argument(
        "--top_k", type=int, default=5, help="Количество результатов для Top@K (default: 5)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Подробный вывод (показать все результаты)"
    )
    parser.add_argument(
        "--benchmark_file",
        type=str,
        default=str(PROJECT_ROOT / "benchmark_questions.txt"),
        help="Путь к файлу с вопросами",
    )
    args = parser.parse_args()

    # 1. Парсим вопросы
    benchmark_path = Path(args.benchmark_file)
    if not benchmark_path.exists():
        print(f"❌ Файл не найден: {benchmark_path}")
        sys.exit(1)

    questions = parse_benchmark(benchmark_path)
    if not questions:
        print("❌ Нет вопросов для бенчмарка")
        sys.exit(1)

    print(f"Загружено {len(questions)} вопросов из {benchmark_path.name}")

    # 2. Инициализируем ретривер
    print("Инициализация ретривера...")
    retriever = Retriever()

    # 3. Прогоняем бенчмарк
    results = run_benchmark(
        retriever=retriever,
        questions=questions,
        top_k=args.top_k,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
