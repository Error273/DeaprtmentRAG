"""
Тест RAG Pipeline: задаём вопрос → получаем ответ от LLM с источниками.

Запуск:
    python scripts/test_pipeline.py
"""

from rag.pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline()

    questions = [
        "Кто руководитель кафедры?",
        "Электронная почта поташева",
        "кабинет марданова",
        "какие языки знает поташев?",
        "кто научный руководитель у Федорова К?"
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Вопрос: {q}")
        print("=" * 60)

        response = pipeline.ask(q)

        print(f"\nОтвет:\n{response.answer}")
        print(f"\nИсточники:")
        for src in response.sources:
            print(f"  [{src.match_type}] {src.title}")
            print(f"    URL: {src.source_url}")
            print(f"    Score: {src.score:.4f} (BM25: {src.bm25_norm:.4f}, Sem: {src.sem_norm:.4f})")


if __name__ == "__main__":
    main()
