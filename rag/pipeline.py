"""
Pipeline: собирает всё вместе — query → retrieve → llm → answer.

Это основная точка входа для RAG-системы.
"""

from dataclasses import dataclass

from rag.retriever import Retriever, RetrievedDocument
from rag.llm import LLM


@dataclass
class RAGResponse:
    """Полный ответ RAG-системы."""
    answer: str                          # Ответ LLM
    sources: list[RetrievedDocument]     # Найденные документы
    query: str                           # Исходный вопрос


class RAGPipeline:
    """
    Полный RAG-пайплайн: поиск - формирование контекста - генерация ответа.

    Использование:
        pipeline = RAGPipeline()
        response = pipeline.ask("Кто заведует кафедрой?")
        print(response.answer)
        for src in response.sources:
            print(f"  - {src.title}: {src.source_url}")
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLM | None = None,
    ):
        print("Инициализация RAG Pipeline...")
        self.retriever = retriever or Retriever()
        self.llm = llm or LLM()
        print("RAG Pipeline готов к работе!")

    def ask(
        self,
        question: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> RAGResponse:
        """
        Полный цикл RAG: поиск → контекст → LLM → ответ.

        Args:
            question: Вопрос пользователя.
            top_k: Сколько документов искать.
            category: Фильтр по категории (опционально).

        Returns:
            RAGResponse с ответом, источниками и исходным запросом.
        """
        # 1. Поиск релевантных документов
        docs = self.retriever.search(query=question, top_k=top_k, category=category)

        # 2. Формируем контекст из найденных документов
        context = self.retriever.format_context(docs)

        # 3. Генерируем ответ через LLM
        answer = self.llm.ask(question=question, context=context)

        return RAGResponse(
            answer=answer,
            sources=docs,
            query=question,
        )

    def ask_stream(
        self,
        question: str,
        top_k: int = 5,
        category: str | None = None,
    ):
        """
        Стриминговый вариант — отдаёт токены по мере генерации.
        Сначала ищет документы, потом стримит ответ.

        Args:
            question: Вопрос пользователя.
            top_k: Сколько документов искать.
            category: Фильтр по категории.

        Yields:
            Кортежи (token: str, sources: list | None).
            sources передаётся только в первом yield, потом None.
        """
        # 1. Поиск
        docs = self.retriever.search(query=question, top_k=top_k, category=category)

        # 2. Контекст
        context = self.retriever.format_context(docs)

        # 3. Стриминг ответа
        first = True
        for token in self.llm.ask_stream(question=question, context=context):
            if first:
                yield token, docs
                first = True
            yield token, None
