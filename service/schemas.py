"""
Pydantic-модели запросов и ответов для FastAPI сервиса.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ── Запросы ──────────────────────────────────────────────────────────

VALID_CATEGORIES = {"main", "news", "people"}

class AskRequest(BaseModel):
    """Запрос к RAG-системе."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Вопрос пользователя о кафедре аэрогидромеханики",
        examples=["Кто заведует кафедрой?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Сколько документов использовать для контекста",
    )
    category: str | None = Field(
        default=None,
        description="Фильтр по категории: 'main', 'news', 'people'. Оставьте пустым для поиска по всем.",
        examples=[None],
    )

    @field_validator("category", mode="before")
    @classmethod
    def clean_category(cls, v):
        """Swagger UI отправляет 'string' как плейсхолдер — превращаем в None."""
        if v is None:
            return None
        v = str(v).strip()
        if not v or v not in VALID_CATEGORIES:
            return None
        return v


# ── Ответы ───────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    """Один источник, использованный для ответа."""
    title: str
    source_url: str
    category: str
    score: float
    match_type: str  # "semantic", "bm25", "hybrid"


class AskResponse(BaseModel):
    """Ответ RAG-системы."""
    answer: str = Field(description="Ответ LLM на вопрос")
    query: str = Field(description="Исходный вопрос")
    sources: list[SourceDocument] = Field(
        description="Список источников, использованных для ответа"
    )


class HealthResponse(BaseModel):
    """Ответ health-эндпоинта."""
    status: str = "ok"
    model: str = Field(description="Используемая LLM-модель")
    qdrant_connected: bool = Field(description="Подключение к Qdrant")


class ErrorResponse(BaseModel):
    """Ответ при ошибке."""
    detail: str
