"""
LLM: обёртка над OpenRouter API для генерации ответов.

Использует openai-совместимый API OpenRouter.
Модель по умолчанию: stepfun/step-3.5-flash:free (бесплатная, быстрая).
"""

from openai import OpenAI

from rag.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SYSTEM_PROMPT,
)


class LLM:
    """
    Обёртка над LLM через OpenRouter.

    Использует openai SDK (OpenRouter совместим с OpenAI API).

    Использование:
        llm = LLM()
        answer = llm.ask("Кто заведует кафедрой?", context="...")
        print(answer)
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        key = api_key or OPENROUTER_API_KEY
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY не задан! "
                "Укажите его в .env файле или передайте в конструктор."
            )

        self.client = OpenAI(
            api_key=key,
            base_url=base_url,
        )
        print(f"LLM инициализирован: {self.model}")

    def ask(self, question: str, context: str) -> str:
        """
        Задать вопрос LLM с контекстом из ретривера.

        Args:
            question: Вопрос пользователя.
            context: Контекст (полные тексты найденных страниц).

        Returns:
            Ответ модели (строка).
        """
        user_message = (
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {question}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content.strip()

    def ask_stream(self, question: str, context: str):
        """
        Стриминговый вариант — отдаёт токены по мере генерации.
        Полезно для Telegram-бота или веб-интерфейса.

        Args:
            question: Вопрос пользователя.
            context: Контекст из ретривера.

        Yields:
            Строковые токены по мере генерации.
        """
        user_message = (
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {question}"
        )

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
