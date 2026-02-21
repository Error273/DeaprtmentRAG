"""
Чанкер: разбивает текст на чанки по количеству символов с сохранением целых предложений.
"""

import re


def split_into_sentences(text: str) -> list[str]:
    """Разбивает текст на предложения."""
    # Разделяем по точке, восклицательному, вопросительному знаку
    # Не разбиваем по точке внутри сокращений (д.ф.-м.н., к.ф.-м.н., им., и т.д.)
    sentences = re.split(r'(?<=[.!?])\s+(?=[А-ЯA-Z\d«\"(])', text)
    return [s.strip() for s in sentences if s.strip()]


def merge_short_chunks(chunks: list[str], min_chunk_size: int = 80) -> list[str]:
    """
    Склеивает слишком короткие чанки с соседними.

    Чанки типа «Трек 2.» или «Кремлевская, д.» бессмысленны для поиска.
    Приклеиваем их к следующему чанку, а если следующего нет — к предыдущему.
    """
    if not chunks:
        return chunks

    merged = []
    carry = ""  # буфер для коротких чанков, ожидающих склейки

    for chunk in chunks:
        if carry:
            chunk = carry + " " + chunk
            carry = ""

        if len(chunk) < min_chunk_size:
            # Слишком короткий — попробуем склеить со следующим
            carry = chunk
        else:
            merged.append(chunk)

    # Если последний чанк был коротким, приклеиваем к предыдущему
    if carry:
        if merged:
            merged[-1] = merged[-1] + " " + carry
        else:
            merged.append(carry)

    return merged


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_chunk_size: int = 80,
) -> list[str]:
    """
    Разбивает текст на чанки по количеству символов с сохранением целых предложений.

    Args:
        text: Исходный текст.
        chunk_size: Желаемый размер чанка в символах.
        chunk_overlap: Размер перекрытия между соседними чанками (в символах).
                       Берутся последние предложения предыдущего чанка,
                       чтобы не терять контекст на стыке.
        min_chunk_size: Минимальный размер чанка. Более короткие склеиваются с соседними.

    Returns:
        Список строк — чанков.
    """
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # Если одно предложение длиннее chunk_size — берём его целиком как отдельный чанк
        if sentence_len > chunk_size and not current_chunk_sentences:
            chunks.append(sentence)
            continue

        # Если добавление предложения превысит лимит — закрываем текущий чанк
        if current_length + sentence_len > chunk_size and current_chunk_sentences:
            chunk_text_str = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text_str)

            # Формируем overlap: берём последние предложения, пока не наберём chunk_overlap символов
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk_sentences):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)

            current_chunk_sentences = overlap_sentences
            current_length = overlap_len

        current_chunk_sentences.append(sentence)
        current_length += sentence_len

    # Последний чанк
    if current_chunk_sentences:
        chunk_text_str = ' '.join(current_chunk_sentences)
        # Не добавляем, если он полностью совпадает с предыдущим (из-за overlap)
        if not chunks or chunk_text_str != chunks[-1]:
            chunks.append(chunk_text_str)

    # Пост-обработка: склеиваем слишком короткие чанки
    chunks = merge_short_chunks(chunks, min_chunk_size)

    return chunks


def chunk_document(
    doc: dict,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    category: str = 'general',
) -> list[dict]:
    """
    Чанкует один документ (JSON из data/cleaned/) и возвращает список чанков с метаданными.

    Args:
        doc: Словарь с полями url, title, content.
        chunk_size: Размер чанка.
        chunk_overlap: Перекрытие.
        category: Категория документа (main, news, people).

    Returns:
        Список словарей-чанков.
    """
    content = doc.get('content', '').strip()
    if not content:
        return []

    text_chunks = chunk_text(content, chunk_size, chunk_overlap)

    result = []
    for i, chunk in enumerate(text_chunks):
        result.append({
            'chunk_id': f'{category}_{doc.get("title", "unknown")}_{i}',
            'text': chunk,
            'metadata': {
                'source_url': doc.get('url', ''),
                'title': doc.get('title', ''),
                'category': category,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
            }
        })

    return result
