import json
import os
import sys
import textwrap

# Принудительно UTF-8, чтобы не падало на Windows cp1251
sys.stdout.reconfigure(encoding='utf-8')


PARSED_DATA_DIR = './cleared_data/'


def format_content(content, width=100):
    """Разбивает длинную строку content на читаемые абзацы."""
    # Разделяем по двойным пробелам или переносам (часто разделители в scraped тексте)
    # Убираем лишние пробелы
    content = content.replace('\\n', '\n').replace('\\t', ' ')
    
    # Разбиваем на части по ключевым разделителям
    parts = [p.strip() for p in content.split('\\\\') if p.strip()]
    
    lines = []
    for part in parts:
        wrapped = textwrap.fill(part.strip(), width=width)
        lines.append(wrapped)
    
    return '\n\n'.join(lines)


def print_json_file(filepath):
    """Читает JSON файл и выводит его содержимое в читаемом виде."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rel_path = os.path.relpath(filepath, PARSED_DATA_DIR)
    
    print('=' * 110)
    print(f'  ФАЙЛ: {rel_path}')
    print('=' * 110)
    
    if 'title' in data:
        print(f'\n  [Заголовок]: {data["title"]}')
    
    if 'url' in data:
        print(f'  [URL]: {data["url"]}')
    
    if 'content' in data:
        print(f'\n  [Содержимое]:')
        print('-' * 110)
        formatted = format_content(data['content'])
        # Добавляем отступ для красоты
        for line in formatted.split('\n'):
            print(f'    {line}')
        print('-' * 110)
    
    # Выводим другие поля, если есть
    other_keys = [k for k in data.keys() if k not in ('url', 'title', 'content')]
    if other_keys:
        print(f'\n  [Дополнительные поля]:')
        for key in other_keys:
            value = data[key]
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + '...'
            print(f'    {key}: {value}')
    
    print('\n')


def main():
    file_count = 0
    
    # Рекурсивно обходим все JSON файлы
    for root, dirs, files in os.walk(PARSED_DATA_DIR):
        for filename in sorted(files):
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                try:
                    print_json_file(filepath)
                    file_count += 1
                except Exception as e:
                    print(f'[ОШИБКА] при чтении {filepath}: {e}\n')
    
    print(f'\nВсего обработано файлов: {file_count}')


if __name__ == '__main__':
    main()