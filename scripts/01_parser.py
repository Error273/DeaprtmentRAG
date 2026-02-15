import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Настройки парсера
BASE_URL = "https://kpfu.ru/math/strctre/mech/fluid"
VISITED_URLS = set()

# Файлы, которые мы не пытаемся парсить как текст
EXCLUDED_EXTENSIONS = ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar', '.jpg', '.png')

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def clean_text(soup):
    """Очищает HTML от лишних тегов и возвращает чистый текст."""
    # Удаляем невидимые и структурные элементы, которые не нужны RAG
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        element.extract()

    # Получаем текст, заменяя теги пробелами
    text = soup.get_text(separator=' ', strip=True)
    # Убираем множественные пробелы и переносы
    text = re.sub(r'\s+', ' ', text)
    return text


def parse_simple_page(url, output_dir):
    """Парсит конкретную страницу и ищет новые ссылки."""
    if url in VISITED_URLS:
        return []

    VISITED_URLS.add(url)
    print(f"Парсинг: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        # КФУ часто использует windows-1251, bs4 обычно справляется, но лучше подстраховаться
        response.encoding = response.apparent_encoding
    except requests.RequestException as e:
        print(f"Ошибка доступа к {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    # Извлекаем заголовок
    title_tag = soup.find('title')
    title = title_tag.get_text(strip=True) if title_tag else "Без названия"

    page_text = clean_text(soup)

    # Сохраняем данные, если есть осмысленный текст
    if len(page_text) > 50:
        save_data(output_dir, url, title, page_text)


def save_data(output_dir, url, title, text):
    """Сохраняет извлеченный текст в формате JSONLines для RAG."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Генерируем безопасное имя файла на основе URL
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', url.replace(BASE_URL, ''))
    if not safe_name or safe_name == '_':
        safe_name = "main_page"

    filepath = os.path.join(output_dir, f"{safe_name}.json")

    data = {
        "url": url,
        "title": title,
        "content": text
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # сперва спарсим основные страницы
    links = ['https://kpfu.ru/math/strctre/mech/fluid',
             'https://kpfu.ru/math/strctre/mech/fluid/istoriya-kafedry',
             'https://kpfu.ru/math/strctre/mech/fluid/metodicheskie-posobiya',
             'https://kpfu.ru/math/strctre/mech/fluid/seminary-i-kruzhki',
             'https://kpfu.ru/math/strctre/mech/fluid/spisok-predlagaemyh-tem-kursovyh-i-diplomnyh-rabot',
             'https://kpfu.ru/math/strctre/mech/fluid/grafik-konsultacij',
             'https://kpfu.ru/math/strctre/mech/fluid/abiturientu-o-bakalavriate',
             'https://kpfu.ru/math/strctre/mech/fluid/abiturientu-o-magistrature'
             ]
    for link in links:
        parse_simple_page(link, '../data/raw/')

    # теперь спарсим сложные страницы (состав, новости)

    # состав
    url = 'https://shelly.kpfu.ru/e-ksu/portal_employee.searchscript?p_search=service&p_office=8190&p_order=1&'
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, 'html.parser')
    for person in soup.find_all('tr'):
        link_element = person.find('a')
        if link_element:
            person_link = link_element['href']
            parse_simple_page(person_link, '../data/raw/people')

    # новости
    n_pages = 13
    url = 'https://kpfu.ru/news_list_content'
    news_links = set()
    for page in range(n_pages):
        data_payload = {"p_sub": "102096",
                        "p_width": "594",
                        "p_all": "1",
                        "p_count": f"{page}",
                        "p_tag_id": "",
                        "p_ctype": "2"}
        response = requests.post(url, data=data_payload)
        response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, 'html.parser')
        # блок каждой новости
        for news_block in soup.find_all('div'):
            if news_block.find('a'):
                news_link = news_block.find('a')['href']
                if not news_link.startswith('https://kpfu.ru/main_page'):
                    news_links.add(news_link)
        time.sleep(1)

    # сохраняем каждую новость
    for url in news_links:
        parse_simple_page(url, '../data/raw/news')
