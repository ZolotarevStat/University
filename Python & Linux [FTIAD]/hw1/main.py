from bs4 import BeautifulSoup
import urllib.request
import os
import shutil
from itertools import compress
from urllib.error import HTTPError
import time
from functools import wraps


def retry_multi(max_retries: int = 5):
    """ Retry a function `max_retries` times.
    Взято отсюда: https://stackoverflow.com/questions/23892210/python-catch-timeout-and-repeat-request
    :param max_retries: количество попыток переподключения
    :return: надо подставить эту функцию с @ перед объявлением требуемой для перезапуска функции
    """

    def retry(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            num_retries = 0
            while num_retries <= max_retries:
                try:
                    ret = func(*args, **kwargs)
                    break
                except HTTPError:
                    if num_retries == max_retries:
                        ret = None
                        break
                    num_retries += 1
                    time.sleep(3)
            return ret

        return wrapper

    return retry


@retry_multi()
def parse_page(url):
    """
        Вспомогательная функция,
            обеспечивает изолированное перевыполнение функции до тех пор,
            пока не получится успешно спарсить текущий url
        :param url: текущий url, который пытаемся спарсить
        :return: объект BeautifulSoup, внутрь которого хотим залезть
    """
    return BeautifulSoup(urllib.request.urlopen(url), 'html.parser')


def get_info(url: str = "https://habr.com", depth: int = 1):
    """
        Основная функция, выполняющая поставленную в ДЗ-1 задачу
        :param url: ссылка на сайт, с которого хотим получить html-файлы
        :param depth: глубина обхода

        :rtype: object сама функция не возвращает ничего,
                    но пишет в директорию со скриптом следующее:
                    1. папка data, в которой лежат пронумерованные html-файлы
                    2. файл urls.txt с адресом всех ссылочек, которые получилось распарсить
    """
    if os.path.exists('urls.txt'):
        os.remove('urls.txt')
    if os.path.exists('data/'):
        shutil.rmtree('data/')
    if not os.path.exists('data/'):
        os.makedirs('data/')
    urls = []
    levels = [1]
    counter = 1
    urls.append(url)

    for level in range(1, depth + 1):
        if level == 1:
            soup = parse_page(url)
            with open('data/' + str(counter) + '.html', 'w', encoding="utf-8") as file:
                file.write(str(soup))
            for link in soup.findAll('a'):
                a = link.get('href')
                if type(a) == str:
                    if not url + a in urls:
                        if 'http' not in a:
                            counter += 1
                            levels.append(level + 1)
                            a = url + a
                            urls.append(a)
                            html = parse_page(a)
                            with open('data/' + str(counter) + '.html', 'w', encoding="utf-8") as file:
                                file.write(str(html))
                if counter % 6 == 0:
                    break
        else:
            for current_url in list(compress(urls, [True if elem in [level] else False for elem in levels])):
                soup = parse_page(current_url)
                for link in soup.findAll('a'):
                    a = link.get('href')
                    if type(a) == str:
                        if 'en/' in a:
                            a = a.replace('en/', '')
                        if not url + a in urls:
                            if 'http' not in a:
                                a = url + a
                                html = parse_page(a)
                                if html is None:
                                    continue
                                counter += 1
                                levels.append(level + 1)
                                urls.append(a)
                                with open('data/' + str(counter) + '.html', 'w', encoding="utf-8") as file:
                                    file.write(str(html))

    with open('urls.txt', 'w') as file:
        for i, row in enumerate(urls):
            s = "".join(map(str, row))
            file.write(str(i + 1) + ' ' + s + '\n')


if __name__ == '__main__':
    get_info(url="https://aanba.ru", depth=2)
