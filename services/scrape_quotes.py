import requests
from bs4 import BeautifulSoup


BASE_URL = 'https://quotes.toscrape.com/page'


class ScrapeQuotes:
    @staticmethod
    def scrape_quotes_pages(pages: int):
        quotes = []

        for i in range(1, pages + 1):
            url = f"{BASE_URL}/{i}"
            response = requests.get(url)

            soup = BeautifulSoup(response.text, 'html.parser')

            quotes_on_page = []

            for quote in soup.select("div.quote"):
                text = quote.select_one("span.text").get_text()
                author = quote.select_one("small.author").get_text()
                tags = quote.select("div.tags a.tag")
                tags_text = ScrapeQuotes.retrieve_tags(tags)

                quotes_on_page.append({
                    "text": text,
                    "author": author,
                    "tags": tags_text,
                })

            quotes.extend(quotes_on_page)

        return quotes

    @staticmethod
    def retrieve_tags(tags: list) -> str:
        tag_text = []
        for tag in tags:
            tag_text.append(tag.get_text())

        tags_text = ", ".join(tag_text)

        return tags_text

