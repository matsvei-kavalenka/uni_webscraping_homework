import requests
from bs4 import BeautifulSoup


BASE_URL = 'https://books.toscrape.com/'
RATING = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


class ScrapeBooks:
    @staticmethod
    def scrape_books_pages(pages: int):
        books = []

        for i in range(1, pages + 1):
            url = f"{BASE_URL}catalogue/page-{i}.html"
            response = requests.get(url)

            soup = BeautifulSoup(response.text, 'html.parser')
            books_on_page = []

            for book in soup.select("article.product_pod"):
                a = book.select_one("h3 a")
                title = a.get("title") or a.get_text(strip=True)

                price_text = book.select_one(".product_price .price_color").get_text(strip=True)
                price_text = price_text.replace("Â", "").replace("£", "")
                price = float(price_text)

                rating = ScrapeBooks.parse_rating(book)

                books_on_page.append({
                    "title": title,
                    "price": price,
                    "rating": rating,
                })

            books.extend(books_on_page)

        return books

    @staticmethod
    def parse_rating(article):
        p = article.select_one("p.star-rating")

        if not p:
            return 0

        classes = p.get("class", [])
        word = next((c for c in classes if c in RATING), None)

        return RATING[word]
