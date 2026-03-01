import pandas as pd

from services.scrape_books import ScrapeBooks
from services.scrape_quotes import ScrapeQuotes
from services.visualise_data import VisualiseBooks, VisualiseQuotes
from utils.write_csv import write_csv
from utils.print_df_info import print_df_info

books_path = "data/books.csv"
quotes_path = "data/quotes.csv"

books = ScrapeBooks.scrape_books_pages(50)
write_csv(books_path, books)

quotes = ScrapeQuotes.scrape_quotes_pages(10)
write_csv(quotes_path, quotes)

# Visualise Books data
df_books = pd.read_csv(books_path)
df_books = df_books.dropna()
print_df_info(df_books, "books")

VisualiseBooks.box_plot_price(df_books)
VisualiseBooks.bar_chart_ratings(df_books)
VisualiseBooks.scatter_price_vs_rating(df_books)

# Visualise Quotes data
df_quotes = pd.read_csv(quotes_path)
df_quotes = df_quotes.dropna()
df_quotes = VisualiseQuotes.prepare_numerical_values(df_quotes)
print_df_info(df_quotes, "quotes")

VisualiseQuotes.box_plot_quote_length(df_quotes)
VisualiseQuotes.bar_chart_top_10_authors(df_quotes)
VisualiseQuotes.scatter_length_vs_tags(df_quotes)
