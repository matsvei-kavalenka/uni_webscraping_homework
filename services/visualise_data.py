import pandas as pd
import matplotlib.pyplot as plt


class VisualiseBooks:
    @staticmethod
    def box_plot_price(df: pd.DataFrame):
        plt.figure()
        plt.boxplot(df["price"])
        plt.title("Book Prices")
        plt.xlabel("Books")
        plt.ylabel("Price")
        plt.savefig("data/book_box_plot.png")
        plt.show()

    @staticmethod
    def bar_chart_ratings(df: pd.DataFrame):
        rating_counts = df["rating"].value_counts().sort_index()

        plt.figure()
        plt.bar(rating_counts.index, rating_counts.values)
        plt.title("Distribution of Book Ratings")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.xticks([1, 2, 3, 4, 5])
        plt.savefig("data/book_bar_chart.png")
        plt.show()

    @staticmethod
    def scatter_price_vs_rating(df: pd.DataFrame):
        plt.figure()
        plt.scatter(df["price"], df["rating"])
        plt.title("Price vs Rating")
        plt.xlabel("Price")
        plt.ylabel("Rating")
        plt.savefig("data/book_scatter_plot.png")
        plt.show()


class VisualiseQuotes:
    @staticmethod
    def prepare_numerical_values(df: pd.DataFrame):
        df["quote_length"] = df["text"].apply(len)

        df["tag_count"] = df["tags"].apply(
            lambda x: len(str(x).split(",")) if pd.notnull(x) else 0
        )

        return df

    @staticmethod
    def box_plot_quote_length(df: pd.DataFrame):
        plt.figure()
        plt.boxplot(df["quote_length"])
        plt.title("Box Plot of Quote Length")
        plt.xlabel("Quotes")
        plt.ylabel("Number of Characters")
        plt.savefig("data/quote_box_plot.png")
        plt.show()

    @staticmethod
    def bar_chart_top_10_authors(df):
        author_counts = df["author"].value_counts().head(10)

        plt.figure()
        plt.bar(author_counts.index, author_counts.values)
        plt.title("Top 10 Authors by Number of Quotes")
        plt.xlabel("Author")
        plt.ylabel("Number of Quotes")
        plt.xticks(rotation=45)
        plt.savefig("data/quote_bar_chart.png")
        plt.show()

    @staticmethod
    def scatter_length_vs_tags(df: pd.DataFrame):
        plt.figure()
        plt.scatter(df["quote_length"], df["tag_count"])
        plt.title("Quote Length vs Number of Tags")
        plt.xlabel("Quote Length")
        plt.ylabel("Number of Tags")
        plt.savefig("data/quote_scatter_plot.png")
        plt.show()
