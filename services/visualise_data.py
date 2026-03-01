import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class VisualiseBooks:
    @staticmethod
    def box_plot_price(df: pd.DataFrame):
        plt.figure()
        plt.boxplot(df["price"])
        plt.title("Book Prices")
        plt.xlabel("Books")
        plt.ylabel("Price")
        plt.savefig("plots/book_box_plot.png")
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
        plt.savefig("plots/book_bar_chart.png")
        plt.show()

    @staticmethod
    def scatter_price_vs_rating(df: pd.DataFrame):
        plt.figure()
        plt.scatter(df["price"], df["rating"])
        plt.title("Price vs Rating")
        plt.xlabel("Price")
        plt.ylabel("Rating")
        plt.savefig("plots/book_scatter_plot.png")
        plt.show()

    @staticmethod
    def word_cloud(df):
        word_cloud = WordCloud(
            width=800,
            height=400,
            background_color="black"
        ).generate_from_frequencies(df)

        plt.figure(figsize=(10, 5))
        plt.imshow(word_cloud)
        plt.axis("off")
        plt.show()

    @staticmethod
    def elbow_graph(k_range, inertia):
        plt.figure()
        plt.plot(k_range, inertia, marker='o')
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method")
        plt.savefig("plots_2/elbow_method.png")
        plt.show()

    @staticmethod
    def silhouette_plot(k_range, silhouette_scores):
        plt.figure()
        plt.plot(k_range, silhouette_scores, marker='o', color='green')
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs K")
        plt.savefig("plots_2/silhouette_scores.png")
        plt.show()

    @staticmethod
    def davies_bouldin_plot(k_range, davies_bouldin_scores):
        plt.figure()
        plt.plot(k_range, davies_bouldin_scores, marker='o', color='red')
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Davies-Bouldin Index")
        plt.title("Davies-Bouldin Index vs K")
        plt.savefig("plots_2/davies_bouldin_scores.png")
        plt.show()

    @staticmethod
    def cluster_visualization(df, labels):
        plt.figure(figsize=(10, 6))
        
        # Plot each genre with its own color and label
        genres = df['Genre'].unique()
        colors = plt.cm.viridis([i / len(genres) for i in range(len(genres))])
        
        for i, genre in enumerate(sorted(genres)):
            mask = df['Genre'] == genre
            plt.scatter(df.loc[mask, 'price'], df.loc[mask, 'rating'], 
                       c=[colors[i]], label=genre, alpha=0.7)
        
        plt.xlabel("Price")
        plt.ylabel("Rating")
        plt.title("Book Cluster Visualization by Genre")
        plt.legend(title='Genre')
        plt.savefig("plots_2/cluster_visualization.png")
        plt.show()

    @staticmethod
    def confusion_matrix_plot(y_test, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", ax=ax)

        ax.set_title(title)
        fig.tight_layout()
        fig.savefig("plots_2/confusion_matrix.png")
        plt.show()
        plt.close(fig)

    @staticmethod
    def tsne_plot(X_tsne, labels, title="t-SNE Visualization"):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.title(title)
        plt.colorbar(scatter, label='Genre Label')
        plt.savefig("plots_2/tsne_visualization.png")
        plt.show()

    @staticmethod
    def mlp_results_table(results):
        print(f"{'Config':<8} {'Hidden Layers':<15} {'Epochs':<8} {'LR':<8} {'Activation':<12} {'Accuracy':<10} {'F1':<10}")
        print("-"*80)
        for i, r in enumerate(results):
            c = r['config']
            print(f"{i+1:<8} {str(c['hidden_layer_sizes']):<15} {c['max_iter']:<8} {c['learning_rate_init']:<8} {c['activation']:<12} {r['accuracy']:<10.4f} {r['f1_score']:<10.4f}")
        print("="*80)


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
        plt.savefig("plots/quote_box_plot.png")
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
        plt.savefig("plots/quote_bar_chart.png")
        plt.show()

    @staticmethod
    def scatter_length_vs_tags(df: pd.DataFrame):
        plt.figure()
        plt.scatter(df["quote_length"], df["tag_count"])
        plt.title("Quote Length vs Number of Tags")
        plt.xlabel("Quote Length")
        plt.ylabel("Number of Tags")
        plt.savefig("plots/quote_scatter_plot.png")
        plt.show()

    @staticmethod
    def word_cloud(df):
        word_cloud = WordCloud(
            width=800,
            height=400,
            background_color="black"
        ).generate_from_frequencies(df)

        plt.figure(figsize=(10, 5))
        plt.imshow(word_cloud)
        plt.axis("off")
        plt.show()
