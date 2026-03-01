import pandas as pd

from services.visualise_data import VisualiseBooks
from utils.find_best_k import find_best_k
from utils.vectorise_data import vectorise_data
from utils.prepare_cluster_params import books_prepare_cluster_params, apply_clustering, assign_genre
from utils.mlp_classifier import train_mlp_classifiers, compute_tsne

books_path = "data/books.csv"

df_books = pd.read_csv(books_path)
df_books = df_books.dropna()

df_books_vectorised = vectorise_data(df_books['title'])
VisualiseBooks.word_cloud(df_books_vectorised)

k_range, inertia, silhouette_scores, davies_bouldin_scores, x_scaled = books_prepare_cluster_params(df_books)

optimal_k = find_best_k(k_range, silhouette_scores, davies_bouldin_scores)

VisualiseBooks.elbow_graph(k_range, inertia)
VisualiseBooks.silhouette_plot(k_range, silhouette_scores)
VisualiseBooks.davies_bouldin_plot(k_range, davies_bouldin_scores)

labels = apply_clustering(x_scaled, optimal_k)

df_clustered = assign_genre(df_books, labels)

df_clustered.to_csv("data/books_clustered.csv", index=False)

VisualiseBooks.cluster_visualization(df_clustered, labels)

results, best_model, best_result, X_test, y_test = train_mlp_classifiers(x_scaled, labels)

VisualiseBooks.mlp_results_table(results)

print(f"\nBest Model Config: {best_result['config']}")
print(f"Best Accuracy: {best_result['accuracy']:.4f}")

VisualiseBooks.confusion_matrix_plot(best_result['y_test'], best_result['y_pred'], "MLP Confusion Matrix (Best Model)")

X_tsne = compute_tsne(x_scaled, labels)
VisualiseBooks.tsne_plot(X_tsne, labels, "t-SNE Visualization of Book Genres")
