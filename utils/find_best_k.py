import numpy as np


def find_best_k(k_range, silhouette_scores, davies_bouldin_scores):
    k_list = list(k_range)
    optimal_k_silhouette = k_list[np.argmax(silhouette_scores)]
    print("Best k (Silhouette):", optimal_k_silhouette)
    optimal_k_db = k_list[np.argmin(davies_bouldin_scores)]
    print("Best k (Davies-Bouldin):", optimal_k_db)
    return optimal_k_silhouette
