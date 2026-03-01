from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

books_features = ['price', 'rating']


def get_genre_label(mean_price, price_thresholds):
    price_low, price_high = price_thresholds
    
    if mean_price >= price_high:
        return "Expensive"
    elif mean_price >= price_low:
        return "Moderate"
    else:
        return "Cheap"


def books_prepare_cluster_params(df_books):
    features = df_books[books_features]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)

    inertia = []
    silhouette_scores = []
    davies_bouldin_scores = []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(x_scaled)

        inertia.append(kmeans.inertia_)

        s_score = silhouette_score(x_scaled, labels)
        silhouette_scores.append(s_score)
        db_score = davies_bouldin_score(x_scaled, labels)
        davies_bouldin_scores.append(db_score)
        print(f"{k}: silhouette score: {s_score}, davies bouldin score: {db_score}")

    return range(2, 11), inertia, silhouette_scores, davies_bouldin_scores, x_scaled


def apply_clustering(x_scaled, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(x_scaled)
    return labels


def assign_genre(df, labels):
    df = df.copy()
    df['cluster'] = labels
    df['Genre_label'] = labels
    
    # Calculate price thresholds based on tertiles
    price_low = df['price'].quantile(0.33)
    price_high = df['price'].quantile(0.66)
    
    # Compute mean price per cluster and assign genre
    cluster_stats = df.groupby('cluster')[['price', 'rating']].mean()
    
    genre_mapping = {}
    for cluster_id in cluster_stats.index:
        mean_price = cluster_stats.loc[cluster_id, 'price']
        genre_mapping[cluster_id] = get_genre_label(mean_price, (price_low, price_high))
    
    df['Genre'] = df['Genre_label'].map(genre_mapping)
    
    print("\nCluster Genre Assignments:")
    for cluster_id, genre in genre_mapping.items():
        stats = cluster_stats.loc[cluster_id]
        print(f"  Cluster {cluster_id}: {genre} (avg price: {stats['price']:.2f})")
    
    return df
