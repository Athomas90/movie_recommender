import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random

# --- Load and prepare the dataset ---
def preprocess_movies(file_path):
    df = pd.read_csv(file_path)
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df['adult'] = df['adult'].astype(str).str.lower().map({'true': 1, 'false': 0})
    df['genres'] = df['genres'].fillna('')
    df['genres_list'] = df['genres'].apply(lambda x: [g.strip() for g in x.split(',')])
    all_genres = set(genre for sublist in df['genres_list'] for genre in sublist)
    for genre in all_genres:
        df[genre] = df['genres_list'].apply(lambda g_list: int(genre in g_list))
    feature_cols = ['vote_average', 'vote_count', 'revenue', 'budget', 'runtime',
                    'popularity', 'release_year', 'adult'] + list(all_genres)
    df_model = df[feature_cols].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_model)
    return df, df_model, scaled, feature_cols

# --- Elbow plot ---
def plot_elbow(scaled_data, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method to Determine Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

# --- Run KMeans and assign clusters ---
def assign_clusters(df, df_model, scaled_data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    df['Cluster'] = -1
    df.loc[df_model.index, 'Cluster'] = labels
    return df, kmeans

# --- Recommender based on cluster ---
def recommend_by_title(title, df):
    title = title.lower()
    if title not in df['title'].str.lower().values:
        return [f"Title '{title}' not found."]
    cluster = df[df['title'].str.lower() == title]['Cluster'].values[0]
    if cluster == -1:
        return ["Title not assigned to a valid cluster."]
    candidates = df[(df['Cluster'] == cluster) & (df['title'].str.lower() != title)]
    return candidates['title'].sample(n=min(5, len(candidates))).tolist()

# --- Random recommender baseline ---
def random_recommendations(df, exclude_title=None):
    candidates = df[df['title'].str.lower() != exclude_title.lower()] if exclude_title else df
    return candidates['title'].sample(n=min(5, len(candidates))).tolist()

# --- Accuracy test comparison ---
def test_recommenders(df, n_trials=10):
    titles = df[df['Cluster'] != -1]['title'].sample(n=n_trials, random_state=42).tolist()
    cluster_hits = 0
    random_hits = 0
    for title in titles:
        title_lower = title.lower()
        cluster = df[df['title'].str.lower() == title_lower]['Cluster'].values[0]
        cluster_recs = recommend_by_title(title, df)
        cluster_recs_clean = [t.lower() for t in cluster_recs if isinstance(t, str)]
        cluster_hits += sum(df[df['title'].str.lower().isin(cluster_recs_clean)]['Cluster'] == cluster)
        rand_recs = random_recommendations(df, exclude_title=title)
        rand_recs_clean = [t.lower() for t in rand_recs]
        random_hits += sum(df[df['title'].str.lower().isin(rand_recs_clean)]['Cluster'] == cluster)
    print(f"Cluster Recommender Accuracy: {cluster_hits / (n_trials * 5):.2%}")
    print(f"Random Recommender Accuracy: {random_hits / (n_trials * 5):.2%}")
