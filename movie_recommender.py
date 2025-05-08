import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random

# --- Load and prepare the dataset ---
def preprocess_movies(file_path):
    df = pd.read_csv(file_path)

    df['genres'] = df['genres'].fillna('')
    df['production_companies'] = df['production_companies'].fillna('')
    df['keywords'] = df['keywords'].fillna('')
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df['decade'] = (df['release_year'] // 10) * 10
    df['decade'] = df['decade'].fillna(0).astype(int).astype(str)
    df['decade'] = df['decade'].replace("0", "Unknown")

    # One-hot encode genres
    df['genres_list'] = df['genres'].apply(lambda x: [g.strip() for g in x.split(',')])
    all_genres = set(g for sublist in df['genres_list'] for g in sublist)
    for genre in all_genres:
        df[genre] = df['genres_list'].apply(lambda g_list: int(genre in g_list)) * 2  # Weighted genre

    # One-hot encode production companies
    df['companies_list'] = df['production_companies'].apply(lambda x: [c.strip() for c in x.split(',')])
    all_companies = set(c for sublist in df['companies_list'] for c in sublist)
    for company in all_companies:
        df[company] = df['companies_list'].apply(lambda c_list: int(company in c_list)) * 3  # Weighted company

    # One-hot encode keywords
    df['keywords_list'] = df['keywords'].apply(lambda x: [k.strip().lower() for k in x.split(',')])
    all_keywords = set(k for sublist in df['keywords_list'] for k in sublist)
    for keyword in all_keywords:
        df[keyword] = df['keywords_list'].apply(lambda k_list: int(keyword in k_list)) * 2  # Weighted keyword

    # One-hot encode decades
    decades = pd.get_dummies(df['decade'], prefix='decade')
    df = pd.concat([df, decades], axis=1)

    # Final features
    feature_cols = list(all_genres) + list(all_companies) + list(all_keywords) + list(decades.columns)
    df_model = df[feature_cols].dropna()

    # Normalize
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

# --- Assign clusters ---
def assign_clusters(df, df_model, scaled_data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    df['Cluster'] = -1
    df.loc[df_model.index, 'Cluster'] = labels
    return df, kmeans

# --- Recommender with fallback to same studio ---
def recommend_by_title(title, df):
    title = title.lower()
    match = df[df['title'].str.lower() == title]
    if match.empty:
        return [f"Title '{title}' not found."]
    cluster = match['Cluster'].values[0]
    studio_str = match['production_companies'].values[0]
    studios = [s.strip() for s in studio_str.split(',') if s.strip()]

    same_cluster = df[(df['Cluster'] == cluster) & (df['title'].str.lower() != title)]

    if studios:
        studio_match = df[df['production_companies'].str.contains(studios[0], na=False, case=False)]
        combined = pd.concat([same_cluster, studio_match]).drop_duplicates('title')
    else:
        combined = same_cluster

    if combined.empty:
        return ["No similar movies found."]
    return combined['title'].sample(n=min(5, len(combined))).tolist()
