import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
import numpy as np

def load_data(file_path='iris.txt'):
    # Load all 5 columns: 4 features + 1 species label
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    try:
        df = pd.read_csv(file_path, header=None, names=columns, sep=',')
        return df
    except FileNotFoundError:
        print("Error: File not found.")
        return None

def run_cluster_validation(df):
    X = df.iloc[:, 0:4]

    inertia = []
    sil_scores = []
    k_range = range(2, 11)

    print("\n--- Validation Metrics ---")
    print(f"{'K':<5} {'Inertia (SSE)':<20} {'Silhouette Score':<20}")

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)

        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

        print(f"{k:<5} {km.inertia_:<20.2f} {silhouette_score(X, km.labels_):<20.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Elbow (Inertia) on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Sum of Squared Errors (Inertia)', color=color)
    ax1.plot(k_range, inertia, marker='o', color=color, label='Elbow (SSE)')
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Silhouette on right y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, sil_scores, marker='s', linestyle='--', color=color, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Elbow Method and Silhouette Score Analysis')
    fig.tight_layout()
    plt.show()

def plot_step(df, centroids, iteration, ax):
    """Helper to plot a single step of the K-means process"""
    sns.scatterplot(
        data=df, x='petal_length', y='petal_width',
        hue='cluster', palette='viridis', s=50, ax=ax, legend=False
    )
    # Plot centroids
    ax.scatter(centroids[:, 2], centroids[:, 3],
               s=200, c='red', marker='X', edgecolors='black', label='Centroids')
    ax.set_title(f'Iteration: {iteration}')
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')

def run_analysis():
    df = load_data()

    run_cluster_validation(df)
    if df is None: return

    X = df.iloc[:, 0:4]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    iterations = [1, 2, 3]

    common_seed = 42

    print("--- Running K-Means Step-by-Step ---")

    for idx, i in enumerate(iterations):
        kmeans = KMeans(n_clusters=3, init='random', n_init=1, max_iter=i, random_state=common_seed)
        kmeans.fit(X)

        df['cluster'] = kmeans.labels_
        centroids = kmeans.cluster_centers_

        plot_step(df, centroids, i, axes[idx])

        if i == 3: # Assuming convergence happens quickly for Iris
            axes[idx].set_title(f'Iteration {i} (Likely Converged)')

    plt.tight_layout()
    plt.show()

    # --- Q1: Comparison to Ground Truth ---
    final_km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_label'] = final_km.fit_predict(X)

    print("\n--- Centroids (Final) ---")
    print(final_km.cluster_centers_)

    ct = pd.crosstab(df['species'], df['cluster_label'])
    print("\n--- Confusion Matrix (Species vs Cluster) ---")
    print(ct)

if __name__ == "__main__":
    run_analysis()
