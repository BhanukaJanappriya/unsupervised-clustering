import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
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
