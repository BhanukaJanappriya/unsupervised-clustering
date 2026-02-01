import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def run_iris_clustering(file_path='iris.txt', sep=','):
    # 1. Load the Data
    # We assume standard CSV format. If your txt file is space-separated,
    # change sep=',' to sep='\s+'
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    try:
        # Loading only the first 4 columns as requested
        # 'header=None' assumes the file doesn't have a top row with names
        df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2, 3], names=feature_names, sep=sep)
        print(f"Successfully loaded {len(df)} samples.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # 2. Initialize and Fit K-Means
    # n_clusters=3 as requested (likely corresponding to Setosa, Versicolor, Virginica)
    # random_state=42 ensures the results are reproducible
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    # Fit the model to the data
    kmeans.fit(df)

    # Get the cluster labels (0, 1, or 2)
    labels = kmeans.labels_

    # Add labels back to the dataframe for analysis
    df['cluster'] = labels

    print("Clustering complete. Centroids:")
    print(kmeans.cluster_centers_)

    # 3. Visualization
    # Since we have 4 dimensions, we visualize 2 at a time.
    # We'll plot Sepal Length vs Sepal Width and Petal Length vs Petal Width.

    plt.figure(figsize=(12, 5))

    # Plot 1: Sepal Dimensions
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='cluster', palette='viridis', s=60)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=200, c='red', marker='X', label='Centroids')
    plt.title('K-Means Clustering: Sepal Dimensions')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()

    # Plot 2: Petal Dimensions
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='cluster', palette='viridis', s=60)
    plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3],
                s=200, c='red', marker='X', label='Centroids')
    plt.title('K-Means Clustering: Petal Dimensions')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_iris_clustering()
