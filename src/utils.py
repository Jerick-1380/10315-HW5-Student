import numpy as np
import gzip
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mtick

def consistent_scale_eigenvectors(V):
    """ Scale the columns of V such that everyone uses a consistent
        set of eigenvectors.

        Input:
        V: numpy ndarray each **column** is an eigenvector

        Returns V. V is modified in place and also returned.

        Implementation based on code from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py::svd_flip
    """
    max_abs_cols = np.argmax(np.abs(V), axis=0)
    signs = np.sign(V[max_abs_cols, range(V.shape[1])])
    V *= signs
    return V

def get_data():
  path = 'data/stock_returns.csv'
  df = pd.read_csv(path, index_col=0)
  return df

def plot_2d_pca(pca_result):
    """
    Plots the 2D PCA-compressed dataset with color-coded clusters and a legend.

    Args:
    - pca_result: pandas DataFrame with a date index and two columns for the principal components.
    """
    pca_array = pca_result.values if isinstance(pca_result, pd.DataFrame) else pca_result
    plt.figure(figsize=(10, 6))
    
    plt.scatter(pca_array[:, 0], pca_array[:, 1], alpha=0.7)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2D PCA Projection")
    plt.grid(True)
    plt.show()


def plot_2d_pca_with_clusters_and_legend(pca_result, labels, K=2):
    """
    Plots the 2D PCA-compressed dataset with color-coded clusters and a legend.

    Args:
    - pca_result: pandas DataFrame with a date index and two columns for the principal components.
    - labels: numpy ndarray of shape (N,) with cluster labels from k-means.
    - K: int, the number of clusters (default is 2).
    """

    pca_array = pca_result.values if isinstance(pca_result, pd.DataFrame) else pca_result
    plt.figure(figsize=(10, 6))
    
    for i in range(K):
        cluster_points = pca_array[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", alpha=0.7)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2D PCA Projection with K-Means Clusters")
    plt.legend(title="Clusters")
    plt.grid(True)
    plt.show()


def run_kmeans_and_plot(df, kmeans, pca, K=4):
    """
    Runs PCA and k-means clustering on the DataFrame, then plots the 2D PCA results with clusters
    and separate histograms showing the yearly distribution for each cluster label, each with its own x-axis.

    Args:
    - df: pandas DataFrame with a date index and features for clustering.
    - K: int, the number of clusters for k-means (default is 4).
    """
    
    X = df.values
    pca_result = pca(X, 2)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    centroids, labels = kmeans(pca_result, K)

    plot_2d_pca_with_clusters_and_legend(pd.DataFrame(pca_result, index=df.index), labels, K)

    fig, axes = plt.subplots(K, 1, figsize=(10, 5 * K))
    for i, label in enumerate(range(K)):  
        years = df.index.year
        year_labels = years[labels == label]

        year_counts = year_labels.value_counts().sort_index().astype(int)

        axes[i].bar(year_counts.index, year_counts.values, color='skyblue', alpha=0.7)
        axes[i].set_title(f"Yearly Distribution of Points in Cluster {label}")
        axes[i].set_xlabel("Year")
        axes[i].set_ylabel("Number of Points")
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)
        
        axes[i].xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.0f}"))
        axes[i].set_xticks(year_counts.index)
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    return pca_result, labels



def svm_train(X, y, C, kernel, tol=1e-3, max_passes=10):
    """
    Train a Support Vector Machine (SVM).
    
    Parameters:
    - X: numpy array, shape (n_samples, n_features), the input data.
    - y: numpy array, shape (n_samples,), the target labels (-1 or 1).
    - C: float, regularization parameter.
    - kernel: callable, kernel function that takes two vectors as input and returns a scalar.
    - tol: float, tolerance for stopping criterion.
    - max_passes: int, max number of passes over data without alpha changing.
    
    Returns:
    - alphas: numpy array, shape (n_samples,), the Lagrange multipliers.
    - b: float, the bias term.
    - support_vectors: list of indices of support vectors.
    """
    n_samples, n_features = X.shape
    alphas = np.zeros(n_samples)
    b = 0
    passes = 0

    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel(X[i], X[j])

    while passes < max_passes:
        alpha_pairs_changed = 0
        for i in range(n_samples):
            E_i = np.dot((alphas * y), K[:, i]) + b - y[i]

            if (y[i] * E_i < -tol and alphas[i] < C) or (y[i] * E_i > tol and alphas[i] > 0):
                j = np.random.choice([n for n in range(n_samples) if n != i])
                E_j = np.dot((alphas * y), K[:, j]) + b - y[j]
                alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                if y[i] != y[j]:
                    L, H = max(0, alphas[j] - alphas[i]), min(C, C + alphas[j] - alphas[i])
                else:
                    L, H = max(0, alphas[i] + alphas[j] - C), min(C, alphas[i] + alphas[j])
                if L == H:
                    continue
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alphas[j] -= y[j] * (E_i - E_j) / eta
                alphas[j] = np.clip(alphas[j], L, H)

                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                b1 = b - E_i - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                b2 = b - E_j - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                alpha_pairs_changed += 1

        if alpha_pairs_changed == 0:
            passes += 1
        else:
            passes = 0

    support_vectors = np.where(alphas > 0)[0]
    return alphas, b, support_vectors

def svm_predict(X_train, y_train, alphas, b, kernel, X_test):
    """
    Make predictions with the trained SVM model.

    Parameters:
    - X_train: numpy array, training data.
    - y_train: numpy array, training labels.
    - alphas: numpy array, the learned Lagrange multipliers.
    - b: float, the bias term.
    - kernel: callable, kernel function.
    - X_test: numpy array, test data.

    Returns:
    - predictions: numpy array, predicted labels for test data.
    """
    y_pred = []
    for x in X_test:
        result = sum(alphas[i] * y_train[i] * kernel(X_train[i], x) for i in range(len(X_train))) + b
        y_pred.append(np.sign(result))
    return np.array(y_pred)


def calculate_accuracy(X_train, y_train, alphas, b, kernel):
    """
    Calculate the accuracy of the SVM model on the training set.

    Parameters:
    - X_train: numpy array, training data.
    - y_train: numpy array, training labels.
    - alphas: numpy array, the learned Lagrange multipliers.
    - b: float, the bias term.
    - kernel: callable, kernel function.

    Returns:
    - accuracy: float, accuracy of the model on the training set (in percentage).
    """
    predictions = svm_predict(X_train, y_train, alphas, b, kernel, X_train)

    correct_predictions = np.sum(predictions == y_train)
    accuracy = (correct_predictions / len(y_train)) * 100

    return accuracy
