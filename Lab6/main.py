import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import DBSCAN

# pd.set_option('display.max_columns', None)

class MyKMeans():
    def __init__(self, n_clusters=3, n_iters=100):
        self.n_clusters = n_clusters
        self.n_iters = n_iters

    def fit(self, X):
        np.random.seed(0)
        self.centers = np.random.uniform(low=X.min(axis = 0),
                                    high=X.max(axis = 0),
                                    size=(self.n_clusters, X.shape[1]))

        for it in range(self.n_iters):
            labels = self.predict(X)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(self.centers == new_centers):
                print(f"Алгоритм сошёлся на {it} итерации")
                break
            self.centers = new_centers

    def predict(self, X):
        labels = pairwise_distances_argmin(X, self.centers)
        return labels


def plt_show(X_filtered, y_filtered, name):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_filtered, cmap='tab10', edgecolor='k')
    plt.colorbar(scatter, label="Класс (y)")
    plt.title(f"{name}")
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv("data_Mar_64.txt", header=None, sep=',', quotechar='"')
    print(data)
    X, y_name = data.iloc[:, 1:].values, data.iloc[:, 0].values
    y = LabelEncoder().fit_transform(y_name)
    print(y)
    X_PCA = PCA(n_components=2, random_state=0).fit_transform(X)
    # print(f"X[0]:\n{X[0]}")
    X_filtered = X_PCA[y < 15]
    y_filtered = y[y < 15]
    print(f"X_PCA[0]:\n{round(X_PCA[0][0], 2), round(X_PCA[0][1], 2)}")
    plt_show(X_filtered, y_filtered, "Метод главных компонент")


    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
    X_filtered = X_tsne[y < 15]
    plt_show(X_filtered, y_filtered, "Стохастическое вложение соседей с t-распределением")
    print(f"X_tsne[0]:\n{round(X_tsne[0][0], 2), round(X_tsne[0][1], 2)}")
    # print(f"X[0]:\n{X[0]}")

    from sklearn import datasets

    n_samples = 1000

    noisy_blobs = datasets.make_blobs(n_samples=n_samples,
                                      cluster_std=[1.0, 3.0, 0.5],
                                      random_state=0)
    X, y = noisy_blobs
    cluster = MyKMeans()
    cluster.fit(X)
    print(f"Кластеризация объектов с гиперпараметром n_iters=100:\n{cluster.predict(X)[:10]}")
    print(f"Объект с индексом 1 относится к кластеру с индексом: {cluster.predict(X[1].reshape(1, -1))}")
    k_means100 = cluster.predict(X)
    plt_show(X, k_means100, "Кластеризация при n_iters=100")

    cluster = MyKMeans(n_clusters=3, n_iters=5)
    cluster.fit(X)
    print(f"Кластеризация объектов с гиперпараметром n_iters=5:\n{cluster.predict(X)[:10]}")
    print(f"Объект с индексом 1 относится к кластеру с индексом: {cluster.predict(X[1].reshape(1, -1))}")
    k_means5 = cluster.predict(X)
    plt_show(X, k_means5, "Кластеризация при n_iters=5")

    print(f"Количество объектов с изменением метки {np.sum(k_means5 != k_means100)}")

    clusters = DBSCAN(eps=0.5).fit_predict(X)
    print(f"Объект с индексом 1 принадлежит кластеру: {clusters[1]}")
    print(f"Полученное количество кластеров равно {len(set(clusters) - {-1})} (без выбросов)")
    print(f"Количество объектов, отнесенных к выбросам: {(clusters == -1).sum()}")
    plt_show(X, clusters, "DBSCAN")


