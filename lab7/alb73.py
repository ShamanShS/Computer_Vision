import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Функция для выполнения KMeans с нуля
def kmeans(X, clusters_count, iterations=100, threshold=1e-4):
    X_len, _ = X.shape
    centroids = X[np.random.choice(X_len, clusters_count, replace=False)]
    labels = np.zeros(X_len, dtype=int)
    
    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(clusters_count)])
        
        if np.all(np.abs(new_centroids - centroids) < threshold):
            break
        centroids = new_centroids
    
    return labels, centroids

# Подсчет матрицы ошибок
def matrixErrors(targets, clusters_tags):
    uniq_count = len(np.unique(targets))
    conf_matrix = np.zeros((uniq_count, uniq_count), dtype=int)
    for true_tag, my_tag in zip(targets, clusters_tags):
        conf_matrix[true_tag, my_tag] += 1
    return conf_matrix

if __name__ == "__main__":
    digits = load_digits()
    data = digits.data
    true_labels = digits.target

    # Применяем свой k-means
    k = 10
    predicted_labels, _ = kmeans(data, k)
    # print(len(predicted_labels))
    cluster_to_label = {}
    for cluster in range(k):
        mask = (predicted_labels == cluster)
        if np.any(mask):
            most_common_label = np.bincount(true_labels[mask]).argmax()
            cluster_to_label[cluster] = most_common_label
    
    predicted_labels_mapped = np.array([cluster_to_label[label] for label in predicted_labels])
    
    # Визуализация
    plt.figure(figsize=(8, 3))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(f'Предсказ: {predicted_labels_mapped[i]}\nИстин: {true_labels[i]}')
        plt.axis('off')
    plt.show()
    
    # Accuracy и матрица ошибок
    accuracy = accuracy_score(true_labels, predicted_labels_mapped)
    print(f"Accuracy: {accuracy:.4f}")
    conf_matrix = matrixErrors(true_labels, predicted_labels_mapped)
    
    # Отображение матриц ошибок
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Reds)
    axes[0].set_title('Матрица ошибок (самописный)')
    axes[0].set_xlabel('Предсказанные метки')
    axes[0].set_ylabel('Истинные метки')
    plt.colorbar(axes[0].imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Reds), ax=axes[0])
    
    # Библиотечный KMeans
    kmeansB = KMeans(n_clusters=10, init='k-means++', random_state=42)
    kmeansB.fit(data)
    predicted_labels = kmeansB.labels_
    
    cluster_to_label = {}
    for cluster in range(k):
        mask = (predicted_labels == cluster)
        if np.any(mask):
            most_common_label = np.bincount(true_labels[mask]).argmax()
            cluster_to_label[cluster] = most_common_label
    
    predicted_labels_mapped = np.array([cluster_to_label[label] for label in predicted_labels])
    accuracy = accuracy_score(true_labels, predicted_labels_mapped)
    print(f"Accuracy (библиотека): {accuracy:.4f}")
    conf_matrix = matrixErrors(true_labels, predicted_labels_mapped)
    
    axes[1].imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Reds)
    axes[1].set_title('Матрица ошибок (библиотека)')
    axes[1].set_xlabel('Предсказанные метки')
    axes[1].set_ylabel('Истинные метки')
    plt.colorbar(axes[1].imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Reds), ax=axes[1])
    
    plt.show()