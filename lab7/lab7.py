from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt

def hist(digits):
    histograms = []
    for image in digits.images:
        parts = np.array_split(image, 4, axis=1)
        histogram_values = np.concatenate([np.histogram(part, bins=16, range=(0, 16))[0] for part in parts])
        histograms.append(histogram_values)
    return np.array(histograms)

def kmeans(X, k, count, tol):
    ci = np.random.choice(len(X), k, replace=False)
    cen = X[ci]
    for _ in range(count):
        d = np.linalg.norm(X[:, np.newaxis] - cen, axis=2)
        labels = np.argmin(d, axis=1)
        new_c = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else cen[j] for j in range(k)])
        if np.linalg.norm(new_c - cen) < tol:
            break
        cen = new_c
    return labels, cen

def match_labels(true_labels, cluster_labels):
    cm = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in cluster_labels])

if __name__ == "__main__":
    digits = load_digits()
    histograms = hist(digits)
    labels, _ = kmeans(histograms, 10, 100, 1e-4)
    
    matched_labels = match_labels(digits.target, labels)
    cm_custom = confusion_matrix(digits.target, matched_labels)
    
    plt.imshow(cm_custom, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Матрица ошибок')
    plt.colorbar()
    plt.show()
    
    kmeans_by_lib = KMeans(n_clusters=10, random_state=42)
    predicted_labels = kmeans_by_lib.fit_predict(digits.data)
    matched_lib_labels = match_labels(digits.target, predicted_labels)
    
    
    cm_lib = confusion_matrix(digits.target, matched_lib_labels)
    plt.imshow(cm_lib, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Матрица ошибок (библиотека)')
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(8, 3))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(f'Число: {digits.target[i]}')
        plt.axis('off')
    plt.show()

    accuracy_custom = np.mean(digits.target == matched_labels)
accuracy_lib = np.mean(digits.target == matched_lib_labels)

print(f'Точность кастомного KMeans: {accuracy_custom:.4f}')
print(f'Точность KMeans из sklearn: {accuracy_lib:.4f}')