import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import sobel


def analyze_clustering(data, title):


    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    predicted_labels = kmeans.labels_

    
    silhouette_avg = silhouette_score(data, predicted_labels)


    print(f"{title}")
    print(f"Средний силуэтный коэффициент: {silhouette_avg:.4f}")

    #матрица ошибок
    mapped_labels = np.zeros_like(predicted_labels)
    for i in range(n_clusters):
        mask = (predicted_labels == i)
        mapped_labels[mask] = mode(true_labels[mask])[0]

    conf_matrix_mapped = confusion_matrix(true_labels, mapped_labels)

    print("Отчет классификации:")
    print(classification_report(true_labels, mapped_labels))

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_mapped, annot=True, fmt='d', cmap='Greens', 
                xticklabels=digits.target_names, yticklabels=digits.target_names)
    plt.xlabel("Предсказанные метки")
    plt.ylabel("Истинные метки")
    plt.title(f"Матрица ошибок ({title})")
    plt.show()

if __name__ == "__main__":

    digits = load_digits()
    data = digits.data
    true_labels = digits.target
    n_clusters = len(np.unique(true_labels)) 
    analyze_clustering(data, "Кластеризация по исходным признакам")

 
    histograms = np.array([np.histogram(image, bins=16, range=(0, 16))[0] for image in digits.images])
    analyze_clustering(histograms, "Кластеризация по гистограмме интенсивности")


    gradient_magnitudes = np.array([sobel(image).ravel() for image in digits.images])
    analyze_clustering(gradient_magnitudes, "Кластеризация по магнитуде градиента")


    color_moments = np.array([[np.mean(image), np.std(image)] for image in digits.images])
    analyze_clustering(color_moments, "Кластеризация по цветовым моментам (среднее, дисперсия)")

