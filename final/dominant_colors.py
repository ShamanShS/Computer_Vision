import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter

def get_dominant_colors(image_path, k=3):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    pixels = image.reshape(-1, 3)
    

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    

    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    

    counts = Counter(labels)
    

    sorted_colors = [colors[i] for i in counts.keys()]
    
    return sorted_colors, counts

def plot_colors(colors, counts):

    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), labels=[f'Color {i+1}: {tuple(map(int, color))}' for i, color in enumerate(colors)], colors=np.array(colors)/255, startangle=90)
    plt.axis('equal')
    plt.show()


image_path = "image/C.jpg"


colors, counts = get_dominant_colors(image_path, k=5)


plot_colors(colors, counts)
