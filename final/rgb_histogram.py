import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_rgb_histogram(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    plt.figure(figsize=(10, 5))
    bins = np.arange(257)
    plt.plot(bins[:-1], np.histogram(r, bins=bins)[0], color='red', label='Red')
    plt.plot(bins[:-1], np.histogram(g, bins=bins)[0], color='green', label='Green')
    plt.plot(bins[:-1], np.histogram(b, bins=bins)[0], color='blue', label='Blue')
    plt.xlabel('Интенсивность')
    plt.ylabel('Количество пикселей')
    plt.title('Гистограмма RGB')
    plt.xlim(0, 255) 
    plt.legend()
    plt.show()


image_path = 'image/C.jpg'  
plot_rgb_histogram(image_path)
