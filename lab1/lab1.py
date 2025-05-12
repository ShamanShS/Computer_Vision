import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def inverte(image):
    return 255 - image

def averaging(image):
    return np.mean(image, axis=2).astype(np.uint8)

def noise(image):
    mean = 0
    stddev = 20  
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

def gauss_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def apply_gauss(image, kernel):
    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    kernel_radius = kernel_size // 2
    blurred_image = np.copy(image)
    np.pad(blurred_image, kernel_radius, mode="edge")
    for i in range(kernel_radius, image_height - kernel_radius):
        for j in range(kernel_radius, image_width - kernel_radius):
            patch = image[i - kernel_radius:i + kernel_radius + 1, j - kernel_radius:j + kernel_radius + 1]
            blurred_image[i, j] = np.sum(patch * kernel)
    return blurred_image[kernel_radius:image_height - kernel_radius, kernel_radius:image_width - kernel_radius]

def gause(image):
    kernel_size = 11
    sigma = 20.0
    return apply_gauss(image, gauss_kernel(kernel_size, sigma))

def histogram_equalization(image):
    histogram, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    cdf = histogram.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    equalized_image = cdf_normalized[image]
    return equalized_image.astype(np.uint8)


if __name__ == "__main__":
    img = Image.open("Image.jpg")
    image_mas =  np.delete(np.array(img), 3, axis=2)
    h, w, i = image_mas.shape
    print(w, h, i)


    img_inv = inverte(np.copy(image_mas))
    img_av = averaging(np.copy(image_mas))
    img_noise = noise(np.copy(img_av))
    img_blur = gause(np.copy(img_av))
    img_eq = histogram_equalization(np.copy(image_mas))

    # unique, counts = np.unique(img_noise, return_counts=True)
    # print(dict(zip(unique, counts))[255])


    plt.figure(figsize=(20, 6))

    plt.subplot(3, 4, 1)
    plt.imshow(image_mas)
    plt.axis('off')
    plt.title('Оригинальное изображение')

    plt.subplot(3, 4, 3)
    plt.imshow(img_inv)
    plt.axis('off')
    plt.title('Инвертированное изображение')

    plt.subplot(3, 4, 7)
    plt.imshow(img_av, cmap='gray')
    plt.axis('off')
    plt.title('Полутоновое изображение')

    plt.subplot(3, 4, 4)
    plt.imshow(img_noise, cmap='gray')
    plt.axis('off')
    plt.title('Изображение с шумом')

    plt.subplot(3, 4, 8)
    values, bins = np.histogram(img_noise, bins=256, range=(0, 255))
    plt.bar(bins[:-1], values, width=1, color='blue', alpha=0.7)
    plt.xlabel('Яркость пикселей')
    plt.ylabel('Частота')
    plt.title('Гистограмма изображения с шумом')

    plt.subplot(3, 4, 6)
    plt.imshow(img_blur, cmap='gray')
    plt.axis('off')
    plt.title('Размытое изображение')

    plt.subplot(3, 4, 2)
    plt.imshow(img_eq, cmap='gray')
    plt.axis('off')
    plt.title('Эквализированное изображение')

    plt.subplot(3, 4, 5)
    histogram = np.histogram(image_mas, bins=256, range=(0, 255))
    values, bins = histogram
    plt.bar(bins[:-1], values, width=1, color='blue', alpha=0.7)
    plt.xlabel('Яркость пикселей')
    plt.ylabel('Частота')
    plt.title('Гистограмма исходного изображения ')

    plt.subplot(3, 4, 5)
    histogram = np.histogram(img_eq, bins=256, range=(0, 255))
    values, bins = histogram
    plt.bar(bins[:-1], values, width=1, color='red', alpha=0.7)
    plt.xlabel('Яркость пикселей')
    plt.ylabel('Частота')
    plt.title('Гистограмма эквализированого изображения ')

    plt.show()