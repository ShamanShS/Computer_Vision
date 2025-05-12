import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import lab3


def kenny(image):
    """Обнаружение границ (аналог Canny)."""
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    blur_image = lab3.gause(gray_image)
    sobel_image = lab3.sobel(blur_image)
    gradient_magnitude = np.sqrt(sobel_image[0]**2 + sobel_image[1]**2)
    gradient_direction = np.arctan2(sobel_image[1], sobel_image[0])
    # quantized_direction = lab3.quantize_directions(gradient_direction)
    # suppressed_image = lab3.non_maximum_suppression(gradient_magnitude, quantized_direction)
    return lab3.hysteresis(gradient_magnitude, 75, 200)


def fill_hough_space(image, flag):
    """Создание кумулятивного массива для прямых (flag=1) или окружностей (flag=0)."""
    rows, cols = image.shape
    diagonal = int(np.sqrt(rows**2 + cols**2))

    if flag == 1:  # Для прямых
        mas = np.zeros((2 * diagonal, 271), dtype=int)
        y, x = np.nonzero(image)
        for q in range(271):
            theta = np.deg2rad(q - 90)
            ro = np.round(x * np.cos(theta) + y * np.sin(theta)).astype(int) + diagonal
            np.add.at(mas[:, q], ro, 1)
    else:  # Для окружностей
        max_radius = diagonal // 2
        mas = np.zeros((cols, rows, max_radius), dtype=int)
        y, x = np.nonzero(image)
        for a in range(cols):
            for b in range(rows):
                r = np.sqrt((x - a)**2 + (y - b)**2).round().astype(int)
                valid = r < max_radius
                np.add.at(mas[a, b], r[valid], 1)

    return mas


def find_peaks(hough_space, num_peaks, threshold=5):
    """Поиск локальных максимумов в пространстве Хафа."""
    peaks = []
    hough_space_copy = hough_space.copy()
    for _ in range(num_peaks):
        peak_idx = np.unravel_index(np.argmax(hough_space_copy), hough_space_copy.shape)
        peaks.append(peak_idx)
        x, y = peak_idx[:2]
        hough_space_copy[max(0, x-threshold):x+threshold+1, max(0, y-threshold):y+threshold+1] = 0
    return peaks


def draw_lines(peaks, original_image, t=2):
    """Нанесение найденных прямых на изображение."""
    result = original_image.copy()
    rows, cols, _ = result.shape
    diagonal = int(np.sqrt(rows**2 + cols**2))
    for ro, theta_idx in peaks:
        theta = np.deg2rad(theta_idx - 90)
        ro -= diagonal  # Возврат к реальной системе координат
        color = np.random.randint(0, 256, 3, dtype=int)
        for y in range(rows):
            x = int((ro - y * np.sin(theta)) / np.cos(theta))
            if 0 <= x < cols:
                result[y, x] = color
    return result


if __name__ == "__main__":
    # Загрузка изображения
    image_path = 'image1.jpg'
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    # Обнаружение границ
    edges = kenny(image_array)

    # Преобразование Хафа
    hough_space = fill_hough_space(edges, flag=1)
    smoothed_hough = lab3.gause(hough_space, sigma=5)

    # Поиск пиков
    peaks = find_peaks(smoothed_hough, num_peaks=10)

    # Рисование прямых
    result_image = draw_lines(peaks, image_array)

    # Визуализация
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title('Исходное изображение')
    plt.imshow(image_array)
    plt.subplot(2, 2, 2)
    plt.title('Обнаруженные границы')
    plt.imshow(edges, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('Аккумуляторное пространство Хафа')
    plt.imshow(smoothed_hough, cmap='hot', aspect='auto')
    plt.subplot(2, 2, 4)
    plt.title('Прямые на изображении')
    plt.imshow(result_image)
    plt.tight_layout()
    plt.show()
