import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
import cv2
import lab3


# def detect_edges(image):
#     """Обнаружение границ с использованием алгоритма Кэнни."""
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     edges = cv2.Canny(gray_image, 100, 200)  # Параметры: пороги 100 и 200
#     return edges

def kenny(image):
    """Обнаружение границ (аналог Canny)."""
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    blur_image = lab3.gause(gray_image)
    sobel_image = lab3.sobel(blur_image)
    gradient_magnitude = np.sqrt(sobel_image[0]**2 + sobel_image[1]**2)
    # gradient_direction = np.arctan2(sobel_image[1], sobel_image[0])
    # quantized_direction = lab3.quantize_directions(gradient_direction)
    # suppressed_image = lab3.non_maximum_suppression(gradient_magnitude, quantized_direction)
    return lab3.hysteresis(gradient_magnitude, 75, 200)


def hough_transform(edges):
    """
    Реализация преобразования Хафа для поиска прямых.
    """
    rows, cols = edges.shape
    diag_len = int(np.sqrt(rows**2 + cols**2))  # Длина диагонали изображения
    thetas = np.deg2rad(np.arange(-90, 91))  # Углы от -90° до +90° (с шагом 1°)
    rhos = np.arange(-diag_len, diag_len + 1)  # Значения ro от -diag_len до +diag_len

    accumulator = np.zeros((2 * diag_len + 1, len(thetas)), dtype=np.float32)

    # Координаты ненулевых точек
    y_idxs, x_idxs = np.nonzero(edges)

    # Заполнение аккумулятора
    for x, y in zip(x_idxs, y_idxs):
        for theta_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta)) + diag_len
            accumulator[rho, theta_idx] += 1

    # Сглаживание аккумулятора
    # smoothed_accumulator = gaussian_filter(accumulator, sigma=2)
    smoothed_accumulator = lab3.gause(accumulator, sigma=2)

    return smoothed_accumulator, thetas, rhos


def find_peaks(accumulator, num_peaks, threshold=5):
    """
    Поиск локальных максимумов в пространстве Хафа.
    """
    peaks = []
    accumulator_copy = accumulator.copy()
    for _ in range(num_peaks):
        peak_idx = np.unravel_index(np.argmax(accumulator_copy), accumulator_copy.shape)
        peaks.append(peak_idx)
        x, y = peak_idx
        accumulator_copy[max(0, x-threshold):x+threshold+1, max(0, y-threshold):y+threshold+1] = 0
    return peaks


def draw_lines(image, peaks, thetas, rhos):
    """
    Рисование найденных прямых на исходном изображении.
    """
    result_image = image.copy()
    for rho_idx, theta_idx in peaks:
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return result_image


if __name__ == "__main__":
    # Загрузка изображения
    image_path = "image.png"  # Укажите путь к изображению
    image = np.array(Image.open(image_path).convert("RGB"))

    # Обнаружение границ
    # edges = detect_edges(image)
    edges = kenny(image)

    # Преобразование Хафа
    accumulator, thetas, rhos = hough_transform(edges)

    # Поиск локальных максимумов
    peaks = find_peaks(accumulator, num_peaks=10)

    # Рисование прямых на изображении
    result_image = draw_lines(image, peaks, thetas, rhos)

    # Визуализация результатов
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title("Исходное изображение")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Обнаруженные границы")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Аккумуляторное пространство Хафа")
    plt.imshow(accumulator, cmap="hot", aspect="auto")
    plt.xlabel("Theta (углы)")
    plt.ylabel("Rho (расстояния)")

    plt.subplot(2, 2, 4)
    plt.title("Найденные прямые")
    plt.imshow(result_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
