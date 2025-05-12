import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image

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

def gause(image, sigma = 5):
    kernel_size = 5
    return apply_gauss(image, gauss_kernel(kernel_size, sigma))



def convolution(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    
    pad_h, pad_w = kh // 2, kw // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    result = np.zeros_like(image, dtype=float)
    

    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + kh, j:j + kw]
            result[i, j] = np.sum(region * kernel)
    
    return result


def sobel(image):

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    gradient_x = convolution(image, sobel_x)
    gradient_y = convolution(image, sobel_y)
   
    return gradient_x, gradient_y


def quantize_directions(gradient_direction):

    gradient_direction_deg = np.degrees(gradient_direction)
    

    directions = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180
    bins = np.array([-157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5])

    indices = np.digitize(gradient_direction_deg, bins, right=True)
    quantized_direction = directions[indices % 8]

    
    return quantized_direction


def non_maximum_suppression(gradient_magnitude, gradient_direction):

    h, w = gradient_magnitude.shape
    suppressed_image = np.zeros((h, w), dtype=np.float32)


    gradient_direction_deg = (np.degrees(gradient_direction) + 360) % 360

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            magnitude = gradient_magnitude[i, j]
            direction = gradient_direction_deg[i, j]


            if (0 <= direction < 22.5) or (157.5 <= direction < 202.5) or (337.5 <= direction <= 360):
                neighbor1, neighbor2 = gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]
            elif (22.5 <= direction < 67.5) or (202.5 <= direction < 247.5):
                neighbor1, neighbor2 = gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]
            elif (67.5 <= direction < 112.5) or (247.5 <= direction < 292.5):
                neighbor1, neighbor2 = gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]
            else:
                neighbor1, neighbor2 = gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]


            if magnitude >= neighbor1 and magnitude >= neighbor2:
                suppressed_image[i, j] = magnitude

    return suppressed_image


def hysteresis(gm, lw, li):
    h, w = gm.shape
    result_image = np.zeros((h, w), dtype=np.uint8)
    strong_edges = deque()
    for i in range(h):
        for j in range(w):
            if gm[i, j] >= li:
                result_image[i, j] = 255
                strong_edges.append((i, j))
    while strong_edges:
        i, j = strong_edges.popleft()
        for x in range(-1, 2):
            for y in range(-1, 2):
                if x == 0 and y == 0:
                    continue
                ni, nj = i + x, j + y
                if 0 <= ni < h and 0 <= nj < w and result_image[ni, nj] == 0 and gm[ni, nj] >= lw:
                    result_image[ni, nj] = 255
                    strong_edges.append((ni, nj))

    return result_image

if __name__ == "__main__":
    
    image1 = Image.open('image1.jpg')
    # image2 = Image.open('image3.jpg')
    image1.load()
    image1_mas =  np.array(image1)
    # image2_mas =  np.array(image2)

    gray_image1 = np.mean(image1_mas, axis=2).astype(np.uint8)
    # gray_image2 = np.mean(image2_mas, axis=2).astype(np.uint8)

    blurimage1 = gause(gray_image1)
    # blurimage2 = gause(gray_image2)

    sobel_image1 = sobel(blurimage1)
    # sobel_image2 = sobel(blurimage2)




    gradient_magnitude1 = np.sqrt(sobel_image1[0]**2 + sobel_image1[1]**2)
    # gradient_magnitude2 = np.sqrt(sobel_image2[0]**2 + sobel_image2[1]**2)

    gradient_direction1 = np.arctan2(sobel_image1[1], sobel_image1[0])
    # gradient_direction2 = np.arctan2(sobel_image2[1], sobel_image2[0])

    quantized_direction1 = quantize_directions(gradient_direction1)
    # quantized_direction2 = quantize_directions(gradient_direction2)

    suppressed_image1 = non_maximum_suppression(gradient_magnitude1, gradient_direction1)
    # suppressed_image2 = non_maximum_suppression(gradient_magnitude2, gradient_direction2)

    result_image4 = hysteresis(gradient_magnitude1, 75, 200)
    # result_image5 = hysteresis(gradient_magnitude2, 75, 200)


    # print("Исходное направление (градусы):")
    # print(np.degrees(gradient_direction1) % 360)
    # print("Квантизированное направление:")
    # for i in (np.degrees(gradient_direction1) % 360):
    #     print(i)
    # print("------------------------------------")
    # print("Исходное направление (градусы):")
    # print(np.degrees(gradient_direction2) % 360)
    # print("Квантизированное направление:")
    # print(quantized_direction2)
    # print("------------------------------------")
    
    plt.figure(figsize=(15, 12))


    plt.subplot(3, 3, 1)
    plt.imshow(image1_mas)
    plt.axis('off')
    plt.title('Оригинальное изображение')
    
    plt.subplot(3, 3, 2)
    plt.imshow(sobel_image1[0], cmap="gray")
    plt.axis('off')
    plt.title('Градиент X изображение 1')

    plt.subplot(3, 3, 3)
    plt.imshow(sobel_image1[1], cmap="gray")
    plt.axis('off')
    plt.title('Градиент Y изображение 1')

    plt.subplot(3, 3, 4)
    plt.imshow(np.sqrt(sobel_image1[0]**2 + sobel_image1[1]**2), cmap="gray")
    plt.axis('off')
    plt.title('Модуль градиента изображения 1')


    plt.subplot(3, 3, 5)
    plt.imshow(gradient_magnitude1, cmap="gray")
    plt.axis('off')
    plt.title('Магнитуда градиента 1')

    plt.subplot(3, 3, 6)
    plt.imshow(gradient_direction1, cmap="gray")
    plt.axis('off')
    plt.title('Направление градиента 1')

    plt.subplot(3, 3, 7)
    plt.imshow(quantized_direction1, cmap="gray")
    plt.axis('off')
    plt.title('Квантизированное напр')
    plt.tight_layout()

    # plt.subplot(2, 2, 3)
    # plt.imshow(gradient_magnitude2, cmap="gray")
    # plt.axis('off')
    # plt.title('Магнитуда градиента 2')

    # plt.subplot(2, 2, 4)
    # plt.imshow(gradient_direction2, cmap="hsv")
    # plt.axis('off')
    # plt.title('Направление градиента 2')


    # plt.figure(figsize=(12, 6))
    plt.subplot(3, 3, 8)
    plt.imshow(suppressed_image1, cmap="gray")
    plt.axis('off')
    plt.title('Изображение 1 после подавления немаксимумов')

    # plt.subplot(1, 2, 2)
    # plt.imshow(suppressed_image2, cmap="gray")
    # plt.axis('off')
    # plt.title('Изображение 2 после подавления немаксимумов')
    # plt.tight_layout()


    plt.subplot(3, 3, 9)
    plt.imshow(result_image4, cmap="gray")
    plt.axis('off')
    plt.title('Границы изображения 1, пороги: 75, 200')

    # plt.subplot(1, 3, 2)
    # plt.imshow(result_image5, cmap="gray")
    # plt.axis('off')
    # plt.title('Границы изображения 2, пороги: 75, 200')
    # plt.tight_layout()


    # plt.figure(figsize=(15, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(gradient_direction1, cmap="gray")
    # plt.axis('off')
    # plt.title('Границы изображения 1, пороги: 75, 200')

    # plt.subplot(1, 2, 2)
    # plt.imshow(quantized_direction1, cmap="gray")
    # plt.axis('off')
    # plt.title('Границы изображения 2, пороги: 75, 200')
    # plt.tight_layout()


    plt.show()
