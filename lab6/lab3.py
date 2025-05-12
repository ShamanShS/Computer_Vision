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

