import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image
import lab3



def kenny(image):
    gray_image1 = np.mean(image1_mas, axis=2).astype(np.uint8)
    blurimage1 = lab3.gause(gray_image1)
    sobel_image1 = lab3.sobel(blurimage1)
    gradient_magnitude1 = np.sqrt(sobel_image1[0]**2 + sobel_image1[1]**2)
    gradient_direction1 = np.arctan2(sobel_image1[1], sobel_image1[0])
    quantized_direction1 = lab3.quantize_directions(gradient_direction1)
    suppressed_image1 = lab3.non_maximum_suppression(gradient_magnitude1, gradient_direction1)
    result_image = lab3.hysteresis(suppressed_image1, 75, 200)
    return result_image



def EquationLine(x, y, theta):
    theta = theta * (np.pi/180)
    return round(x * np.cos(theta) + y * np.sin(theta))



def FillCumMas(image):
    rows, cols = image.shape
    diagonal = round(np.sqrt(rows**2 + cols**2))
    mas = np.zeros([diagonal, 271])
    Q = np.linspace(0, 270, 271).astype(int)
    for y in range(rows):
        for x in range(cols):
            if image[y, x] != 0:
                for q in Q:
                    ro = EquationLine(x, y, q - 90)
                    mas[ro, q] += 1
    
    return (mas.astype(int))



def FindPeak(matr, number_of_peaks):
    matr_copy = matr.copy()
    rows, cols = matr.shape
    peaks = []
    for k in range(number_of_peaks):
        peak = np.amax(matr_copy)
        for i in range(rows):
            for j in range(cols):
                if matr_copy[i, j] == peak:
                    peaks.append([i, j])
                    matr_copy[i, j] = 0
                    NullAround(i, j, 5,  matr_copy)

    return peaks, matr_copy

def NullAround(i, j, k, matr):
    try:
        matr[i - k:i + k, j - k:j + k] = 0
    except:
        pass

def DrawLine(_peaks, picture, t = 2):
    new = picture.copy()    
    rows, cols, t = picture.shape
    for i in range(len(_peaks)):
        peak = _peaks[i]
        ro = peak[0]
        theta = (peak[1] - 90) * (np.pi/180)
        color = np.random.randint(0, 256, 3)
        for y in range(rows):
            for x in range(cols):
                if ro + t >  x * np.cos(theta) + y * np.sin(theta) and ro - t <  x * np.cos(theta) + y * np.sin(theta):
                        new[y, x] = color

    return new

if __name__ == "__main__":
 
    image1 = Image.open('image11.png')    
    image1.load()
    image1_mas =  np.array(image1)

    result_image = kenny(image1_mas)
    print("A")

    m = FillCumMas(result_image)
    # mg = lab3.gause(m, 2)
    print("B")
    cum_mas_sum_max, matr = FindPeak(m, 10)

    print(cum_mas_sum_max)
    new = DrawLine(cum_mas_sum_max,image1_mas, 2)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1), plt.title('Исходное изображение'), plt.imshow(image1_mas, cmap='gray')
    plt.subplot(2, 2, 2), plt.title('Границы'), plt.imshow(result_image, cmap='gray')
    plt.subplot(2, 2, 3), plt.title('Аккумулятор'), plt.imshow(m, cmap='hot', aspect='auto')
    plt.subplot(2, 2, 4), plt.title('Прямые на изображении'), plt.imshow(new[..., ::-1])
    plt.tight_layout()
    plt.show()