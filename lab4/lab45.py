import lab3
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def kenny(image):

    gray_image = np.mean(image, axis=2).astype(np.uint8)
    blur_image = lab3.gause(gray_image)
    sobel_image = lab3.sobel(blur_image)
    gradient_magnitude = np.sqrt(sobel_image[0]**2 + sobel_image[1]**2)
    gradient_direction = np.arctan2(sobel_image[1], sobel_image[0])
    quantized_direction = lab3.quantize_directions(gradient_direction)
    suppressed_image = lab3.non_maximum_suppression(gradient_magnitude, quantized_direction)
    return lab3.hysteresis(suppressed_image, 75, 200)

def EquationLine(x, y, theta):
    theta = theta * (np.pi / 180)
    return round(x * np.cos(theta) + y * np.sin(theta))
    
def ConstructionLine(x, ro, q):
    return round((-x * np.cos(q) + ro ) / np.sin(q))

def fill_cumulative_array(image):
    rows, cols = image.shape
    diagonal = round(np.sqrt(rows**2 + cols**2))

    res = np.zeros([diagonal, 271])

    Q = np.linspace(0, 270, 271).astype(int)
    for y in range(rows):
        for x in range(cols):
            if image[y, x] != 0:
                for q in Q:
                    rho = EquationLine(x, y, q - 90)
                    res[rho, q] += 1

    return res.astype(int)

def NullAround(i, j, k, mx):
    try:
        mx[i - k:i + k, j - k:j + k] = 0
    except:
        pass


def FindPeak(img, number_of_peaks):
    res = img.copy()
    rows, cols = img.shape
    peaks = []
    for _ in range(number_of_peaks):
        peak = np.amax(res)
        for i in range(rows):
            for j in range(cols):
                if res[i, j] == peak:
                    peaks.append([i, j])
                    res[i, j] = 0
                    NullAround(i, j, 5,  res)

    return peaks

def DrawLine(_peaks, picture, picture3, t = 3):
    new = picture.copy()    
    rows, cols = picture3.shape
    print(f"len(_peaks) = {len(_peaks)}")
    for i in range(len(_peaks)):
        peak = _peaks[i]
        rho = peak[0]
        theta = (peak[1] - 90) * (np.pi/180)
        color = np.random.randint(0, 256, 3)
        for y in range(rows):
            for x in range(cols):
                if rho + t >  x * np.cos(theta) + y * np.sin(theta) and rho - t <  x * np.cos(theta) + y * np.sin(theta):
                    new[y, x] = color

    return new

def ShowImage(*images):
    numOfImages = len(images)
    fig, axes = plt.subplots(1, numOfImages)

    if numOfImages == 1:
        axes = [axes]

    for i in range(numOfImages):
        axes[i].imshow(images[i], cmap='gray')

    fig.set_figwidth(10 * numOfImages)    
    fig.set_figheight(10) 

    plt.show()

if __name__ == '__main__':
    fig, ax = plt.subplots()
    img = Image.open('image.png')

    img.load()
    image_data = np.array(img, dtype='uint8')
    h = kenny(image_data)
    print("A")
    m = fill_cumulative_array(h)
    mg = lab3.gause(m, sigma=3)
    print("B")
    plt.imshow(mg, cmap='hot', aspect="auto")
    peaks = FindPeak(mg, 10)
    print("C")
    new_test = DrawLine(peaks, image_data, h)
    ShowImage(image_data, new_test, h)
