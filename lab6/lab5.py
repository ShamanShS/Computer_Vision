import math
import numpy
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def convert(img):
    original = np.array(img)
    data = np.array(original)
    h, w = data.shape[0], data.shape[1]
    res = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            res[i, j] = np.sum(data[i, j]) / 3
    return res

def specialp(gray_img, t=30):
    h, w = gray_img.shape[0], gray_img.shape[1]
    special = []
    circle = [[3, 0], [3, 1], [2, 2], [1, 3], [0, 3], [-1, 3], [-2, 2], [-3, 1], [-3, 0], [-3, -1], [-2, -2], [-1, -3], [0, -3], [1, -3], [2, -2], [3, -1]]
    for i in range(3, h - 3):
        for j in range(3, w - 3):
            fl = [gray_img[i, j + 3], gray_img[i + 3, j], gray_img[i, j - 3], gray_img[i - 3, j]]
            a = gray_img[i, j]
            num1 = 0
            num2 = 0
            for b in fl:
                if a >= b + t:
                    num1 += 1
                if b >= a + t:
                    num2 += 1
            if num1 < 3 and num2 < 3:
                continue
            num1, num2 = 0, 0
            for k in range(-9, len(circle)):
                b = gray_img[i + circle[k][0], j + circle[k][1]]
                if a >= b + t:
                    num1 += 1
                    if num1 > 9:
                        special.append([i, j])
                        break
                else:
                    num1 = 0
                if b >= a + t:
                    num2 += 1
                    if num2 > 9:
                        special.append([i, j])
                        break
                else:
                    num2 = 0
    return special

def gaus(gray_img):
    fs = (5, 5)
    fc = (2, 2)
    filter = np.zeros(fs, dtype=float)
    sigma = 1
    c = 1 / (2 * np.pi * sigma * sigma)
    for i in range(5):
        for j in range(5):
            filter[i, j] = c * np.exp(-((j - 2) ** 2 + (i - 2) ** 2) / (2 * sigma * sigma))
    k = 0
    for i in range(5):
        for j in range(5):
            k += filter[i, j]
    for i in range(5):
        for j in range(5):
            filter[i, j] /= k
    nextf = np.zeros((gray_img.shape[0] + fc[0] * 2, gray_img.shape[1] + fc[1] * 2))
    org = np.array(nextf.astype(np.uint8))
    org[fc[0]: -fc[0], fc[1]:-fc[1]] = gray_img
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            nextf[i + fc[0], j + fc[1]] = \
                np.sum(org[i: i + fs[0], j: j + fs[1]] * filter)
    nextf = nextf[fc[0]: -fc[0], fc[1]:-fc[1]]
    blur_img = nextf
    return blur_img

def sobel(blur_img):
    f_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    f_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    xX = convolve2d(blur_img, f_x) / 8
    yY = convolve2d(blur_img, f_y) / 8
    return xX, yY

def magnitude(xX, yY):
    magn = np.sqrt(xX ** 2 + yY ** 2)
    magn = magn.astype(np.uint8)
    return magn

def sumR(xX, yY, special, filter):
    Rs = []
    center = filter.shape[0] // 2
    k = 0.04
    ix2 = xX ** 2
    ixiy = xX * yY
    iy2 = yY ** 2
    for l in range(len(special)):
        y = special[l][0] - center
        x = special[l][1] - center
        m = np.zeros((2, 2))
        for u in range(filter.shape[0]):
            for v in range(filter.shape[1]):
                if y + u < xX.shape[0] and x + v < xX.shape[1]:
                    a = np.array([[ix2[y + u, x + v], ixiy[y + u, x + v]], [ixiy[y + u, x + v], iy2[y + u, x + v]]])
                    m += filter[u, v] * a
        R = np.linalg.det(m) - k * np.trace(m) ** 2
        if R > 0:
            Rs.append([R, l])
    Rs = sorted(Rs, key=lambda Rs: Rs[0], reverse=True)
    R_sorted = Rs[:700]
    special2 = []
    for i in R_sorted:
        special2.append([special[i[1]][0], special[i[1]][1]])
    return Rs, R_sorted, special2


def visp(img, kyypoint, R_sorted):
    k = 0
    for i in R_sorted:
        img[kyypoint[i[1]][0], kyypoint[i[1]][1]] = [255, 0, 0]
        img[kyypoint[i[1]][0] + 1, kyypoint[i[1]][1]] = [255, 0, 0]
        img[kyypoint[i[1]][0] - 1, kyypoint[i[1]][1]] = [255, 0, 0]
        img[kyypoint[i[1]][0], kyypoint[i[1]][1] + 1] = [255, 0, 0]
        img[kyypoint[i[1]][0], kyypoint[i[1]][1] - 1] = [255, 0, 0]
        k += 1
    return img

def angless(R_sorted, special,gray_img):
    radius = 31
    count1 = 0
    count2 = 0
    angles = []
    for i in R_sorted:
        x, y = special[i[1]][0], special[i[1]][1]
        for k in range(-radius, radius):
            for l in range(-radius, radius):
                if (k ** 2 + l ** 2 < radius ** 2):
                    if 0 < x + k < gray_img.shape[0] and 0 < y + l < gray_img.shape[1]:
                        count1 += l * gray_img[x + k, y + l]
                        count2 += k * gray_img[x + k, y + l]
        angles.append([x, y, round(math.atan2(count2, count1) / (np.pi / 15)) * (np.pi / 15)])
        count1 = 0
        count2 = 0
    return angles


def brief(R_sorted, special, gray_img, angles, blur_img):
    lines = []
    l = 0
    for i in R_sorted:
        line = np.zeros((256), dtype=int)
        num = 0
        angle = angles[l][2]
        l += 1
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = (rotation @ p.T).astype(int).T
        points2 = (rotation @ p2.T).astype(int).T
        for k in range(len(points)):
            if 0 < special[i[1]][0] + points[k][0] < gray_img.shape[0] and 0 < special[i[1]][1] + points[k][1] < gray_img.shape[1] and 0 <  special[i[1]][0] + points2[k][0] < gray_img.shape[0] and 0 <  special[i[1]][1] + points2[k][1] < gray_img.shape[1]:
                if blur_img[special[i[1]][0] + points[k][0], special[i[1]][1] + points[k][1]] < blur_img[special[i[1]][0] + points2[k][0], special[i[1]][1] + points2[k][1]]:
                    line[num] = 1
                else:
                    line[num] = 0
                num += 1
        lines.append(line)
    return lines, special


def descriptor(img):
    gray_img = convert(img)
    # print(gray_img)
    special = specialp(gray_img)
    blur_img = gaus(gray_img)
    xX, yY = sobel(blur_img)
    magn = magnitude(xX, yY)
    filter_size = (5, 5)
    filter = np.zeros(filter_size, dtype=float)
    sigma = 1
    c = 1 / (2 * np.pi * sigma * sigma)
    for i in range(5):
        for j in range(5):
            filter[i, j] = c * np.exp(-((j - 2) ** 2 + (i - 2) ** 2) / (2 * sigma * sigma))
    k = 0
    for i in range(5):
        for j in range(5):
            k += filter[i, j]
    for i in range(5):
        for j in range(5):
            filter[i, j] /= k

    Rs, R_sorted, special2 = sumR(xX, yY, special, filter)
    res=visp(np.array(img), special, R_sorted)
    angles = angless(R_sorted, special, gray_img)
    lines, angles = brief(R_sorted, special, gray_img, angles, blur_img)
    # print(lines)
    return lines, special2,res


if __name__ == "__main__":

    img = Image.open('imgmin.png').convert("RGB")
    """коробка"""
    img_scene = Image.open('img.png').convert("RGB")
    """много коробок"""
    original = np.array(img)
    original2 = np.array(img_scene)

    width, height = img.width, img.height
    width //= 2
    height //= 2

    img2 = img.resize((width, height))
    """коробка уменьшенная"""
    # print(img2)
    width2, height2 = img_scene.width, img_scene.height
    width2 //= 2
    height2 //= 2
    img_scene2 = img_scene.resize((width2, height2))
    """много коробок уменьшенная"""

    p = []
    p2 = []
    for num in range(256):
        k = np.random.normal(0, 31/5)
        l = np.random.normal(0, 31/5)
        n = np.random.normal(0, 31/5)
        m = np.random.normal(0, 31/5)
        p.append([k, l])
        p2.append([n, m])
    p = np.array(p)
    p2 = np.array(p2)

    desk1, keypoints1,res = descriptor(img)
    desk2, keypoints2,res2 = descriptor(img2)
    desk1s, keypoints1s,res_scene = descriptor(img_scene)
    desk2s, keypoints2s,res_scene2 = descriptor(img_scene2)
    # print(desk1)

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(res)
    plt.axis('off')
    plt.title('Контрольные точки изображения 1')

    plt.subplot(2, 2, 2)
    plt.imshow(res_scene)
    plt.axis('off')
    plt.title('Контрольные точки изображения 2')
    plt.show()

    np.save('harris1.npy', keypoints1)
    np.save('descriptors1.npy', desk1)

    np.save('harris2.npy', keypoints1s)
    np.save('descriptors2.npy', desk1s)

    np.save('harris1min.npy', keypoints2)
    np.save('descriptors1min.npy', desk2)
    
    np.save('harris2min.npy', keypoints2s)
    np.save('descriptors2min.npy', desk2s)