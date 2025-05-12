from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

def seg(cbimage):
    h, w = cbimage.shape
    visited = np.zeros_like(cbimage)
    segimage = np.zeros((h, w, 3), dtype=np.uint8)
    # seed_value = cbimage[0, 0]
    for seed_x in range(h):
        for seed_y in range(w):
            seed_value = cbimage[seed_x, seed_y]
            if not visited[seed_x, seed_y] and cbimage[seed_x, seed_y] == seed_value:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                queue = [(seed_x, seed_y)]
                while queue:
                    x, y = queue.pop(0)
                    if visited[x, y]:
                        continue
                    segimage[x, y] = color
                    visited[x, y] = 1
                    for dx in range(-1, 2):
                      for dy in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < h and 0 <= ny < w and cbimage[nx, ny] == seed_value:
                                queue.append((nx, ny))
    return segimage




def histogram_segmentation1(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    histogram = np.histogram(gray_image, bins=256, range=(0, 256))[0]
    pad = 5
    # local_min = np.where((histogram[:-2] > histogram[1:-1]) & (histogram[1:-1] < histogram[2:]))[0] + 1
    local_min = []
    for i in range(pad, len(histogram) - pad, pad):
        if histogram[i] == min(histogram[i - pad: i + pad]):
            local_min.append(i)

    li = [(local_min[i] + local_min[i + 1]) // 2 for i in range(len(local_min) - 1)]
    li.insert(0, 0)
    li.append(255)

    binary_image = np.zeros_like(gray_image, dtype=np.uint8)

    for i in range(len(li) - 1):
        mask = (gray_image >= li[i]) & (gray_image < li[i + 1])
        binary_image[mask] = i * 255 / (len(li) - 1)

    return binary_image

def histogram_segmentation(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    histogram = np.histogram(gray_image, bins=256, range=(0, 256))[0]
    binary_image = np.zeros_like(gray_image, dtype=np.uint8)
    pad = 1
    local_min = []
    for i in range(1, len(histogram)):
        if histogram[i - 1] > histogram[i] and histogram[i] < histogram[i + 1]:
            local_min.append(i)

    li = []
    for i in range(len(local_min) - 1):
        li.append((local_min[i] + local_min[i + 1]) // 2)

        
    li.insert(0, 0)
    li.append(255)
    for i in range(len(li) - 1):
        cur = li[i]
        next = li[i + 1]
        mask = np.zeros_like(gray_image, dtype=np.uint8)
        for a in range(gray_image.shape[0]):
            for j in range(gray_image.shape[1]):
                if cur <= gray_image[a, j] <= next:
                    mask[a, j] = 255
        binary_image += mask * (i * 255 // (len(li) - 1))
    return binary_image


def image_work2(image):
    himage = histogram_segmentation1(image)
    segment = seg(himage)
    return himage, segment



if __name__ == "__main__":

    image1 = Image.open('image1.jpg')
    image2 = Image.open('image.png')

    img_mas1 = np.array(image1)
    img_mas2 = np.array(image2)

    himage1,segment1, = image_work2(img_mas1)
    himage2,segment2, = image_work2(img_mas2)


    plt.figure(figsize=(15, 12))

    plt.subplot(3, 2, 1)
    plt.imshow(img_mas1)
    plt.axis('off')
    plt.title('Ориг')

    plt.subplot(3, 2, 2)
    plt.imshow(img_mas2)
    plt.axis('off')
    plt.title('Ориг')

    plt.subplot(3, 2, 3)
    plt.imshow(himage1, cmap='gray')
    plt.axis('off')
    plt.title('Гистограммный метод для изображения 1')

    plt.subplot(3, 2, 4)
    plt.imshow(himage2, cmap='gray')
    plt.axis('off')
    plt.title('Гистограммный метод для изображения 2')

    plt.subplot(3, 2, 5)
    plt.imshow(segment1, cmap='gray')
    plt.axis('off')
    plt.title('Второй метод сегментации для изображения 1')

    plt.subplot(3, 2, 6)
    plt.imshow(segment2, cmap='gray')
    plt.axis('off')
    plt.title('Второй метод сегментации для изображения 2')

    # plt.subplot(4, 2, 5)
    # plt.imshow(himage12, cmap='gray')
    # plt.axis('off')
    # plt.title('Второй метод сегментации для изображения 2')

    # plt.subplot(4, 2, 6)
    # plt.imshow(himage22, cmap='gray')
    # plt.axis('off')
    # plt.title('Второй метод сегментации для изображения 2')

    # plt.subplot(4, 2, 7)
    # plt.imshow(segment12, cmap='gray')
    # plt.axis('off')
    # plt.title('Второй метод сегментации для изображения 2')

    # plt.subplot(4, 2, 8)
    # plt.imshow(segment22, cmap='gray')
    # plt.axis('off')
    # plt.title('Второй метод сегментации для изображения 2')

    plt.show()