from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random


def otsu(image):
    gimage = np.mean(image, axis=2).astype(np.uint8)
    histogram = np.histogram(gimage, bins=256, range=(0, 256))[0]
    cumulhist = np.cumsum(histogram)
    tp = gimage.size
    max_variance = 0
    threshold = 0

    for t in range(256):
        w1 = cumulhist[t] / tp
        w2 = 1 - w1
        if cumulhist[t] == 0 or cumulhist[t] == tp:
            continue
        mean1 = np.sum(np.arange(0, t + 1) * histogram[:t + 1]) / cumulhist[t]
        mean2 = np.sum(np.arange(t + 1, 256) * histogram[t + 1:]) / (tp - cumulhist[t])
        variance = w1 * w2 * (mean1 - mean2) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = t

    bimage = (gimage > threshold).astype(np.uint8)
    return bimage

def count_segments(binary_image):
    height, width = binary_image.shape
    visited = np.zeros_like(binary_image, dtype=bool)
    num_segments = 0  

    for i in range(height):
        for j in range(width):
            if visited[i, j] or binary_image[i, j] == 0:
                continue
            num_segments += 1
            queue = [(i, j)] 

            while queue:
                x, y = queue.pop() 
                if visited[x, y]:
                    continue
                visited[x, y] = True  
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx == 0 and dy == 0: 
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny] and binary_image[nx, ny] == 1:
                            queue.append((nx, ny))

    return num_segments


def salt_pepper(image, ks=3):
    h, w = image.shape  
    fimage = np.copy(image)
    maskS = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]], dtype=np.uint8) 
    maskP = np.array([[255, 255, 255], [255, 0, 255], [255, 255, 255]], dtype=np.uint8) 

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            patch = image[i - 1: i + 2, j - 1:j + 2]
            if np.array_equal(maskS, patch):
                fimage[i, j] = 0
            if np.array_equal(maskP, patch):
                fimage[i, j] = 255


    return fimage


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







def image_work1(image):

    bimage = otsu(image)
    seg_count = count_segments(bimage)
    cbimage = salt_pepper(bimage)
    # segimage, seg_count = seg_and_count(cbimage)
    segimage = seg(cbimage)

    return bimage, seg_count, cbimage, segimage



if __name__ == "__main__":

    image1 = Image.open('map_56.png')
    image2 = Image.open('image2.jpg')

    image1_mas =  np.array(image1)
    image2_mas =  np.array(image2)

    bimage1, seg_count1, cbimage1, segimage1 = image_work1(np.copy(image1_mas))
    bimage2, seg_count2, cbimage2, segimage2 = image_work1(np.copy(image2_mas))



    plt.figure(figsize=(15, 12))


    plt.subplot(4, 2, 1)
    plt.imshow(image1_mas)
    plt.axis('off')
    plt.title('Изображение 1')

    plt.subplot(4, 2, 2)
    plt.imshow(image2_mas)
    plt.axis('off')
    plt.title('Изображение 2')

    plt.subplot(4, 2, 3)
    plt.imshow(bimage1, cmap='gray')
    plt.axis('off')
    plt.title(f'Бинарное изображение 1 \n (Сегментов: {seg_count1})')

    plt.subplot(4, 2, 4)
    plt.imshow(bimage2, cmap='gray')
    plt.axis('off')
    plt.title(f'Бинарное изображение 1 \n (Сегментов: {seg_count2})')

    plt.subplot(4, 2, 5)
    plt.imshow(cbimage1, cmap='gray')
    plt.axis('off')
    plt.title('Изображение без шума 1')

    plt.subplot(4, 2, 6)
    plt.imshow(cbimage2, cmap='gray')
    plt.axis('off')
    plt.title('Изображение без шума 2')

    plt.subplot(4, 2, 7)
    plt.imshow(segimage1)
    plt.axis('off')
    plt.title('Сегментированное изображение 1')

    plt.subplot(4, 2, 8)
    plt.imshow(segimage2)
    plt.axis('off')
    plt.title('Сегментированное изображение 2')



    plt.show()
