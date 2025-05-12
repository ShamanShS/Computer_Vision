import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def blur(img, size, intensity):
    radius = size
    size = size * 2 + 1
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * intensity) ** 2) * np.exp(-((x * (1 - size)) ** 2 + (y * (1 - size)) ** 2) / (2 * intensity ** 2)),
        (size, size)
    )
    kernel = kernel / np.sum(kernel)
    height, width = img.shape
    res = np.copy(img)
    np.pad(res, radius, mode='constant', constant_values=0)
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            roi = img[i - radius:i + radius + 1, j - radius:j + radius + 1]
            res[i, j] = np.sum(roi * kernel)
    return res[radius:height - radius, radius: width - radius].astype('uint8')

def diagonal(x, y, image, threshold, offsets):
    if image[y + offsets[0][1], x + offsets[0][0]] + threshold < image[y, x] < image[y + offsets[8][1], x + offsets[8][0]] - threshold:
        return False
    if image[y + offsets[8][1], x + offsets[8][0]] + threshold < image[y, x] < image[y + offsets[0][1], x + offsets[0][0]] - threshold:
        return False
    if image[y + offsets[12][1], x + offsets[12][0]] + threshold < image[y, x] < image[y + offsets[4][1], x + offsets[4][0]] - threshold:
        return False
    if image[y + offsets[4][1], x + offsets[4][0]] + threshold < image[y, x] < image[y + offsets[12][1], x + offsets[12][0]] - threshold:
        return False
    return True

def fast(x, y, image, threshold, n):
    candidate = image[y, x]
    offsets = [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3)
    ]
    
    if not diagonal(x, y, image, threshold, offsets):
      return False
    for i in range(0, n-1):
        offsets.append(offsets[i])
    flag1 = False
    flag2 = False

    count_bright = 0
    count_dark = 0
    
    for i, j in offsets:
        p = image[y + j, x + i]
        if candidate + threshold <= p:
            count_bright += 1
            if count_bright == n:
                break
            if not flag1:
                flag1 = True
            if flag2:
                flag2 = False
                count_dark = 0
        elif candidate - threshold >= p:
            count_dark += 1
            if count_dark == n:
                break
            if not flag2:
                flag2 = True
            if flag1:
                flag1 = False
                count_bright = 0
    return (count_bright >= n) or (count_dark >= n)

def filter_corners(keypoints, image, k=0.04):
    h, w = image.shape
    res = []
    zero_x = []
    zero_y = []
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * 20 ** 2)) * np.exp(-((x - (5 - 1) / 2) ** 2 + (y - (5 - 1) / 2) ** 2) / (2 * 20 ** 2)),
        (5, 5)
    )
    for x, y in keypoints:
        if x >= 3 and x < w - 3 and y >= 3 and y < h - 3:
            M = np.zeros((2, 2))
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    weight = kernel[dx + 2, dy + 2]
                    Ix = (image[y + dy, x + dx + 1] - image[y + dy, x + dx - 1]) / 2
                    Iy = (image[y + dy + 1, x + dx] - image[y + dy - 1, x + dx]) / 2
                    M[0, 0] += weight * Ix * Ix
                    M[0, 1] += weight * Ix * Iy
                    M[1, 0] += weight * Ix * Iy
                    M[1, 1] += weight * Iy * Iy
            det_M = np.linalg.det(M)
            trace_M = np.trace(M)
            R = det_M - k * trace_M**2
            if R > 100:
                res.append((x, y))
            else:
                zero_x.append(x)
                zero_y.append(y)

    return res, zero_x, zero_y

def orientation(keypoints, image):
    orientations = []
    for x, y in keypoints:
        M = np.zeros((2, 2))
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                Ix = (image[y + dy, x + dx + 1] - image[y + dy, x + dx - 1]) / 2
                Iy = (image[y + dy + 1, x + dx] - image[y + dy - 1, x + dx]) / 2
                M[0, 0] += Ix * Ix
                M[0, 1] += Ix * Iy
                M[1, 0] += Ix * Iy
                M[1, 1] += Iy * Iy
        angle = np.arctan2(M[0, 1], M[1, 0])
        orientations.append(angle)
    return orientations

def generate_pairs(n, p, seed=0):
    np.random.seed(seed)
    return np.random.normal(0, (p**2)/25, (n, 2, 2))

def compute_orientation(orientation, generated):
    pairs = []
    for o in generated:

        rot_of1 = rotate(o[0], orientation)
        rot_of2 = rotate(o[1], orientation)

        point1 = (rot_of1[0], rot_of1[1])
        point2 = (rot_of2[0], rot_of2[1])

        pairs.append((point1, point2))
    return pairs

def rotate(vector, angle):
    x, y = vector
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)
    return (new_x, new_y)

def select_pairs(orientation, orientation_pairs):
    num = len(orientation_pairs)
    ind = int(orientation / (2 * np.pi) * num) % num
    res = orientation_pairs[ind]
    return res

def brief(image, key_point, pairs, key_point_orientations, filtered_corners):
    x, y = key_point

    descriptor = np.zeros(len(pairs[0]), dtype=np.uint8)

    indx = np.where((filtered_corners == key_point).all(axis=1))[0][0]
    orientation = key_point_orientations[indx]
    selected = select_pairs(orientation, orientation_pairs)

    for i, (p1, p2) in enumerate(selected):
        x1, y1 = x + int(p1[0]), y + int(p1[1])
        x2, y2 = x + int(p2[0]), y + int(p2[1])

        if 0 <= i < len(descriptor):
            if x1 >= 0 and x1 < image.shape[1] and y1 >= 0 and y1 < image.shape[0] and x2 >= 0 and x2 < image.shape[1] and y2 >= 0 and y2 < image.shape[0]:
                if image[y1, x1] < image[y2, x2]:
                    descriptor[i] = 1

    return descriptor

if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2)
    image = Image.open('image22.png')    
    image.load()
    image_data =  np.array(image)
    grayscale = np.mean(image_data, axis=2).astype(np.uint8)
    threshold = 40
    corners = []
    for y in range(3, grayscale.shape[0] - 3):
        for x in range(3, grayscale.shape[1] - 3):
            if fast(x, y, grayscale, threshold, 12):
                corners.append((x, y))
    c = np.array(corners)


    ax[0][0].imshow(grayscale, cmap='gray')
    ax[0][0].scatter(c[:, 0], c[:, 1], c='r', s=10)
    ax[0][0].axis('off')

    filtered_pnts, zero_x1, zero_y1 = filter_corners(corners, grayscale, k=0.04)
    filtered_corners = np.array(filtered_pnts)

    ax[0][1].imshow(grayscale, cmap='gray')
    ax[0][1].scatter(filtered_corners[:, 0], filtered_corners[:, 1], c='orange', s=10)
    ax[0][1].axis('off')

    ax[1][0].imshow(grayscale, cmap='gray')
    ax[1][0].scatter(zero_x1, zero_y1, c='yellow', s=10)
    ax[1][0].axis('off')

    orientations = orientation(filtered_corners, grayscale)
    blur_img = blur(grayscale, 2, 4.5)

    n = 256
    p = 31
    g = generate_pairs(n, p)
    orientation_pairs = []
    for i in range(30):
        angle = i * (2 * np.pi / 30)
        pairs = compute_orientation(angle, g)
        orientation_pairs.append(pairs)

    descriptors = []
    for pnt in filtered_corners:
        descriptor = brief(blur_img, pnt, orientation_pairs, orientations, filtered_corners)
        descriptors.append(descriptor)
    dsc = np.array(descriptors, dtype=np.uint8)

    result = np.repeat(blur_img[:, :, np.newaxis], 3, axis=2)
    for i in range(len(filtered_corners)):
        key = filtered_corners[i]
        descriptor = dsc[i]
        x, y = map(int, key)
        x, y = int(x), int(y)
        if any(descriptor):
            result[y-1:y+2, x-1:x+2] = [0, 255, 0]
    
    ax[1][1].imshow(result, cmap='gray')
    ax[1][1].axis('off')
    plt.show()
    np.savetxt('descriptorsM.txt', descriptors, fmt='%d', delimiter='')