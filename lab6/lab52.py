import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import lab3


def diagonal(x, y, image, threshold, offsets):
    if image[y + offsets[0][1], x + offsets[0][0]] + threshold < image[y, x] < image[y + offsets[8][1], x + offsets[8][0]] - threshold:
        if image[y + offsets[4][1], x + offsets[4][0]] + threshold < image[y, x] < image[y + offsets[12][1], x + offsets[12][0]] - threshold:
            return False
    if image[y + offsets[8][1], x + offsets[8][0]] + threshold < image[y, x] < image[y + offsets[0][1], x + offsets[0][0]] - threshold:
        if image[y + offsets[12][1], x + offsets[12][0]] + threshold < image[y, x] < image[y + offsets[4][1], x + offsets[4][0]] - threshold:
            return False
    if image[y + offsets[12][1], x + offsets[12][0]] + threshold < image[y, x] < image[y + offsets[4][1], x + offsets[4][0]] - threshold:
        if image[y + offsets[8][1], x + offsets[8][0]] + threshold < image[y, x] < image[y + offsets[0][1], x + offsets[0][0]] - threshold:
            return False
    if image[y + offsets[4][1], x + offsets[4][0]] + threshold < image[y, x] < image[y + offsets[12][1], x + offsets[12][0]] - threshold:
        if image[y + offsets[0][1], x + offsets[0][0]] + threshold < image[y, x] < image[y + offsets[8][1], x + offsets[8][0]] - threshold:
            return False
    return True


def fast(x, y, image, threshold, n):
    if (x == 44 and y == 11):
        print(1)
    candidate = image[y, x]
    offsets = [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3)
    ]
    # offsets = [
    #     (0, 3), (-1, 3), (-2, 2)
    # ]
    
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
        else:
            flag1 = False
            flag2 = False
            count_bright = 0
            count_dark = 0
    return (count_bright >= n) or (count_dark >= n)

def filter_keypoints(keypoints, image, k=0.04):
    h, w = image.shape
    filtered_keypoints = []
    zero_x = []
    zero_y = []
    kernel = lab3.gauss_kernel(5, 3)
    all_point = []
    N = 200
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
            # all_point.append([R, (x, y)])
            if R > 1e-5:
                filtered_keypoints.append((x, y))
            else:
                zero_x.append(x)
                zero_y.append(y)

    # all_point.sort(key=lambda point: point[0], reverse=False)
    # for i in range(N):
    #     if i < len(all_point):
    #         filtered_keypoints.append(all_point[i][1])

    return filtered_keypoints,zero_x,zero_y

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

def generateb(n, p, seed=0):
    np.random.seed(seed)  # Устанавливаем зерно для генератора случайных чисел
    pairs = np.random.normal(0, (p**2)/25, (n, 2, 2))
    return pairs

def genero(orienang, n_pairs):
    pairs = []
    for o in n_pairs:
        rot_of1 = rotate(o[0], orienang)
        rot_of2 = rotate(o[1], orienang)

        point1 = (rot_of1[0], rot_of1[1])
        point2 = (rot_of2[0], rot_of2[1])

        pairs.append((point1, point2))
    return pairs

def rotate(vector, angle):
    x, y = vector
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)
    return (new_x, new_y)

def selectpa(orientation, orientation_pairs):
    num = len(orientation_pairs)
    ind = int(orientation / (2 * np.pi) * num) % num
    sp = orientation_pairs[ind]
    return sp

def brief(image, keypoint, pairs, keypoint_orientations, filtered_keypoints, orientation_pairs):
    x, y = keypoint
    descriptor = np.zeros(len(pairs[0]), dtype=np.uint8)
    ind = np.where((filtered_keypoints == keypoint).all(axis=1))[0][0]
    orientation = keypoint_orientations[ind]
    sp = selectpa(orientation, orientation_pairs)
    for i, (p1, p2) in enumerate(sp):
        x1, y1 = x + int(p1[0]), y + int(p1[1])
        x2, y2 = x + int(p2[0]), y + int(p2[1])
        if 0 <= i < len(descriptor):
            if x1 >= 0 and x1 < image.shape[1] and y1 >= 0 and y1 < image.shape[0] and x2 >= 0 and x2 < image.shape[1] and y2 >= 0 and y2 < image.shape[0]:
                if image[y1, x1] < image[y2, x2]:
                    descriptor[i] = 1
    return descriptor

if __name__ == "__main__":

    image = Image.open('box.png')    
    # image = Image.open('box.png')  
    image.load()
    image_mas =  np.array(image)
    gray_image = np.array(image)
    threshold = 30
    keypoints1 = []
    for y in range(3, gray_image.shape[0] - 3):
        for x in range(3, gray_image.shape[1] - 3):
            if fast(x, y, gray_image, threshold,12):
                keypoints1.append((x, y))
    key4 = np.array(keypoints1)


    plt.figure(figsize=(16, 8))
    plt.imshow(gray_image, cmap='gray')
    plt.scatter(key4[:, 0], key4[:, 1], c='r', s=1)
    plt.axis('off')
    plt.title(f'Порог {threshold}')

    filtered_key1, zero_x1, zero_y1 = filter_keypoints(keypoints1, gray_image, k=0.04)

    filtered_keypoints1 = np.array(filtered_key1)

    
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.scatter(filtered_keypoints1[:, 0], filtered_keypoints1[:, 1], c='r', s=1)
    plt.axis('off')
    plt.title(f'Отфильтрованные ключевые точки порог {threshold}')

    plt.subplot(1, 2, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.scatter(zero_x1, zero_y1, c='r', s=1)
    plt.axis('off')
    plt.title('Точки которые не прошли отбор')



    keypoint_orientations = orientation(filtered_keypoints1, gray_image)
    blur_img = lab3.gause(gray_image, 20)

    n = 256
    p = 31
    brief_pairs = generateb(n, p)

    orientation_pairs = []
    for i in range(30):
        orienang = i * (2 * np.pi / 30)
        pairs = genero(orienang, brief_pairs)
        orientation_pairs.append(pairs)
    

    descriptors = []
    for key in filtered_keypoints1:
        descriptor = brief(blur_img, key, orientation_pairs, keypoint_orientations, filtered_keypoints1, orientation_pairs)
        descriptors.append(descriptor)
    descript = np.array(descriptors, dtype=np.uint8)

    result = np.repeat(blur_img[:, :, np.newaxis], 3, axis=2)
    for i in range(len(filtered_keypoints1)):
        key = filtered_keypoints1[i]
        descriptor = descript[i]
        x, y = map(int, key)
        x, y = int(x), int(y)
        if any(descriptor):
            result[y-1:y+2, x-1:x+2] = [0, 255, 0]
    
    plt.figure(figsize=(16, 8))
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.show()
    # np.save('harris2min.npy', filtered_keypoints1)
    # np.save('descriptors2min.npy', descriptors)
    