import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import lab3


def Hamming(disk1, disk2):
    d = 0
    for i in range(len(disk1)):
        if disk1[i] != disk2[i]:
            d += 1
    return d

def LoveTest(key, disk1, disk2):
    ind_min1, ind_min2 = -1, -1
    min1, min2 = np.inf, np.inf
    for i, vectors in enumerate([disk1, disk2]):
        for j, vec in enumerate(vectors):
            d = Hamming(key, vec)
            ind = j + (0 if i == 0 else len(disk1))
            if min1 == np.inf:
                min1 = d
                ind_min1 = ind
            elif d < min1:
                min2 = min1
                min1 = d
                ind_min2 = ind_min1
                ind_min1 = ind
            elif min1 <= d < min2:
                min2 = d
                ind_min2 = ind
    if min2 != 0:
        result = [ind_min1, ind_min2, min1/min2]
    else:
        result = [ind_min1, ind_min2, 0]

    return result

def matche(desk1, desk1m, desk2, desk2m, key, key2, scale=2):
    matchpcop = []
    match_point = []
    count = 0
    for i, j in zip([desk1, desk1m], [key, key2]):
        for a, keyy in zip(i, j):
            R = LoveTest(a, desk2, desk2m)
            if R[2] < 0.7:
                matchpcop.append(R)
                if count < len(desk1):
                    match_point.append([keyy[0], keyy[1]])
                else:
                    match_point.append([keyy[0] * scale, keyy[1] * scale])
            count += 1
    return matchpcop, match_point

def poimg(img, match_point, color=(255, 0, 0), r=2):
    img_copy = np.array(img)
    for i in match_point:
        img_copy[i[0]-r:i[0]+r+1, i[1]-r:i[1]+r+1] = color
    return img_copy

def matchp(img_scene, matchpcop, key1, key1m, color=(255, 0, 0), r=1):
    img = np.array(img_scene)
    matchp = []
    for i in matchpcop:
        if i[0] < len(key1):
            matchp.append(key1[i[0]])
        else:
            matchp.append([key1m[i[0] - len(key1)][0] * 2, key1m[i[1] - len(key1)][1] * 2])
    for i in matchp:
        img[i[0]-r:i[0]+r+1, i[1]-r:i[1]+r+1] = color
    return img, matchp

def concat(img, imgc):
    concat_img = np.zeros((max(img.shape[0], imgc.shape[0]), img.shape[1]+imgc.shape[1], 3), dtype=np.uint8)
    concat_img[:img.shape[0], :img.shape[1]] = img
    concat_img[:imgc.shape[0], img.shape[1]:] = imgc
    return concat_img

def transpose_matrix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    t = [[matrix[j][i] for j in range(rows)] for i in range(cols)]
    return t

def create_matrix_A(fl):
    A = np.zeros((9, 9))
    for i in range(3):
        offset = 0
        for k in range(3):
            for j in range(3):
                A[i * 3 + k, j + offset] = fl[i, j]
            offset += 3
    return A

def RANSAC(match, match_scene, num_iterations=1000):
    maxi = 0
    match2 = np.column_stack((match, np.ones(len(match))))
    matchc2 = np.column_stack((match_scene, np.ones(len(match_scene))))
    for i in range(num_iterations):
        ind = np.random.randint(0, len(match), 3)
        before = np.column_stack((match[ind], np.ones(3)))
        after = np.column_stack((match_scene[ind], np.ones(3)))
        A = create_matrix_A(before)
        if np.isclose(np.linalg.det(A), 0):
            continue
        x = np.linalg.solve(A, after.flatten())
        H = x.reshape((3, 3))
        afterr = np.matmul(H, match2.T).T
        dist = np.sqrt(np.sum((afterr - matchc2) ** 2, axis=1))
        inliers_count = np.count_nonzero(dist < 3)
        if inliers_count > maxi:
            line = np.argwhere(dist < 3)
            maxi = inliers_count
            inl = match2[dist < 3], matchc2[dist < 3]
    return inl, line

def proekt(inl):
    A = np.zeros((inl[1].shape[0]*3, 9))
    for i in range(inl[1].shape[0]):
        offset = 0
        for k in range(3):
            for j in range(3):
                A[i*3 + k, j+offset] = inl[0][i][j]
            offset += 3
    b = inl[1].flatten()
    xX = np.dot(np.linalg.pinv(A), b)
    H = xX.reshape((3, 3))
    return H

def tp(H, inlm):
    tps = np.matmul(H, transpose_matrix(inlm))
    tps = transpose_matrix(tps)
    tps = np.round(tps).astype(np.int64)
    return tps

def pi(img):
    polygon = np.array([[0, 0, 1], [0, img.shape[1], 1], [img.shape[0], img.shape[1], 1], [img.shape[0], 0, 1]])
    new_polygon = tp(H, polygon)
    return polygon, new_polygon



if __name__ == "__main__":
    image1 = Image.open('imgmin.png').convert("RGB")    
    image1.load()
    image_mas1 =  np.array(image1)

    image2 = Image.open('img.png').convert("RGB")    
    image2.load()
    image_mas2 =  np.array(image2)

    harris_point1 = np.load('harris1.npy')
    harris_point2 = np.load('harris2.npy')
    harris_point1m = np.load('harris1min.npy')
    harris_point2m = np.load('harris2min.npy')
    descriptors1 = np.load('descriptors1.npy')
    descriptors2 = np.load('descriptors2.npy')
    descriptors1m = np.load('descriptors1min.npy')
    descriptors2m = np.load('descriptors2min.npy')
    

    matchpcop, match_point = matche(descriptors1, descriptors1m,descriptors2 , descriptors2m, harris_point1, harris_point1m)
    print(matchpcop)
    print(match_point)
    img_copy = poimg(image1, match_point)

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(img_copy)
    plt.axis('off')
    plt.title('Изображение 1 с общими точками')

    img_scene_copy, match_scene_point = matchp(image2, matchpcop, harris_point2, harris_point2m)

    plt.subplot(1, 2, 2)
    plt.imshow(img_scene_copy)
    plt.axis('off')
    plt.title('Изображение 2 с общими точками')
    plt.show()


    concat_img = concat(img_copy, img_scene_copy)

    inl, line = RANSAC(np.array(match_point), np.array(match_scene_point))
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(concat_img)
    for i in line:
        p1, p2 = match_point[i[0]], match_scene_point[i[0]]
        plt.plot([p1[1], p2[1]+image1.width], [p1[0], p2[0]], color='green')
    plt.axis('off')
    plt.title('Объединенное изображение с линиями')

    H = proekt(inl)
    print(inl)
    polygon,new_polygon=pi(img_copy)

    plt.subplot(1, 2, 2)
    plt.imshow(concat_img)
    for i in range(new_polygon.shape[0]):
        plt.plot([polygon[i-1, 1], polygon[i, 1]], [polygon[i-1, 0], polygon[i, 0]], c='red')
        plt.plot([new_polygon[i-1, 1]+image1.width, new_polygon[i, 1]+image1.width], [new_polygon[i-1, 0], new_polygon[i, 0]], c='blue')
    plt.axis('off')
    plt.title('Объединенное изображение с полигонами')
    plt.show()