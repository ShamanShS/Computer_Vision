import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import lab3


def HammindDistance(first, second):
    return np.sum(np.absolute(first - second))

def FindSimilarPoint(special_points1, special_points2, all_desc1, desc2, border):

    similar_points = []
    
    # Для каждой специальной точки первого изображения 
    for i in range(len(all_desc1)):
        possible_points = {}  
        print(len(all_desc1))
        
        for j in range(len(all_desc1[i])):
            
            # проверяем точки второго изображения
            for k in range(len(desc2)):
                d1, d2 = all_desc1[i, j], desc2[k]
                
                difference = HammindDistance(d1, d2)
                if difference <= border:
                    possible_points[difference] = [special_points1[i], special_points2[k]]
                    
        LoweTest(possible_points, similar_points)
        print(i)
    return similar_points

def LoweTest(points, array):
    if len(points) == 0:
        return
    
    points = sorted(points.items(), key=lambda x: x[0])[0:2]
    
    if len(points) == 1:
        p1, p2 = points[0][1][0], points[0][1][1]
        array.append([p1, p2])
        # print(f'{p1}, {p2}, r = {points[0][0]}')
    else:
        r1, r2 = points[0][0], points[1][0]
        R = r1 / r2
        
        p1, p2 = points[0][1][0], points[0][1][1]
        if R < 0.8:
            array.append([p1, p2])
            # print(f'{p1}, {p2}, r1 / r2 = {R}')
            
            
def RANSAC(N, similar_points):
    inlier = 0
    M_best, T_best = 0, 0
    
    for i in range(N):
        rand_indexes = np.random.randint(0, len(similar_points), 3)
        pair1, pair2, pair3 = similar_points[rand_indexes[0]], similar_points[rand_indexes[1]], similar_points[rand_indexes[2]]
        
        A = np.array([
            [pair1[0, 0], pair1[0, 1], 0, 0, 1, 0],
            [0, 0, pair1[0, 0], pair1[0, 1], 0, 1],
            [pair2[0, 0], pair2[0, 1], 0, 0, 1, 0],
            [0, 0, pair2[0, 0], pair2[0, 1], 0, 1],
            [pair3[0, 0], pair3[0, 1], 0, 0, 1, 0],
            [0, 0, pair3[0, 0], pair3[0, 1], 0, 1]
        ])
        b = np.array([
            pair1[1, 0],
            pair1[1, 1],
            pair2[1, 0],
            pair2[1, 1],
            pair3[1, 0],
            pair3[1, 1]
        ])
        
        if np.linalg.det(A.T @ A) == 0:
            continue
        x = np.linalg.inv(A.T @ A) @ A.T @ b
        inlier_i = 0
        for pair in similar_points:
            p1, p2 = pair[0], pair[1]
            
            M, T = np.array([[x[0], x[1]], [x[2], x[3]]]), np.array([x[4], x[5]])
            p1 = (M @ p1 + T).astype(int)
            if np.array_equiv(p1, p2):
                inlier_i += 1
        if inlier_i > inlier:
            inlier = inlier_i
            M_best, T_best  = M, T

    return M_best, T_best
    
def ShowPairsPoint(semitone, points_pair, w2):
    if len(semitone.shape) == 3:
        rows, cols, depth = semitone.shape
    else:
        rows, cols = semitone.shape
    
    # перевод картинки в трёхмерное состояние
    new = np.ones((rows, cols, 3)).astype(int)
    color = 144, 24, 219
    for i in range(rows):
        for j in range(cols):
            new[i, j, 0] = semitone[i, j]
            new[i, j, 1] = semitone[i, j]
            new[i, j, 2] = semitone[i, j]
    
    # отрисовка соединённых пар точек    
    for pair in points_pair:
        x1 = pair[0][0]
        y1 = pair[0][1]
        x2 = pair[1][0]
        y2 = pair[1][1] + w2
        
        new[x1, y1] = color
        new[x2, y2] = color
        
        if x1 < x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        
        for x in range(x2, x1 + 1):
            y = int(np.round(((x - x1) / (x2 - x1)) * (y2 - y1) + y1))
            new[x, y] = color
            
    return new

def ConcatenateImage(image1, image2):
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    
    if h1 > h2:
        new = np.zeros([h1, w1+w2])
    else:
        new = np.zeros([h2, w1+w2])
    new[0:h1, 0:w1] = image1
    new[0:h2, w1:w2+w1] = image2
    
    return new, w1

def ShowImage(*images):
    numOfImages = len(images)
    fig, axes = plt.subplots(1, numOfImages)
    if numOfImages == 1:
        axes = [axes]
    for i in range(numOfImages):
            axes[i].imshow(images[i], cmap='gray')
    fig.set_figwidth(10*numOfImages)    
    fig.set_figheight(10) 

    plt.show()



if __name__ == "__main__":
    image = Image.open('box.png')
    imageBig = Image.open('box_in_scene.png')
    img1 = np.array(image)
    img2 = np.array(imageBig)

    harris_point2 = np.load('harris2.npy')
    harris_point1 = np.load('harris1.npy')
    descriptors2 = np.load('descriptors2.npy')
    descriptors1 = np.load('descriptors1.npy')

    similar_points = FindSimilarPoint(harris_point2, harris_point1, descriptors2, descriptors1, 35)
    print(similar_points)
    concatenate, w1 = ConcatenateImage(img1, img2)

    show_pairs_points = ShowPairsPoint(concatenate,similar_points, w1 )
    ShowImage(show_pairs_points)