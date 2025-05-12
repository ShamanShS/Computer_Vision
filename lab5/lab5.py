import numpy as np
from scipy.ndimage import convolve
import lab3
from PIL import Image





def FAST(image,r, t, n):
    rows, cols = image.shape
    
    special_points, fast_img = [], np.zeros(image.shape)
    
    # проход по всем пикселям
    for i in range(r, rows - r):
        for j in range(r, cols - r):    
            
            # выбор точек круга брезенхема
            p1 = image[i, j + 3 ]
            p2 = image[i+1, j + 3 ]
            p3 = image[i+2, j + 2 ]
            p4 = image[i+3, j + 1 ]
            p5 = image[i+3, j ]
            p6 = image[i+3, j - 1 ]
            p7 = image[i+2, j - 2 ]
            p8 = image[i+1, j - 3 ]
            p9 = image[i, j - 3 ]
            p10 = image[i-1, j - 3 ]
            p11 = image[i-2, j - 2 ]
            p12 = image[i-3, j - 1 ]
            p13 = image[i-3, j ]
            p14 = image[i-3, j + 1 ]
            p15 = image[i-2, j + 2 ]
            p16 = image[i-1, j + 3 ]
            
            
            cur_I = image[i, j]
            
            # проверка диаметров
            
            if (p1 + t < cur_I < p9 - t) or (p9 +t < cur_I < p1 - t):
                continue 
            if (p13 + t < cur_I < p5 - t) or (p5 +t < cur_I < p13 - t):
                continue
                
            brez_circle = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16]
            
            size = len(brez_circle)
            
            # поиск n последовательности точек
            for u in range(size):
                
                isMore, count = False, 0
                
                if brez_circle[u] > cur_I + t:
                    isMore, count = True, 1
                if brez_circle[u] < cur_I - t and isMore is True:
                    isMore, count = False, 1
                   
                for v in range(1, n):
                    if brez_circle[(u + v) % size] > cur_I + t and isMore is False:
                        break
                    if brez_circle[(u + v) % size] > cur_I + t and isMore is True:
                        count += 1
                    if brez_circle[(u + v) % size] < cur_I - t and isMore is True:
                        break
                    if brez_circle[(u + v) % size] < cur_I - t and isMore is False:
                        count += 1
                if count == n:
                    special_points.append([i, j])
                    fast_img[i, j] = 255
                    break
                    
    return special_points, fast_img

def Harris(image, special_points,k, N):
    R_arr = {}
    
    sobel_gy = np.array((
        (-1, 0, 1), 
        (-2, 0, 2), 
        (-1, 0, 1)
    ))
    sobel_gx = np.array((
        (-1, -2, -1), 
        (0, 0 ,0), 
        (1, 2, 1)
    ))
    
    # проход по всем точкам из FAST
    for p in special_points:
        M = np.zeros([2, 2])
        
        # проход по точкам из окна 5х5
        w = lab3.gauss_kernel(1, 5)
        for i in range(-2, 3):
            for j in range(-2, 3):
                _x, _y = p[0] + i, p[1] + j
                
                # подсчет частных производных в точке
                Ix, Iy = 0, 0
                
                Ix = np.sum(image[_x - 1:_x + 2, _y - 1: _y + 2] * sobel_gx)
                Iy = np.sum(image[_x - 1:_x + 2, _y - 1: _y + 2] * sobel_gy)

                # вычисление M
                I = np.array([[Ix ** 2, Ix * Iy], [Ix * Iy, Iy ** 2]])
                M = M + w[i + 2, j + 2] * I
                
        # подсчет R
        l1, l2 = np.linalg.eigvals(M) #[0], np.linalg.eigvals(M)[1]
        det_M, trace_M = l1 * l2, l1 + l2
        R = det_M - k * (trace_M ** 2)
        if R > 0:
            R_arr[R] = [p[0], p[1]]
            
    new = dict(sorted(R_arr.items(), reverse=True, key=lambda x: x[0]))
    
    return list(new.values())[:N]


def FindOrienation(image, X, Y, R = 31):
    area, teta = [[0, 0]], 0
    x_c = X + R
    y_c = Y + R
    
    w_borders = np.zeros(shape=(image.shape[0] + 2 * R, image.shape[1] + 2 * R))
    w_borders[R:-R, R:-R] = image
    
    # заполняем область вокруг особой точки
    # рассматриваем каждый радиус r до R
    for r in range(1, R + 1):
        for _x in range(-r, r + 1):
            if r ** 2 - _x ** 2 < 0:
                continue
            _y1 = int(np.round(np.sqrt(r ** 2 - _x ** 2)))
            _y2 = int(np.round(-np.sqrt(r ** 2 - _x ** 2)))
            if -r <= _y1 <= r:
                area.append([_x, _y1])
            if -r <= _y2 <= r:
                area.append([_x, _y2])
        for _y in range(-r, r + 1):
            if r ** 2 - _y ** 2 < 0:
                continue
            _x1 = int(np.round(np.sqrt(r ** 2 - _y ** 2)))
            _x2 = int(np.round(-np.sqrt(r ** 2 - _y ** 2)))
            if -r <= _x1 <= r:
                area.append([_x1, _y])
            if -r <= _x2 <= r:
                area.append([_x2, _y])
    area = np.unique(area, axis=0)
    
    # вычисляем моменты
    m00, m01, m10 = 0, 0, 0
    for p in area:
        I = w_borders[p[0] + x_c, p[1] + y_c]
        m00 += (p[0] ** 0) * (p[1] ** 0) * I
        m01 += (p[0] ** 0) * (p[1] ** 1) * I
        m10 += (p[0] ** 1) * (p[1] ** 0) * I
    teta_return = np.arctan2(m01, m10)
    if teta_return < 0:
        return np.arctan2(m01, m10) + np.pi * 2
    return np.arctan2(m01, m10)


def RotationMatrix(teta):
    return np.array([
        [ np.cos(teta), np.sin(teta)], 
        [-np.sin(teta), np.cos(teta)]
        ])


def BRIEF(image, special_points, orientations, p, n):
    
    descriptors, all_descriptors = [], np.zeros((len(special_points), 30, n))
    

    shift = round(p / 2)
    
    frame = np.zeros(shape=(image.shape[0] + p - 1, image.shape[1] + p - 1))
    # print(frame.shape)
    frame[shift-1:-shift+1, shift-1:-shift+1] = image
    
    # набор углов с шагом в 15 градусов
    angles = np.zeros(30)
    angles[0] = 0
    for i in range(1, len(angles)):
            angles[i] = angles[i - 1] + 2 * np.pi / 30
    
    
    # проходимся по всем особым точкам
    pattern_points = []
    for k in range(len(special_points)):
        x_c, y_c = special_points[k][0] + shift, special_points[k][1] + shift
        teta_c = orientations[k]
        area = [[0, 0]]

        # рассматриваем область радиуса shift
        for r in range(1, shift + 1):
            for _x in range(x_c - r, x_c + r + 1):
                if r ** 2 - (_x - x_c) ** 2 < 0:
                    continue
                _y1 = int(np.round(np.sqrt(r ** 2 - (_x - x_c) ** 2) + y_c))
                _y2 = int(np.round(-np.sqrt(r ** 2 - (_x - x_c) ** 2) + y_c))
                if 0 <= _y1 < frame.shape[1]:
                    area.append([_x - x_c, _y1 - y_c])
                if 0 <= _y2 < frame.shape[1]:
                    area.append([_x - x_c, _y2 - y_c])
            for _y in range(y_c - r, y_c + r + 1):
                if r ** 2 - (_y - y_c) ** 2 < 0:
                    continue
                _x1 = int(np.round(np.sqrt(r ** 2 - (_y - y_c) ** 2) + x_c))
                _x2 = int(np.round(-np.sqrt(r ** 2 - (_y - y_c) ** 2) + x_c))
                if 0 <= _x1 < frame.shape[0]:
                    area.append([_x1 - x_c, _y - y_c])
                if 0 <= _x2 < frame.shape[0]:
                    area.append([_x2 - x_c, _y - y_c])
        area = np.unique(area, axis=0)
        
        # пары точек рандомные
        S = np.zeros((2, n, 2))
        for i in range(n):
            if k == 0:
                while True:
                    rand = np.random.normal(0, p ** 2 / 25, 4).astype(int)
                    
                    if [rand[0], rand[1]] in area.tolist() and [rand[2], rand[3]] in area.tolist():
                        if not [rand[0], rand[1]] in pattern_points and not [rand[2], rand[3]] in pattern_points:
                            pattern_points.append([rand[0], rand[1]])
                            pattern_points.append([rand[2], rand[3]])
                            S[0, i] = [rand[0], rand[1]]
                            S[1, i] = [rand[2], rand[3]]
                            break
            else:
                u1, u2 = pattern_points[2 * i], pattern_points[2 * i + 1]
                S[0, i] = [u1[0], u1[1]]
                S[1, i] = [u2[0], u2[1]]
                
        # округляем угол
        min_dif, round_teta_c = 100000000, 0
        for a in angles:
            dif = np.abs(a - teta_c)
            if dif < min_dif:
                min_dif = dif
                round_teta_c = a
        teta_c = round_teta_c
        
        for a in range(len(angles)):
            _S1, _S2 = S[0].T, S[1].T
            S1 = (RotationMatrix(angles[a]) @ _S1).astype(int)
            S2 = (RotationMatrix(angles[a]) @ _S2).astype(int)
            bin_row = np.zeros(shape=n)
            for i in range(n):
                p1, p2 = S1.T[i], S2.T[i]
                if frame[x_c + p1[0], y_c + p1[1]] < frame[x_c + p2[0], y_c + p2[1]]:
                    bin_row[i] = 1
            if teta_c == angles[a]:
                descriptors.append(list(bin_row))
                all_descriptors[k, a] = bin_row
            else:
                all_descriptors[k, a] = bin_row
    return descriptors, all_descriptors


def ShowPoint(image,semitone, points):
    if len(image.shape) == 3:
        rows, cols, depth = image.shape
    else:
        rows, cols = image.shape
    new = np.ones((rows, cols, 3)).astype(int)
    color = 144, 24, 219
    for i in range(rows):
        for j in range(cols):
            new[i, j, 0] = semitone[i, j]
            new[i, j, 1] = semitone[i, j]
            new[i, j, 2] = semitone[i, j]
            
    for i in range(len(points)):
        new[points[i][0], points[i][1]] = color
    return new





if __name__ == "__main__":

    image1 = Image.open('image1.jpg')    
    image1.load()
    image1_mas =  np.array(image1)
    gray_image1 = np.mean(image1_mas, axis=2).astype(np.uint8)

    points, fast = FAST(gray_image1, 3,  20, 12)
    print(len(points))