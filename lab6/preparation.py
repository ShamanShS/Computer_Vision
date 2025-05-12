import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import lab52
import lab3




def descriptor(img):
    gray_image = img
    threshold = 40
    keypoints1 = []
    for y in range(3, gray_image.shape[0] - 3):
        for x in range(3, gray_image.shape[1] - 3):
            if lab52.fast(x, y, gray_image, threshold,12):
                keypoints1.append((x, y))
    key = np.array(keypoints1)

    filtered_key, zero_x, zero_y = lab52.filter_keypoints(keypoints1, gray_image, k=0.04)

    filtered_keypoints = np.array(filtered_key)

    keypoint_orientations = lab52.orientation(filtered_keypoints, gray_image)
    blur_img = lab3.gause(gray_image, 20)

    n = 256
    p = 31
    brief_pairs = lab52.generateb(n, p)

    orientation_pairs = []
    for i in range(30):
        orienang = i * (2 * np.pi / 30)
        pairs = lab52.genero(orienang, brief_pairs)
        orientation_pairs.append(pairs)
    

    descriptors = []
    for key in filtered_keypoints:
        descriptor = lab52.brief(blur_img, key, orientation_pairs, keypoint_orientations, filtered_keypoints, orientation_pairs)
        descriptors.append(descriptor)
    descript = np.array(descriptors)

    for i in range(len(filtered_keypoints)):
        key = filtered_keypoints[i]
        descriptor = descript[i]


    return filtered_keypoints, descriptors


    


if __name__ == "__main__":
    image = Image.open('box.png')
    imageBig = Image.open('box_in_scene.png')
    img1 = np.array(image)
    img2 = np.array(imageBig)

    width, height = image.width, image.height
    width //= 2
    height //= 2
    img1min = image.resize((width, height))

    width2, height2 = imageBig.width, imageBig.height
    width2 //= 2
    height2 //= 2
    img2min = imageBig.resize((width2, height2))

    harris1, descr1 = descriptor(img1)
    # print(descr1)
    harris2, descr2 = descriptor(img2)
    harris1min, descr1min = descriptor(np.array(img1min))
    harris2min, descr2min = descriptor(np.array(img2min))
    
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.scatter(harris1[:, 0], harris1[:, 1], c='r', s=1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.scatter(harris2[:, 0], harris2[:, 1], c='r', s=1)
    plt.axis('off')
    plt.show()

    # np.save('harris1.npy', harris1)
    # np.save('descriptors1.npy', descr1)

    # np.save('harris2.npy', harris2)
    # np.save('descriptors2.npy', descr2)

    # np.save('harris1min.npy', harris1min)
    # np.save('descriptors1min.npy', descr1min)
    
    # np.save('harris2min.npy', harris2min)
    # np.save('descriptors2min.npy', descr2min)





