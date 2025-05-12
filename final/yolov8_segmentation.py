import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


model = YOLO("yolov8s-seg.pt")  


image_path = "image/img.jpg"  
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


results = model(img_rgb)


masks = results[0].masks.cpu().numpy()


segmentation_image = img_rgb.copy()
for mask in masks:
    segmentation_image[mask == 1] = [0, 255, 0] 


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmentation_image)
plt.title("Segmentation")
plt.axis("off")

plt.show()
