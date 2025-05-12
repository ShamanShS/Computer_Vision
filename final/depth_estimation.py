import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


model_type = "DPT_Large" 
midas = torch.hub.load("intel-isl/MiDaS", model_type)


midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


image_path = "image/C.jpg"  
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)




input_batch = transform(img).squeeze(-1)

print(input_batch.shape)

# Оценка глубины
with torch.no_grad():
    prediction = midas(input_batch)

# Преобразование глубины в изображение
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze()

output = prediction.cpu().numpy()

# Визуализация глубины


# plt.figure(figsize=(100, 100))
# plt.subplot(1, 2, 1)
# plt.imshow(normalized_img)
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(1, 2, 2)

# plt.imshow(output, cmap="inferno")
# plt.axis("off")
# # # # plt.savefig('Depth.jpg')
# # # # plt.savefig('Depth.jpg',dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()
# cmap = plt.get_cmap('inferno')
# if output.dtype != np.uint8:
#     image = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)  # Нормализация
#     image = np.uint8(image)  # Преобразуем в uint8
# colored_image_bgr = cv2.cvtColor((colored_image[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# cv2.imwrite("temp.jpg", output)
# img33 = cv2.imread("temp.jpg")
# print(output.dtype)
normalized_img = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# uint8_img = (output * 255).astype(np.uint8)
color_mapped_image = cv2.applyColorMap(normalized_img, cv2.COLORMAP_INFERNO)
cv2.imwrite("depth.jpg", color_mapped_image)

