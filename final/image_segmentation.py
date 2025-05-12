import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Загрузка предобученной модели DeepLabV3
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# Преобразования для изображения
preprocess = T.Compose([
    T.Resize((520, 520)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка изображения
image_path = "image/img.jpg"  # Укажите путь к вашему изображению
input_image = Image.open(image_path)
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Создаем batch размерности

# Сегментация изображения
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# Преобразование сегментации в изображение
palette = np.random.randint(0, 256, (21, 3), dtype=np.uint8)
segmentation_image = palette[output_predictions.numpy()]

# Визуализация оригинального изображения и сегментации
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmentation_image)
plt.title("Segmentation")
plt.axis("off")

plt.show()
