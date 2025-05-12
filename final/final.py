import torch
from torchvision import models, transforms
from PIL import Image
import requests

# Загружаем предобученную модель ResNet-50
model = models.resnet50(pretrained=True)
model.eval()  # Переводим в режим инференса (без обучения)

# Загрузка списка классов для ImageNet
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.split("\n")

# Функция для предобработки изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Изменение размера
    transforms.ToTensor(),           # Преобразование в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

# Загружаем изображение
image_path = "image/C.jpg"  # Укажите путь к вашему изображению
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Добавляем batch размерности

# Классификация изображения
with torch.no_grad():
    outputs = model(image)

# Получаем топ-5 предсказанных классов
probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

print("Топ-5 предсказанных классов:")
for i in range(5):
    print(f"{labels[top5_catid[i]]}: {top5_prob[i].item()*100:.2f}%")
