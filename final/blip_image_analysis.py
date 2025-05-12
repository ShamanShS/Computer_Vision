from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Загрузка предобученной модели и процессора BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Загрузка изображения
image_url = "image/blurred_img.jpg"  # Замените на URL вашего изображения
image = Image.open(image_url)

# Подготовка изображения для модели
inputs = processor(image, return_tensors="pt")

# Генерация описания изображения
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("Описание изображения:", caption)
