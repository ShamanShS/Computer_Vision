import cv2

def blur_image(input_image_path, output_image_path, ksize=(15, 15)):
    # Загрузка изображения
    image = cv2.imread(input_image_path)
    
    # Применение размытия (Gaussian Blur)
    blurred_image = cv2.GaussianBlur(image, ksize, 0)
    
    # Сохранение результата
    cv2.imwrite(output_image_path, blurred_image)
    print(f"Сохранено размытое изображение: {output_image_path}")

# Пример использования
input_image_path = "image/imgchb.jpg"  # Укажите путь к вашему изображению
output_image_path = "image/blurred_img.jpg"  # Укажите путь для сохранения размытого изображения
blur_image(input_image_path, output_image_path)
