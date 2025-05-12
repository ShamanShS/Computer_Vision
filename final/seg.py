from ultralytics import YOLO
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
import numpy as np


model_yolo = YOLO("yolov8n-seg.pt")  
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def describe_object(image_crop, texts):

    inputs = processor_clip(text=texts, images=image_crop, return_tensors="pt", padding=True)
    outputs = model_clip(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return texts[probs.argmax().item()]

image = cv2.imread("image/C.jpg")
image2 = cv2.imread("depth.jpg")
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертация в RGB
# image = image_rgb

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_img, (5, 5), 20)
blurred_3ch = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
image_rgb = blurred_3ch
image = image_rgb


results = model_yolo(image_rgb)


text_labels = [
    "person", "dog", "tree", 
    "building", "horse","wolf"
]


for result in results:
    if result.masks is not None:
        for i, mask in enumerate(result.masks):

            contour = mask.xy[0].astype(np.int32).reshape((-1, 1, 2))  # Убрали .cpu()
            

            x, y, w, h = cv2.boundingRect(contour)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
            
            if x2 <= x1 or y2 <= y1:
                continue  


            crop = image_rgb[y1:y2, x1:x2]
            mask_array = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            offset_contour = contour - (x1, y1)
            cv2.fillPoly(mask_array, [offset_contour], 255)
            masked_crop = cv2.bitwise_and(crop, crop, mask=mask_array)
            

            clip_label = describe_object(
                Image.fromarray(masked_crop), 
                text_labels
            )
            

            cls_id = int(result.boxes[i].cls.item())
            cls_name = model_yolo.names[cls_id]
            

            hue = (i * 50) % 180
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]), 
                cv2.COLOR_HSV2BGR
            )[0][0].tolist()


            cv2.drawContours(image, [contour], -1, color, 2)
            overlay = image.copy()
            cv2.fillPoly(overlay, [contour], color)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            

            label = f"{cls_name} ({clip_label})"
            cv2.putText(image, label, (x1, y1-10 if y1>20 else y1+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.drawContours(image2, [contour], -1, color, 2)
            overlay = image2.copy()
            cv2.fillPoly(overlay, [contour], color)
            cv2.addWeighted(overlay, 0.3, image2, 0.7, 0, image2)
            

            label = f"{cls_name} ({clip_label})"
            cv2.putText(image2, label, (x1, y1-10 if y1>20 else y1+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


cv2.imwrite("segmented_imageE.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imwrite("segmented_image2.jpg", image2)
cv2.imshow("Segmentation Result", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
