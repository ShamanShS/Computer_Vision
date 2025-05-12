import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np


model = VGG16(weights='imagenet')


img_path = 'img.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

for _, label, probability in decode_predictions(preds, top=3)[0]:
    print(f"{label}: {probability:.2f}")