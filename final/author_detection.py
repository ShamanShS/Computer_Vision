import tensorflow as tf
from tf.keras.preprocessing.image import ImageDataGenerator
from tf.keras.models import Sequential
from tf.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Создание генератора данных
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Загрузка данных
train_generator = datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Создание модели
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Сохранение модели
model.save('author_detection_model.h5')
