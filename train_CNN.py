# train.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# Đường dẫn tới dữ liệu 
train_dir = 'data/train/'
validation_dir = 'data/validation/'

# Tạo pipeline để đọc và xử lý dữ liệu với augmentation cho tập train
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load dữ liệu từ thư mục, kích thước ảnh 150x150, batch size 32
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Xây dựng mô hình CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(8, activation='softmax')
])

# Compile mô hình
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Huấn luyện mô hình và lưu lại lịch sử huấn luyện
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,  # Đặt số lượng epoch
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Lưu mô hình sau khi huấn luyện xong
model.save('animal_classification_model.h5')

# Lưu lịch sử huấn luyện vào'history.pkl'
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Đánh giá 
evaluation = model.evaluate(validation_generator)
print(f'Loss: {evaluation[0]}, Accuracy: {evaluation[1]}')
