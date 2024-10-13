
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

model = tf.keras.models.load_model('animal_classification_model.h5')

val_dir = 'data/validation/'

animal_classes = ['butterfly', 'chicken', 'cow', 'dog', 'horse', 'sheep', 'spider', 'squirrel']

# Hàm dự đoán loài động vật
def predict_animal(image_path):
    img = image.load_img(image_path, target_size=(150, 150)) 
    img_array = image.img_to_array(img)                       
    img_array = np.expand_dims(img_array, axis=0)            
    img_array /= 255.0                                        
    predictions = model.predict(img_array)                   
    predicted_class = np.argmax(predictions[0])               # Lấy lớp có giá trị dự đoán cao nhất
    confidence = predictions[0][predicted_class]              # Xác suất của lớp được dự đoán
    return animal_classes[predicted_class], confidence        # Trả về tên loài động vật và xác suất

# Hàm hiển thị ảnh 
def display_image_with_prediction(image_path, predicted_animal, confidence):
    img = image.load_img(image_path)
    img_array = np.array(img)
    
    fig, ax = plt.subplots(1)
    ax.imshow(img_array)

    # Thêm khung và chữ lên ảnh
    rect = patches.Rectangle((5, 5), 100, 40, linewidth=3, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)
    plt.text(10, 25, f'{predicted_animal} {confidence:.2f}', color='blue', fontsize=14, weight='bold')

    plt.axis('off')  # Tắt trục
    plt.show()

# Dự đoán
for animal in os.listdir(val_dir):
    animal_dir = os.path.join(val_dir, animal)
    if os.path.isdir(animal_dir):
        print(f"\nDự đoán cho loài: {animal}")
        # Lấy danh sách ảnh 
        image_files = os.listdir(animal_dir)
        # Lấy ngẫu nhiên 10 ảnh 
        selected_images = random.sample(image_files, min(10, len(image_files)))

        for img_file in selected_images:
            img_path = os.path.join(animal_dir, img_file)
            predicted_animal, confidence = predict_animal(img_path)
            print(f"Ảnh: {img_file} | Dự đoán: {predicted_animal} ({confidence:.2f})")
            display_image_with_prediction(img_path, predicted_animal, confidence)
