import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_dir = '../Data/'
target_class_size = 1500
class_names = os.listdir(dataset_dir)
class_names.sort()
X = []
y = []

datagen = ImageDataGenerator(
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

for class_id, class_name in enumerate(class_names):
    class_path = os.path.join(dataset_dir, class_name)
    
    if os.path.isdir(class_path):
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_images = []

        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            class_images.append(image)

        current_class_size = len(class_images)

        if current_class_size > target_class_size:
            class_images = class_images[:target_class_size]

        elif current_class_size < target_class_size:
            num_to_augment = target_class_size - current_class_size
            augment_count = 0
            for idx, image in enumerate(class_images):
                image_exp = np.expand_dims(image, axis=0)
                for batch in datagen.flow(image_exp, batch_size=1):
                    aug_image = batch[0].astype('uint8')
                    save_path = os.path.join(class_path, f"aug_{augment_count}.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                    class_images.append(aug_image)
                    augment_count += 1
                    if augment_count >= num_to_augment:
                        break
                if augment_count >= num_to_augment:
                    break

        X.extend(class_images)
        y.extend([class_id] * len(class_images))

X = np.array(X)
y = np.array(y)

print(f"Total images after balancing: {len(X)}")
