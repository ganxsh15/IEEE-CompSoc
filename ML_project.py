import cv2
import os

labels = ['A', 'B', 'C']  # Add more signs
for label in labels:
    os.makedirs(f'data/{label}', exist_ok=True)

    cap = cv2.VideoCapture(0)
    print(f"Collecting images for {label}")
    count = 0

    while count < 200:
        ret, frame = cap.read()
        cv2.putText(frame, f'Label: {label}, Image: {count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Capture", frame)
        
        key = cv2.waitKey(1)
        if key == ord('c'):
            cv2.imwrite(f'data/{label}/{count}.jpg', frame)
            count += 1
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = 64

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    'data',
    target_size=(img_size, img_size),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    'data',
    target_size=(img_size, img_size),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32,
    subset='validation'
)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(train_gen, epochs=10, validation_data=val_gen)
model.save('sign_language_model.h5')
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('sign_language_model.h5')
class_names = list(train_gen.class_indices.keys())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    roi = frame[100:300, 100:300]  # region of interest box
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, img_size, img_size, 1)

    result = model.predict(reshaped)
    label = class_names[np.argmax(result)]

    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    cv2.putText(frame, f'Prediction: {label}', (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
