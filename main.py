# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from gtts import gTTS
# import random
# import time

# # -----------------------------
# # 1. PATHS AND PARAMETERS
# # -----------------------------
# DATASET_PATH = "dataset"
# IMG_SIZE = 128
# BATCH_SIZE = 16
# EPOCHS = 25

# # -----------------------------
# # 2. DATASET LOADING
# # -----------------------------
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2
# )

# train_data = datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# val_data = datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# # -----------------------------
# # 3. MODEL ARCHITECTURE (CNN)
# # -----------------------------
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(len(train_data.class_indices), activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# # -----------------------------
# # 4. TRAIN THE MODEL
# # -----------------------------
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# history = model.fit(
#     train_data,
#     epochs=EPOCHS,
#     validation_data=val_data,
#     callbacks=[early_stop]
# )

# # -----------------------------
# # 5. SAVE MODEL AND LABELS
# # -----------------------------
# model.save("sign_model.h5")

# # Save class indices
# labels = {v: k for k, v in train_data.class_indices.items()}
# np.save("labels.npy", labels)
# print("✅ Model and labels saved successfully!")

# # -----------------------------
# # 6. REAL-TIME DETECTION
# # -----------------------------
# def speak_text(text):
#     tts = gTTS(text=text, lang='en')
#     tts.save("speech.mp3")
#     os.system("start speech.mp3" if os.name == 'nt' else "mpg321 speech.mp3")

# def predict_sign(frame):
#     img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#     img = np.expand_dims(img, axis=0) / 255.0
#     prediction = model.predict(img)
#     label_index = np.argmax(prediction)
#     confidence = np.max(prediction)
#     label = labels[label_index]
#     return label, confidence

# cap = cv2.VideoCapture(0)
# print("🎥 Starting webcam... Press 'q' to quit.")

# last_prediction = ""
# last_spoken_time = time.time()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip frame
#     frame = cv2.flip(frame, 1)
#     label, conf = predict_sign(frame)

#     # Display prediction
#     cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (30, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow("Sign2Speech", frame)

#     # Speak only if new label or after 3 seconds
#     if conf > 0.85 and (label != last_prediction or time.time() - last_spoken_time > 3):
#         print(f"🖐 Recognized: {label}")
#         speak_text(label)
#         last_prediction = label
#         last_spoken_time = time.time()

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("🛑 Webcam closed.")








# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import mediapipe as mp
# import time

# # -----------------------------
# # 1. CONFIGURATION
# # -----------------------------
# DATASET_PATH = "dataset"
# IMG_SIZE = 128
# BATCH_SIZE = 16
# EPOCHS = 25

# # Mediapipe initialization
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
# mp_drawing = mp.solutions.drawing_utils

# # -----------------------------
# # 2. HAND CROPPING FUNCTION
# # -----------------------------
# def extract_hand(frame, draw=False):
#     """Detect and crop only the hand region using MediaPipe."""
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb)

#     if results.multi_hand_landmarks and draw:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     if not results.multi_hand_landmarks:
#         return None

#     for hand_landmarks in results.multi_hand_landmarks:
#         h, w, _ = frame.shape
#         x_coords = [lm.x for lm in hand_landmarks.landmark]
#         y_coords = [lm.y for lm in hand_landmarks.landmark]
#         x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
#         y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

#         # Padding to make sure full hand is captured
#         pad = 30
#         x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
#         x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

#         cropped = frame[y_min:y_max, x_min:x_max]
#         if cropped.size == 0:
#             return None

#         cropped = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
#         return cropped
#     return None

# # -----------------------------
# # 3. DATASET PREPROCESSING PIPELINE
# # -----------------------------
# print("🧹 Preprocessing dataset with MediaPipe hand cropping...")

# def preprocess_dataset(input_dir, output_dir="processed_dataset"):
#     os.makedirs(output_dir, exist_ok=True)

#     for class_name in os.listdir(input_dir):
#         class_dir = os.path.join(input_dir, class_name)
#         if not os.path.isdir(class_dir):
#             continue

#         output_class_dir = os.path.join(output_dir, class_name)
#         os.makedirs(output_class_dir, exist_ok=True)

#         for img_file in os.listdir(class_dir):
#             img_path = os.path.join(class_dir, img_file)
#             img = cv2.imread(img_path)
#             if img is None:
#                 continue

#             hand_img = extract_hand(img)
#             if hand_img is not None:
#                 save_path = os.path.join(output_class_dir, img_file)
#                 cv2.imwrite(save_path, hand_img)

#     return output_dir

# PROCESSED_DATASET_PATH = preprocess_dataset(DATASET_PATH)
# print("✅ Hand-only dataset created successfully!")

# # -----------------------------
# # 4. LOAD DATA FOR TRAINING
# # -----------------------------
# datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# train_data = datagen.flow_from_directory(
#     PROCESSED_DATASET_PATH,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# val_data = datagen.flow_from_directory(
#     PROCESSED_DATASET_PATH,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# # -----------------------------
# # 5. CNN MODEL
# # -----------------------------
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(len(train_data.class_indices), activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # -----------------------------
# # 6. TRAIN THE MODEL
# # -----------------------------
# print("🧠 Training the model...")
# history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data, callbacks=[early_stop])

# # -----------------------------
# # 7. SAVE MODEL AND LABELS
# # -----------------------------
# model.save("sign_model.keras")  # modern keras format
# labels = {v: k for k, v in train_data.class_indices.items()}
# np.save("labels.npy", labels)
# print("✅ Model and labels saved successfully!")

# # -----------------------------
# # 8. REAL-TIME PREDICTION LOOP WITH HAND LANDMARKS
# # -----------------------------
# def predict_sign(frame):
#     hand_img = extract_hand(frame, draw=True)  # 👈 Draws hand skeleton
#     if hand_img is None:
#         return None, 0
#     img = np.expand_dims(hand_img, axis=0) / 255.0
#     prediction = model.predict(img)
#     label_index = np.argmax(prediction)
#     confidence = np.max(prediction)
#     label = labels[label_index]
#     return label, confidence

# cap = cv2.VideoCapture(0)
# print("🎥 Starting webcam... Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     label, conf = predict_sign(frame)

#     if label:
#         cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (30, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         print(f"🖐 Recognized: {label} ({conf*100:.2f}%)")
#     else:
#         cv2.putText(frame, "No hand detected", (30, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow("Sign2Speech (Hand Tracking)", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.applications import MobileNetV2
# import mediapipe as mp
# from cvzone.SelfiSegmentationModule import SelfiSegmentation
# from collections import deque
# import time

# # -----------------------------
# # CONFIGURATION
# # -----------------------------
# DATASET_PATH = "dataset"
# PROCESSED_PATH = "processed_dataset"
# IMG_SIZE = 160
# BATCH_SIZE = 16
# EPOCHS = 25
# SMOOTHING_WINDOW = 5

# # Initialize Mediapipe + Segmentation
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
# mp_drawing = mp.solutions.drawing_utils
# segmentor = SelfiSegmentation()

# # -----------------------------
# # HAND EXTRACTION FUNCTION
# # -----------------------------
# def extract_hand(frame, draw=False, remove_bg=True):
#     """Detect and crop only the hand region using MediaPipe + optional background removal."""
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb)
#     if not results.multi_hand_landmarks:
#         return None

#     if remove_bg:
#         frame = segmentor.removeBG(frame, (255, 255, 255))  # ✅ FIXED

#     h, w, _ = frame.shape
#     for hand_landmarks in results.multi_hand_landmarks:
#         x_coords = [lm.x for lm in hand_landmarks.landmark]
#         y_coords = [lm.y for lm in hand_landmarks.landmark]
#         x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
#         y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

#         pad = 30
#         x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
#         x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

#         cropped = frame[y_min:y_max, x_min:x_max]
#         if cropped.size == 0:
#             return None

#         cropped = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

#         if draw:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#         return cropped
#     return None


# # -----------------------------
# # DATASET PREPROCESSING
# # -----------------------------
# def preprocess_dataset(input_dir, output_dir=PROCESSED_PATH):
#     os.makedirs(output_dir, exist_ok=True)
#     print("🧹 Preprocessing dataset with MediaPipe hand detection + background removal...")

#     for class_name in os.listdir(input_dir):
#         class_dir = os.path.join(input_dir, class_name)
#         if not os.path.isdir(class_dir):
#             continue

#         output_class_dir = os.path.join(output_dir, class_name)
#         os.makedirs(output_class_dir, exist_ok=True)

#         for img_file in os.listdir(class_dir):
#             img_path = os.path.join(class_dir, img_file)
#             img = cv2.imread(img_path)
#             if img is None:
#                 continue

#             hand_img = extract_hand(img, draw=False, remove_bg=True)
#             if hand_img is not None:
#                 save_path = os.path.join(output_class_dir, img_file)
#                 cv2.imwrite(save_path, hand_img)

#     print("✅ Preprocessing complete! Hand-only dataset ready.")
#     return output_dir

# PROCESSED_DATASET_PATH = preprocess_dataset(DATASET_PATH)

# # -----------------------------
# # DATA AUGMENTATION & GENERATOR
# # -----------------------------
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     brightness_range=[0.5, 1.5],
#     zoom_range=0.2,
#     rotation_range=10,
#     horizontal_flip=True
# )

# train_data = datagen.flow_from_directory(
#     PROCESSED_DATASET_PATH,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# val_data = datagen.flow_from_directory(
#     PROCESSED_DATASET_PATH,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# # -----------------------------
# # TRANSFER LEARNING MODEL (MobileNetV2)
# # -----------------------------
# base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# x = GlobalAveragePooling2D()(base.output)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.4)(x)
# output = Dense(len(train_data.class_indices), activation='softmax')(x)
# model = Model(inputs=base.input, outputs=output)

# for layer in base.layers:
#     layer.trainable = False  # freeze base layers initially

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # -----------------------------
# # TRAIN THE MODEL
# # -----------------------------
# print("🧠 Training MobileNetV2 model...")
# history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data, callbacks=[early_stop])
# model.save("sign_model_mobilenet.keras")

# labels = {v: k for k, v in train_data.class_indices.items()}
# np.save("labels.npy", labels)
# print("✅ Model trained and saved successfully!")

# # -----------------------------
# # REAL-TIME PREDICTION LOOP (With Temporal Smoothing)
# # -----------------------------
# def predict_sign(frame):
#     hand_img = extract_hand(frame, draw=True, remove_bg=True)
#     if hand_img is None:
#         return None, 0
#     img = np.expand_dims(hand_img, axis=0) / 255.0
#     prediction = model.predict(img)
#     label_index = np.argmax(prediction)
#     confidence = np.max(prediction)
#     label = labels[label_index]
#     return label, confidence

# cap = cv2.VideoCapture(0)
# print("🎥 Starting webcam... Press 'q' to quit.")

# predictions = deque(maxlen=SMOOTHING_WINDOW)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     label, conf = predict_sign(frame)

#     if label:
#         predictions.append(label)
#         if len(predictions) == predictions.maxlen:
#             final_label = max(set(predictions), key=predictions.count)
#         else:
#             final_label = label

#         cv2.putText(frame, f"{final_label} ({conf*100:.1f}%)", (30, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         print(f"🖐 Recognized: {final_label} ({conf*100:.2f}%)")
#     else:
#         cv2.putText(frame, "No hand detected", (30, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow("ISL-to-Text (MobileNetV2)", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
import mediapipe as mp
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_PATH = "dataset"
PROCESSED_PATH = "processed_dataset"
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 25

# Mediapipe & background remover
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
segmentor = SelfiSegmentation()

# -----------------------------
# HAND EXTRACTION
# -----------------------------
def extract_hand(frame, remove_bg=True):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None

    if remove_bg:
        frame = segmentor.removeBG(frame, (255, 255, 255))

    h, w, _ = frame.shape
    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
        pad = 30
        x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
        x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)
        cropped = frame[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return None
        return cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
    return None

# -----------------------------
# DATASET PREPROCESSING
# -----------------------------
def preprocess_dataset(input_dir, output_dir=PROCESSED_PATH):
    os.makedirs(output_dir, exist_ok=True)
    print("🧹 Preprocessing dataset...")

    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            hand_img = extract_hand(img)
            if hand_img is not None:
                cv2.imwrite(os.path.join(output_class_dir, img_file), hand_img)
    print("✅ Preprocessing complete!")
    return output_dir

PROCESSED_PATH = preprocess_dataset(DATASET_PATH)

# -----------------------------
# DATA GENERATORS
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    brightness_range=[0.5, 1.5],
    zoom_range=0.2,
    rotation_range=10,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    PROCESSED_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    PROCESSED_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# -----------------------------
# MODEL: MobileNetV2
# -----------------------------
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(len(train_data.class_indices), activation='softmax')(x)
model = Model(inputs=base.input, outputs=output)

for layer in base.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("🧠 Training MobileNetV2 model...")
history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data, callbacks=[early_stop])

# -----------------------------
# SAVE MODEL AND LABELS
# -----------------------------
model.save("sign_model_mobilenet.keras")
labels = {v: k for k, v in train_data.class_indices.items()}
np.save("labels.npy", labels)
print("✅ Model trained and saved successfully!")
