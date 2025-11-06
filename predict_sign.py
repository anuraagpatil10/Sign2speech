import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from collections import deque

# -----------------------------
# LOAD MODEL AND LABELS
# -----------------------------
model = tf.keras.models.load_model("sign_model_mobilenet.keras")
labels = np.load("labels.npy", allow_pickle=True).item()

# -----------------------------
# CONFIGURATION
# -----------------------------
IMG_SIZE = 160
SMOOTHING_WINDOW = 5

# Initialize Mediapipe + background remover
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils
segmentor = SelfiSegmentation()

# -----------------------------
# HAND EXTRACTION FUNCTION (for prediction)
# -----------------------------
def extract_hand(frame, draw=False, remove_bg=True):
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
        cropped = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
        if draw:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return cropped
    return None

# -----------------------------
# PREDICTION LOOP
# -----------------------------
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


def predict_sign(frame):
    # Create a copy for drawing (so predictions use the clean frame)
    draw_frame = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    hand_img = None

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            # 🟢 Draw on draw_frame (not on the one used for prediction)
            mp_drawing.draw_landmarks(draw_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
            pad = 30
            x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
            x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

            # Draw bounding box on draw_frame
            cv2.rectangle(draw_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # ✳️ Crop clean hand (from original frame, no drawings)
            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                continue
            hand_img = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

    if hand_img is None:
        return None, 0, draw_frame  # Return frame for display

    # Feed clean image to model
    img = np.expand_dims(hand_img, axis=0) / 255.0
    prediction = model.predict(img)
    label_index = np.argmax(prediction)
    confidence = np.max(prediction)
    label = labels[label_index]

    # Return both prediction and frame-with-skeleton
    return label, confidence, draw_frame


cap = cv2.VideoCapture(0)
print("Starting Sign2Speech (Loaded Model)... Press 'q' to quit.")

predictions = deque(maxlen=SMOOTHING_WINDOW)

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
#         print(f"Recognized: {final_label} ({conf*100:.2f}%)")
#     else:
#         cv2.putText(frame, "No hand detected", (30, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow("Sign2Speech (Live Prediction)", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    label, conf, draw_frame = predict_sign(frame)

    if label:
        predictions.append(label)
        if len(predictions) == predictions.maxlen:
            final_label = max(set(predictions), key=predictions.count)
        else:
            final_label = label

        cv2.putText(draw_frame, f"{final_label} ({conf*100:.1f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Recognized: {final_label} ({conf*100:.2f}%)")
    else:
        cv2.putText(draw_frame, "No hand detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the version with skeletons
    cv2.imshow("Sign-Translate", draw_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
