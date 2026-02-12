import os
import argparse
import numpy as np
import tensorflow as tf
import cv2

# ---------- CONFIG ----------
IMG_SIZE = 224
WEIGHT_BASELINE = 0.6
WEIGHT_SECOND = 0.4

BASELINE_MODEL_PATH = "baseline_model/model_baseline.keras"
SECOND_MODEL_PATH = "ensemble_model/model_second.keras"


# ---------- LOAD MODELS ----------
baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
second_model = tf.keras.models.load_model(SECOND_MODEL_PATH)


# ---------- IMAGE PREPROCESS ----------
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


# ---------- PREDICT ----------
def ensemble_predict(image_path):
    img = preprocess_image(image_path)

    p1 = baseline_model.predict(img)[0][0]
    p2 = second_model.predict(img)[0][0]

    final_prob = WEIGHT_BASELINE * p1 + WEIGHT_SECOND * p2

    label = "Fractured" if final_prob >= 0.5 else "Non-Fractured"

    return final_prob, label


# ---------- RUN ON FOLDER ----------
def predict_folder(folder):
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, file)
            prob, label = ensemble_predict(path)
            print(f"{file} → {label} (prob={prob:.3f})")


# ---------- MAIN ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder with X-ray images")
    args = parser.parse_args()

    predict_folder(args.input)