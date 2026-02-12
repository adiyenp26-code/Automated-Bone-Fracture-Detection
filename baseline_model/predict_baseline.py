import os
import json
import argparse
import numpy as np
import tensorflow as tf
import cv2


RESULTS_DIR = os.path.join("baseline_model", "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "model.keras")
CLASS_MAP_PATH = os.path.join(RESULTS_DIR, "class_map.json")

IMG_SIZE = (224, 224)


def load_and_preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to image OR folder of images")
    args = parser.parse_args()

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    # Invert mapping: 0 -> fractured, 1 -> non_fractured (depending on your folders)
    inv_map = {v: k for k, v in class_map.items()}

    input_path = args.input

    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    else:
        files = [input_path]

    print("\n--- Baseline Predictions ---")
    for fp in files:
        x = load_and_preprocess(fp)
        prob = float(model.predict(x, verbose=0).ravel()[0])
        pred = 1 if prob >= 0.5 else 0
        label = inv_map.get(pred, str(pred))
        print(f"{os.path.basename(fp)} -> {label} (prob={prob:.4f})")


if __name__ == "__main__":
    main()