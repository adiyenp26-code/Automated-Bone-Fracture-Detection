import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMG_SIZE = 224  # OK for MobileNetV2 / EfficientNetB0 (resize always)


def list_images(p: Path):
    if p.is_file() and p.suffix.lower() in VALID_EXTS:
        return [p]
    if p.is_dir():
        return sorted([x for x in p.rglob("*") if x.suffix.lower() in VALID_EXTS])
    return []


def preprocess(img_path: Path):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise ValueError(f"Could not read image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = (rgb.astype(np.float32) / 255.0)[None, ...]  # (1,224,224,3)
    return x, rgb


def probs_from_model(model, x):
    """
    Returns 2-class probabilities [p_class0, p_class1]
    Works for:
      - softmax outputs shape (2,)
      - sigmoid outputs shape (1,)
    """
    p = model.predict(x, verbose=0)[0]

    # scalar
    if np.ndim(p) == 0:
        p1 = float(p)
        p = np.array([1.0 - p1, p1], dtype=np.float32)

    # sigmoid-like [p]
    elif len(p) == 1:
        p1 = float(p[0])
        p = np.array([1.0 - p1, p1], dtype=np.float32)

    # softmax-like [p0,p1]
    else:
        p = np.array(p, dtype=np.float32)
        if p.shape[0] != 2:
            raise ValueError(f"Model output is not binary (got shape {p.shape}).")

    # normalize just in case
    s = float(p.sum()) + 1e-8
    return p / s


def confidence_gap(p2):
    s = np.sort(p2)
    return float(s[-1] - s[-2])


def find_keras_files(search_dirs):
    found = []
    for d in search_dirs:
        d = Path(d)
        if d.exists():
            found.extend(list(d.rglob("*.keras")))
    return sorted(found)


def detect_class_index(project_root: Path, user_data_dir: str):
    """
    We want to know which probability index corresponds to 'fractured'.
    Keras folder ordering is alphabetical by folder name.
    If folders are ['fractured','non_fractured'] => fractured is usually class 0.
    """
    data_dir = Path(user_data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    fractured = data_dir / "fractured"
    nonfract = data_dir / "non_fractured"

    if fractured.exists() and nonfract.exists():
        class_names = sorted(["fractured", "non_fractured"])
        frac_index = class_names.index("fractured")
        return frac_index, class_names

    # fallback: assume the usual alphabetical ordering
    # (this is the most common case anyway)
    class_names = ["fractured", "non_fractured"]
    frac_index = 0
    return frac_index, class_names


# ---------- Grad-CAM (optional) ----------
def find_last_conv_layer(model):
    # Try direct conv layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # Try nested models
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                return find_last_conv_layer(layer)
            except Exception:
                pass
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def gradcam_heatmap(model, img_tensor, class_index):
    last_conv_name = find_last_conv_layer(model)
    last_conv_layer = model.get_layer(last_conv_name)

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        # If sigmoid output, preds shape might be (1,1)
        if preds.shape[-1] == 1:
            # map class_index: fractured -> 1, non_fractured -> 0
            # probability of class 1 is preds[:,0], class 0 is 1-preds[:,0]
            p1 = preds[:, 0]
            chosen = p1 if class_index == 1 else (1.0 - p1)
            loss = chosen
        else:
            loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay(rgb_img, heatmap, alpha=0.40):
    hm = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
    hm_u8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    out = (1 - alpha) * rgb_img + alpha * hm_color
    return np.clip(out, 0, 255).astype(np.uint8)


def save_heatmap_panel(out_path: Path, rgb, ov1, ov2, header):
    fig = plt.figure(figsize=(12, 4))
    plt.suptitle(header)

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(ov1)
    ax2.set_title("Grad-CAM (Model 1)")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(ov2)
    ax3.set_title("Grad-CAM (Model 2)")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder of X-ray images (e.g. test_images)")
    ap.add_argument("--out", default="inference_outputs", help="Output folder for CSV + heatmaps")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for fractured decision")
    ap.add_argument("--w1", type=float, default=0.5, help="Weight for model 1")
    ap.add_argument("--w2", type=float, default=0.5, help="Weight for model 2")
    ap.add_argument("--model1", default="", help="Path to model 1 (.keras)")
    ap.add_argument("--model2", default="", help="Path to model 2 (.keras)")
    ap.add_argument("--data_dir", default="data", help="Dataset folder (to detect class order)")
    ap.add_argument("--save_heatmaps", action="store_true", help="Save Grad-CAM heatmaps")
    args = ap.parse_args()

    project_root = Path.cwd()

    # Detect fractured class index
    fractured_index, class_names = detect_class_index(project_root, args.data_dir)
    nonfract_index = 1 - fractured_index

    print(f"[INFO] Class order assumed: {class_names}")
    print(f"[INFO] fractured_index={fractured_index} | non_fractured_index={nonfract_index}")

    # Output folders
    out_dir = Path(args.out)
    heat_dir = out_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_heatmaps:
        heat_dir.mkdir(parents=True, exist_ok=True)

    # Resolve models
    if not args.model1 or not args.model2:
        # Auto-find as fallback
        search_dirs = [
            project_root / "ensemble_model",
            project_root / "baseline_model",
            project_root / "results",
            project_root,
        ]
        candidates = find_keras_files(search_dirs)
        if len(candidates) < 2:
            raise RuntimeError(
                "Could not find 2 .keras files automatically. "
                "Run with --model1 and --model2 pointing to your models."
            )
        m1_path, m2_path = candidates[-2], candidates[-1]
    else:
        m1_path, m2_path = Path(args.model1), Path(args.model2)

    if str(m1_path.resolve()) == str(m2_path.resolve()):
        raise RuntimeError(
            f"You selected the SAME model twice:\n{m1_path}\nPick two different .keras files."
        )

    print(f"[LOAD] Model 1: {m1_path}")
    print(f"[LOAD] Model 2: {m2_path}")
    model1 = tf.keras.models.load_model(str(m1_path))
    model2 = tf.keras.models.load_model(str(m2_path))

    # Input images
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    images = list_images(input_path)
    if not images:
        raise RuntimeError(f"No images found in: {input_path}")

    rows = []
    for img in images:
        x, rgb = preprocess(img)

        p1 = probs_from_model(model1, x)
        p2 = probs_from_model(model2, x)

        p_ens = (args.w1 * p1) + (args.w2 * p2)
        p_ens = p_ens / (float(p_ens.sum()) + 1e-8)

        fractured_prob = float(p_ens[fractured_index])
        nonfract_prob = float(p_ens[nonfract_index])
        label = "fractured" if fractured_prob >= args.threshold else "non_fractured"
        conf = confidence_gap(p_ens)

        heatmap_path = ""
        if args.save_heatmaps:
            class_idx_for_cam = fractured_index if label == "fractured" else nonfract_index
            hm1 = gradcam_heatmap(model1, tf.convert_to_tensor(x), class_idx_for_cam)
            hm2 = gradcam_heatmap(model2, tf.convert_to_tensor(x), class_idx_for_cam)
            ov1 = overlay(rgb, hm1)
            ov2 = overlay(rgb, hm2)
            out_hm = heat_dir / f"{img.stem}_heatmap.png"
            save_heatmap_panel(
                out_hm,
                rgb,
                ov1,
                ov2,
                header=f"{img.name} | {label} | p(fractured)={fractured_prob:.3f}",
            )
            heatmap_path = str(out_hm)

        rows.append(
            {
                "file": str(img),
                "predicted_label": label,
                "p_fractured_ensemble": fractured_prob,
                "p_nonfractured_ensemble": nonfract_prob,
                "confidence_gap": conf,
                "p_fractured_model1": float(p1[fractured_index]),
                "p_fractured_model2": float(p2[fractured_index]),
                "threshold_used": args.threshold,
                "w1": args.w1,
                "w2": args.w2,
                "class_order": ",".join(class_names),
                "fractured_index": fractured_index,
                "model1_path": str(m1_path),
                "model2_path": str(m2_path),
                "heatmap_path": heatmap_path,
            }
        )

        print(f"[OK] {img.name} -> {label} | p(fractured)={fractured_prob:.3f} | conf={conf:.3f}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== DONE ===")
    print(f"CSV saved to: {csv_path}")
    if args.save_heatmaps:
        print(f"Heatmaps saved to: {heat_dir}")


if __name__ == "__main__":
    main()