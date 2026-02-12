import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# -------------------------
# Helpers
# -------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_plot(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def compute_class_weights(train_gen):
    """
    Basic class weighting for imbalanced datasets.
    class indices: { 'fractured': 0, 'non_fractured': 1 } (depends on classes order)
    """
    labels = train_gen.classes
    total = len(labels)
    # counts per class index
    unique, counts = np.unique(labels, return_counts=True)
    count_dict = dict(zip(unique, counts))
    # weight = total / (2 * count_class)
    class_weight = {}
    for cls in unique:
        class_weight[int(cls)] = total / (len(unique) * count_dict[cls])
    return class_weight

def build_model(backbone_name: str, input_shape=(224,224,3)):
    if backbone_name == "mobilenetv2":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone_name == "efficientnetb0":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unknown backbone_name")

    base.trainable = False  # transfer learning baseline

    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_history(hist, out_path_prefix: str, title_prefix: str):
    # Accuracy
    plt.figure(figsize=(9,5))
    plt.plot(hist.history["accuracy"], marker="o", label="Train")
    plt.plot(hist.history["val_accuracy"], marker="o", label="Validation")
    plt.title(f"Training vs Validation Accuracy ({title_prefix})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    save_plot(out_path_prefix + "_accuracy.png")

    # Loss
    plt.figure(figsize=(9,5))
    plt.plot(hist.history["loss"], marker="o", label="Train")
    plt.plot(hist.history["val_loss"], marker="o", label="Validation")
    plt.title(f"Training vs Validation Loss ({title_prefix})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    save_plot(out_path_prefix + "_loss.png")

def plot_confusion(cm, labels, out_path, title):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=20)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    save_plot(out_path)

def youden_threshold(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thr[idx]), fpr, tpr, thr

def compute_clinical_metrics(cm):
    # cm format for binary:
    # [[TN, FP],
    #  [FN, TP]] if you use labels order [0,1] for negative,positive
    TN, FP, FN, TP = cm.ravel()

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # recall positive
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv         = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    accuracy    = (TP + TN) / (TP + TN + FP + FN) if (TP+TN+FP+FN)>0 else 0.0

    return {
        "TP": int(TP), "FP": int(FP), "TN": int(TN), "FN": int(FN),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "accuracy": float(accuracy),
    }

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data", help="Path to data/ with fractured/ non_fractured/")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--results_dir", type=str, default="ensemble/results")
    ap.add_argument("--w_mobilenet", type=float, default=0.6)
    ap.add_argument("--w_effnet", type=float, default=0.4)
    args = ap.parse_args()

    ensure_dir(args.results_dir)

    # Sanity check weights
    w_sum = args.w_mobilenet + args.w_effnet
    w1 = args.w_mobilenet / w_sum
    w2 = args.w_effnet / w_sum

    img_size = (args.img_size, args.img_size)

    # Data loaders
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True
    )

    # IMPORTANT: explicit class order for consistency
    classes = ["fractured", "non_fractured"]

    train_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
        classes=classes
    )

    val_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        classes=classes
    )

    class_weight = compute_class_weights(train_gen)

    # Build and train MobileNetV2
    print("\n=== Training Model A: MobileNetV2 ===")
    model_a = build_model("mobilenetv2", input_shape=(args.img_size, args.img_size, 3))
    hist_a = model_a.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weight,
        verbose=1
    )
    model_a_path = os.path.join(args.results_dir, "model_mobilenetv2.keras")
    model_a.save(model_a_path)

    plot_history(hist_a, os.path.join(args.results_dir, "mobilenetv2"), "MobileNetV2")

    # Build and train EfficientNetB0
    print("\n=== Training Model B: EfficientNetB0 ===")
    model_b = build_model("efficientnetb0", input_shape=(args.img_size, args.img_size, 3))
    hist_b = model_b.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weight,
        verbose=1
    )
    model_b_path = os.path.join(args.results_dir, "model_efficientnetb0.keras")
    model_b.save(model_b_path)

    plot_history(hist_b, os.path.join(args.results_dir, "efficientnetb0"), "EfficientNetB0")

    # -------------------------
    # Evaluate Ensemble on validation
    # -------------------------
    val_gen.reset()
    y_true = val_gen.classes.astype(int)

    # Probabilities from each model
    probs_a = model_a.predict(val_gen, verbose=0).reshape(-1)
    probs_b = model_b.predict(val_gen, verbose=0).reshape(-1)

    # Weighted ensemble
    probs_ens = (w1 * probs_a) + (w2 * probs_b)

    # ROC + AUC
    fpr, tpr, thresholds = roc_curve(y_true, probs_ens)
    roc_auc = auc(fpr, tpr)

    # Choose threshold using Youden’s J (clinical-style)
    best_thr, fpr2, tpr2, thr2 = youden_threshold(y_true, probs_ens)

    # Default threshold for reporting (0.5) + also note best_thr
    thr_report = 0.5
    y_pred = (probs_ens >= thr_report).astype(int)

    # Confusion matrix (need ordering TN/FP/FN/TP)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    # Save confusion matrix plot
    plot_confusion(
        cm,
        labels=["fractured", "non_fractured"],
        out_path=os.path.join(args.results_dir, "confusion_matrix_ensemble.png"),
        title="Confusion Matrix (Ensemble)"
    )

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=["fractured", "non_fractured"],
        digits=4
    )
    with open(os.path.join(args.results_dir, "classification_report_ensemble.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Clinical metrics
    clinical = compute_clinical_metrics(cm)
    with open(os.path.join(args.results_dir, "clinical_metrics_ensemble.txt"), "w", encoding="utf-8") as f:
        f.write("Clinical Metrics (Ensemble)\n")
        f.write(f"Threshold used for classification: {thr_report}\n\n")
        f.write(f"TP: {clinical['TP']}\nFP: {clinical['FP']}\nTN: {clinical['TN']}\nFN: {clinical['FN']}\n\n")
        f.write(f"Sensitivity (Recall, fractured): {clinical['sensitivity']:.4f}\n")
        f.write(f"Specificity (non-fractured):     {clinical['specificity']:.4f}\n")
        f.write(f"Precision:                       {clinical['precision']:.4f}\n")
        f.write(f"NPV:                             {clinical['npv']:.4f}\n")
        f.write(f"Accuracy:                        {clinical['accuracy']:.4f}\n")
        f.write(f"ROC-AUC (ensemble probs):        {roc_auc:.4f}\n")

    # ROC curve plot
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0,1],[0,1], linestyle="--", label="Chance")
    plt.scatter([fpr2[np.argmax(tpr2 - fpr2)]], [tpr2[np.argmax(tpr2 - fpr2)]], s=80,
                label=f"Suggested thr={best_thr:.3f}")
    plt.title("ROC Curve (Ensemble)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.grid(True)
    plt.legend()
    save_plot(os.path.join(args.results_dir, "roc_curve_ensemble.png"))

    # Threshold note
    with open(os.path.join(args.results_dir, "threshold_note_ensemble.txt"), "w", encoding="utf-8") as f:
        f.write("Threshold Note (Ensemble)\n\n")
        f.write(f"1) Standard threshold used for reporting = 0.50\n")
        f.write("   Reason: widely used default; easy to interpret; consistent across experiments.\n\n")
        f.write(f"2) ROC-based suggested threshold (Youden’s J) = {best_thr:.4f}\n")
        f.write("   Meaning: maximizes (Sensitivity - False Positive Rate) on this validation split.\n")
        f.write("   In clinical workflows, threshold can be tuned depending on whether missing fractures\n")
        f.write("   (FN) is worse than false alarms (FP).\n\n")
        f.write("   Recommendation: keep 0.50 for baseline reporting, but discuss tuning as a future step.\n")

    # Save ensemble metadata
    meta = {
        "model_a": "mobilenetv2",
        "model_b": "efficientnetb0",
        "weights": {"mobilenetv2": w1, "efficientnetb0": w2},
        "report_threshold": thr_report,
        "youden_threshold": best_thr,
        "roc_auc": float(roc_auc),
        "class_indices": val_gen.class_indices
    }
    with open(os.path.join(args.results_dir, "ensemble_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Ensemble training complete.")
    print(f"Models saved:\n - {model_a_path}\n - {model_b_path}")
    print(f"Results saved in: {args.results_dir}")

if __name__ == "__main__":
    main()