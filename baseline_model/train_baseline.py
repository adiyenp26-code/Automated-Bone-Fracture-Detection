import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_accuracy_plot(history, out_path: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history.history.get("accuracy", []), marker="o", label="Train")
    plt.plot(history.history.get("val_accuracy", []), marker="o", label="Validation")
    plt.title("Training vs Validation Accuracy (Baseline)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_loss_plot(history, out_path: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history.history.get("loss", []), marker="o", label="Train")
    plt.plot(history.history.get("val_loss", []), marker="o", label="Validation")
    plt.title("Training vs Validation Loss (Baseline)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_confusion_matrix_plot(cm, class_names, out_path: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix (Baseline)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=20)
    plt.yticks(range(len(class_names)), class_names)

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_baseline_model(img_size: int, num_classes: int, lr: float) -> tf.keras.Model:
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base.trainable = False  # baseline = freeze backbone

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to dataset folder containing fractured/ and non_fractured/")
    parser.add_argument("--out_dir", type=str, default=os.path.join("baseline_model", "results"),
                        help="Where to save outputs")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Data folder not found: {args.data_dir}\n"
            "Expected structure:\n"
            "data/fractured/\n"
            "data/non_fractured/\n"
        )

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Data loading (folder names become class labels) ----
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=args.val_split,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.10,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=args.val_split
    )

    train_data = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=args.seed
    )

    val_data = val_datagen.flow_from_directory(
        args.data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    class_names = list(train_data.class_indices.keys())
    class_indices = train_data.class_indices
    print("Class indices:", class_indices)

    # ---- Class weights to handle imbalance ----
    y_train = train_data.classes
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print("Class weights:", class_weight)

    # ---- Build model ----
    model = build_baseline_model(args.img_size, train_data.num_classes, args.lr)

    model_path = os.path.join(args.out_dir, "model_baseline.keras")
    callbacks = [
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    # ---- Train ----
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # ---- Save learning curves ----
    save_accuracy_plot(history, os.path.join(args.out_dir, "accuracy.png"))
    save_loss_plot(history, os.path.join(args.out_dir, "loss.png"))

    # ---- Evaluate on validation ----
    val_data.reset()
    y_prob = model.predict(val_data, verbose=1)         # shape: (N, num_classes)
    y_pred = np.argmax(y_prob, axis=1)                  # predicted class index
    y_true = val_data.classes                           # true class index

    # Standard confusion matrix + classification report
    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix_plot(cm, class_names, os.path.join(args.out_dir, "confusion_matrix.png"))

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # =========================================================
    # Add-ons:
    # 1) Sensitivity/Specificity summary (clinical_metrics.txt)
    # 2) ROC curve (roc_curve.png)
    # 3) Threshold note (threshold_note.txt) + suggested threshold
    # =========================================================

    # Identify fractured class index robustly
    if "fractured" not in class_indices or "non_fractured" not in class_indices:
        raise ValueError(
            "Folder names must be exactly 'fractured' and 'non_fractured' inside data/.\n"
            f"Found: {list(class_indices.keys())}"
        )

    fractured_idx = class_indices["fractured"]

    # Binary ground truth: fractured = 1, non_fractured = 0
    y_true_bin = (y_true == fractured_idx).astype(int)

    # Probability of the positive class (fractured)
    p_fractured = y_prob[:, fractured_idx]

    # Default threshold
    default_thr = 0.50
    y_pred_bin_default = (p_fractured >= default_thr).astype(int)
    cm_def = confusion_matrix(y_true_bin, y_pred_bin_default)
    tn, fp, fn, tp = cm_def.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0   # TPR / recall
    specificity = tn / (tn + fp) if (tn + fp) else 0.0   # TNR
    precision = tp / (tp + fp) if (tp + fp) else 0.0     # PPV
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    # ROC + AUC
    fpr, tpr, thr_list = roc_curve(y_true_bin, p_fractured)
    roc_auc = auc(fpr, tpr)

    # Suggested threshold using Youden’s J (TPR - FPR)
    j_scores = tpr - fpr
    best_i = int(np.argmax(j_scores))
    best_thr = float(thr_list[best_i])
    best_tpr = float(tpr[best_i])
    best_fpr = float(fpr[best_i])
    best_spec = 1.0 - best_fpr

    # Save ROC curve image
    roc_path = os.path.join(args.out_dir, "roc_curve.png")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.scatter([best_fpr], [best_tpr], label=f"Suggested thr={best_thr:.2f}", s=60)
    plt.title("ROC Curve (Baseline)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # Save clinical metrics text
    metrics_path = os.path.join(args.out_dir, "clinical_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Clinical Metrics Summary (Baseline)\n")
        f.write("==================================\n\n")
        f.write(f"Positive class: fractured (index {fractured_idx})\n")
        f.write(f"Default threshold used: {default_thr:.2f}\n\n")
        f.write("Binary confusion matrix at default threshold (fractured=1):\n")
        f.write(f"TP={tp}, FN={fn}, TN={tn}, FP={fp}\n\n")
        f.write(f"Sensitivity / Recall (TPR): {sensitivity:.4f}\n")
        f.write(f"Specificity (TNR):          {specificity:.4f}\n")
        f.write(f"Precision (PPV):            {precision:.4f}\n")
        f.write(f"NPV:                        {npv:.4f}\n")
        f.write(f"ROC-AUC:                    {roc_auc:.4f}\n\n")
        f.write("Suggested threshold (Youden’s J maximization):\n")
        f.write(f"  threshold:   {best_thr:.4f}\n")
        f.write(f"  sensitivity: {best_tpr:.4f}\n")
        f.write(f"  specificity: {best_spec:.4f}\n")

    # Save threshold note
    note_path = os.path.join(args.out_dir, "threshold_note.txt")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("Threshold Note (Baseline)\n")
        f.write("=========================\n\n")
        f.write("This model outputs a probability for the 'fractured' class.\n")
        f.write("A default threshold of 0.50 is used to convert probability into a binary decision.\n\n")
        f.write("Why 0.50?\n")
        f.write("- It is the standard starting point when the costs of false positives and false negatives\n")
        f.write("  are assumed equal.\n\n")
        f.write("How to tune for biomedical use:\n")
        f.write("- If the priority is to miss as few fractures as possible (higher sensitivity), lower the threshold.\n")
        f.write("- If the priority is to reduce false alarms (higher specificity), raise the threshold.\n\n")
        f.write("Suggested threshold (from ROC, Youden’s J):\n")
        f.write(f"- {best_thr:.4f} (maximizes sensitivity + specificity - 1 on this validation split)\n\n")
        f.write("Important note:\n")
        f.write("- Threshold tuning should be validated on an independent test set before any real clinical use.\n")

    print("\nSaved baseline outputs to:", args.out_dir)
    print("Model:", model_path)
    print("Clinical metrics:", metrics_path)
    print("ROC curve:", roc_path)
    print("Threshold note:", note_path)


if __name__ == "__main__":
    main()