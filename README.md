# Automated Bone Fracture Detection: Transfer Learning and Ensemble Deep Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

This repository implements a deep learning framework for automated binary classification of bone fractures from radiographic X-ray images. By leveraging transfer learning and ensemble techniques, I aim to enhance diagnostic reliability in medical imaging, with a focus on handling class imbalance and prioritizing clinically relevant metrics like sensitivity and specificity.

**Note:** This is an engineering evaluation study and is **not intended for clinical use or deployment**.

## 🎯 Problem Formulation

I frame the task as a binary supervised classification problem:

X → {Fractured, Non-Fractured}

Where:
- X is a radiographic X-ray image resized to 224×224 pixels.
- The dataset exhibits class imbalance, which I address through targeted techniques.
- Evaluation includes both threshold-dependent (e.g., confusion matrices) and threshold-independent (e.g., ROC curves) analyses to ensure a balanced sensitivity–specificity trade-off.

This setup simulates real-world medical scenarios where missing a fracture (false negative) can have severe consequences, while over-diagnosis (false positive) increases unnecessary interventions.

## 🗂️ Dataset

The dataset is sourced from Kaggle: [Bone Fracture Dataset](https://www.kaggle.com/datasets/orvile/bone-fracture-dataset).

Expected directory structure:
```
dataset/
├── fractured/
└── non_fractured/
```

### Preprocessing Pipeline
- **Resizing**: All images are resized to 224×224 pixels for model compatibility.
- **Normalization**: Pixel values scaled to [0, 1] for stable training.
- **Class Weighting**: Applied to counter imbalance and prevent bias toward the majority class.
- **Data Augmentation**: Controlled transformations including rotation (±20°), translation (up to 10%), zoom (up to 10%), and horizontal flips to improve generalization without introducing artifacts.

**Important:** The dataset is not included in this repository due to licensing and size constraints. Download it from the Kaggle link and organize it as shown above.

## 🧠 Methodology

I implemented and compared two modeling strategies:

### 1. Baseline Model: MobileNetV2 (Transfer Learning)
- **Architecture**: Pretrained on ImageNet with a frozen convolutional backbone initially, followed by fine-tuning.
- **Custom Head**:
  - Global Average Pooling.
  - Dense layer with ReLU activation.
  - Dropout for regularization.
  - Softmax output for binary classification.
- **Optimization**:
  - Loss: Categorical Cross-Entropy (with class weights).
  - Optimizer: Adam.
  - Early stopping and learning rate scheduling for convergence.

This baseline provides a lightweight, efficient reference to assess the impact of imbalance mitigation.

### 2. Ensemble Model: Weighted Probability Fusion of MobileNetV2 and EfficientNetB0
- **Individual Models**: Both pretrained on ImageNet and trained independently under identical preprocessing.
- **Fusion Mechanism**:
  P_{ensemble} = w_1 P_{MobileNetV2} + w_2 P_{EfficientNetB0}
  - Weights (w_1, w_2) are configurable via `ensemble_meta.json` (default: equal weighting).
- **Rationale**: Combines complementary feature representations to reduce variance and improve robustness, especially under class imbalance.

This ensemble approach leverages decision-level diversity for better overall performance without significantly increasing complexity.

## 📊 Results and Evaluation

All results are pre-computed and stored in the repository for immediate inspection—no retraining required. I prioritize clinically meaningful metrics over raw accuracy, including:
- Sensitivity (true positive rate) for fracture detection.
- Specificity (true negative rate) for avoiding false alarms.
- ROC-AUC for threshold-independent separability.
- Confusion matrices to reveal decision asymmetry.

### Baseline Model Results
Location: `baseline_model/Results/`
- `accuracy.png` & `loss.png`: Training/validation curves.
- `confusion_matrix.png`: Visualizes predictions at the selected threshold.
- `roc_curve.png`: ROC curve with AUC.
- `classification_report.txt`: Precision, recall, F1-score.
- `clinical_metrics.txt`: Sensitivity, specificity, NPV/PPV.
- `threshold_note.txt`: Rationale for threshold selection.

**Key Observations**: High sensitivity but some asymmetry due to imbalance, with a conservative bias toward predicting fractures.

### Ensemble Model Results
Location: `ensemble_model//Results/`
- Individual curves: `mobilenetv2_accuracy.png`, `mobilenetv2_loss.png`, `efficientnetb0_accuracy.png`, `efficientnetb0_loss.png`.
- Fused outputs: `confusion_matrix_ensemble.png`, `roc_curve_ensemble.png`, `classification_report_ensemble.txt`, `clinical_metrics_ensemble.txt`, `threshold_note_ensemble.txt`.

**Key Improvements**: Reduced false negatives and positives, improved symmetry, and greater threshold robustness—demonstrating the value of ensemble fusion.

## 🧪 Simulation-Based Inference

Running Predictions on New X-rays

- Place unseen X-ray images inside:
  test_images/

Test the models in a deployment-like scenario using:
```
python predict_ensemble_clinical.py --input test_images
```

This script:
1. Loads pretrained `.keras` models.
2. Applies preprocessing.
3. Computes fused probabilities.
4. Outputs predictions with confidence scores.

Results are saved in `inference_outputs/`. This is for simulation only—not clinical validation.

## 🧭 Code Navigation Guide

### Baseline Pipeline
- `train_baseline.py`: Handles data loading, preprocessing, weighting, model training, and evaluation.
- `predict_baseline.py`: Performs inference on new images using the trained model.

### Ensemble Pipeline
- `train_ensemble.py`: Trains MobileNetV2 and EfficientNetB0 independently.
- `predict_ensemble.py`: Inference for individual models.
- `predict_ensemble_clinical.py`: Fuses probabilities for ensemble predictions.
- `ensemble_meta.json`: Configures fusion weights.

All scripts are modular, and designed for easy extension.

## 💻 System Requirements

- **Python**: ≥ 3.9
- **Key Libraries**: TensorFlow/Keras, NumPy, scikit-learn, Matplotlib (full list in `requirements.txt`).
- **Hardware**:
  - CPU-supported; GPU recommended for faster training.
  - ≥ 8 GB RAM.

Install dependencies:
```
pip install -r requirements.txt
```

## 🔁 Reproducibility

To replicate:
1. Clone the repo:
   ```
   git clone https://github.com/YOUR_USERNAME/bone-fracture-detection.git
   cd bone-fracture-detection
   ```
2. Install dependencies (as above).
3. Download and organize the dataset from Kaggle.
4. Run training scripts or inspect pre-saved results.

I ensure reproducibility through fixed seeds, detailed documentation, and pre-computed artifacts.

## 🔬 Key Findings

- The baseline model excels in sensitivity but shows imbalance-induced bias.
- Ensemble fusion significantly reduces errors (false positives/negatives) and enhances robustness.
- Gains stem from complementary representations and variance reduction at the decision level.
- Emphasis on clinical metrics reveals practical insights beyond accuracy.

## 🚀 Future Directions

- Validate on diverse, external datasets.
- Extend to multi-anatomical or severity-based classification.
- Incorporate explainability (e.g., attention maps).
- Explore cost-sensitive learning for imbalanced scenarios.
- Conduct prospective studies with independent test sets.

## 📜 License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

---

If you find this useful, let's open an issue for discussions.
