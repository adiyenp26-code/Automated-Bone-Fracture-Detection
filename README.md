Automated Bone Fracture Detection Using Transfer Learning and Ensemble Deep Neural Networks

Abstract

This repository presents a structured implementation and evaluation of a deep learning–based framework for automated binary fracture classification from radiographic X-ray images. A baseline transfer learning model (MobileNetV2) is developed and compared against an ensemble architecture combining MobileNetV2 and EfficientNetB0 through weighted probability fusion.

The study evaluates model behaviour under class imbalance using clinically relevant metrics, confusion matrix analysis, and threshold-independent ROC evaluation. The objective is to examine robustness, sensitivity–specificity balance, and decision stability under controlled experimental conditions.

This work constitutes an engineering evaluation study and does not represent a clinical diagnostic system.

Problem Formulation

The task is framed as supervised binary classification:

X→{Fractured,Non-Fractured}

where 𝑋 denotes a radiographic image resized to 224×224 pixels.

The dataset exhibits class imbalance, reflecting realistic screening distributions. Consequently, evaluation extends beyond overall accuracy to include sensitivity, specificity, and negative predictive value.

Dataset

The dataset was sourced from a publicly available Kaggle repository: (https://www.kaggle.com/datasets/orvile/bone-fracture-dataset)

Images are organised as:

fractured/
non_fractured/


Preprocessing steps:

Resizing to 224×224
Pixel normalisation to [0,1]
Class weighting during optimisation
Controlled augmentation (rotation, translation, zoom, horizontal flipping)
The dataset is not included in this repository due to licensing and size constraints.

Methodology

Baseline Model: MobileNetV2

The baseline architecture employs MobileNetV2 as a pretrained convolutional backbone initialised with ImageNet weights.

Configuration:

Frozen convolutional base during initial training
Global Average Pooling
Dense layer (ReLU activation)
Dropout regularisation
Two-class softmax output

Training utilised categorical cross-entropy loss with the Adam optimiser. Class weighting was incorporated to mitigate imbalance-induced bias.

Ensemble Model: Weighted Probability Fusion

To improve robustness, a second backbone (EfficientNetB0) was trained independently under identical preprocessing and optimisation conditions.

The ensemble operates at the probability level:

#​

P
EfficientNetB0
	​


Weights were selected based on validation stability and discriminative behaviour.

The ensemble strategy avoids feature concatenation and preserves independent representational biases of each architecture.
