# Tomato Leaf Disease Classification on Real Data (PlantVillage)

## Overview

A Convolutional Neural Network trained from scratch on real photographs of tomato leaves to classify them as healthy, bacterial spot, or late blight — an early-detection tool for crop disease. Farmers often detect crop diseases too late, once damage has already spread across a field; automated classification from a single leaf photo lets them act before it spreads further.

## Dataset

- **Source:** [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) (Hughes & Salathé, 2015; Mohanty et al., 2016), obtained via git sparse-checkout from the official GitHub repository.
- **5,627 real images** across 3 classes: `Bacterial_spot` (2,127), `Late_blight` (1,909), `healthy` (1,591) — scoped from the full 38-class PlantVillage dataset (tomato-only) to keep CPU training time practical.
- Original images 256x256 RGB, resized to 80x80 for training.

## Data Preparation

- Images loaded and resized to 80x80, pixel values normalized to [0,1].
- Class labels encoded 0/1/2 (alphabetical folder order: `Bacterial_spot`, `Late_blight`, `healthy`).
- Data augmentation (random flip, rotation, zoom, contrast) applied during training only.
- Stratified 70% / 15% / 15% split (train / validation / test).
- Images loaded fully into memory as NumPy arrays rather than a streaming `tf.data` pipeline — a deliberate speed trade-off at this dataset size.

## Exploratory Data Analysis

- Class balance check and visual sample inspection per class.
- The 3 classes are reasonably balanced (2,127 / 1,909 / 1,591), avoiding a severe class-imbalance problem.
- Visually distinct disease signatures: bacterial spot shows small dark spots, late blight shows large brown patches, healthy leaves are uniformly green.

## Model

- A single CNN built from scratch (3 convolutional blocks + dense layers) — no pretrained/transfer-learning weights were used, since ImageNet weights could not be downloaded in this environment.
- Adam optimizer, image size 80x80, batch size 64, up to 12 epochs, early stopping (patience=3, monitoring validation loss).
- No systematic hyperparameter search was performed.

## Results

**Test accuracy:** 85.7% | **Test loss:** 0.411

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Bacterial_spot | 0.96 | 0.87 | 0.91 |
| Late_blight | 0.89 | 0.73 | 0.80 |
| healthy | 0.74 | 1.00 | 0.85 |

**Confusion matrix highlights:** Bacterial_spot — 276 correct, 25 → Late_blight, 18 → healthy. Late_blight — 10 → Bacterial_spot, 210 correct, 67 → healthy. Healthy — 1 → Bacterial_spot, 0 → Late_blight, 238 correct.

Training curves showed overfitting starting around epoch 2 (validation loss rising while training loss kept falling); early stopping correctly restored the best-validation-epoch weights rather than the final epoch's weights.

**Most common error:** confusing `Late_blight` with `healthy` (67 cases) — plausible, since some early-stage late blight leaves are still mostly green. The `healthy` class shows perfect recall (1.00) but lower precision (0.74), meaning some diseased leaves are misclassified as healthy.

## Tech Stack

- **Python**
- **TensorFlow/Keras** — CNN model
- **NumPy, Pillow** — image loading and processing
- **scikit-learn** — train/test split, classification report, confusion matrix
- **Matplotlib** — visualizations
- **Jupyter Notebook**

## Files

- `Tomato_Disease_CNN_RealData.ipynb` — full notebook
- `plantvillage_data.zip` — the real dataset (tomato subset)
- `sample_leaves.png` — one sample image per class
- `training_curves.png` — training/validation accuracy and loss over epochs
- `confusion_matrix.png` — confusion matrix on the held-out test set
- `sample_predictions.png` — model predictions on real test images

## Limitations

- Only 3 of the 38 PlantVillage classes were used (tomato-related only), to keep CPU training time practical.
- No transfer learning was used, due to no network access to pretrained weights.
- Only a single CNN architecture was tried; no comparison models were built.
- Real-photo accuracy (85.7%) is notably lower than the ~99% typically seen on synthetic-image versions of similar datasets — a useful illustration of the real-vs-synthetic-data accuracy gap.

## Possible Next Steps

- Try transfer learning (e.g. MobileNet, EfficientNet) once pretrained weights are accessible.
- Extend to more of the 38 PlantVillage classes, or to other crops.
- Investigate the healthy/Late_blight confusion further with targeted data augmentation or a higher-resolution input.
