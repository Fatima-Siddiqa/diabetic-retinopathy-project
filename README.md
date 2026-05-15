# DR-Scope: Diabetic Retinopathy Grading — Extended

**Course:** CS-4112 Deep Learning | Assignment 3  
**Institution:** FAST-NUCES, Islamabad — Department of Artificial Intelligence & Data Science  
**Authors:** Eman Ali (23i-2564) · Fatima Siddiqa (23i-2543) · Mariam Shaiq (23i-3250)  
**Supervisor:** Dr. Zohair Ahmed

---

## Overview

This repository contains the full implementation for Assignment 3, which extends the Assignment 2 reproduction of [Anand et al. (2024)](https://doi.org/10.3389/frai.2024.1396160) — *Smart Grading of Diabetic Retinopathy: An Intelligent Recommendation-Based Fine-Tuned EfficientNetB0 Framework* — with two principled novelties:

1. **Ben Graham Retinal Contrast Enhancement** — a preprocessing step that amplifies microaneurysms, haemorrhages, and neovascular features by subtracting a Gaussian-blurred version of the image from itself.
2. **CORAL Ordinal Loss** — replaces the standard cross-entropy loss with the Consistent Rank Logits framework (Cao et al., 2020), which decomposes the 5-grade DR severity problem into 4 ordered binary sub-tasks, guaranteeing rank-consistent predictions and structurally penalising clinically dangerous grade skips.

Cross-dataset generalisation is also evaluated on **Messidor-2** to assess domain robustness beyond APTOS 2019.

---

## Repository Structure

```
project-root/
├── README.md
├── requirements.txt
├── train.py               # Training entry point (baseline or CORAL mode)
├── inference.py           # Single-image or batch inference
├── config.yaml            # All hyper-parameters and paths
├── Efficient Net extended with CORAL +BG.ipynb        # Main notebook (Sections 1–13)
│
├── data/
│   └── sample_data.csv    # 8 sample rows from APTOS 2019 (id_code, diagnosis)
│
├── notebooks/
│   └── 01_inference_demo.ipynb   # Step-by-step inference + results visualisation
│
├── src/
│   ├── model.py           # EfficientNet-B0/B2, CORALHead, GradCAM, coral_loss
│   ├── dataset.py         # DRDataset, DRDatasetBG, Ben Graham transforms
│   └── utils.py           # train_one_epoch, evaluate, compute_metrics, etc.
│
├── results/
│   ├── baseline_metrics.json    # Reproduced EfficientNet-B0 baseline metrics
│   ├── improved_metrics.json    # CORAL model metrics (APTOS + Messidor-2)
│   └── training_log.csv         # Epoch-by-epoch loss / accuracy / QWK
│
└── checkpoints/
    └── best_model.pth     # Best checkpoint by validation accuracy
```
---

## Notebook Sections

| Section | Description |
|---------|-------------|
| 1 | Mount Google Drive & download APTOS 2019 dataset via `gdown` |
| 2 | Install libraries & imports |
| 3 | Configuration (`Config` class — all hyperparameters in one place) |
| 4 | Exploratory Data Analysis — class distribution, sample images |
| 5 | Dataset & DataLoaders — `APTOSDataset`, stratified split, `WeightedRandomSampler` |
| 6 | Model architecture — EfficientNet-B0 + custom classification head |
| 7 | Training — two-stage fine-tuning (Stage 1: head-only; Stage 2: full fine-tune) |
| 8 | Evaluation & results — metrics, training curves, confusion matrix |
| 9 | GradCAM explainability — heatmaps for all 5 DR grades |
| 10 | **[A3]** Ben Graham preprocessing — `APTOSDatasetBG`, enhanced DataLoaders |
| 11 | **[A3]** CORAL model & training — `CORALHead`, `coral_loss`, two-stage training |
| 12 | **[A3]** Final comparison — B2 baseline vs. CORAL + Ben Graham (QWK, F1, AUC) |
| 13 | **[A3]** Messidor-2 cross-dataset evaluation — zero-shot generalisation |

---

## Datasets

### APTOS 2019 Blindness Detection
- **Source:** [Kaggle — APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection)
- **Size:** 3,662 labelled retinal fundus images
- **Labels:** 5-class DR severity (0: No DR → 4: Proliferative DR)
- **Split:** 80% train (2,929) / 20% validation (733), stratified

| Grade | Severity | Count | % |
|-------|----------|-------|---|
| 0 | No DR | 1805 | 49.3% |
| 1 | Mild | 370 | 10.1% |
| 2 | Moderate | 999 | 27.3% |
| 3 | Severe | 193 | 5.3% |
| 4 | Proliferative | 295 | 8.1% |

### Messidor-2
- **Source:** [Messidor Project](https://www.adcis.net/en/third-party/messidor2/)
- **Size:** 1,748 retinal fundus images with adjudicated DR severity grades
- **Use:** Zero-shot cross-dataset evaluation only — no Messidor-2 images used during training

---

## Model Architecture

**Backbone:** EfficientNet-B0 pretrained on ImageNet

**B2 Baseline Head (Cross-Entropy):**
```
Dropout(p=0.4) → Linear(1280 → 512) → ReLU → Dropout(p=0.3) → Linear(512 → 5)
```

**CORAL Head (Ordinal Loss):**
```
Dropout(p=0.4) → Linear(1280 → 512) → ReLU → Dropout(p=0.3) → CORALHead(512 → 4)
```
The `CORALHead` uses a single shared weight vector `Linear(512 → 1, bias=False)` plus 4 learned bias scalars — one per ordinal threshold. Predictions are rank-consistent by construction.

---

## Key Methods

### Ben Graham Preprocessing
```python
def ben_graham_enhance(pil_img, sigma=30):
    img = np.array(pil_img.convert("RGB"))
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    enhanced = cv2.addWeighted(img, 4, blurred, -4, 128)
    return Image.fromarray(np.clip(enhanced, 0, 255).astype(np.uint8))
```
Applied to all training and validation images before the standard augmentation pipeline.

### CORAL Loss
```python
def coral_loss(outputs, targets, num_classes=5):
    sets = [targets > i for i in range(num_classes - 1)]
    loss = sum(
        F.binary_cross_entropy(outputs[:, i], sets[i].float())
        for i in range(num_classes - 1)
    )
    return loss / (num_classes - 1)
```

### Grade Prediction from CORAL outputs
```python
predicted_grade = (outputs > 0.5).sum(dim=1)  # guaranteed in {0, 1, 2, 3, 4}
```

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Image size | 224 × 224 |
| Batch size | 16 |
| Stage 1 LR | 1e-3 |
| Stage 2 LR | 1e-5 |
| Weight decay | 1e-4 |
| Dropout (head) | 0.4 |
| Dropout (intermediate) | 0.3 |
| Optimiser | AdamW |
| LR Scheduler | Cosine Annealing |
| Early stopping patience | 5 epochs |
| Stage 1 epochs | 5 (head only) |
| Stage 2 max epochs | 15 (full fine-tune) |
| Class imbalance | WeightedRandomSampler |
| Random seed | 42 |

**Stage 1** freezes the backbone and trains only the custom head, preventing destruction of pretrained ImageNet features by the randomly initialised head at a high learning rate.  
**Stage 2** unfreezes the full network and fine-tunes end-to-end at a reduced learning rate with cosine annealing.

---

## Results

### APTOS 2019 — B2 Baseline vs. CORAL + Ben Graham

| Metric | B2 Baseline | CORAL + BG |
|--------|-------------|------------|
| Accuracy | 0.7572 | 0.7418 |
| Macro F1 | 0.6059 | 0.6341 |
| Macro Precision | 0.5973 | 0.6124 |
| Macro Recall | 0.6388 | 0.6712 |
| **QWK** | **0.7814** | **0.8237** |
| AUC (Macro OvR) | 0.9138 | 0.9204 |

### Per-Class — CORAL + BG (APTOS 2019 Validation)

| Grade | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| No DR (0) | 0.93 | 0.94 | 0.93 | 361 |
| Mild (1) | 0.48 | 0.58 | 0.53 | 74 |
| Moderate (2) | 0.74 | 0.61 | 0.67 | 200 |
| Severe (3) | 0.42 | 0.72 | 0.53 | 39 |
| Proliferative (4) | 0.48 | 0.47 | 0.47 | 59 |

### Cross-Dataset — CORAL + BG on Messidor-2

| Metric | APTOS 2019 | Messidor-2 |
|--------|-----------|------------|
| Accuracy | 0.7418 | 0.6631 |
| Macro F1 | 0.6341 | 0.5573 |
| QWK | 0.8237 | 0.7104 |
| AUC | 0.9204 | 0.8791 |

---

## Setup & Usage

### Requirements
```
torch
torchvision
scikit-learn
pandas
numpy
matplotlib
seaborn
pillow
opencv-python
gdown
```

Install all dependencies:
```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn pillow opencv-python gdown
```

### Running on Google Colab (Recommended)

1. Open `Assignment_3_updated.ipynb` in Google Colab
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Mount Google Drive when prompted (checkpoints are saved to Drive)
4. Run all cells in order — dataset is downloaded automatically via `gdown` in Section 1
5. Checkpoints are saved to `Config.OUTPUT_DIR` (`/content/drive/MyDrive/aptos_outputs/`)

### Running Locally

1. Download APTOS 2019 from [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection) and place images in `data/aptos2019/train_images/`
2. Update `Config.DATA_DIR`, `Config.IMG_DIR`, and `Config.OUTPUT_DIR` in Section 3
3. Run sections sequentially. To run only the A3 extensions, Sections 10–13 depend on the trained B2 model from Sections 5–8

---

## Evaluation Metrics

| Metric | Why Used |
|--------|----------|
| **QWK** | Primary metric — penalises large ordinal errors quadratically; official APTOS 2019 competition metric |
| Macro F1 | Equal weight across all 5 classes; not inflated by class imbalance |
| Macro Recall | Critical for patient safety — tracks severe DR detection rate |
| AUC (OvR) | Ranking ability across all classes regardless of threshold |
| Accuracy | Reported for comparability with original paper only |

> **Note:** Accuracy is intentionally secondary. A model that predicts Grade 0 for every image achieves ~49% accuracy on APTOS 2019 while being clinically useless. QWK is immune to this pathology.

---

## GradCAM Explainability

GradCAM visualisations are generated in Sections 9 (B2 model) and 11 (CORAL model) using the final convolutional block of EfficientNet-B0. For each of the 5 DR grades, one representative validation image is displayed as a three-panel figure: original fundus image · raw heatmap · blended overlay.

---

## References

- Anand et al. (2024). *Smart Grading of Diabetic Retinopathy: An Intelligent Recommendation-Based Fine-Tuned EfficientNetB0 Framework.* Frontiers in AI, vol. 7. https://doi.org/10.3389/frai.2024.1396160
- Cao, Mirjalili & Raschka (2020). *Rank Consistent Ordinal Regression for Neural Networks.* Pattern Recognition Letters, 140, 325–331.
- Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML 2019.
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV 2017.
- Graham (2015). *Kaggle Diabetic Retinopathy Detection Competition Report.* University of Warwick.
- APTOS 2019 Dataset: https://www.kaggle.com/c/aptos2019-blindness-detection
- Messidor-2 Dataset: https://www.adcis.net/en/third-party/messidor2/

---
