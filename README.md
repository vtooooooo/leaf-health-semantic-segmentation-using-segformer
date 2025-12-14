# 🌿 Leaf Health Semantic Segmentation Using SegFormer

This repository contains an end-to-end deep learning pipeline for **semantic segmentation of leaf health**, classifying each pixel into **background**, **healthy leaf**, or **dry/damaged leaf**.
The project fine-tunes **SegFormer-B0**, a transformer-based segmentation architecture, with extensive augmentation, hyperparameter tuning, and a combined **CE + Dice loss** objective to handle class imbalance.

---

## 📌 Project Overview

* Built a semantic segmentation model to analyze leaf health at **pixel level**, supporting early plant disease detection.
* Used **SegFormer-B0 (MiT encoder + MLP decoder)** for efficient multi-scale feature extraction.
* Introduced a **modern technique (Dice Loss)** to improve performance on minority classes (dry leaf tissue).
* Achieved strong quantitative and qualitative improvements through tuning and loss engineering.

---

## 📁 Dataset

* ~200 real leaf images captured using a mobile camera.
* Pixel-level masks created using CVAT.
* Mask labels remapped to `{0: background, 1: healthy_leaf, 2: dry_leaf}`.
* All images resized to **512×512**.
* Splits:

  * **80%** Training
  * **10%** Validation
  * **10%** Test
* Example dataset structure (from report, page 4):

  * Image 027 → Mask 027
  * Image 193 → Mask 193
  * Image 161 → Mask 161


---

## 🛠️ Preprocessing & Augmentation

Strong augmentations applied using **Albumentations**:

* Affine transforms
* Horizontal/vertical flips
* Random brightness/contrast
* Gaussian noise + motion blur
* Grid + elastic distortions
* Coarse dropout
* Normalization (ImageNet mean & std)

These steps significantly improved generalization on a small dataset.

---

## 🧠 Model Architecture

SegFormer-B0 with:

* **MiT transformer encoder** (hierarchical, multi-scale features)
* **Lightweight MLP decoder**
* Final head modified to output **3 classes**
* Logits upsampled to match mask resolution

Architecture visual (from PPT, slide 8):


---

## 🎯 Loss Functions

### 1. **Cross-Entropy Loss (CE)**

Used for initial fine-tuning.

### 2. **Dice Loss** (recent technique)

Improves overlap and is effective for imbalanced classes.

### 3. **Combined Loss**

```
Total_Loss = 0.5 * CE + 0.5 * Dice
```

This significantly improved dry-leaf segmentation (see report page 7).


---

## 🔍 Hyperparameter Tuning

We performed **Random Search** across:

* Learning rate: `1e-5 → 5e-4`
* Batch size: `4, 8, 16`
* Weight decay: `0, 0.01, 0.05`
* Optimizers: `AdamW`, `SGD`

**Best configuration** (used for fine-tuning):

* LR = **1e-4**
* Batch = **8**
* Weight Decay = **0.05**
* Optimizer = **AdamW**

(From report page 5, and PPT slide 12.)
 

---

## 📈 Training & Validation Performance

From CE-only fine-tuning (report page 6):

* Training loss consistently decreased
* Validation loss dropped steadily
* Validation IoU improved across epochs

These charts are included in this repo and shown in the report.


---

## 🧪 Results

### **Quantitative**

* CE fine-tuning IoU ≈ **0.83**
* Strong segmentation for both healthy and dry leaf regions
* Dice loss further improved:

  * Dry leaf true positives ↑
  * False negatives ↓
  * Sharper boundaries
    (Report pages 7–8)


### **Qualitative**

* Before training: model fails to segment properly (PPT slide 11)
* After CE finetuning: clean segmentation on healthy/dry regions
* After CE + Dice: best minority-class performance


---

## 🚀 Inference

Example inference workflow is included in the notebooks:

1. Load trained model
2. Preprocess input image
3. Predict mask
4. Visualize segmentation or overlay on original image

Also supports running predictions from Google Drive.

---

## 📦 Repository Structure

```
├── data/                 # (Optionally store dataset paths or mounting info)
├── models/               # Saved SegFormer models
├── notebooks/            # Training, tuning, and inference notebooks
├── scripts/              # Utility functions (preprocessing, metrics, prediction)
├── README.md             # Project documentation
```

---

## 🔗 Links

Dataset, notebook, and model links are available in the report (page 9).


---

## 🧑‍🤝‍🧑 Team

* Lakshman Rajith
* Vittu Darshan
* Yogananda Manjunath

(Project title & team list from PPT slide 2.)


---

## 📝 References

Key papers on SegFormer, Dice Loss, ViT, FCN, and Albumentations are included in the report bibliography (page 8).


---

If you want, I can also generate:
✅ A GitHub cover image
✅ Badges (Python version, Model size, License, etc.)
✅ A short tagline for the repository
Just tell me!
