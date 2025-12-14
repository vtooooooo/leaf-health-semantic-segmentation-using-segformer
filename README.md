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

---

## 📈 Training & Validation Performance

From CE-only fine-tuning (report page 6):

* Training loss consistently decreased
* Validation loss dropped steadily
* Validation IoU improved across epochs


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

## 🔗 Links

* Dataset: [Download Dataset](https://unhnewhaven-my.sharepoint.com/personal/vdars1_unh_newhaven_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fvdars1%5Funh%5Fnewhaven%5Fedu%2FDocuments%2FDeep%20Learning%20Project%2Fdedata&ga=1)
* Notebook: [Notebook](https://colab.research.google.com/drive/18htI1d1acHcNqF-s0VMoTZ0xUqnZFkNC?usp=sharing)
* Working Link: [Live Working](https://huggingface.co/spaces/darshanrdas/leaf_segmentation3)


---

## 📝 References

* E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers,” arXiv preprint arXiv:2105.15203, 2021. https://arxiv.org/abs/2105.15203
* F. Milletari, N. Navab, and S. A. Ahmadi, “V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation,” arXiv preprint arXiv:1606.04797, 2016. https://arxiv.org/abs/1606.04797
* J. Long, E. Shelhamer, and T. Darrell, “Fully Convolutional Networks for Semantic Segmentation,” arXiv preprint arXiv:1411.4038, 2015. https://arxiv.org/abs/1411.4038
* A. Dosovitskiy et al., “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale,” arXiv preprint arXiv:2010.11929, 2020. https://arxiv.org/abs/2010.11929
* A. Buslaev, V. I. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, and A. A. Kalinin, “Albumentations: Fast and Flexible Image Augmentations,” Information, vol. 11, no. 2, 2020. https://www.mdpi.com/2078-2489/11/2/125
* I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” arXiv preprint arXiv:1711.05101, 2019. https://arxiv.org/abs/1711.05101

---
