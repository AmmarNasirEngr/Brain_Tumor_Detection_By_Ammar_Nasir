<div align="center">

## 🧠 Brain Tumor Detection with YOLOv8 
#Prepared by Ammar Nasir
This notebook fine-tunes a YOLOv8s model on an MRI brain tumor dataset to detect and localize tumors in medical images. The pipeline covers: dependency setup → dataset loading → model training → validation → inference on new MRI scans → ONNX export for deployment.


*Fine-tuning YOLOv8s on MRI scans for real-time brain tumor localization*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmmarNasirEngr/brain-tumor-detection/blob/main/Brain_Tumor_Detection_By_Ammar_Nasir.ipynb)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
&nbsp;
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00CFFF?style=flat)
&nbsp;
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat&logo=pytorch&logoColor=white)
&nbsp;
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=flat&logo=nvidia&logoColor=white)

</div>

---

## 📌 Overview

This project fine-tunes a **YOLOv8s** model (pretrained on COCO) on a labeled MRI brain tumor dataset to **detect and localize tumors across 3 classes** in real time. Built and trained in Google Colab on a Tesla T4 GPU.

The pipeline is fully reproducible — from dataset download to ONNX export — structured in clean, phase-by-phase cells that anyone can follow and run end-to-end.

---

## 🗂️ Pipeline

```
Phase 1  →  Install Dependencies        ultralytics · gdown · roboflow
Phase 2  →  Imports
Phase 3  →  GPU Check
Phase 4  →  Dataset Setup               Mount Drive · set extract_dir
Phase 5  →  Dataset Verification        File counts per split
Phase 6  →  Load YOLOv8s               Pretrained on COCO (yolov8s.pt)
Phase 7  →  Fine-Tune                   5 epochs · AdamW · fraction=0.1
Phase 8  →  Validate                    model.val() on 1,980 images
Phase 9  →  Visualize Results           Confusion matrix · PR · F1 curves
Phase 10 →  Inference                   Predict on test MRI images
Phase 11 →  Export to ONNX             Deployment-ready model
```

---

## 📦 Dataset

| Property | Details |
|---|---|
| Format | YOLOv8 (bounding boxes + labels) |
| Image Size | 640 × 640 px |
| Classes | `label0` · `label1` · `label2` |
| Train Images | 693 |
| Validation Images | 1,980 |
| Source | Google Drive |
| Drive File ID | `1R-Ckl-CwGD4Xuj6WEEwUQkdO8uCgB19E` |

> Dataset is downloaded automatically in the notebook via `gdown`. No manual download required.

---

## ⚙️ Model & Hyperparameters

| Parameter | Value |
|---|---|
| Architecture | YOLOv8s |
| Pretrained On | COCO |
| Parameters | 11.1M |
| GFLOPs | 28.4 |
| Epochs | 5 |
| Image Size | 640 |
| Batch Size | 16 |
| Optimizer | AdamW (`lr=0.001`, `momentum=0.937`) |
| Data Fraction | 0.1 (10% — baseline run) |
| Hardware | Tesla T4 · 14GB VRAM |
| Training Time | ~0.054 hours |

---

## 📊 Results

> Validated on **1,980 images** · **4,381 instances** · `best.pt` checkpoint

| Metric | All Classes | label0 | label1 | label2 |
|---|:---:|:---:|:---:|:---:|
| **Precision** | 0.408 | 0.341 | 0.488 | 0.401 |
| **Recall** | 0.357 | 0.211 | 0.602 | 0.257 |
| **mAP@0.5** | 0.322 | 0.193 | 0.548 | 0.227 |
| **mAP@0.5:0.95** | 0.126 | 0.062 | 0.244 | 0.074 |

**Inference speed:** `0.2ms` preprocess · `4.3ms` inference · `2.5ms` postprocess per image on T4.

> ⚠️ **Baseline run only.** `fraction=0.1` + 5 epochs is an intentional lightweight experiment — `label1` already hits mAP@0.5 of **0.548** and Recall of **0.602**. Full-data training at 50–100 epochs is expected to push all three classes significantly higher.

---

## 🚀 Quickstart

### 1. Open in Colab

Click the badge at the top, or upload `Brain_Tumor_Detection_By_Ammar_Nasir.ipynb` manually.  
**Runtime → Change runtime type → T4 GPU** before running.

### 2. Mount Your Drive

The dataset lives in Google Drive. In Phase 4, set:

```python
extract_dir = '/content/drive/MyDrive/Brain Tumor Detection/unzipped_brain_tumor_dataset'
```

Adjust this path if your Drive folder name differs.

### 3. Run All Phases

Each phase is self-contained. Run top-to-bottom. Training takes ~3–4 minutes on T4.

### 4. Test on Your Own MRI

In Phase 10, update:

```python
test_image_path = '/content/drive/MyDrive/.../your_mri_image.jpg'
```

---

## 🗃️ Output Structure

After training, results are saved to:

```
runs/
└── detect/
    └── brain_tumor_project/
        └── yolov8_brain_tumor/
            ├── weights/
            │   ├── best.pt          ← Best checkpoint (use this for inference)
            │   └── last.pt
            ├── results.png          ← Loss & metric curves across epochs
            ├── confusion_matrix.png
            ├── BoxPR_curve.png
            ├── BoxF1_curve.png
            ├── val_batch*.jpg
            └── train_batch*.jpg
```

---

## 📤 Export

The notebook exports the trained model to **ONNX** for production deployment:

```python
model.export(format="onnx")
```

Compatible with **ONNX Runtime**, **TensorRT**, and edge inference pipelines.

---

## 🧰 Requirements

```bash
pip install ultralytics gdown roboflow
```

PyTorch with CUDA is recommended. CPU-only inference works but is significantly slower.

| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.10 + CUDA 12.8 |
| Ultralytics | 8.4.51 |
| gdown | latest |

---

## 👤 Author

**Ammar Nasir**  
AI Engineer

[![GitHub](https://img.shields.io/badge/GitHub-AmmarNasirEngr-181717?style=flat&logo=github)](https://github.com/AmmarNasirEngr)

---

<div align="center">

*If this project helped you, consider giving it a ⭐*

</div>
