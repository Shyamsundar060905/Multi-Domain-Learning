# 🌍 Domain Generalization for Remote Sensing Image Classification
### Using Residual Adapters with a Shared ResNet-50 Backbone

---

## 📌 Overview
Deep learning models for image classification often suffer from **domain shift**—a sharp drop in performance when evaluated on data from unseen or different distributions. This problem is especially severe in **remote sensing**, where images vary across satellites, sensors, resolutions, and geographic regions.

This project proposes a **parameter-efficient multi-domain learning framework** that replaces multiple domain-specific CNNs with a **single unified model** using **residual adapters**. Instead of training one heavy model per dataset, we share a frozen ResNet-50 backbone and introduce lightweight, domain-specific adapter modules for specialization.

---

## ✨ Key Features
- Single unified model for multiple domains  
- Frozen ResNet-50 backbone pretrained on ImageNet  
- Lightweight residual adapters for domain specialization  
- Domain-specific Batch Normalization layers  
- Separate classifier heads per dataset  
- Significant reduction in total trainable parameters  

---

## 🧠 Methodology

### Architecture
- **Backbone:** ResNet-50 (ImageNet pretrained, frozen)
- **Adapters:** Bottleneck-style residual adapters
- **Insertion Layers:** Layer 3 and Layer 4 only
- **Normalization:** Domain-specific BatchNorm
- **Classifier:** One fully connected head per domain

Early layers learn domain-invariant features, while deeper layers adapt to domain-specific semantics via adapters.

---

## 📊 Datasets Used
The model is trained and evaluated across four diverse remote sensing datasets:

| Dataset     | Description                          | Classes |
|------------|--------------------------------------|---------|
| EuroSAT    | Sentinel-2 satellite imagery         | 10      |
| PatternNet | High-resolution aerial scenes        | 38      |
| MLRS       | Mixed aerial & satellite imagery     | 46      |
| ADVANCE    | Real-world satellite images          | 13      |

All datasets are used in **RGB format** with consistent preprocessing.

---

## ⚙️ Training Strategy

### Optimization
- **Loss:** Domain-specific Cross Entropy
- **Optimizer:** Adam (separate optimizer per domain)
- **Learning Rate:** `1e-3`
- **Weight Decay:** `1e-4`

### Training Rules
- Backbone parameters remain frozen
- Only adapters, BatchNorm layers, and classifier heads are trained
- Domain-aware routing selects the appropriate adapter stack

---

## 📈 Results

### Classification Accuracy

| Dataset     | Single-Domain (%) | Multi-Domain (%) |
|------------|------------------|------------------|
| EuroSAT    | 97.64            | 96.83            |
| PatternNet | 99.17            | 98.95            |
| MLRS       | 94.57            | 94.41            |
| ADVANCE    | 93.56            | 93.24            |

> Performance drop is **less than 1%**, despite using a single shared model.

---

### Parameter Efficiency

| Setup | Trainable Parameters |
|------|----------------------|
| 4 × Single-Domain Models | ~94M |
| **Multi-Domain (Adapters)** | **28.5M** |

This results in **~70% reduction** in total parameters.

---

## 🔍 Feature Visualization
t-SNE visualization of penultimate-layer features shows **clear separation between domains**, confirming that adapters learn domain-specific representations while preserving shared features.

---

## 🧪 Implementation Details
- Framework: **PyTorch**
- Backbone: **ResNet-50**
- Input normalization: ImageNet mean & std
- Data augmentation:
  - Random horizontal flips
  - Rotations
  - Color jitter

---

## 🚀 Why This Project Matters
- Scales efficiently to many domains
- Reduces memory and deployment cost
- Suitable for edge devices and large-scale systems
- Strong foundation for continual and multi-modal learning

---

## 🔮 Future Work
- Dynamic or shared adapter selection
- Continual learning (EWC, LwF)
- Adapter fusion across related domains
- Extension to multispectral, SAR, and temporal data
- Transformer backbones with adapters

---

## 👨‍💻 Authors
**Shyamsundar Paramasivam**  
**Yash Singhal**  
Department of Computer Science & Engineering  
LNMIIT Jaipur

---

## 📚 References
- Rebuffi et al., *Learning Multiple Visual Domains with Residual Adapters*, NeurIPS 2017  
- Rebuffi et al., *Efficient Parametrization of Multi-Domain CNNs*, CVPR 2018  
