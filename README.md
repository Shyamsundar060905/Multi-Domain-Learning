🌍 Domain Generalization for Remote Sensing Image Classification

Using Residual Adapters with a Shared ResNet-50 Backbone

📌 Overview

Deep learning models for image classification often suffer from domain shift—a drop in performance when evaluated on data from unseen or different distributions. This problem is especially severe in remote sensing, where images vary across satellites, sensors, resolutions, and geographic regions.

This project proposes a parameter-efficient multi-domain learning framework that replaces multiple domain-specific CNNs with a single unified model using residual adapters. Instead of training one heavy model per dataset, we share a frozen ResNet-50 backbone and introduce lightweight, domain-specific adapter modules for specialization.

✨ Key Features

✅ Single unified model for multiple remote sensing datasets

✅ Frozen ResNet-50 backbone pretrained on ImageNet

✅ Lightweight residual adapters for domain-specific specialization

✅ Domain-specific Batch Normalization and classifier heads

✅ Significant reduction in total parameters

✅ Competitive accuracy compared to separate models

🧠 Methodology
Architecture

Backbone: ResNet-50 (pretrained on ImageNet, frozen)

Adapters:

Bottleneck-style residual adapters

Inserted only in Layer 3 and Layer 4

Normalization: Domain-specific BatchNorm layers

Classifier: Separate fully connected head per domain

Why adapters?
Early layers learn general visual features, while deeper layers capture domain-specific semantics. Adapters allow domain customization without modifying shared parameters.

📊 Datasets Used

The model is trained and evaluated across four diverse remote sensing domains:

Dataset	Type	Classes
EuroSAT	Sentinel-2 satellite imagery	10
PatternNet	High-resolution aerial scenes	38
MLRS	Mixed aerial & satellite	46
ADVANCE	Real-world satellite images	13

All datasets are used in RGB format, with consistent preprocessing and augmentation.

⚙️ Training Strategy

Shared backbone remains frozen

Only adapters, BatchNorm layers, and classifiers are trained

Domain-aware routing selects the correct adapter and classifier

Loss: Domain-specific Cross Entropy

Optimizer: Adam (per-domain)

Learning rate: 1e-3

Weight decay: 1e-4

| Dataset    | Type                          | Classes |
| ---------- | ----------------------------- | ------- |
| EuroSAT    | Sentinel-2 satellite imagery  | 10      |
| PatternNet | High-resolution aerial scenes | 38      |
| MLRS       | Mixed aerial & satellite      | 46      |
| ADVANCE    | Real-world satellite images   | 13      |


➡️ Performance drop < 1%, despite using a single shared model.

Parameter Efficiency
Setup	Trainable Parameters
4 × Single-Domain Models	~94M
Multi-Domain (Adapters)	28.5M

✅ ~70% parameter reduction compared to training four separate models.

🔍 Feature Visualization

t-SNE analysis of penultimate-layer features shows clear domain separation, confirming that adapters successfully learn domain-specific representations while preserving a shared feature space.

🧪 Implementation Details

Framework: PyTorch

Backbone: ResNet-50

Input normalization: ImageNet mean & std

Augmentations:

Random horizontal flip

Rotation

Color jitter

🚀 Why This Matters

Scales efficiently to many domains

Suitable for resource-constrained deployment (edge, satellites, drones)

Avoids maintaining multiple heavy models

Strong foundation for continual learning and multi-modal remote sensing

🔮 Future Work

Dynamic or shared adapter selection

Continual learning methods (EWC, LwF)

Adapter fusion for related domains

Extension to multispectral, SAR, or temporal data

Transformer-based backbones with adapters

📚 References

Rebuffi et al., Learning Multiple Visual Domains with Residual Adapters, NeurIPS 2017

Rebuffi et al., Efficient Parametrization of Multi-Domain CNNs, CVPR 2018

EuroSAT, PatternNet, MLRS datasets
