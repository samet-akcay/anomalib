---
title: "Anomalib v1: A Comprehensive Redesign of the Open-Source Visual Anomaly Detection Library"
tags:
  - Python
  - anomaly detection
  - unsupervised learning
  - zero/few-shot learning
  - deep learning
  - vision language models
  - computer vision
authors:
  # Order not final
  - name: Samet Akcay
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Dick Ameln
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Ashwin Vaidya
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Harim Kang
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Intel
    index: 1
date: 15 July 2024
bibliography: paper.bib
---

# Summary

Anomalib is a novel visual anomaly detection and localization library. This open-source library provides state-of-the-art algorithms from the literature and a set of tools to design custom anomaly detection algorithms via a plug-and-play approach. With a focus on reproducibility and modularity, Anomalib offers:

- Off-the-shelf state-of-the-art anomaly detection algorithms that achieve top performance on benchmarks
- Components to design custom algorithms tailored to specific needs
- Additional tools including experiment trackers, visualizers, and hyper-parameter optimizers
- Inference capabilities such model optimization and quantization for real-time deployment

# Statement of Need

Anomaly detection is a growing research area in machine learning, focusing on distinguishing between normal and anomalous samples in datasets. Unlike supervised approaches, anomaly detection algorithms rely solely on normal samples during training, making them well-suited for real-world applications in industrial, medical, and security domains where clearly defined anomalous classes are often lacking.

The increasing number of publications and techniques in anomaly detection highlights the need for a unified library for benchmarking algorithms. While supervised tasks have seen various libraries emerge over the past years [@mmdetection; @detectron], the unsupervised anomaly detection domain lacks similar comprehensive efforts.

Existing anomaly detection libraries often focus on single algorithms, lack performance optimizations, or do not include deep learning techniques [@Zhao2019PyOD:Detection]. This makes it challenging to utilize these implementations for out-of-the-box comparison of the most recent algorithms on a given dataset.

Anomalib addresses these issues by providing a comprehensive, modular, and extensible framework for anomaly detection tasks, facilitating both research and practical applications in this crucial field.

# Design and Implementation

Anomalib is built on four key design principles that guide its development and functionality:

1. **Reproducibility**: Anomalib prioritizes the replication of published anomaly detection results, ensuring trustworthy findings and enabling objective comparisons between models.

2. **Extensibility**: The library is designed to be customizable and expandable, allowing users to tailor functionality to their specific needs.

3. **Modularity**: Anomalib's modular design allows users to create custom algorithms by combining pre-existing functional building blocks.

4. **Real-Time Performance**: The library focuses on the efficient deployment of models, providing interfaces for real-time inference using either GPU (via PyTorch [@PyTorch]) or CPU (via OpenVINO [@OpenVINO]).

Anomalib's architecture facilitates a complete workflow from data processing to model deployment through several key components:

1. **Data Handling**: Supports various image, video, and 3D datasets, including custom datasets and synthetic anomaly generation.

2. **Pre-processing**: Uses the `torchvision v2` [@torchvision] for image transformations and implements image tiling [@Rolih_2024_CVPR] for high-resolution images.

3. **Model Architecture**: Divided into modular components (e.g., feature extraction, dimensionality reduction) and complete algorithms (e.g., PatchCore [@Roth2021TowardsDetection], PADIM [@Defard2021PaDiM:Localization]).

4. **Post-processing**: Includes thresholding, normalization, and visualization of anomaly detection results.

5. **Deployment**: Supports model optimization and deployment using OpenVINO and ONNX formats.

Anomalib uses Lightning [@Falcon2020PyTorchLightning] for efficient model implementation and training, ensuring compatibility with a wide range of deep learning tools and practices.

# Usage and Tools

Anomalib provides a comprehensive set of tools and interfaces for ease of use and flexibility in anomaly detection tasks.

## Command Line Interface (CLI)

Anomalib offers a powerful CLI built on top of Lightning CLI, with commands for training, testing, inference, export, hyperparameter optimization, and benchmarking:

```bash
anomalib fit --config model_config.yaml
anomalib test --ckpt_path model.ckpt --config model_config.yaml
anomalib predict --weights model.bin --metadata metadata.json
anomalib export openvino --input_model model.onnx
anomalib hpo --backend comet --config sweep.yaml
anomalib benchmark --config benchmark_config.yaml
```

## Python API

For more flexibility, Anomalib can be used via its Python API:

```python
from anomalib.data import Mvtec
from anomalib.models import Patchcore
from anomalib.engine import Engine

datamodule = Mvtec(category="bottle")
model = Patchcore(backbone="resnet18")
engine = Engine()

engine.fit(datamodule, model)
engine.test(datamodule, model)
engine.export(model, "openvino")

```

## Additional Tools

Anomalib also provides:

- Hyperparameter optimization support using Comet [@CometML] and Weights & Biases [@Biewald2020ExperimentBiases]
- Benchmarking capabilities for comparing models and datasets
- A Gradio-based user interface for easy model inference

# Performance

Anomalib has been shown to reproduce state-of-the-art performance on standard benchmarks such as MVTec AD [@MVTec] and VisA [@VisA], which demonstrates its effectiveness and reliability for visual anomaly detection tasks. The library's benchmarking tools allow users to easily compare different models across various datasets to facilitate research and development in the field.

# Conclusions

Anomalib is a comprehensive, open-source library for deep learning-based visual anomaly detection. It provides a unified framework for training, benchmarking, deploying, and developing anomaly detection models. The library's design emphasizes reproducibility, extensibility, and real-time performance, making it suitable for both research and practical applications.

As an open-source project, Anomalib welcomes community contributions and aims to continuously update with the latest state-of-the-art techniques in the field. Future work will focus on extending its capabilities to other modalities such as audio to further broaden its applicability.

By providing a standardized, extensible platform for anomaly detection, Anomalib aims to accelerate research in this field and facilitate the adoption of advanced anomaly detection techniques in real-world applications.

References
