# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Example showing how to use the EfficientAd model.

EfficientAd is a fast and accurate model for anomaly detection,
particularly well-suited for industrial inspection tasks.
"""

from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import EfficientAd

# 1. Basic Usage
# Initialize with default settings
model = EfficientAd()

# 2. Custom Configuration
# Configure model parameters
model = EfficientAd(
    teacher_out_channels=384,  # Number of teacher output channels
    student_out_channels=384,  # Number of student output channels
    padding_same=True,  # Use same padding in convolutions
)

# 3. Training Pipeline
# Set up the complete training pipeline
datamodule = MVTec(
    root="./datasets/MVTec",
    category="bottle",
    train_batch_size=32,
)

# Initialize training engine with specific settings
engine = Engine(
    max_epochs=20,
    accelerator="auto",  # Automatically detect GPU/CPU
    devices=1,  # Number of devices to use
)

# Train the model
engine.fit(
    model=model,
    datamodule=datamodule,
)
