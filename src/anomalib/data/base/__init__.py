"""Base classes for custom dataset and datamodules."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .datamodule import DataModule
from .dataset import Dataset
from .depth import DepthDataset
from .video import VideoDataModule, VideoDataset

__all__ = [
    "Dataset",
    "DataModule",
    "VideoDataset",
    "VideoDataModule",
    "DepthDataset",
]
