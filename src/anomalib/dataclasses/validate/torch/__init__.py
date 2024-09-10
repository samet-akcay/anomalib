"""Validate IO torch.Tensor data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .depth import DepthBatchValidator, DepthValidator
from .image import ImageBatchValidator, ImageValidator
from .video import VideoBatchValidator, VideoValidator

__all__ = [
    # Item validators
    "DepthValidator",
    "ImageValidator",
    "VideoValidator",
    # Batch validators
    "DepthBatchValidator",
    "ImageBatchValidator",
    "VideoBatchValidator",
]
