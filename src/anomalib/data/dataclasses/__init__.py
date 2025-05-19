"""Anomalib dataclasses.

This module provides a collection of Pydantic models used throughout the Anomalib
library for representing and managing various types of data related to anomaly
detection tasks.

The models are designed to handle both PyTorch tensors and NumPy arrays, with
built-in validation and type conversion capabilities. They are organized into
base models and specific implementations for different data types:

Base Models
~~~~~~~~~~

- :class:`TensorField`: Base model for tensor fields
- :class:`BaseItem`: Base model for single data items
- :class:`BaseBatch`: Base model for batched data items

Image Models
~~~~~~~~~~

- :class:`ImageItem`: Model for single image items
- :class:`ImageBatch`: Model for batched image items

Video Models
~~~~~~~~~~

- :class:`VideoItem`: Model for single video items
- :class:`VideoBatch`: Model for batched video items

Depth Models
~~~~~~~~~~

- :class:`DepthItem`: Model for single depth items
- :class:`DepthBatch`: Model for batched depth items

Example:
    Create and use an image item:

    >>> from anomalib.data.dataclasses import ImageItem
    >>> import torch
    >>> item = ImageItem(
    ...     image=torch.rand(3, 224, 224),
    ...     gt_label=torch.tensor(0),
    ...     image_path="path/to/image.jpg"
    ... )
    >>> item.image.shape
    torch.Size([3, 224, 224])

    Convert to numpy:
    >>> numpy_item = item.to_numpy()
    >>> type(numpy_item.image)
    <class 'numpy.ndarray'>
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data.dataclasses.base import BaseBatch, BaseItem, TensorField
from anomalib.data.dataclasses.models import (
    DepthBatch,
    DepthItem,
    ImageBatch,
    ImageItem,
    VideoBatch,
    VideoItem,
)

__all__ = [
    # Base
    "TensorField",
    "BaseItem",
    "BaseBatch",
    # Image
    "ImageItem",
    "ImageBatch",
    # Video
    "VideoItem",
    "VideoBatch",
    # Depth
    "DepthItem",
    "DepthBatch",
]
