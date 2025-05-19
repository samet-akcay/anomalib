"""Specific dataclass models for Anomalib using Pydantic.

This module provides specific Pydantic models for handling image, video, and depth
data in Anomalib. These models extend the base models with additional validation
and fields specific to each data type.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

import numpy as np
import torch
from pydantic import Field, field_validator
from torchvision.tv_tensors import Image, Mask, Video

from anomalib.data.dataclasses.base import BaseBatch, BaseItem


class ImageItem(BaseItem):
    """Model for single image items."""

    image: Optional[Union[torch.Tensor, np.ndarray]] = Field(
        default=None,
        description="Input image data of shape (C, H, W) or (H, W, C)",
    )
    image_path: Optional[str] = Field(default=None, description="Path to image file")
    mask_path: Optional[str] = Field(default=None, description="Path to mask file for segmentation tasks")

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: Any) -> Union[torch.Tensor, np.ndarray]:
        """Validate image data.

        Args:
            v: Image data to validate

        Returns:
            Validated image data

        Raises:
            ValueError: If image data is invalid
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if len(v.shape) != 3:
                raise ValueError(f"Image must have 3 dimensions (C, H, W) or (H, W, C), got {len(v.shape)}")
            return v
        raise ValueError(f"Invalid image type: {type(v)}")


class ImageBatch(BaseBatch):
    """Model for batched image items."""

    image: Optional[Union[torch.Tensor, np.ndarray]] = Field(
        default=None,
        description="Batch of input images of shape (B, C, H, W) or (B, H, W, C)",
    )
    image_path: Optional[list[str]] = Field(default=None, description="List of paths to image files")

    @field_validator("image")
    @classmethod
    def validate_batch_image(cls, v: Any) -> Union[torch.Tensor, np.ndarray]:
        """Validate batched image data.

        Args:
            v: Batched image data to validate

        Returns:
            Validated batched image data

        Raises:
            ValueError: If batched image data is invalid
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if len(v.shape) != 4:
                raise ValueError(f"Batch image must have 4 dimensions (B, C, H, W) or (B, H, W, C), got {len(v.shape)}")
            return v
        raise ValueError(f"Invalid batch image type: {type(v)}")


class VideoItem(BaseItem):
    """Model for single video items."""

    image: Optional[Union[torch.Tensor, np.ndarray]] = Field(
        default=None,
        description="Input video data of shape (T, C, H, W) or (T, H, W, C)",
    )
    video_path: Optional[str] = Field(default=None, description="Path to video file")
    frames: Optional[Union[torch.Tensor, np.ndarray]] = Field(
        default=None,
        description="Individual frames of shape (T, C, H, W) or (T, H, W, C)",
    )
    last_frame: Optional[Union[torch.Tensor, np.ndarray]] = Field(
        default=None,
        description="Last frame of shape (C, H, W) or (H, W, C)",
    )

    @field_validator("image", "frames")
    @classmethod
    def validate_video(cls, v: Any) -> Union[torch.Tensor, np.ndarray]:
        """Validate video data.

        Args:
            v: Video data to validate

        Returns:
            Validated video data

        Raises:
            ValueError: If video data is invalid
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if len(v.shape) != 4:
                raise ValueError(f"Video must have 4 dimensions (T, C, H, W) or (T, H, W, C), got {len(v.shape)}")
            return v
        raise ValueError(f"Invalid video type: {type(v)}")

    @field_validator("last_frame")
    @classmethod
    def validate_last_frame(cls, v: Any) -> Union[torch.Tensor, np.ndarray]:
        """Validate last frame data.

        Args:
            v: Last frame data to validate

        Returns:
            Validated last frame data

        Raises:
            ValueError: If last frame data is invalid
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if len(v.shape) != 3:
                raise ValueError(f"Last frame must have 3 dimensions (C, H, W) or (H, W, C), got {len(v.shape)}")
            return v
        raise ValueError(f"Invalid last frame type: {type(v)}")


class VideoBatch(BaseBatch):
    """Model for batched video items."""

    image: Optional[Union[torch.Tensor, np.ndarray]] = Field(
        default=None,
        description="Batch of input videos of shape (B, T, C, H, W) or (B, T, H, W, C)",
    )
    video_path: Optional[list[str]] = Field(default=None, description="List of paths to video files")
    frames: Optional[Union[torch.Tensor, np.ndarray]] = Field(
        default=None,
        description="Individual frames of shape (B, T, C, H, W) or (B, T, H, W, C)",
    )
    last_frame: Optional[Union[torch.Tensor, np.ndarray]] = Field(
        default=None,
        description="Last frames of shape (B, C, H, W) or (B, H, W, C)",
    )

    @field_validator("image", "frames")
    @classmethod
    def validate_batch_video(cls, v: Any) -> Union[torch.Tensor, np.ndarray]:
        """Validate batched video data.

        Args:
            v: Batched video data to validate

        Returns:
            Validated batched video data

        Raises:
            ValueError: If batched video data is invalid
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if len(v.shape) != 5:
                raise ValueError(
                    f"Batch video must have 5 dimensions (B, T, C, H, W) or (B, T, H, W, C), got {len(v.shape)}"
                )
            return v
        raise ValueError(f"Invalid batch video type: {type(v)}")

    @field_validator("last_frame")
    @classmethod
    def validate_batch_last_frame(cls, v: Any) -> Union[torch.Tensor, np.ndarray]:
        """Validate batched last frame data.

        Args:
            v: Batched last frame data to validate

        Returns:
            Validated batched last frame data

        Raises:
            ValueError: If batched last frame data is invalid
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if len(v.shape) != 4:
                raise ValueError(
                    f"Batch last frame must have 4 dimensions (B, C, H, W) or (B, H, W, C), got {len(v.shape)}"
                )
            return v
        raise ValueError(f"Invalid batch last frame type: {type(v)}")


class DepthItem(ImageItem):
    """Model for single depth items."""

    depth_map: Optional[Union[torch.Tensor, np.ndarray]] = Field(
        default=None,
        description="Depth map of shape (H, W)",
    )
    depth_path: Optional[str] = Field(default=None, description="Path to depth map file")

    @field_validator("depth_map")
    @classmethod
    def validate_depth_map(cls, v: Any) -> Union[torch.Tensor, np.ndarray]:
        """Validate depth map data.

        Args:
            v: Depth map data to validate

        Returns:
            Validated depth map data

        Raises:
            ValueError: If depth map data is invalid
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if len(v.shape) != 2:
                raise ValueError(f"Depth map must have 2 dimensions (H, W), got {len(v.shape)}")
            return v
        raise ValueError(f"Invalid depth map type: {type(v)}")


class DepthBatch(ImageBatch):
    """Model for batched depth items."""

    depth_map: Optional[Union[torch.Tensor, np.ndarray]] = Field(
        default=None,
        description="Batch of depth maps of shape (B, H, W)",
    )
    depth_path: Optional[list[str]] = Field(default=None, description="List of paths to depth map files")

    @field_validator("depth_map")
    @classmethod
    def validate_batch_depth_map(cls, v: Any) -> Union[torch.Tensor, np.ndarray]:
        """Validate batched depth map data.

        Args:
            v: Batched depth map data to validate

        Returns:
            Validated batched depth map data

        Raises:
            ValueError: If batched depth map data is invalid
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if len(v.shape) != 3:
                raise ValueError(f"Batch depth map must have 3 dimensions (B, H, W), got {len(v.shape)}")
            return v
        raise ValueError(f"Invalid batch depth map type: {type(v)}")
