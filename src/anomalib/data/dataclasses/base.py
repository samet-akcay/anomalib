"""Base dataclasses for Anomalib using Pydantic.

This module provides base Pydantic models for handling both torch and numpy data
in Anomalib. The models include validation and type conversion capabilities.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator
from torchvision.tv_tensors import Image, Mask, Video


class TorchNumpyField(BaseModel):
    """Base model for fields that can handle both PyTorch tensors and NumPy arrays.

    This class provides automatic validation and conversion between PyTorch tensors
    and NumPy arrays. It ensures that fields can accept either type and maintains
    type consistency throughout the model.
    """

    @field_validator("*", mode="before")
    @classmethod
    def validate_tensor(cls, v: Any) -> torch.Tensor | np.ndarray:  # noqa: ANN401
        """Validate and convert input to tensor.

        Args:
            v: Input value to validate

        Returns:
            Validated tensor (torch.Tensor or np.ndarray)

        Raises:
            ValueError: If input cannot be converted to tensor
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            return v
        if isinstance(v, (list, tuple)):
            try:
                return torch.tensor(v)
            except (TypeError, ValueError):
                return np.array(v)
        msg = f"Cannot convert {type(v)} to tensor"
        raise ValueError(msg)


class InputFields(TorchNumpyField):
    """Input fields for data items."""

    image: torch.Tensor | np.ndarray | None = Field(default=None, description="Input image/video data")
    image_path: str | None = Field(default=None, description="Path to image file")

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: Any) -> torch.Tensor | np.ndarray:  # noqa: ANN401
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
            if len(v.shape) not in {2, 3, 4}:
                msg = f"Image must have 2-4 dimensions, got {len(v.shape)}"
                raise ValueError(msg)
            return v
        msg = f"Invalid image type: {type(v)}"
        raise ValueError(msg)


class AnnotationFields(TorchNumpyField):
    """Annotation fields for data items."""

    label: torch.Tensor | np.ndarray | None = Field(default=None, description="Ground truth label")
    mask: torch.Tensor | np.ndarray | None = Field(default=None, description="Ground truth mask")

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: Any) -> torch.Tensor | np.ndarray:  # noqa: ANN401
        """Validate label data.

        Args:
            v: Label data to validate

        Returns:
            Validated label data

        Raises:
            ValueError: If label data is invalid
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if v.ndim > 1:
                msg = f"Label must be 0D or 1D, got {v.ndim}D"
                raise ValueError(msg)
            return v
        if isinstance(v, (int, float)):
            return torch.tensor(v) if torch.is_tensor(v) else np.array(v)
        msg = f"Invalid label type: {type(v)}"
        raise ValueError(msg)

    @field_validator("mask")
    @classmethod
    def validate_mask(cls, v: Any) -> torch.Tensor | np.ndarray:  # noqa: ANN401
        """Validate mask data.

        Args:
            v: Mask data to validate

        Returns:
            Validated mask data

        Raises:
            ValueError: If mask data is invalid
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if len(v.shape) not in {2, 3}:
                msg = f"Mask must have 2-3 dimensions, got {len(v.shape)}"
                raise ValueError(msg)
            return v
        msg = f"Invalid mask type: {type(v)}"
        raise ValueError(msg)


class PredictionFields(TorchNumpyField):
    """Prediction fields for data items."""

    score: float | None = Field(default=None, description="Predicted anomaly score")
    label: int | None = Field(default=None, description="Predicted anomaly label")
    mask: torch.Tensor | np.ndarray | None = Field(default=None, description="Predicted anomaly mask")
    anomaly_map: torch.Tensor | np.ndarray | None = Field(default=None, description="Anomaly map")

    @field_validator("mask")
    @classmethod
    def validate_mask(cls, v: Any) -> torch.Tensor | np.ndarray:  # noqa: ANN401
        """Validate prediction mask data.

        Args:
            v: Prediction mask data to validate

        Returns:
            Validated prediction mask data

        Raises:
            ValueError: If prediction mask data is invalid
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if len(v.shape) not in {2, 3}:
                msg = f"Prediction mask must have 2-3 dimensions, got {len(v.shape)}"
                raise ValueError(msg)
            return v
        msg = f"Invalid prediction mask type: {type(v)}"
        raise ValueError(msg)

    @field_validator("anomaly_map")
    @classmethod
    def validate_anomaly_map(cls, v: Any) -> torch.Tensor | np.ndarray:  # noqa: ANN401
        """Validate anomaly map data.

        Args:
            v: Anomaly map data to validate

        Returns:
            Validated anomaly map data

        Raises:
            ValueError: If anomaly map data is invalid
        """
        if v is None:
            return v
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if len(v.shape) not in {2, 3}:
                msg = f"Anomaly map must have 2-3 dimensions, got {len(v.shape)}"
                raise ValueError(msg)
            return v
        msg = f"Invalid anomaly map type: {type(v)}"
        raise ValueError(msg)


class BaseItem(TorchNumpyField):
    """Base model for single data items."""

    input: InputFields = Field(default_factory=InputFields, description="Input fields")
    annotation: AnnotationFields = Field(default_factory=AnnotationFields, description="Annotation fields")
    prediction: PredictionFields = Field(default_factory=PredictionFields, description="Prediction fields")

    # For backward compatibility and direct access
    @property
    def image(self) -> torch.Tensor | np.ndarray | None:
        """Get the image from input fields."""
        return self.input.image

    @image.setter
    def image(self, value: torch.Tensor | np.ndarray | None) -> None:
        """Set the image in input fields."""
        self.input.image = value

    @property
    def image_path(self) -> str | None:
        """Get the image path from input fields."""
        return self.input.image_path

    @image_path.setter
    def image_path(self, value: str | None) -> None:
        """Set the image path in input fields."""
        self.input.image_path = value

    @property
    def gt_label(self) -> torch.Tensor | np.ndarray | None:
        """Get the ground truth label from annotation fields."""
        return self.annotation.label

    @gt_label.setter
    def gt_label(self, value: torch.Tensor | np.ndarray | None) -> None:
        """Set the ground truth label in annotation fields."""
        self.annotation.label = value

    @property
    def gt_mask(self) -> torch.Tensor | np.ndarray | None:
        """Get the ground truth mask from annotation fields."""
        return self.annotation.mask

    @gt_mask.setter
    def gt_mask(self, value: torch.Tensor | np.ndarray | None) -> None:
        """Set the ground truth mask in annotation fields."""
        self.annotation.mask = value

    @property
    def pred_score(self) -> float | None:
        """Get the prediction score from prediction fields."""
        return self.prediction.score

    @pred_score.setter
    def pred_score(self, value: float | None) -> None:
        """Set the prediction score in prediction fields."""
        self.prediction.score = value

    @property
    def pred_label(self) -> int | None:
        """Get the prediction label from prediction fields."""
        return self.prediction.label

    @pred_label.setter
    def pred_label(self, value: int | None) -> None:
        """Set the prediction label in prediction fields."""
        self.prediction.label = value

    @property
    def pred_mask(self) -> torch.Tensor | np.ndarray | None:
        """Get the prediction mask from prediction fields."""
        return self.prediction.mask

    @pred_mask.setter
    def pred_mask(self, value: torch.Tensor | np.ndarray | None) -> None:
        """Set the prediction mask in prediction fields."""
        self.prediction.mask = value

    @property
    def anomaly_map(self) -> torch.Tensor | np.ndarray | None:
        """Get the anomaly map from prediction fields."""
        return self.prediction.anomaly_map

    @anomaly_map.setter
    def anomaly_map(self, value: torch.Tensor | np.ndarray | None) -> None:
        """Set the anomaly map in prediction fields."""
        self.prediction.anomaly_map = value

    def to_numpy(self) -> "BaseItem":
        """Convert all tensor fields to numpy arrays.

        Returns:
            New item with numpy arrays
        """
        data = self.model_dump()
        for field in ["input", "annotation", "prediction"]:
            for key, value in data[field].items():
                if isinstance(value, torch.Tensor):
                    data[field][key] = value.cpu().numpy()
        return self.__class__(**data)

    def to_torch(self) -> "BaseItem":
        """Convert all numpy arrays to torch tensors.

        Returns:
            New item with torch tensors
        """
        data = self.model_dump()
        for field in ["input", "annotation", "prediction"]:
            for key, value in data[field].items():
                if isinstance(value, np.ndarray):
                    data[field][key] = torch.from_numpy(value)
        return self.__class__(**data)


class BaseBatch(TorchNumpyField):
    """Base model for batched data items."""

    input: InputFields = Field(default_factory=InputFields, description="Input fields")
    annotation: AnnotationFields = Field(default_factory=AnnotationFields, description="Annotation fields")
    prediction: PredictionFields = Field(default_factory=PredictionFields, description="Prediction fields")

    @property
    def image(self) -> torch.Tensor | np.ndarray | None:
        """Get the input image."""
        return self.input.image

    @image.setter
    def image(self, value: torch.Tensor | np.ndarray | None) -> None:
        """Set the input image."""
        self.input.image = value

    @property
    def image_path(self) -> str | None:
        """Get the input image path."""
        return self.input.image_path

    @image_path.setter
    def image_path(self, value: str | None) -> None:
        """Set the input image path."""
        self.input.image_path = value

    @property
    def gt_label(self) -> torch.Tensor | np.ndarray | None:
        """Get the ground truth label."""
        return self.annotation.label

    @gt_label.setter
    def gt_label(self, value: torch.Tensor | np.ndarray | None) -> None:
        """Set the ground truth label."""
        self.annotation.label = value

    @property
    def gt_mask(self) -> torch.Tensor | np.ndarray | None:
        """Get the ground truth mask."""
        return self.annotation.mask

    @gt_mask.setter
    def gt_mask(self, value: torch.Tensor | np.ndarray | None) -> None:
        """Set the ground truth mask."""
        self.annotation.mask = value

    @property
    def pred_score(self) -> float | None:
        """Get the prediction score."""
        return self.prediction.score

    @pred_score.setter
    def pred_score(self, value: float | None) -> None:
        """Set the prediction score."""
        self.prediction.score = value

    @property
    def pred_label(self) -> int | None:
        """Get the prediction label."""
        return self.prediction.label

    @pred_label.setter
    def pred_label(self, value: int | None) -> None:
        """Set the prediction label."""
        self.prediction.label = value

    @property
    def pred_mask(self) -> torch.Tensor | np.ndarray | None:
        """Get the prediction mask."""
        return self.prediction.mask

    @pred_mask.setter
    def pred_mask(self, value: torch.Tensor | np.ndarray | None) -> None:
        """Set the prediction mask."""
        self.prediction.mask = value

    @property
    def anomaly_map(self) -> torch.Tensor | np.ndarray | None:
        """Get the anomaly map."""
        return self.prediction.anomaly_map

    @anomaly_map.setter
    def anomaly_map(self, value: torch.Tensor | np.ndarray | None) -> None:
        """Set the anomaly map."""
        self.prediction.anomaly_map = value

    def to_numpy(self) -> "BaseBatch":
        """Convert all tensor fields to numpy arrays.

        Returns:
            New batch with numpy arrays
        """
        data = self.model_dump()
        for field in ["input", "annotation", "prediction"]:
            for key, value in data[field].items():
                if isinstance(value, torch.Tensor):
                    data[field][key] = value.cpu().numpy()
        return self.__class__(**data)

    def to_torch(self) -> "BaseBatch":
        """Convert all numpy arrays to torch tensors.

        Returns:
            New batch with torch tensors
        """
        data = self.model_dump()
        for field in ["input", "annotation", "prediction"]:
            for key, value in data[field].items():
                if isinstance(value, np.ndarray):
                    data[field][key] = torch.from_numpy(value)
        return self.__class__(**data)
