"""Anomaly map computation for Student-Teacher Feature Pyramid Matching model.

This module implements functionality to generate anomaly heatmaps by comparing
features between a pre-trained teacher network and a student network that learns
to match the teacher's representations.

The anomaly maps are generated by:
1. Computing cosine similarity between teacher and student features
2. Converting similarity scores to anomaly scores via L2 norm
3. Upscaling anomaly scores to original image size
4. Combining multiple layer scores via element-wise multiplication

Example:
    >>> from anomalib.models.image.stfpm.anomaly_map import AnomalyMapGenerator
    >>> generator = AnomalyMapGenerator()
    >>> teacher_features = {"layer1": torch.randn(1, 64, 32, 32)}
    >>> student_features = {"layer1": torch.randn(1, 64, 32, 32)}
    >>> anomaly_map = generator.compute_anomaly_map(
    ...     teacher_features,
    ...     student_features,
    ...     image_size=(256, 256)
    ... )

See Also:
    - :class:`AnomalyMapGenerator`: Main class for generating anomaly maps
    - :func:`compute_layer_map`: Function to compute per-layer anomaly scores
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class AnomalyMapGenerator(nn.Module):
    """Generate anomaly heatmaps by comparing teacher and student features.

    This class implements functionality to generate anomaly maps by comparing
    feature representations between a pre-trained teacher network and a student
    network. The comparison is done via cosine similarity and L2 distance.

    The anomaly map generation process involves:
    1. Computing cosine similarity between teacher-student feature pairs
    2. Converting similarity scores to anomaly scores using L2 norm
    3. Upscaling the scores to original image size
    4. Combining multiple layer scores via element-wise multiplication

    Example:
        >>> from anomalib.models.image.stfpm.anomaly_map import AnomalyMapGenerator
        >>> generator = AnomalyMapGenerator()
        >>> teacher_features = {"layer1": torch.randn(1, 64, 32, 32)}
        >>> student_features = {"layer1": torch.randn(1, 64, 32, 32)}
        >>> anomaly_map = generator.compute_anomaly_map(
        ...     teacher_features,
        ...     student_features,
        ...     image_size=(256, 256)
        ... )

    See Also:
        - :func:`compute_layer_map`: Function to compute per-layer anomaly scores
        - :func:`compute_anomaly_map`: Function to combine layer scores
    """

    def __init__(self) -> None:
        """Initialize pairwise distance metric."""
        super().__init__()
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)

    @staticmethod
    def compute_layer_map(
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
        image_size: tuple[int, int] | torch.Size,
    ) -> torch.Tensor:
        """Compute anomaly map for a single feature layer.

        The layer map is computed by:
        1. Normalizing teacher and student features
        2. Computing L2 distance between normalized features
        3. Upscaling the distance map to original image size

        Args:
            teacher_features (torch.Tensor): Features from teacher network with
                shape ``(B, C, H, W)``
            student_features (torch.Tensor): Features from student network with
                matching shape
            image_size (tuple[int, int] | torch.Size): Target size for upscaling
                in format ``(H, W)``

        Returns:
            torch.Tensor: Anomaly scores for the layer, upscaled to
                ``image_size``
        """
        norm_teacher_features = F.normalize(teacher_features)
        norm_student_features = F.normalize(student_features)

        layer_map = 0.5 * torch.norm(norm_teacher_features - norm_student_features, p=2, dim=-3, keepdim=True) ** 2
        return F.interpolate(layer_map, size=image_size, align_corners=False, mode="bilinear")

    def compute_anomaly_map(
        self,
        teacher_features: dict[str, torch.Tensor],
        student_features: dict[str, torch.Tensor],
        image_size: tuple[int, int] | torch.Size,
    ) -> torch.Tensor:
        """Compute overall anomaly map by combining multiple layer maps.

        The final anomaly map is generated by:
        1. Computing per-layer anomaly maps via :func:`compute_layer_map`
        2. Combining layer maps through element-wise multiplication

        Args:
            teacher_features (dict[str, torch.Tensor]): Dictionary mapping layer
                names to teacher feature tensors
            student_features (dict[str, torch.Tensor]): Dictionary mapping layer
                names to student feature tensors
            image_size (tuple[int, int] | torch.Size): Target size for the
                anomaly map in format ``(H, W)``

        Returns:
            torch.Tensor: Final anomaly map with shape ``(B, 1, H, W)`` where
                ``B`` is batch size and ``(H, W)`` matches ``image_size``
        """
        batch_size = next(iter(teacher_features.values())).shape[0]
        anomaly_map = torch.ones(batch_size, 1, image_size[0], image_size[1])
        for layer in teacher_features:
            layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer], image_size)
            anomaly_map = anomaly_map.to(layer_map.device)
            anomaly_map *= layer_map

        return anomaly_map

    def forward(self, **kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate anomaly map from teacher and student features.

        Expects the following keys in ``kwargs``:
        - ``teacher_features``: Dictionary of teacher network features
        - ``student_features``: Dictionary of student network features
        - ``image_size``: Target size for the anomaly map

        Args:
            kwargs (dict[str, torch.Tensor]): Keyword arguments containing
                required inputs

        Example:
            >>> generator = AnomalyMapGenerator()
            >>> anomaly_map = generator(
            ...     teacher_features=teacher_features,
            ...     student_features=student_features,
            ...     image_size=(256, 256)
            ... )

        Raises:
            ValueError: If required keys are missing from ``kwargs``

        Returns:
            torch.Tensor: Anomaly map with shape ``(B, 1, H, W)``
        """
        if not ("teacher_features" in kwargs and "student_features" in kwargs):
            msg = f"Expected keys `teacher_features` and `student_features. Found {kwargs.keys()}"
            raise ValueError(msg)

        teacher_features: dict[str, torch.Tensor] = kwargs["teacher_features"]
        student_features: dict[str, torch.Tensor] = kwargs["student_features"]
        image_size: tuple[int, int] | torch.Size = kwargs["image_size"]

        return self.compute_anomaly_map(teacher_features, student_features, image_size)
