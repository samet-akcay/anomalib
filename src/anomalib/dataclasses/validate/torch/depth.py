"""Validate torch depth data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision.tv_tensors import Image, Mask

from anomalib.dataclasses.validate.path import validate_path


class DepthValidator:
    """Validate torch.Tensor data for depth maps."""

    @staticmethod
    def validate_image(depth_map: torch.Tensor) -> Image:
        """Validate and convert a depth image tensor.

        Args:
            depth_map (torch.Tensor): Input depth image tensor.

        Returns:
            Image: Validated and converted depth image.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the image shape is invalid.

        Examples:
            >>> import torch
            >>> depth_map = torch.rand(1, 224, 224)
            >>> validated_image = DepthValidator.validate_image(depth_map)
            >>> isinstance(validated_image, Image)
            True
            >>> validated_image.shape
            torch.Size([1, 224, 224])
        """
        if not isinstance(depth_map, torch.Tensor):
            msg = f"Depth map must be a torch.Tensor, got {type(depth_map)}."
            raise TypeError(msg)

        if depth_map.ndim not in {2, 3}:
            msg = f"Depth map must have shape [H, W] or [1, H, W], got shape {depth_map.shape}."
            raise ValueError(msg)

        if depth_map.ndim == 3 and depth_map.shape[0] != 1:
            msg = f"Depth map must have 1 channel, got {depth_map.shape[0]}."
            raise ValueError(msg)

        return Image(depth_map.unsqueeze(0) if depth_map.ndim == 2 else depth_map)

    @staticmethod
    def validate_gt_mask(mask: torch.Tensor | None) -> Mask | None:
        """Validate the ground truth mask for a depth image.

        Args:
            mask (torch.Tensor | None): Input ground truth mask.

        Returns:
            Mask | None: Validated ground truth mask, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> import torch
            >>> mask = torch.randint(0, 2, (1, 224, 224))
            >>> validated_mask = DepthValidator.validate_gt_mask(mask)
            >>> isinstance(validated_mask, Mask)
            True
            >>> validated_mask.shape
            torch.Size([224, 224])
        """
        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor):
            msg = f"Ground truth mask must be a torch.Tensor, got {type(mask)}."
            raise TypeError(msg)
        if mask.ndim not in {2, 3}:
            msg = f"Ground truth mask must have shape [H, W] or [1, H, W], got shape {mask.shape}."
            raise ValueError(msg)
        if mask.ndim == 3 and mask.shape[0] != 1:
            msg = f"Ground truth mask must have 1 channel, got {mask.shape[0]}."
            raise ValueError(msg)

        return Mask(mask.squeeze(0) if mask.ndim == 3 else mask, dtype=torch.bool)

    @staticmethod
    def validate_gt_label(label: int | torch.Tensor | None) -> torch.Tensor | None:
        """Validate the ground truth label for a depth image.

        Args:
            label (int | torch.Tensor | None): Input ground truth label.

        Returns:
            torch.Tensor | None: Validated ground truth label as a boolean tensor, or None.

        Raises:
            TypeError: If the input is not an int or torch.Tensor.
            ValueError: If the label is not 0 or 1, or if the tensor shape is invalid.

        Examples:
            >>> validated_label = DepthValidator.validate_gt_label(1)
            >>> validated_label
            tensor(True)

            >>> import torch
            >>> tensor_label = torch.tensor(0)
            >>> validated_tensor_label = DepthValidator.validate_gt_label(tensor_label)
            >>> validated_tensor_label
            tensor(False)
        """
        if label is None:
            return None

        if isinstance(label, int):
            if label not in {0, 1}:
                msg = f"Ground truth label must be 0 or 1, got {label}."
                raise ValueError(msg)
            return torch.tensor(bool(label))

        if isinstance(label, torch.Tensor):
            if label.ndim != 0:
                msg = f"Ground truth label must be a scalar tensor, got shape {label.shape}."
                raise ValueError(msg)
            if label.item() not in {0, 1}:
                msg = f"Ground truth label must be 0 or 1, got {label.item()}."
                raise ValueError(msg)
            return label.to(torch.bool)

        msg = f"Ground truth label must be an int or torch.Tensor, got {type(label)}."
        raise TypeError(msg)

    @staticmethod
    def validate_image_path(path: str | None) -> str | None:
        """Validate the image path for a depth image.

        Args:
            path (str | None): Input image path.

        Returns:
            str | None: Validated image path, or None.

        Examples:
            >>> path = "/path/to/depth_image.png"
            >>> validated_path = DepthValidator.validate_image_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(path) if path is not None else None

    @staticmethod
    def validate_mask_path(path: str | None) -> str | None:
        """Validate the mask path for a depth image.

        Args:
            path (str | None): Input mask path.

        Returns:
            str | None: Validated mask path, or None.

        Examples:
            >>> path = "/path/to/depth_mask.png"
            >>> validated_path = DepthValidator.validate_mask_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(path) if path is not None else None

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | None) -> Mask | None:
        """Validate the anomaly map for a depth image.

        Args:
            anomaly_map (torch.Tensor | None): Input anomaly map.

        Returns:
            Mask | None: Validated anomaly map as a Mask, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> import torch
            >>> anomaly_map = torch.rand(1, 224, 224)
            >>> validated_map = DepthValidator.validate_anomaly_map(anomaly_map)
            >>> isinstance(validated_map, Mask)
            True
            >>> validated_map.shape
            torch.Size([224, 224])
        """
        return DepthValidator.validate_gt_mask(anomaly_map)

    @staticmethod
    def validate_pred_mask(mask: torch.Tensor | None) -> Mask | None:
        """Validate the prediction mask for a depth image.

        Args:
            mask (torch.Tensor | None): Input prediction mask.

        Returns:
            Mask | None: Validated prediction mask, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> import torch
            >>> mask = torch.randint(0, 2, (1, 224, 224))
            >>> validated_mask = DepthValidator.validate_pred_mask(mask)
            >>> isinstance(validated_mask, Mask)
            True
            >>> validated_mask.shape
            torch.Size([224, 224])
        """
        return DepthValidator.validate_gt_mask(mask)

    @staticmethod
    def validate_pred_label(label: int | torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction label for a depth image.

        Args:
            label (int | torch.Tensor | None): Input prediction label.

        Returns:
            torch.Tensor | None: Validated prediction label as a boolean tensor, or None.

        Raises:
            TypeError: If the input is not an int or torch.Tensor.
            ValueError: If the label is not 0 or 1, or if the tensor shape is invalid.

        Examples:
            >>> validated_label = DepthValidator.validate_pred_label(0)
            >>> validated_label
            tensor(False)

            >>> import torch
            >>> tensor_label = torch.tensor(1)
            >>> validated_tensor_label = DepthValidator.validate_pred_label(tensor_label)
            >>> validated_tensor_label
            tensor(True)
        """
        return DepthValidator.validate_gt_label(label)

    @staticmethod
    def validate_pred_score(score: float | torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction score for a depth image.

        Args:
            score (float | torch.Tensor | None): Input prediction score.

        Returns:
            torch.Tensor | None: Validated prediction score as a tensor, or None.

        Raises:
            TypeError: If the input is not a float or torch.Tensor.
            ValueError: If the score is not between 0 and 1, or if the tensor shape is invalid.

        Examples:
            >>> validated_score = DepthValidator.validate_pred_score(0.7)
            >>> validated_score
            tensor(0.7000)

            >>> import torch
            >>> tensor_score = torch.tensor(0.3)
            >>> validated_tensor_score = DepthValidator.validate_pred_score(tensor_score)
            >>> validated_tensor_score
            tensor(0.3000)
        """
        if score is None:
            return None

        if isinstance(score, float):
            if not 0 <= score <= 1:
                msg = f"Prediction score must be between 0 and 1, got {score}."
                raise ValueError(msg)
            return torch.tensor(score)

        if isinstance(score, torch.Tensor):
            if score.ndim != 0:
                msg = f"Prediction score must be a scalar tensor, got shape {score.shape}."
                raise ValueError(msg)
            if not 0 <= score.item() <= 1:
                msg = f"Prediction score must be between 0 and 1, got {score.item()}."
                raise ValueError(msg)
            return score.to(torch.float32)

        msg = f"Prediction score must be a float or torch.Tensor, got {type(score)}."
        raise TypeError(msg)

    @staticmethod
    def validate_depth_map(depth_map: torch.Tensor) -> Image:
        """Validate and convert a depth map tensor.

        Args:
            depth_map (torch.Tensor): Input depth map tensor.

        Returns:
            Image: Validated and converted depth map.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the depth map shape is invalid.

        Examples:
            >>> import torch
            >>> depth_map = torch.rand(1, 224, 224)
            >>> validated_depth_map = DepthValidator.validate_depth_map(depth_map)
            >>> isinstance(validated_depth_map, Image)
            True
            >>> validated_depth_map.shape
            torch.Size([1, 224, 224])
        """
        return DepthValidator.validate_image(depth_map)

    @staticmethod
    def validate_depth_path(path: str | None) -> str | None:
        """Validate the depth map path.

        Args:
            path (str | None): Input depth map path.

        Returns:
            str | None: Validated depth map path, or None.

        Examples:
            >>> path = "/path/to/depth_map.png"
            >>> validated_path = DepthValidator.validate_depth_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(path) if path is not None else None


class DepthBatchValidator:
    """Validate torch.Tensor data for batches of depth maps."""

    @staticmethod
    def validate_image(depth_map: torch.Tensor) -> Image:
        """Validate and convert a depth map tensor.

        Args:
            depth_map (torch.Tensor): Input depth map tensor.

        Returns:
            Image: Validated and converted batch of depth maps.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the depth map shape is invalid.

        Examples:
            >>> import torch
            >>> tensor = torch.rand(4, 1, 224, 224)  # Batch of 4 depth maps
            >>> validated_depth_map = DepthBatchValidator.validate_image(tensor)
            >>> isinstance(validated_depth_map, Image)
            True
            >>> validated_depth_map.shape
            torch.Size([1, 224, 224])
        """
        if not isinstance(depth_map, torch.Tensor):
            msg = f"Depth maps must be a torch.Tensor, got {type(depth_map)}."
            raise TypeError(msg)

        if depth_map.ndim != 4:
            msg = f"Depth maps must have shape [N, C, H, W], got shape {depth_map.shape}."
            raise ValueError(msg)

        if depth_map.shape[1] != 1:
            msg = f"Depth maps must have 1 channel, got {depth_map.shape[1]}."
            raise ValueError(msg)

        return Image(depth_map.to(torch.float32))

    @staticmethod
    def validate_gt_mask(mask: torch.Tensor | None) -> Mask | None:
        """Validate a batch of ground truth masks for depth images.

        Args:
            mask (torch.Tensor | None): Input batch of ground truth masks.

        Returns:
            Mask | None: Validated batch of ground truth masks, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> mask = torch.randint(0, 2, (1, 224, 224))  # Batch of 4 masks
            >>> validated_mask = DepthBatchValidator.validate_gt_mask(mask)
            >>> isinstance(validated_mask, Mask)
            True
            >>> validated_mask.shape
            torch.Size([224, 224])
        """
        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor):
            msg = f"Ground truth masks must be a torch.Tensor, got {type(mask)}."
            raise TypeError(msg)
        if mask.ndim not in {3, 4}:
            msg = f"Ground truth masks must have shape [N, H, W] or [N, 1, H, W], got shape {mask.shape}."
            raise ValueError(msg)
        if mask.ndim == 4 and mask.shape[1] != 1:
            msg = f"Ground truth masks must have 1 channel, got {mask.shape[1]}."
            raise ValueError(msg)

        return Mask(mask.squeeze(1) if mask.ndim == 4 else mask, dtype=torch.bool)

    @staticmethod
    def validate_gt_label(labels: torch.Tensor | list[int] | None) -> torch.Tensor | None:
        """Validate the ground truth labels for a batch of depth images.

        Args:
            labels (torch.Tensor | list[int] | None): Input ground truth labels.

        Returns:
            torch.Tensor | None: Validated ground truth labels as a boolean tensor, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor or list[int].
            ValueError: If the labels shape is invalid.

        Examples:
            >>> labels = torch.tensor([0, 1, 1, 0])
            >>> validated_labels = DepthBatchValidator.validate_gt_label(labels)
            >>> validated_labels
            tensor([False,  True,  True, False])
        """
        if labels is None:
            return None
        if isinstance(labels, list):
            labels = torch.tensor(labels)
        if not isinstance(labels, torch.Tensor):
            msg = f"Ground truth labels must be a torch.Tensor or list[int], got {type(labels)}."
            raise TypeError(msg)
        if labels.ndim != 1:
            msg = f"Ground truth labels must be 1-dimensional, got shape {labels.shape}."
            raise ValueError(msg)
        return labels.to(torch.bool)

    @staticmethod
    def validate_image_path(image_paths: list[str] | None) -> list[str] | None:
        """Validate the image paths for a batch of depth images.

        Args:
            image_paths (list[str] | None): Input image paths.

        Returns:
            list[str] | None: Validated image paths, or None.

        Examples:
            >>> paths = ["/path/to/depth1.png", "/path/to/depth2.png"]
            >>> validated_paths = DepthBatchValidator.validate_image_path(paths)
            >>> validated_paths == paths
            True
        """
        if image_paths is None:
            return None
        return [validate_path(path) for path in image_paths]

    @staticmethod
    def validate_mask_path(mask_paths: list[str] | None) -> list[str] | None:
        """Validate the mask paths for a batch of depth images.

        Args:
            mask_paths (list[str] | None): Input mask paths.

        Returns:
            list[str] | None: Validated mask paths, or None.

        Examples:
            >>> paths = ["/path/to/mask1.png", "/path/to/mask2.png"]
            >>> validated_paths = DepthBatchValidator.validate_mask_path(paths)
            >>> validated_paths == paths
            True
        """
        if mask_paths is None:
            return None
        return [validate_path(path) for path in mask_paths]

    @staticmethod
    def validate_anomaly_map(anomaly_maps: torch.Tensor | None) -> Mask | None:
        """Validate a batch of anomaly maps for depth images.

        Args:
            anomaly_maps (torch.Tensor | None): Input batch of anomaly maps.

        Returns:
            Mask | None: Validated batch of anomaly maps as a Mask, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> anomaly_maps = torch.rand(4, 1, 224, 224)  # Batch of 4 anomaly maps
            >>> validated_maps = DepthBatchValidator.validate_anomaly_map(anomaly_maps)
            >>> isinstance(validated_maps, Mask)
            True
            >>> validated_maps.shape
            torch.Size([4, 224, 224])
        """
        return DepthBatchValidator.validate_gt_mask(anomaly_maps)

    @staticmethod
    def validate_pred_mask(pred_masks: torch.Tensor | None) -> Mask | None:
        """Validate a batch of prediction masks for depth images.

        Args:
            pred_masks (torch.Tensor | None): Input batch of prediction masks.

        Returns:
            Mask | None: Validated batch of prediction masks, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> pred_masks = torch.randint(0, 2, (4, 1, 224, 224))  # Batch of 4 prediction masks
            >>> validated_masks = DepthBatchValidator.validate_pred_mask(pred_masks)
            >>> isinstance(validated_masks, Mask)
            True
            >>> validated_masks.shape
            torch.Size([4, 224, 224])
        """
        return DepthBatchValidator.validate_gt_mask(pred_masks)

    @staticmethod
    def validate_pred_label(labels: torch.Tensor | list[int] | None) -> torch.Tensor | None:
        """Validate the prediction labels for a batch of depth images.

        Args:
            labels (torch.Tensor | list[int] | None): Input prediction labels.

        Returns:
            torch.Tensor | None: Validated prediction labels as a boolean tensor, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor or list[int].
            ValueError: If the labels shape is invalid.

        Examples:
            >>> labels = torch.tensor([1, 0, 0, 1])
            >>> validated_labels = DepthBatchValidator.validate_pred_label(labels)
            >>> validated_labels
            tensor([ True, False, False,  True])
        """
        return DepthBatchValidator.validate_gt_label(labels)

    @staticmethod
    def validate_pred_score(scores: torch.Tensor | list[float] | None) -> torch.Tensor | None:
        """Validate the prediction scores for a batch of depth images.

        Args:
            scores (torch.Tensor | list[float] | None): Input prediction scores.

        Returns:
            torch.Tensor | None: Validated prediction scores as a tensor, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor or list[float].
            ValueError: If the scores are not between 0 and 1, or if the tensor shape is invalid.

        Examples:
            >>> scores = torch.tensor([0.1, 0.9, 0.3, 0.7])
            >>> validated_scores = DepthBatchValidator.validate_pred_score(scores)
            >>> validated_scores
            tensor([0.1000, 0.9000, 0.3000, 0.7000])
        """
        if scores is None:
            return None
        if isinstance(scores, list):
            scores = torch.tensor(scores)
        if not isinstance(scores, torch.Tensor):
            msg = f"Prediction scores must be a torch.Tensor or list[float], got {type(scores)}."
            raise TypeError(msg)
        if scores.ndim != 1:
            msg = f"Prediction scores must be 1-dimensional, got shape {scores.shape}."
            raise ValueError(msg)
        if not torch.all((scores >= 0) & (scores <= 1)):
            msg = "Prediction scores must be between 0 and 1."
            raise ValueError(msg)
        return scores.to(torch.float32)

    @staticmethod
    def validate_depth_map(depth_maps: torch.Tensor) -> Image:
        """Validate and convert a batch of depth map tensors.

        Args:
            depth_maps (torch.Tensor): Input batch of depth map tensors.

        Returns:
            Image: Validated and converted batch of depth maps.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the depth map shape is invalid.

        Examples:
            >>> import torch
            >>> tensor = torch.rand(4, 1, 224, 224)  # Batch of 4 depth maps
            >>> validated_depth_maps = DepthBatchValidator.validate_depth_map(tensor)
            >>> isinstance(validated_depth_maps, Image)
            True
            >>> validated_depth_maps.shape
            torch.Size([4, 1, 224, 224])
        """
        return DepthBatchValidator.validate_image(depth_maps)

    @staticmethod
    def validate_depth_path(depth_paths: list[str] | None) -> list[str] | None:
        """Validate the depth map paths for a batch of depth images.

        Args:
            depth_paths (list[str] | None): Input depth map paths.

        Returns:
            list[str] | None: Validated depth map paths, or None.

        Examples:
            >>> paths = ["/path/to/depth1.png", "/path/to/depth2.png"]
            >>> validated_paths = DepthBatchValidator.validate_depth_path(paths)
            >>> validated_paths == paths
            True
        """
        if depth_paths is None:
            return None
        return [validate_path(path) for path in depth_paths]
