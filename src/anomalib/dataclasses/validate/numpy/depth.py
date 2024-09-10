"""Numpy.ndarray validation functions for depth data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from anomalib.dataclasses.validate.path import validate_path


class DepthValidator:
    """Validate numpy.ndarray data for depth maps."""

    @staticmethod
    def validate_depth_map(depth_map: np.ndarray) -> np.ndarray:
        """Validate the depth map.

        Args:
            depth_map (np.ndarray): Input depth map.

        Returns:
            np.ndarray: Validated depth map.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the depth map shape is invalid.

        Examples:
            >>> import numpy as np
            >>> depth_map = np.random.rand(224, 224, 1)
            >>> validated_map = DepthValidator.validate_depth_map(depth_map)
            >>> validated_map.shape
            (224, 224)
        """
        if not isinstance(depth_map, np.ndarray):
            msg = f"Depth map must be a numpy.ndarray, got {type(depth_map)}."
            raise TypeError(msg)

        if depth_map.ndim not in {2, 3}:
            msg = f"Depth map must have shape [H, W] or [H, W, 1], got shape {depth_map.shape}."
            raise ValueError(msg)

        if depth_map.ndim == 3 and depth_map.shape[-1] != 1:
            msg = f"Depth map must have 1 channel, got {depth_map.shape[-1]}."
            raise ValueError(msg)

        return np.squeeze(depth_map)

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate and convert a depth image array.

        Args:
            image (np.ndarray): Input depth image array.

        Returns:
            np.ndarray: Validated and converted depth image.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the image shape is invalid.

        Examples:
            >>> import numpy as np
            >>> image = np.random.rand(224, 224, 1)
            >>> validated_image = DepthValidator.validate_image(image)
            >>> validated_image.shape
            (224, 224, 1)
        """
        if not isinstance(image, np.ndarray):
            msg = f"Image must be a numpy.ndarray, got {type(image)}."
            raise TypeError(msg)

        if image.ndim not in {2, 3}:
            msg = f"Depth image must have shape [H, W] or [H, W, 1], got shape {image.shape}."
            raise ValueError(msg)

        if image.ndim == 3 and image.shape[-1] != 1:
            msg = f"Depth image must have 1 channel, got {image.shape[-1]}."
            raise ValueError(msg)

        return image if image.ndim == 3 else image[..., np.newaxis]

    @staticmethod
    def validate_gt_mask(mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask for a depth image.

        Args:
            mask (np.ndarray | None): Input ground truth mask.

        Returns:
            np.ndarray | None: Validated ground truth mask, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> import numpy as np
            >>> mask = np.random.randint(0, 2, (224, 224, 1))
            >>> validated_mask = DepthValidator.validate_gt_mask(mask)
            >>> validated_mask.shape
            (224, 224)
        """
        if mask is None:
            return None
        if not isinstance(mask, np.ndarray):
            msg = f"Ground truth mask must be a numpy.ndarray, got {type(mask)}."
            raise TypeError(msg)
        if mask.ndim not in {2, 3}:
            msg = f"Ground truth mask must have shape [H, W] or [H, W, 1], got shape {mask.shape}."
            raise ValueError(msg)
        if mask.ndim == 3 and mask.shape[-1] != 1:
            msg = f"Ground truth mask must have 1 channel, got {mask.shape[-1]}."
            raise ValueError(msg)

        return np.squeeze(mask).astype(bool)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map for a depth image.

        Args:
            anomaly_map (np.ndarray | None): Input anomaly map.

        Returns:
            np.ndarray | None: Validated anomaly map, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> import numpy as np
            >>> anomaly_map = np.random.rand(224, 224, 1)
            >>> validated_map = DepthValidator.validate_anomaly_map(anomaly_map)
            >>> validated_map.shape
            (224, 224)
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, np.ndarray):
            msg = f"Anomaly map must be a numpy.ndarray, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim not in {2, 3}:
            msg = f"Anomaly map must have shape [H, W] or [H, W, 1], got shape {anomaly_map.shape}."
            raise ValueError(msg)
        if anomaly_map.ndim == 3 and anomaly_map.shape[-1] != 1:
            msg = f"Anomaly map must have 1 channel, got {anomaly_map.shape[-1]}."
            raise ValueError(msg)

        return np.squeeze(anomaly_map)

    @staticmethod
    def validate_gt_label(label: int | None) -> bool | None:
        """Validate the ground truth label for a depth image.

        Args:
            label (int | None): Input ground truth label.

        Returns:
            bool | None: Validated ground truth label as a boolean, or None.

        Raises:
            TypeError: If the input is not an int.
            ValueError: If the label is not 0 or 1.

        Examples:
            >>> validated_label = DepthValidator.validate_gt_label(1)
            >>> validated_label
            True
        """
        if label is None:
            return None
        if not isinstance(label, int):
            msg = f"Ground truth label must be an int, got {type(label)}."
            raise TypeError(msg)
        if label not in {0, 1}:
            msg = f"Ground truth label must be 0 or 1, got {label}."
            raise ValueError(msg)
        return bool(label)

    @staticmethod
    def validate_pred_score(score: float | None) -> float | None:
        """Validate the prediction score for a depth image.

        Args:
            score (float | None): Input prediction score.

        Returns:
            float | None: Validated prediction score, or None.

        Raises:
            TypeError: If the input is not a float.
            ValueError: If the score is not between 0 and 1.

        Examples:
            >>> validated_score = DepthValidator.validate_pred_score(0.7)
            >>> validated_score
            0.7
        """
        if score is None:
            return None
        if not isinstance(score, float):
            msg = f"Prediction score must be a float, got {type(score)}."
            raise TypeError(msg)
        if not 0 <= score <= 1:
            msg = f"Prediction score must be between 0 and 1, got {score}."
            raise ValueError(msg)
        return score

    @staticmethod
    def validate_pred_label(label: int | None) -> bool | None:
        """Validate the prediction label for a depth image.

        Args:
            label (int | None): Input prediction label.

        Returns:
            bool | None: Validated prediction label as a boolean, or None.

        Raises:
            TypeError: If the input is not an int.
            ValueError: If the label is not 0 or 1.

        Examples:
            >>> validated_label = DepthValidator.validate_pred_label(0)
            >>> validated_label
            False
        """
        return DepthValidator.validate_gt_label(label)  # Same validation as gt_label

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


class DepthBatchValidator:
    """Validate numpy.ndarray data for batches of depth maps."""

    @staticmethod
    def validate_depth_map(depth_maps: np.ndarray) -> np.ndarray:
        """Validate and convert a batch of depth map arrays.

        Args:
            depth_maps (np.ndarray): Input batch of depth map arrays.

        Returns:
            np.ndarray: Validated and converted batch of depth maps.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the depth map shape is invalid.

        Examples:
            >>> import numpy as np
            >>> array = np.random.rand(4, 224, 224, 1)  # Batch of 4 depth maps
            >>> validated_depth_maps = DepthBatchValidator.validate_depth_map(array)
            >>> validated_depth_maps.shape
            (4, 224, 224)
        """
        if not isinstance(depth_maps, np.ndarray):
            msg = f"Depth maps must be a numpy.ndarray, got {type(depth_maps)}."
            raise TypeError(msg)

        if depth_maps.ndim not in {3, 4}:
            msg = f"Depth maps must have shape [N, H, W] or [N, H, W, 1], got shape {depth_maps.shape}."
            raise ValueError(msg)

        if depth_maps.ndim == 4 and depth_maps.shape[-1] != 1:
            msg = f"Depth maps must have 1 channel, got {depth_maps.shape[-1]}."
            raise ValueError(msg)

        return np.squeeze(depth_maps, axis=-1).astype(np.float32)

    @staticmethod
    def validate_image(images: np.ndarray) -> np.ndarray:
        """Validate and convert a batch of depth image arrays.

        Args:
            images (np.ndarray): Input batch of depth image arrays.

        Returns:
            np.ndarray: Validated and converted batch of depth images.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the image shape is invalid.

        Examples:
            >>> import numpy as np
            >>> array = np.random.rand(4, 224, 224, 1)  # Batch of 4 depth images
            >>> validated_images = DepthBatchValidator.validate_image(array)
            >>> validated_images.shape
            (4, 224, 224, 1)
        """
        if not isinstance(images, np.ndarray):
            msg = f"Images must be a numpy.ndarray, got {type(images)}."
            raise TypeError(msg)

        if images.ndim != 4:
            msg = f"Depth images must have shape [N, H, W, 1], got shape {images.shape}."
            raise ValueError(msg)

        if images.shape[-1] != 1:
            msg = f"Depth images must have 1 channel, got {images.shape[-1]}."
            raise ValueError(msg)

        return images.astype(np.float32)

    @staticmethod
    def validate_gt_label(labels: np.ndarray | list[int] | None) -> np.ndarray | None:
        """Validate the ground truth labels for a batch of depth images.

        Args:
            labels (np.ndarray | list[int] | None): Input ground truth labels.

        Returns:
            np.ndarray | None: Validated ground truth labels as a boolean array, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray or list[int].
            ValueError: If the labels shape is invalid.

        Examples:
            >>> labels = np.array([0, 1, 1, 0])
            >>> validated_labels = DepthBatchValidator.validate_gt_label(labels)
            >>> validated_labels
            array([False,  True,  True, False])
        """
        if labels is None:
            return None
        if isinstance(labels, list):
            labels = np.array(labels)
        if not isinstance(labels, np.ndarray):
            msg = f"Ground truth labels must be a numpy.ndarray or list[int], got {type(labels)}."
            raise TypeError(msg)
        if labels.ndim != 1:
            msg = f"Ground truth labels must be 1-dimensional, got shape {labels.shape}."
            raise ValueError(msg)
        return labels.astype(bool)

    @staticmethod
    def validate_gt_mask(masks: np.ndarray | None) -> np.ndarray | None:
        """Validate a batch of ground truth masks for depth images.

        Args:
            masks (np.ndarray | None): Input batch of ground truth masks.

        Returns:
            np.ndarray | None: Validated batch of ground truth masks, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> masks = np.random.randint(0, 2, (4, 224, 224, 1))  # Batch of 4 masks
            >>> validated_masks = DepthBatchValidator.validate_gt_mask(masks)
            >>> validated_masks.shape
            (4, 224, 224)
        """
        if masks is None:
            return None
        if not isinstance(masks, np.ndarray):
            msg = f"Ground truth masks must be a numpy.ndarray, got {type(masks)}."
            raise TypeError(msg)
        if masks.ndim not in {3, 4}:
            msg = f"Ground truth masks must have shape [N, H, W] or [N, H, W, 1], got shape {masks.shape}."
            raise ValueError(msg)
        if masks.ndim == 4 and masks.shape[-1] != 1:
            msg = f"Ground truth masks must have 1 channel, got {masks.shape[-1]}."
            raise ValueError(msg)
        return np.squeeze(masks, axis=-1).astype(bool)

    @staticmethod
    def validate_anomaly_map(anomaly_maps: np.ndarray | None) -> np.ndarray | None:
        """Validate a batch of anomaly maps for depth images.

        Args:
            anomaly_maps (np.ndarray | None): Input batch of anomaly maps.

        Returns:
            np.ndarray | None: Validated batch of anomaly maps, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> anomaly_maps = np.random.rand(4, 224, 224, 1)  # Batch of 4 anomaly maps
            >>> validated_maps = DepthBatchValidator.validate_anomaly_map(anomaly_maps)
            >>> validated_maps.shape
            (4, 224, 224)
        """
        return DepthBatchValidator.validate_gt_mask(anomaly_maps)  # Same validation as gt_mask

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
    def validate_pred_scores(pred_scores: np.ndarray | list[float]) -> np.ndarray:
        """Validate the prediction scores for a batch of depth images.

        Args:
            pred_scores (np.ndarray | list[float]): Input prediction scores.

        Returns:
            np.ndarray: Validated prediction scores.

        Raises:
            TypeError: If the input is neither a numpy.ndarray nor a list of floats.
            ValueError: If the prediction scores are not 1-dimensional.

        Examples:
            >>> scores = np.array([0.8, 0.2, 0.6, 0.4])
            >>> validated_scores = DepthBatchValidator.validate_pred_scores(scores)
            >>> validated_scores
            array([0.8, 0.2, 0.6, 0.4], dtype=float32)
        """
        if isinstance(pred_scores, list):
            pred_scores = np.array(pred_scores)
        if not isinstance(pred_scores, np.ndarray):
            msg = f"Prediction scores must be a numpy.ndarray or list[float], got {type(pred_scores)}."
            raise TypeError(msg)
        if pred_scores.ndim != 1:
            msg = f"Prediction scores must be 1-dimensional, got shape {pred_scores.shape}."
            raise ValueError(msg)
        return pred_scores.astype(np.float32)

    @staticmethod
    def validate_pred_labels(pred_labels: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction labels for a batch of depth images.

        Args:
            pred_labels (np.ndarray | None): Input prediction labels.

        Returns:
            np.ndarray | None: Validated prediction labels as a boolean array, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the prediction labels are not 1-dimensional.

        Examples:
            >>> labels = np.array([1, 0, 1, 0])
            >>> validated_labels = DepthBatchValidator.validate_pred_labels(labels)
            >>> validated_labels
            array([ True, False,  True, False])
        """
        if pred_labels is None:
            return None
        if not isinstance(pred_labels, np.ndarray):
            msg = f"Predicted labels must be a numpy.ndarray, got {type(pred_labels)}."
            raise TypeError(msg)
        if pred_labels.ndim != 1:
            msg = f"Predicted labels must be 1-dimensional, got shape {pred_labels.shape}."
            raise ValueError(msg)
        return pred_labels.astype(bool)

    @staticmethod
    def validate_pred_masks(pred_masks: np.ndarray | None) -> np.ndarray | None:
        """Validate a batch of prediction masks for depth images.

        Args:
            pred_masks (np.ndarray | None): Input batch of prediction masks.

        Returns:
            np.ndarray | None: Validated batch of prediction masks, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> pred_masks = np.random.randint(0, 2, (4, 224, 224, 1))  # Batch of 4 prediction masks
            >>> validated_masks = DepthBatchValidator.validate_pred_masks(pred_masks)
            >>> validated_masks.shape
            (4, 224, 224)
        """
        return DepthBatchValidator.validate_gt_mask(pred_masks)  # We can reuse the gt_mask validation

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction score array for a batch of depth items.

        Args:
            pred_score (np.ndarray | None): Prediction score array to validate.

        Returns:
            np.ndarray | None: Validated prediction score array.

        Raises:
            ValueError: If the prediction score array has an invalid shape or type.

        Examples:
            >>> import numpy as np
            >>> scores = np.array([0.1, 0.5, 0.9])
            >>> validated_scores = DepthBatchValidator.validate_pred_score(scores)
            >>> print(validated_scores)
            [0.1 0.5 0.9]
            >>> print(validated_scores.dtype)
            float32

            >>> invalid_scores = np.array([[0.1], [0.5], [0.9]])
            >>> DepthBatchValidator.validate_pred_score(invalid_scores)
            Traceback (most recent call last):
                ...
            ValueError: Prediction score must be a 1D array, got shape (3, 1).
        """
        if pred_score is None:
            return None
        if not isinstance(pred_score, np.ndarray):
            msg = f"Prediction score must be a numpy array, got {type(pred_score)}."
            raise TypeError(msg)
        if pred_score.ndim != 1:
            msg = f"Prediction score must be a 1D array, got shape {pred_score.shape}."
            raise ValueError(msg)
        return pred_score.astype(np.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask array for a batch of depth items.

        Args:
            pred_mask (np.ndarray | None): Prediction mask array to validate.

        Returns:
            np.ndarray | None: Validated prediction mask array.

        Raises:
            ValueError: If the prediction mask array has an invalid shape or type.

        Examples:
            >>> import numpy as np
            >>> masks = np.random.rand(5, 224, 224)  # 5 masks of size 224x224
            >>> validated_masks = DepthBatchValidator.validate_pred_mask(masks)
            >>> print(validated_masks.shape)
            (5, 224, 224)
            >>> print(validated_masks.dtype)
            float32

            >>> invalid_masks = np.random.rand(5, 224, 224, 1)
            >>> DepthBatchValidator.validate_pred_mask(invalid_masks)
            Traceback (most recent call last):
                ...
            ValueError: Prediction mask must be a 3D array [B, H, W], got shape (5, 224, 224, 1).
        """
        if pred_mask is None:
            return None
        if not isinstance(pred_mask, np.ndarray):
            msg = f"Prediction mask must be a numpy array, got {type(pred_mask)}."
            raise TypeError(msg)
        if pred_mask.ndim != 3:
            msg = f"Prediction mask must be a 3D array [B, H, W], got shape {pred_mask.shape}."
            raise ValueError(msg)
        return pred_mask.astype(np.float32)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label array for a batch of depth items.

        Args:
            pred_label (np.ndarray | None): Prediction label array to validate.

        Returns:
            np.ndarray | None: Validated prediction label array.

        Raises:
            ValueError: If the prediction label array has an invalid shape or type.

        Examples:
            >>> import numpy as np
            >>> labels = np.array([0, 1, 1, 0, 1])
            >>> validated_labels = DepthBatchValidator.validate_pred_label(labels)
            >>> print(validated_labels)
            [0 1 1 0 1]
            >>> print(validated_labels.dtype)
            int32

            >>> invalid_labels = np.array([[0], [1], [1]])
            >>> DepthBatchValidator.validate_pred_label(invalid_labels)
            Traceback (most recent call last):
                ...
            ValueError: Prediction label must be a 1D array, got shape (3, 1).
        """
        if pred_label is None:
            return None
        if not isinstance(pred_label, np.ndarray):
            msg = f"Prediction label must be a numpy array, got {type(pred_label)}."
            raise TypeError(msg)
        if pred_label.ndim != 1:
            msg = f"Prediction label must be a 1D array, got shape {pred_label.shape}."
            raise ValueError(msg)
        return pred_label.astype(np.int32)
