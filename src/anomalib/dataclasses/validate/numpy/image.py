"""Numpy.ndarray validation functions for image data.

Sections:
    - Item-level image validation
    - Batch-level image validation
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from anomalib.dataclasses.validate.path import validate_path


class ImageValidator:
    """Validate numpy.ndarray data for images."""

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate and convert the input image array.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            np.ndarray: Validated and converted image.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the image shape or number of channels is invalid.

        Examples:
            >>> import numpy as np
            >>> from anomalib.dataclasses.validators.numpy.image import ImageValidator
            >>> array = np.random.rand(224, 224, 3)
            >>> validated_image = ImageValidator.validate_image(array)
            >>> validated_image.shape
            (1, 224, 224, 3)
            >>> validated_image.dtype
            dtype('float32')
        """
        if not isinstance(image, np.ndarray):
            msg = f"Image must be a numpy.ndarray, got {type(image)}."
            raise TypeError(msg)

        if image.ndim not in {3, 4}:
            msg = f"Image must have shape [H, W, C] or [N, H, W, C], got shape {image.shape}."
            raise ValueError(msg)

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)  # add batch dimension

        if image.shape[-1] not in {1, 3, 4}:
            msg = f"Invalid number of channels: {image.shape[-1]}. Expected 1, 3, or 4."
            raise ValueError(msg)

        return image.astype(np.float32)

    @staticmethod
    def validate_gt_label(label: int | np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth label.

        Args:
            label (int | np.ndarray | None): Input ground truth label.

        Returns:
            np.ndarray | None: Validated ground truth label as a boolean array, or None.

        Raises:
            TypeError: If the input is neither an integer nor a numpy.ndarray.
            ValueError: If the label shape or dtype is invalid.

        Examples:
            >>> import numpy as np
            >>> from anomalib.dataclasses.validators.numpy.image import ImageValidator
            >>> label_int = 1
            >>> validated_label = ImageValidator.validate_gt_label(label_int)
            >>> validated_label
            array(True)
            >>> label_array = np.array(0)
            >>> validated_label = ImageValidator.validate_gt_label(label_array)
            >>> validated_label
            array(False)
        """
        if label is None:
            return None
        if isinstance(label, int):
            return np.array(label, dtype=bool)
        if isinstance(label, np.ndarray):
            if label.ndim != 0:
                msg = f"Ground truth label must be a scalar, got shape {label.shape}."
                raise ValueError(msg)
            if np.issubdtype(label.dtype, np.floating):
                msg = f"Ground truth label must be boolean or integer, got {label.dtype}."
                raise ValueError(msg)
            return label.astype(bool)
        msg = f"Ground truth label must be an integer or a numpy.ndarray, got {type(label)}."
        raise TypeError(msg)

    @staticmethod
    def validate_gt_mask(mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask.

        Args:
            mask (np.ndarray | None): Input ground truth mask.

        Returns:
            np.ndarray | None: Validated ground truth mask, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> import numpy as np
            >>> from anomalib.dataclasses.validators.numpy.image import ImageValidator
            >>> mask = np.random.randint(0, 2, (224, 224, 1))
            >>> validated_mask = ImageValidator.validate_gt_mask(mask)
            >>> validated_mask.shape
            (224, 224)
            >>> validated_mask.dtype
            dtype('bool')
        """
        if mask is None:
            return None
        if not isinstance(mask, np.ndarray):
            msg = f"Ground truth mask must be a numpy.ndarray, got {type(mask)}."
            raise TypeError(msg)
        if mask.ndim not in {2, 3}:
            msg = f"Ground truth mask must have shape [H, W] or [H, W, 1], got shape {mask.shape}."
            raise ValueError(msg)
        if mask.ndim == 3:
            if mask.shape[-1] != 1:
                msg = f"Ground truth mask must have 1 channel, got {mask.shape[-1]}."
                raise ValueError(msg)
            mask = np.squeeze(mask, axis=-1)
        return mask.astype(bool)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map.

        Args:
            anomaly_map (np.ndarray | None): Input anomaly map.

        Returns:
            np.ndarray | None: Validated anomaly map, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> import numpy as np
            >>> from anomalib.dataclasses.validators.numpy.image import ImageValidator
            >>> anomaly_map = np.random.rand(224, 224, 1)
            >>> validated_map = ImageValidator.validate_anomaly_map(anomaly_map)
            >>> validated_map.shape
            (224, 224)
            >>> validated_map.dtype
            dtype('float32')
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, np.ndarray):
            msg = f"Anomaly map must be a numpy.ndarray, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim not in {2, 3}:
            msg = f"Anomaly map must have shape [H, W] or [H, W, 1], got shape {anomaly_map.shape}."
            raise ValueError(msg)
        if anomaly_map.ndim == 3:
            if anomaly_map.shape[-1] != 1:
                msg = f"Anomaly map must have 1 channel, got {anomaly_map.shape[-1]}."
                raise ValueError(msg)
            anomaly_map = np.squeeze(anomaly_map, axis=-1)
        return anomaly_map.astype(np.float32)

    @staticmethod
    def validate_image_path(image_path: str | None) -> str | None:
        """Validate the image path.

        Args:
            image_path (str | None): Input image path.

        Returns:
            str | None: Validated image path, or None.

        Examples:
            >>> from anomalib.dataclasses.validators.numpy.image import ImageValidator
            >>> path = "/path/to/image.jpg"
            >>> validated_path = ImageValidator.validate_image_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(image_path) if image_path else None

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path.

        Args:
            mask_path (str | None): Input mask path.

        Returns:
            str | None: Validated mask path, or None.

        Examples:
            >>> from anomalib.dataclasses.validators.numpy.image import ImageValidator
            >>> path = "/path/to/mask.png"
            >>> validated_path = ImageValidator.validate_mask_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(mask_path) if mask_path else None

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | float) -> np.ndarray:
        """Validate the prediction score.

        Args:
            pred_score (np.ndarray | float): Input prediction score.

        Returns:
            np.ndarray: Validated prediction score.

        Raises:
            TypeError: If the input is neither a float nor a numpy.ndarray.
            ValueError: If the prediction score is not a scalar.

        Examples:
            >>> import numpy as np
            >>> from anomalib.dataclasses.validators.numpy.image import ImageValidator
            >>> score = 0.8
            >>> validated_score = ImageValidator.validate_pred_score(score)
            >>> validated_score
            array(0.8, dtype=float32)
            >>> score_array = np.array(0.6)
            >>> validated_score = ImageValidator.validate_pred_score(score_array)
            >>> validated_score
            array(0.6, dtype=float32)
        """
        if isinstance(pred_score, float):
            pred_score = np.array(pred_score)
        if not isinstance(pred_score, np.ndarray):
            msg = f"Prediction score must be a numpy.ndarray or float, got {type(pred_score)}."
            raise TypeError(msg)
        pred_score = np.squeeze(pred_score)
        if pred_score.ndim != 0:
            msg = f"Prediction score must be a scalar, got shape {pred_score.shape}."
            raise ValueError(msg)
        return pred_score.astype(np.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask.

        Args:
            pred_mask (np.ndarray | None): Input prediction mask.

        Returns:
            np.ndarray | None: Validated prediction mask, or None.

        Examples:
            >>> import numpy as np
            >>> from anomalib.dataclasses.validators.numpy.image import ImageValidator
            >>> mask = np.random.randint(0, 2, (224, 224, 1))
            >>> validated_mask = ImageValidator.validate_pred_mask(mask)
            >>> validated_mask.shape
            (224, 224)
            >>> validated_mask.dtype
            dtype('bool')
        """
        return ImageValidator.validate_gt_mask(pred_mask)  # We can reuse the gt_mask validation

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label.

        Args:
            pred_label (np.ndarray | None): Input prediction label.

        Returns:
            np.ndarray | None: Validated prediction label as a boolean array, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the prediction label is not a scalar.

        Examples:
            >>> import numpy as np
            >>> from anomalib.dataclasses.validators.numpy.image import ImageValidator
            >>> label = np.array(1)
            >>> validated_label = ImageValidator.validate_pred_label(label)
            >>> validated_label
            array(True)
        """
        if pred_label is None:
            return None
        if not isinstance(pred_label, np.ndarray):
            msg = f"Predicted label must be a numpy.ndarray, got {type(pred_label)}."
            raise TypeError(msg)
        pred_label = np.squeeze(pred_label)
        if pred_label.ndim != 0:
            msg = f"Predicted label must be a scalar, got shape {pred_label.shape}."
            raise ValueError(msg)
        return pred_label.astype(bool)


class ImageBatchValidator:
    """Validate numpy.ndarray data for batches of images."""

    @staticmethod
    def validate_image(images: np.ndarray) -> np.ndarray:
        """Validate and convert a batch of image arrays.

        Args:
            images (np.ndarray): Input batch of image arrays.

        Returns:
            np.ndarray: Validated and converted batch of images.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the image shape or number of channels is invalid.

        Examples:
            >>> import numpy as np
            >>> array = np.random.rand(4, 224, 224, 3)  # Batch of 4 RGB images
            >>> validated_images = ImageBatchValidator.validate_image(array)
            >>> validated_images.shape
            (4, 224, 224, 3)
        """
        if not isinstance(images, np.ndarray):
            msg = f"Images must be a numpy.ndarray, got {type(images)}."
            raise TypeError(msg)

        if images.ndim != 4:
            msg = f"Images must have shape [N, H, W, C], got shape {images.shape}."
            raise ValueError(msg)

        if images.shape[-1] not in {1, 3, 4}:
            msg = f"Invalid number of channels: {images.shape[-1]}. Expected 1, 3, or 4."
            raise ValueError(msg)

        return images.astype(np.float32)

    @staticmethod
    def validate_gt_label(labels: np.ndarray | list[int] | None) -> np.ndarray | None:
        """Validate the ground truth labels for a batch of images.

        Args:
            labels (np.ndarray | list[int] | None): Input ground truth labels.

        Returns:
            np.ndarray | None: Validated ground truth labels as a boolean array, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray or list[int].
            ValueError: If the labels shape is invalid.

        Examples:
            >>> labels = np.array([0, 1, 1, 0])
            >>> validated_labels = ImageBatchValidator.validate_gt_label(labels)
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
        """Validate a batch of ground truth masks for images.

        Args:
            masks (np.ndarray | None): Input batch of ground truth masks.

        Returns:
            np.ndarray | None: Validated batch of ground truth masks, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> masks = np.random.randint(0, 2, (4, 224, 224, 1))  # Batch of 4 masks
            >>> validated_masks = ImageBatchValidator.validate_gt_mask(masks)
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
        """Validate a batch of anomaly maps for images.

        Args:
            anomaly_maps (np.ndarray | None): Input batch of anomaly maps.

        Returns:
            np.ndarray | None: Validated batch of anomaly maps, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> anomaly_maps = np.random.rand(4, 224, 224, 1)  # Batch of 4 anomaly maps
            >>> validated_maps = ImageBatchValidator.validate_anomaly_map(anomaly_maps)
            >>> validated_maps.shape
            (4, 224, 224)
        """
        return ImageBatchValidator.validate_gt_mask(anomaly_maps)  # Same validation as gt_mask

    @staticmethod
    def validate_image_path(image_paths: list[str] | None) -> list[str] | None:
        """Validate the image paths for a batch of images.

        Args:
            image_paths (list[str] | None): Input image paths.

        Returns:
            list[str] | None: Validated image paths, or None.

        Examples:
            >>> paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
            >>> validated_paths = ImageBatchValidator.validate_image_path(paths)
            >>> validated_paths == paths
            True
        """
        if image_paths is None:
            return None
        return [validate_path(path) for path in image_paths]

    @staticmethod
    def validate_mask_path(mask_paths: list[str] | None) -> list[str] | None:
        """Validate the mask paths for a batch of images.

        Args:
            mask_paths (list[str] | None): Input mask paths.

        Returns:
            list[str] | None: Validated mask paths, or None.

        Examples:
            >>> paths = ["/path/to/mask1.png", "/path/to/mask2.png"]
            >>> validated_paths = ImageBatchValidator.validate_mask_path(paths)
            >>> validated_paths == paths
            True
        """
        if mask_paths is None:
            return None
        return [validate_path(path) for path in mask_paths]

    @staticmethod
    def validate_pred_score(pred_scores: np.ndarray | list[float]) -> np.ndarray:
        """Validate the prediction scores for a batch of images.

        Args:
            pred_scores (np.ndarray | list[float]): Input prediction scores.

        Returns:
            np.ndarray: Validated prediction scores.

        Raises:
            TypeError: If the input is neither a numpy.ndarray nor a list of floats.
            ValueError: If the prediction scores are not 1-dimensional.

        Examples:
            >>> scores = np.array([0.8, 0.2, 0.6, 0.4])
            >>> validated_scores = ImageBatchValidator.validate_pred_scores(scores)
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
    def validate_pred_label(pred_labels: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction labels for a batch of images.

        Args:
            pred_labels (np.ndarray | None): Input prediction labels.

        Returns:
            np.ndarray | None: Validated prediction labels as a boolean array, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the prediction labels are not 1-dimensional.

        Examples:
            >>> labels = np.array([1, 0, 1, 0])
            >>> validated_labels = ImageBatchValidator.validate_pred_labels(labels)
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
    def validate_pred_mask(pred_masks: np.ndarray | None) -> np.ndarray | None:
        """Validate a batch of prediction masks for images.

        Args:
            pred_masks (np.ndarray | None): Input batch of prediction masks.

        Returns:
            np.ndarray | None: Validated batch of prediction masks, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> pred_masks = np.random.randint(0, 2, (4, 224, 224, 1))  # Batch of 4 prediction masks
            >>> validated_masks = ImageBatchValidator.validate_pred_masks(pred_masks)
            >>> validated_masks.shape
            (4, 224, 224)
        """
        return ImageBatchValidator.validate_gt_mask(pred_masks)  # We can reuse the gt_mask validation
