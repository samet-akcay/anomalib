"""Numpy-based dataclasses for Anomalib.

This module provides numpy-based implementations of the generic dataclasses
used in Anomalib. These classes are designed to work with numpy arrays for
efficient data handling and processing in anomaly detection tasks.

The module includes the following main classes:

- NumpyItem: Represents a single item in Anomalib datasets using numpy arrays.
- NumpyBatch: Represents a batch of items in Anomalib datasets using numpy arrays.
- NumpyImageItem: Represents a single image item with additional image-specific fields.
- NumpyImageBatch: Represents a batch of image items with batch operations.
- NumpyVideoItem: Represents a single video item with video-specific fields.
- NumpyVideoBatch: Represents a batch of video items with video-specific operations.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from anomalib.dataclasses.validate.numpy.depth import DepthBatchValidator, DepthValidator
from anomalib.dataclasses.validate.numpy.image import ImageBatchValidator, ImageValidator
from anomalib.dataclasses.validate.numpy.video import VideoBatchValidator, VideoValidator

from .generic import (
    BatchIterateMixin,
    _DepthInputFields,
    _GenericBatch,
    _GenericItem,
    _ImageInputFields,
    _VideoInputFields,
)


@dataclass
class NumpyItem(_GenericItem[np.ndarray, np.ndarray, np.ndarray, str]):
    """Dataclass for a single item in Anomalib datasets using numpy arrays.

    This class extends _GenericItem for numpy-based data representation. It includes
    both input data (e.g., images, labels) and output data (e.g., predictions,
    anomaly maps) as numpy arrays. It is suitable for numpy-based processing
    pipelines in Anomalib.
    """


@dataclass
class NumpyBatch(_GenericBatch[np.ndarray, np.ndarray, np.ndarray, list[str]]):
    """Dataclass for a batch of items in Anomalib datasets using numpy arrays.

    This class extends _GenericBatch for batches of numpy-based data. It represents
    multiple data points for batch processing in anomaly detection tasks. It includes
    an additional dimension for batch size in all tensor-like fields.
    """


@dataclass
class NumpyImageItem(_ImageInputFields[str], NumpyItem):
    """Dataclass for a single image item in Anomalib datasets using numpy arrays.

    This class combines _ImageInputFields and NumpyItem for image-based anomaly detection.
    It includes image-specific fields and validation methods to ensure proper formatting
    for Anomalib's image-based models.

    Examples:
        >>> item = NumpyImageItem(
        ...     image=np.random.rand(224, 224, 3),
        ...     gt_label=np.array(1),
        ...     gt_mask=np.random.rand(224, 224) > 0.5,
        ...     anomaly_map=np.random.rand(224, 224),
        ...     pred_score=np.array(0.7),
        ...     pred_label=np.array(1),
        ...     image_path="path/to/image.jpg"
        ... )

        >>> # Access fields
        >>> image = item.image
        >>> label = item.gt_label
        >>> path = item.image_path
    """

    def validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validate the image array."""
        # assert image.ndim == 3, f"Expected 3D image, got {image.ndim}D image."
        # if image.shape[0] == 3:
        #     image = image.transpose(1, 2, 0)
        # return image
        return ImageValidator.validate_image(image)

    def validate_gt_label(self, gt_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth label array."""
        return ImageValidator.validate_gt_label(gt_label)

    def validate_gt_mask(self, gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask array."""
        return ImageValidator.validate_gt_mask(gt_mask)

    def validate_mask_path(self, mask_path: str | None) -> str | None:
        """Validate the mask path."""
        return ImageValidator.validate_mask_path(mask_path)

    def validate_anomaly_map(self, anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map array."""
        # if anomaly_map is None:
        #     return None
        # assert isinstance(anomaly_map, np.ndarray), f"Anomaly map must be a numpy array, got {type(anomaly_map)}."
        # assert anomaly_map.ndim in {
        #     2,
        #     3,
        # }, f"Anomaly map must have shape [H, W] or [1, H, W], got shape {anomaly_map.shape}."
        # if anomaly_map.ndim == 3:
        #     assert (
        #         anomaly_map.shape[0] == 1
        #     ), f"Anomaly map with 3 dimensions must have 1 channel, got {anomaly_map.shape[0]}."
        #     anomaly_map = anomaly_map.squeeze(0)
        # return anomaly_map.astype(np.float32)
        return ImageBatchValidator.validate_anomaly_map(anomaly_map)

    def validate_pred_score(self, pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction score array."""
        # if pred_score is None:
        #     return None
        # if pred_score.ndim == 1:
        #     assert len(pred_score) == 1, f"Expected single value for pred_score, got {len(pred_score)}."
        #     pred_score = pred_score[0]
        # return pred_score
        return ImageValidator.validate_pred_score(pred_score)

    def validate_pred_mask(self, pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask array."""
        return ImageValidator.validate_pred_mask(pred_mask)

    def validate_pred_label(self, pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label array."""
        return ImageValidator.validate_pred_label(pred_label)

    def validate_image_path(self, image_path: str | None) -> str | None:
        """Validate the image path."""
        return ImageValidator.validate_image_path(image_path)


@dataclass
class NumpyImageBatch(BatchIterateMixin[NumpyImageItem], _ImageInputFields[list[str]], NumpyBatch):
    """Dataclass for a batch of image items in Anomalib datasets using numpy arrays.

    This class combines BatchIterateMixin, _ImageInputFields, and NumpyBatch for batches
    of image data. It supports batch operations and iteration over individual NumpyImageItems.
    It ensures proper formatting for Anomalib's image-based models.

    Examples:
        >>> batch = NumpyImageBatch(
        ...     image=np.random.rand(32, 224, 224, 3),
        ...     gt_label=np.random.randint(0, 2, (32,)),
        ...     gt_mask=np.random.rand(32, 224, 224) > 0.5,
        ...     anomaly_map=np.random.rand(32, 224, 224),
        ...     pred_score=np.random.rand(32),
        ...     pred_label=np.random.randint(0, 2, (32,)),
        ...     image_path=["path/to/image_{}.jpg".format(i) for i in range(32)]
        ... )

        >>> # Access batch fields
        >>> images = batch.image
        >>> labels = batch.gt_label
        >>> paths = batch.image_path

        >>> # Iterate over items in the batch
        >>> for item in batch:
        ...     process_item(item)
    """

    item_class = NumpyImageItem

    def validate_image(self, image: np.ndarray) -> np.ndarray:
        return image

    def validate_gt_label(self, gt_label: np.ndarray) -> np.ndarray:
        return gt_label

    def validate_gt_mask(self, gt_mask: np.ndarray) -> np.ndarray:
        return gt_mask

    def validate_mask_path(self, mask_path: list[str]) -> list[str]:
        return mask_path

    def validate_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        return anomaly_map

    def validate_pred_score(self, pred_score: np.ndarray) -> np.ndarray:
        return pred_score

    def validate_pred_mask(self, pred_mask: np.ndarray) -> np.ndarray:
        return pred_mask

    def validate_pred_label(self, pred_label: np.ndarray) -> np.ndarray:
        return pred_label

    def validate_image_path(self, image_path: list[str]) -> list[str]:
        return image_path


@dataclass
class NumpyVideoItem(_VideoInputFields[np.ndarray, np.ndarray, np.ndarray, str], NumpyItem):
    """Dataclass for a single video item in Anomalib datasets using numpy arrays.

    This class combines _VideoInputFields and NumpyItem for video-based anomaly detection.
    It includes video-specific fields and validation methods to ensure proper formatting
    for Anomalib's video-based models.
    """

    def validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validate the video array."""
        return VideoValidator.validate_image(image)

    def validate_gt_label(self, gt_label: np.ndarray) -> np.ndarray:
        """Validate the video ground truth label array."""
        return VideoValidator.validate_gt_label(gt_label)

    def validate_gt_mask(self, gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the video ground truth mask array."""
        return VideoValidator.validate_gt_mask(gt_mask)

    def validate_mask_path(self, mask_path: str | None) -> str | None:
        """Validate the video mask path."""
        return VideoValidator.validate_mask_path(mask_path)

    def validate_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        """Validate the video anomaly map array."""
        return VideoValidator.validate_anomaly_map(anomaly_map)

    def validate_pred_score(self, pred_score: np.ndarray) -> np.ndarray:
        """Validate the video prediction score array."""
        return VideoValidator.validate_pred_score(pred_score)

    def validate_pred_mask(self, pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the video prediction mask array."""
        return VideoValidator.validate_pred_mask(pred_mask)

    def validate_pred_label(self, pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the video prediction label array."""
        return VideoValidator.validate_pred_label(pred_label)

    def validate_original_image(self, original_image: np.ndarray) -> np.ndarray:
        """Validate the video original image array."""
        return VideoValidator.validate_original_image(original_image)

    def validate_video_path(self, video_path: str | None) -> str | None:
        """Validate the video path."""
        return VideoValidator.validate_video_path(video_path)

    def validate_target_frame(self, target_frame: np.ndarray) -> np.ndarray:
        """Validate the video target frame array."""
        return VideoValidator.validate_target_frame(target_frame)

    def validate_frames(self, frames: np.ndarray) -> np.ndarray:
        """Validate the video frames array."""
        return VideoValidator.validate_frames(frames)

    def validate_last_frame(self, last_frame: np.ndarray) -> np.ndarray:
        """Validate the video last frame array."""
        return VideoValidator.validate_last_frame(last_frame)


@dataclass
class NumpyVideoBatch(
    BatchIterateMixin[NumpyVideoItem],
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, list[str]],
    NumpyBatch,
):
    """Dataclass for a batch of video items in Anomalib datasets using numpy arrays.

    This class combines BatchIterateMixin, _VideoInputFields, and NumpyBatch for batches
    of video data. It supports batch operations and iteration over individual NumpyVideoItems.
    It ensures proper formatting for Anomalib's video-based models.
    """

    item_class = NumpyVideoItem

    def validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validate the image array."""
        return VideoBatchValidator.validate_image(image)

    def validate_gt_label(self, gt_label: np.ndarray | list[list[int]] | None) -> np.ndarray | None:
        """Validate the ground truth label array."""
        return VideoBatchValidator.validate_gt_label(gt_label)

    def validate_gt_mask(self, gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask array."""
        return VideoBatchValidator.validate_gt_mask(gt_mask)

    def validate_mask_path(self, mask_path: list[str] | None) -> list[str] | None:
        """Validate the mask path."""
        return VideoBatchValidator.validate_mask_path(mask_path)

    def validate_anomaly_map(self, anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map array."""
        return VideoBatchValidator.validate_anomaly_map(anomaly_map)

    def validate_pred_score(self, pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction score array."""
        return VideoBatchValidator.validate_pred_score(pred_score)

    def validate_pred_mask(self, pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask array."""
        return VideoBatchValidator.validate_pred_mask(pred_mask)

    def validate_pred_label(self, pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label array."""
        return VideoBatchValidator.validate_pred_label(pred_label)

    def validate_original_image(self, original_image: np.ndarray) -> np.ndarray:
        """Validate the original image array."""
        return VideoBatchValidator.validate_original_image(original_image)

    def validate_video_path(self, video_path: list[str]) -> list[str]:
        """Validate the video path."""
        return VideoBatchValidator.validate_video_path(video_path)

    def validate_target_frame(self, target_frame: np.ndarray) -> np.ndarray:
        """Validate the target frame array."""
        return VideoBatchValidator.validate_target_frame(target_frame)

    def validate_frames(self, frames: np.ndarray) -> np.ndarray:
        """Validate the frames array."""
        return VideoBatchValidator.validate_frames(frames)

    def validate_last_frame(self, last_frame: np.ndarray) -> np.ndarray:
        """Validate the last frame array."""
        return VideoBatchValidator.validate_last_frame(last_frame)


@dataclass
class NumpyDepthItem(_DepthInputFields[np.ndarray, str], NumpyItem):
    """Dataclass for a single depth item in Anomalib datasets using numpy arrays.

    This class combines _DepthInputFields and NumpyItem for depth-based anomaly detection.
    It includes depth-specific fields and validation methods to ensure proper formatting
    for Anomalib's depth-based models.

    Examples:
        >>> item = NumpyDepthItem(
        ...     image=np.random.rand(224, 224, 3),
        ...     depth_map=np.random.rand(224, 224),
        ...     gt_label=np.array(1),
        ...     gt_mask=np.random.rand(224, 224) > 0.5,
        ...     anomaly_map=np.random.rand(224, 224),
        ...     pred_score=np.array(0.7),
        ...     pred_label=np.array(1),
        ...     image_path="path/to/image.jpg"
        ... )

        >>> # Access fields
        >>> image = item.image
        >>> depth = item.depth_map
        >>> label = item.gt_label
        >>> path = item.image_path
    """

    def validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validate the depth image."""
        return DepthValidator.validate_image(image)

    def validate_gt_label(self, gt_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth label."""
        return DepthValidator.validate_gt_label(gt_label)

    def validate_gt_mask(self, gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask."""
        return DepthValidator.validate_gt_mask(gt_mask)

    def validate_mask_path(self, mask_path: str | None) -> str | None:
        """Validate the mask path."""
        return DepthValidator.validate_mask_path(mask_path)

    def validate_anomaly_map(self, anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map."""
        return DepthValidator.validate_anomaly_map(anomaly_map)

    def validate_pred_score(self, pred_score: np.ndarray | float | None) -> np.ndarray | float | None:
        """Validate the prediction score."""
        return DepthValidator.validate_pred_score(pred_score)

    def validate_pred_mask(self, pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask."""
        return DepthValidator.validate_pred_mask(pred_mask)

    def validate_pred_label(self, pred_label: np.ndarray | int | None) -> np.ndarray | int | None:
        """Validate the prediction label."""
        return DepthValidator.validate_pred_label(pred_label)

    def validate_image_path(self, image_path: str | None) -> str | None:
        """Validate the image path."""
        return DepthValidator.validate_image_path(image_path)

    def validate_depth_map(self, depth_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the depth map."""
        return DepthValidator.validate_depth_map(depth_map)


@dataclass
class NumpyDepthBatch(BatchIterateMixin[NumpyDepthItem], _DepthInputFields[np.ndarray, list[str]], NumpyBatch):
    """Dataclass for a batch of depth items in Anomalib datasets using numpy arrays.

    This class combines BatchIterateMixin, _DepthInputFields, and NumpyBatch for batches
    of depth data. It supports batch operations and iteration over individual NumpyDepthItems.
    It ensures proper formatting for Anomalib's depth-based models.

    Examples:
        >>> batch = NumpyDepthBatch(
        ...     image=np.random.rand(32, 224, 224, 3),
        ...     depth_map=np.random.rand(32, 224, 224),
        ...     gt_label=np.random.randint(0, 2, (32,)),
        ...     gt_mask=np.random.rand(32, 224, 224) > 0.5,
        ...     anomaly_map=np.random.rand(32, 224, 224),
        ...     pred_score=np.random.rand(32),
        ...     pred_label=np.random.randint(0, 2, (32,)),
        ...     image_path=["path/to/image_{}.jpg".format(i) for i in range(32)]
        ... )

        >>> # Access batch fields
        >>> images = batch.image
        >>> depths = batch.depth_map
        >>> labels = batch.gt_label
        >>> paths = batch.image_path

        >>> # Iterate over items in the batch
        >>> for item in batch:
        ...     process_item(item)
    """

    item_class = NumpyDepthItem

    def validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validate the image array."""
        return DepthBatchValidator.validate_image(image)

    def validate_gt_label(self, gt_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth label."""
        return DepthBatchValidator.validate_gt_label(gt_label)

    def validate_gt_mask(self, gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask."""
        return DepthBatchValidator.validate_gt_mask(gt_mask)

    def validate_mask_path(self, mask_path: list[str] | None) -> list[str] | None:
        """Validate the mask path."""
        return DepthBatchValidator.validate_mask_path(mask_path)

    def validate_anomaly_map(self, anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map."""
        return DepthBatchValidator.validate_anomaly_map(anomaly_map)

    def validate_pred_score(self, pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction score."""
        return DepthBatchValidator.validate_pred_score(pred_score)

    def validate_pred_mask(self, pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask."""
        return DepthBatchValidator.validate_pred_mask(pred_mask)

    def validate_pred_label(self, pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label."""
        return DepthBatchValidator.validate_pred_label(pred_label)

    def validate_image_path(self, image_path: list[str] | None) -> list[str] | None:
        """Validate the image path."""
        return DepthBatchValidator.validate_image_path(image_path)

    def validate_depth_map(self, depth_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the depth map."""
        return DepthBatchValidator.validate_depth_map(depth_map)
