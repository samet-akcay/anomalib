"""Numpy.ndarray validation functions for video data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from anomalib.dataclasses.validate.path import validate_path

from .image import ImageBatchValidator, ImageValidator


class VideoValidator:
    """Validate numpy.ndarray data for individual videos."""

    @staticmethod
    def validate_image(video: np.ndarray) -> np.ndarray:
        """Validate and convert the input video array.

        Args:
            video (np.ndarray): Input video array.

        Returns:
            np.ndarray: Validated and converted video array.

        Examples:
            >>> import numpy as np
            >>> array = np.random.rand(10, 224, 224, 3)  # [T, H, W, C]
            >>> validated_video = VideoValidator.validate_image(array)
            >>> validated_video.shape
            (10, 224, 224, 3)
        """
        if not isinstance(video, np.ndarray):
            msg = f"Video must be a numpy.ndarray, got {type(video)}."
            raise TypeError(msg)
        if video.ndim != 4:
            msg = f"Video must have shape [T, H, W, C], got shape {video.shape}."
            raise ValueError(msg)
        if video.shape[-1] not in {1, 3}:
            msg = f"Invalid number of channels: {video.shape[-1]}. Expected 1 or 3."
            raise ValueError(msg)
        return video.astype(np.float32)

    @staticmethod
    def validate_gt_mask(mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask for video.

        Args:
            mask (np.ndarray | None): Input ground truth mask.

        Returns:
            np.ndarray | None: Validated ground truth mask, or None.

        Examples:
            >>> import numpy as np
            >>> mask = np.random.randint(0, 2, (10, 224, 224))  # [T, H, W]
            >>> validated_mask = VideoValidator.validate_gt_mask(mask)
            >>> validated_mask.shape
            (10, 224, 224)
        """
        if mask is None:
            return None
        if not isinstance(mask, np.ndarray):
            msg = f"Mask must be a numpy.ndarray, got {type(mask)}."
            raise TypeError(msg)
        if mask.ndim != 3:
            msg = f"Mask must have shape [T, H, W], got shape {mask.shape}."
            raise ValueError(msg)
        return mask.astype(bool)

    @staticmethod
    def validate_gt_label(label: np.ndarray | int | None) -> np.ndarray | None:
        """Validate the ground truth label for video.

        Args:
            label (np.ndarray | int | None): Input ground truth label.

        Returns:
            np.ndarray | None: Validated ground truth label, or None.

        Examples:
            >>> import numpy as np
            >>> label = np.array([1, 0, 1])
            >>> validated_label = VideoValidator.validate_gt_label(label)
            >>> validated_label
            array([1, 0, 1])
        """
        if label is None:
            return None
        if isinstance(label, int):
            label = np.array([label])
        if not isinstance(label, np.ndarray):
            msg = f"Label must be a numpy.ndarray or int, got {type(label)}."
            raise TypeError(msg)
        return label.astype(np.int32)

    @staticmethod
    def validate_original_image(video: np.ndarray) -> np.ndarray:
        """Validate and convert the original input video array.

        Args:
            video (np.ndarray): Original input video array.

        Returns:
            np.ndarray: Validated and converted original video.

        Examples:
            >>> import numpy as np
            >>> array = np.random.rand(10, 224, 224, 3)  # [T, H, W, C]
            >>> validated_video = VideoValidator.validate_original_image(array)
            >>> validated_video.shape
            (10, 224, 224, 3)
        """
        return VideoValidator.validate_image(video)

    @staticmethod
    def validate_video_path(path: str | None) -> str | None:
        """Validate the video path.

        Args:
            path (str | None): Input video path.

        Returns:
            str | None: Validated video path, or None.

        Examples:
            >>> path = "/path/to/video.mp4"
            >>> validated_path = VideoValidator.validate_video_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(path) if path is not None else None

    @staticmethod
    def validate_target_frame(target_frame: np.ndarray) -> np.ndarray:
        """Validate the target frame of a video.

        Args:
            target_frame (np.ndarray): Input target frame.

        Returns:
            np.ndarray: Validated target frame.

        Examples:
            >>> import numpy as np
            >>> frame = np.random.rand(224, 224, 3)
            >>> validated_frame = VideoValidator.validate_target_frame(frame)
            >>> validated_frame.shape
            (224, 224, 3)
        """
        return ImageValidator.validate_image(target_frame)

    @staticmethod
    def validate_frames(frames: np.ndarray) -> np.ndarray:
        """Validate the frames of a video.

        Args:
            frames (np.ndarray): Input video frames.

        Returns:
            np.ndarray: Validated video frames.

        Examples:
            >>> import numpy as np
            >>> frames = np.random.rand(10, 224, 224, 3)
            >>> validated_frames = VideoValidator.validate_frames(frames)
            >>> validated_frames.shape
            (10, 224, 224, 3)
        """
        return VideoValidator.validate_image(frames)

    @staticmethod
    def validate_last_frame(last_frame: np.ndarray) -> np.ndarray:
        """Validate the last frame of a video.

        Args:
            last_frame (np.ndarray): Input last frame.

        Returns:
            np.ndarray: Validated last frame.

        Examples:
            >>> import numpy as np
            >>> frame = np.random.rand(224, 224, 3)
            >>> validated_frame = VideoValidator.validate_last_frame(frame)
            >>> validated_frame.shape
            (224, 224, 3)
        """
        return ImageValidator.validate_image(last_frame)

    @staticmethod
    def validate_image_path(image_path: str | None) -> str | None:
        """Validate the image path for video.

        Args:
            image_path (str | None): Input image path.

        Returns:
            str | None: Validated image path, or None.

        Examples:
            >>> path = "/path/to/image.jpg"
            >>> validated_path = VideoValidator.validate_image_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(image_path) if image_path else None

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path for video.

        Args:
            mask_path (str | None): Input mask path.

        Returns:
            str | None: Validated mask path, or None.

        Examples:
            >>> path = "/path/to/mask.png"
            >>> validated_path = VideoValidator.validate_mask_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(mask_path) if mask_path else None

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | float) -> np.ndarray:
        """Validate the prediction score for video.

        Args:
            pred_score (np.ndarray | float): Input prediction score.

        Returns:
            np.ndarray: Validated prediction score.

        Examples:
            >>> import numpy as np
            >>> score = np.array([0.8, 0.2, 0.6])
            >>> validated_score = VideoValidator.validate_pred_score(score)
            >>> validated_score
            array([0.8, 0.2, 0.6], dtype=float32)
        """
        return ImageValidator.validate_pred_score(pred_score)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map for video.

        Args:
            anomaly_map (np.ndarray | None): Input anomaly map.

        Returns:
            np.ndarray | None: Validated anomaly map, or None.

        Examples:
            >>> import numpy as np
            >>> anomaly_map = np.random.rand(10, 224, 224)  # [T, H, W]
            >>> validated_map = VideoValidator.validate_anomaly_map(anomaly_map)
            >>> validated_map.shape
            (10, 224, 224)
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, np.ndarray):
            msg = f"Anomaly map must be a numpy.ndarray, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim != 3:
            msg = f"Anomaly map must have shape [T, H, W], got shape {anomaly_map.shape}."
            raise ValueError(msg)
        return anomaly_map.astype(np.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask for video.

        Args:
            pred_mask (np.ndarray | None): Input prediction mask.

        Returns:
            np.ndarray | None: Validated prediction mask, or None.

        Examples:
            >>> import numpy as np
            >>> mask = np.random.randint(0, 2, (10, 224, 224))  # [T, H, W]
            >>> validated_mask = VideoValidator.validate_pred_mask(mask)
            >>> validated_mask.shape
            (10, 224, 224)
        """
        return VideoValidator.validate_gt_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label for video.

        Args:
            pred_label (np.ndarray | None): Input prediction label.

        Returns:
            np.ndarray | None: Validated prediction label, or None.

        Examples:
            >>> import numpy as np
            >>> label = np.array([1, 0, 1])
            >>> validated_label = VideoValidator.validate_pred_label(label)
            >>> validated_label
            array([ True, False,  True])
        """
        return ImageValidator.validate_pred_label(pred_label)


class VideoBatchValidator:
    """Validate numpy.ndarray data for batches of videos."""

    @staticmethod
    def validate_image(videos: np.ndarray) -> np.ndarray:
        """Validate and convert a batch of video arrays.

        Args:
            videos (np.ndarray): Input batch of video arrays.

        Returns:
            np.ndarray: Validated and converted batch of videos.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the video shape or number of channels is invalid.

        Examples:
            >>> import numpy as np
            >>> array = np.random.rand(4, 16, 224, 224, 3)  # Batch of 4 videos, 16 frames each
            >>> validated_videos = VideoBatchValidator.validate_image(array)
            >>> validated_videos.shape
            (4, 16, 224, 224, 3)
        """
        if not isinstance(videos, np.ndarray):
            msg = f"Videos must be a numpy.ndarray, got {type(videos)}."
            raise TypeError(msg)
        if videos.ndim != 5:
            msg = f"Videos must have shape [N, T, H, W, C], got shape {videos.shape}."
            raise ValueError(msg)
        if videos.shape[-1] not in {1, 3}:
            msg = f"Invalid number of channels: {videos.shape[-1]}. Expected 1 or 3."
            raise ValueError(msg)
        return videos.astype(np.float32)

    @staticmethod
    def validate_gt_label(gt_label: np.ndarray | list[list[int]] | None) -> np.ndarray | None:
        """Validate the ground truth labels for a batch of videos.

        Args:
            gt_label (np.ndarray | List[List[int]] | None): Input ground truth labels.

        Returns:
            np.ndarray | None: Validated ground truth labels as a boolean array, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray or list[list[int]].
            ValueError: If the labels shape is invalid.

        Examples:
            >>> labels = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0]])
            >>> validated_labels = VideoBatchValidator.validate_gt_label(labels)
            >>> validated_labels
            array([[False,  True,  True],
                   [ True, False,  True],
                   [False, False,  True],
                   [ True,  True, False]])
        """
        if gt_label is None:
            return None
        if isinstance(gt_label, list):
            gt_label = np.array(gt_label)
        if not isinstance(gt_label, np.ndarray):
            msg = f"Ground truth labels must be a numpy.ndarray or list[list[int]], got {type(gt_label)}."
            raise TypeError(msg)
        if gt_label.ndim != 2:
            msg = f"Ground truth labels must be 2-dimensional [N, T], got shape {gt_label.shape}."
            raise ValueError(msg)
        return gt_label.astype(bool)

    @staticmethod
    def validate_gt_mask(masks: np.ndarray | None) -> np.ndarray | None:
        """Validate a batch of ground truth masks for videos.

        Args:
            masks (np.ndarray | None): Input batch of ground truth masks.

        Returns:
            np.ndarray | None: Validated batch of ground truth masks, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> masks = np.random.randint(0, 2, (4, 16, 224, 224, 1))  # Batch of 4 video masks, 16 frames each
            >>> validated_masks = VideoBatchValidator.validate_gt_mask(masks)
            >>> validated_masks.shape
            (4, 16, 224, 224)
        """
        if masks is None:
            return None
        if not isinstance(masks, np.ndarray):
            msg = f"Ground truth masks must be a numpy.ndarray, got {type(masks)}."
            raise TypeError(msg)
        if masks.ndim not in {4, 5}:
            msg = f"Ground truth masks must have shape [N, T, H, W] or [N, T, H, W, 1], got shape {masks.shape}."
            raise ValueError(msg)
        if masks.ndim == 5 and masks.shape[-1] != 1:
            msg = f"Ground truth masks must have 1 channel, got {masks.shape[-1]}."
            raise ValueError(msg)
        return np.squeeze(masks, axis=-1).astype(bool)

    @staticmethod
    def validate_anomaly_map(anomaly_maps: np.ndarray | None) -> np.ndarray | None:
        """Validate a batch of anomaly maps for videos.

        Args:
            anomaly_maps (np.ndarray | None): Input batch of anomaly maps.

        Returns:
            np.ndarray | None: Validated batch of anomaly maps, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> anomaly_maps = np.random.rand(4, 16, 224, 224, 1)
            >>> validated_maps = VideoBatchValidator.validate_anomaly_map(anomaly_maps)
            >>> validated_maps.shape
            (4, 16, 224, 224)
        """
        return VideoBatchValidator.validate_gt_mask(anomaly_maps)  # Same validation as gt_mask

    @staticmethod
    def validate_image_path(video_paths: list[str] | None) -> list[str] | None:
        """Validate the video paths for a batch of videos.

        Args:
            video_paths (list[str] | None): Input video paths.

        Returns:
            list[str] | None: Validated video paths, or None.

        Examples:
            >>> paths = ["/path/to/video1.mp4", "/path/to/video2.mp4"]
            >>> validated_paths = VideoBatchValidator.validate_image_path(paths)
            >>> validated_paths == paths
            True
        """
        if video_paths is None:
            return None
        return [validate_path(path) for path in video_paths]

    @staticmethod
    def validate_mask_path(mask_paths: list[str] | None) -> list[str] | None:
        """Validate the mask paths for a batch of videos.

        Args:
            mask_paths (list[str] | None): Input mask paths.

        Returns:
            list[str] | None: Validated mask paths, or None.

        Examples:
            >>> paths = ["/path/to/mask1.mp4", "/path/to/mask2.mp4"]
            >>> validated_paths = VideoBatchValidator.validate_mask_path(paths)
            >>> validated_paths == paths
            True
        """
        if mask_paths is None:
            return None
        return [validate_path(path) for path in mask_paths]

    @staticmethod
    def validate_pred_score(pred_scores: np.ndarray | list[list[float]]) -> np.ndarray:
        """Validate the prediction scores for a batch of videos.

        Args:
            pred_scores (np.ndarray | list[list[float]]): Input prediction scores.

        Returns:
            np.ndarray: Validated prediction scores.

        Raises:
            TypeError: If the input is neither a numpy.ndarray nor a list of lists of floats.
            ValueError: If the prediction scores are not 2-dimensional.

        Examples:
            >>> scores = np.array([[0.8, 0.2, 0.6], [0.4, 0.9, 0.1]])
            >>> validated_scores = VideoBatchValidator.validate_pred_scores(scores)
            >>> validated_scores
            array([[0.8, 0.2, 0.6],
                   [0.4, 0.9, 0.1]], dtype=float32)
        """
        if isinstance(pred_scores, list):
            pred_scores = np.array(pred_scores)
        if not isinstance(pred_scores, np.ndarray):
            msg = f"Prediction scores must be a numpy.ndarray or list[list[float]], got {type(pred_scores)}."
            raise TypeError(msg)
        if pred_scores.ndim != 2:
            msg = f"Prediction scores must be 2-dimensional [N, T], got shape {pred_scores.shape}."
            raise ValueError(msg)
        return pred_scores.astype(np.float32)

    @staticmethod
    def validate_pred_label(pred_labels: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction labels for a batch of videos.

        Args:
            pred_labels (np.ndarray | None): Input prediction labels.

        Returns:
            np.ndarray | None: Validated prediction labels as a boolean array, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the prediction labels are not 2-dimensional.

        Examples:
            >>> labels = np.array([[1, 0, 1], [0, 1, 1]])
            >>> validated_labels = VideoBatchValidator.validate_pred_labels(labels)
            >>> validated_labels
            array([[ True, False,  True],
                   [False,  True,  True]])
        """
        if pred_labels is None:
            return None
        if not isinstance(pred_labels, np.ndarray):
            msg = f"Predicted labels must be a numpy.ndarray, got {type(pred_labels)}."
            raise TypeError(msg)
        if pred_labels.ndim != 2:
            msg = f"Predicted labels must be 2-dimensional [N, T], got shape {pred_labels.shape}."
            raise ValueError(msg)
        return pred_labels.astype(bool)

    @staticmethod
    def validate_pred_mask(pred_masks: np.ndarray | None) -> np.ndarray | None:
        """Validate a batch of prediction masks for videos.

        Args:
            pred_masks (np.ndarray | None): Input batch of prediction masks.

        Returns:
            np.ndarray | None: Validated batch of prediction masks, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> # Batch of 4 video prediction masks, 16 frames each
            >>> pred_masks = np.random.randint(0, 2, (4, 16, 224, 224, 1))
            >>> validated_masks = VideoBatchValidator.validate_pred_masks(pred_masks)
            >>> validated_masks.shape
            (4, 16, 224, 224)
        """
        return VideoBatchValidator.validate_gt_mask(pred_masks)  # We can reuse the gt_mask validation

    @staticmethod
    def validate_original_image(original_images: np.ndarray) -> np.ndarray:
        """Validate the original images for a batch of videos.

        Args:
            original_images (np.ndarray): Input batch of original images.

        Returns:
            np.ndarray: Validated batch of original images.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the original image shape is invalid.

        Examples:
            >>> original_images = np.random.rand(4, 224, 224, 3)
            >>> validated_images = VideoBatchValidator.validate_original_images(original_images)
            >>> validated_images.shape
            (4, 224, 224, 3)
        """
        return VideoBatchValidator.validate_image(original_images)

    @staticmethod
    def validate_video_path(video_paths: list[str]) -> list[str]:
        """Validate the video paths for a batch of videos.

        Args:
            video_paths (list[str]): Input video paths.

        Returns:
            list[str]: Validated video paths.

        Examples:
            >>> paths = ["/path/to/video1.mp4", "/path/to/video2.mp4"]
            >>> validated_paths = VideoBatchValidator.validate_video_paths(paths)
            >>> validated_paths == paths
            True
        """
        return [validate_path(path) for path in video_paths]

    @staticmethod
    def validate_target_frame(target_frames: np.ndarray) -> np.ndarray:
        """Validate the target frames for a batch of videos.

        Args:
            target_frames (np.ndarray): Input batch of target frames.

        Returns:
            np.ndarray: Validated batch of target frames.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the target frame shape is invalid.

        Examples:
            >>> target_frames = np.random.rand(4, 224, 224, 3)
            >>> validated_frames = VideoBatchValidator.validate_target_frames(target_frames)
            >>> validated_frames.shape
            (4, 224, 224, 3)
        """
        return ImageBatchValidator.validate_image(target_frames)

    @staticmethod
    def validate_frames(frames: np.ndarray) -> np.ndarray:
        """Validate the frames for a batch of videos.

        Args:
            frames (np.ndarray): Input batch of frames.

        Returns:
            np.ndarray: Validated batch of frames.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the frame shape is invalid.

        Examples:
            >>> frames = np.random.rand(4, 224, 224, 3)
            >>> validated_frames = VideoBatchValidator.validate_frames(frames)
            >>> validated_frames.shape
            (4, 224, 224, 3)
        """
        return VideoBatchValidator.validate_image(frames)

    @staticmethod
    def validate_last_frame(last_frames: np.ndarray) -> np.ndarray:
        """Validate the last frames for a batch of videos.

        Args:
            last_frames (np.ndarray): Input batch of last frames.

        Returns:
            np.ndarray: Validated batch of last frames.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the last frame shape is invalid.

        Examples:
            >>> last_frames = np.random.rand(4, 224, 224, 3)
            >>> validated_frames = VideoBatchValidator.validate_last_frames(last_frames)
            >>> validated_frames.shape
            (4, 224, 224, 3)
        """
        return VideoBatchValidator.validate_target_frame(last_frames)
