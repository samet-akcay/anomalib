"""Validate torch video data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
import torch
from torchvision.tv_tensors import Image, Mask, Video

from anomalib.dataclasses.validate.path import validate_path

from .image import ImageBatchValidator, ImageValidator


class VideoValidator:
    """Validate torch.Tensor data for videos."""

    @staticmethod
    def validate_image(video: torch.Tensor) -> Video:
        """Validate and convert the input video tensor.

        Args:
            video (torch.Tensor): Input video tensor.

        Returns:
            Video: Validated and converted video.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the video shape or number of channels is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import VideoValidator
            >>> tensor = torch.rand(10, 3, 224, 224)  # [T, C, H, W]
            >>> validated_video = VideoValidator.validate_image(tensor)
            >>> isinstance(validated_video, Video)
            True
            >>> validated_video.shape
            torch.Size([10, 3, 224, 224])
        """
        if not isinstance(video, torch.Tensor):
            msg = f"Video must be a torch.Tensor, got {type(video)}."
            raise TypeError(msg)

        if video.ndim != 4:
            msg = f"Video must have shape [T, C, H, W], got shape {video.shape}."
            raise ValueError(msg)

        if video.shape[1] not in {1, 3}:
            msg = f"Video must have 1 or 3 channels, got {video.shape[1]}."
            raise ValueError(msg)

        return Video(video)

    @staticmethod
    def validate_gt_label(gt_label: int | torch.Tensor | Sequence[int] | None) -> torch.Tensor | None:
        """Validate the ground truth label for video.

        Args:
            gt_label (int | torch.Tensor | Sequence[int] | None): Input ground truth label.

        Returns:
            torch.Tensor | None: Validated ground truth label as a boolean tensor, or None.

        Raises:
            TypeError: If the input is not of the expected types.
            ValueError: If the label shape or dtype is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import VideoValidator
            >>> label = [0, 1, 1, 0]
            >>> validated_label = VideoValidator.validate_gt_label(label)
            >>> validated_label
            tensor([False,  True,  True, False])
        """
        if gt_label is None:
            return None
        if isinstance(gt_label, int):
            return torch.tensor([gt_label], dtype=torch.bool)
        if isinstance(gt_label, Sequence):
            gt_label = torch.tensor(gt_label)
        if not isinstance(gt_label, torch.Tensor):
            msg = (
                "Ground truth label must be an integer, a sequence of integers, or a torch.Tensor, "
                f"got {type(gt_label)}."
            )
            raise TypeError(msg)
        if gt_label.ndim != 1:
            msg = f"Ground truth label must be a 1-dimensional vector, got shape {gt_label.shape}."
            raise ValueError(msg)
        if torch.is_floating_point(gt_label):
            msg = f"Ground truth label must be boolean or integer, got {gt_label.dtype}."
            raise ValueError(msg)
        return gt_label.bool()

    @staticmethod
    def validate_gt_mask(gt_mask: torch.Tensor | None) -> Mask | None:
        """Validate the ground truth mask for video.

        Args:
            gt_mask (torch.Tensor | None): Input ground truth mask.

        Returns:
            Mask | None: Validated ground truth mask, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import VideoValidator
            >>> mask = torch.randint(0, 2, (10, 1, 224, 224))  # [T, C, H, W]
            >>> validated_mask = VideoValidator.validate_gt_mask(mask)
            >>> isinstance(validated_mask, Mask)
            True
            >>> validated_mask.shape
            torch.Size([10, 224, 224])
        """
        if gt_mask is None:
            return None
        if not isinstance(gt_mask, torch.Tensor):
            msg = f"Ground truth mask must be a torch.Tensor, got {type(gt_mask)}."
            raise TypeError(msg)

        if gt_mask.ndim not in {2, 3, 4}:
            msg = (
                "Ground truth mask must have shape [H, W], [T, H, W], [1, H, W], or [T, 1, H, W], "
                f"got shape {gt_mask.shape}."
            )
            raise ValueError(msg)

        if gt_mask.ndim == 3 and gt_mask.shape[0] == 1:
            # Case: [1, H, W] -> [H, W]
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.ndim == 4:
            if gt_mask.shape[1] != 1:
                msg = f"Ground truth mask must have 1 channel, got {gt_mask.shape[1]}."
                raise ValueError(msg)
            # Case: [T, 1, H, W] -> [T, H, W]
            gt_mask = gt_mask.squeeze(1)

        return Mask(gt_mask, dtype=torch.bool)

    @staticmethod
    def validate_original_image(video: torch.Tensor) -> Video:
        """Validate and convert the original input video tensor.

        Args:
            video (torch.Tensor): Original input video tensor.

        Returns:
            Video: Validated and converted original video.

        Examples:
            >>> import torch
            >>> tensor = torch.rand(10, 3, 224, 224)  # [T, C, H, W]
            >>> validated_video = VideoValidator.validate_original_image(tensor)
            >>> isinstance(validated_video, Video)
            True
            >>> validated_video.shape
            torch.Size([10, 3, 224, 224])
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
    def validate_target_frame(target_frame: torch.Tensor) -> torch.Tensor:
        """Validate the target frame of a video.

        Args:
            target_frame (torch.Tensor): Input target frame.

        Returns:
            torch.Tensor: Validated target frame.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import VideoValidator
            >>> frame = torch.rand(3, 224, 224)
            >>> validated_frame = VideoValidator.validate_target_frame(frame)
            >>> validated_frame.shape
            torch.Size([3, 224, 224])
        """
        return ImageValidator.validate_image(target_frame.unsqueeze(0)).squeeze(0)

    @staticmethod
    def validate_frames(frames: torch.Tensor) -> torch.Tensor:
        """Validate the frames of a video.

        Args:
            frames (torch.Tensor): Input video frames.

        Returns:
            torch.Tensor: Validated video frames.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import VideoValidator
            >>> frames = torch.rand(10, 3, 224, 224)
            >>> validated_frames = VideoValidator.validate_frames(frames)
            >>> validated_frames.shape
            torch.Size([10, 3, 224, 224])
        """
        return VideoValidator.validate_image(frames)

    @staticmethod
    def validate_last_frame(last_frame: torch.Tensor) -> torch.Tensor:
        """Validate the last frame of a video.

        Args:
            last_frame (torch.Tensor): Input last frame.

        Returns:
            torch.Tensor: Validated last frame.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import VideoValidator
            >>> frame = torch.rand(3, 224, 224)
            >>> validated_frame = VideoValidator.validate_last_frame(frame)
            >>> validated_frame.shape
            torch.Size([3, 224, 224])
        """
        return ImageValidator.validate_image(last_frame.unsqueeze(0)).squeeze(0)

    @staticmethod
    def validate_image_path(image_path: str | None) -> str | None:
        """Validate the image path for video.

        Args:
            image_path (str | None): Input image path.

        Returns:
            str | None: Validated image path, or None.

        Examples:
            >>> from anomalib.dataclasses.validators import VideoValidator
            >>> path = "/path/to/video.mp4"
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
            >>> from anomalib.dataclasses.validators import VideoValidator
            >>> path = "/path/to/mask.mp4"
            >>> validated_path = VideoValidator.validate_mask_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(mask_path) if mask_path else None

    @staticmethod
    def validate_pred_score(pred_score: torch.Tensor | float) -> torch.Tensor:
        """Validate the prediction score for video.

        Args:
            pred_score (torch.Tensor | float): Input prediction score.

        Returns:
            torch.Tensor: Validated prediction score.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import VideoValidator
            >>> score = torch.tensor([0.8, 0.2, 0.6])
            >>> validated_score = VideoValidator.validate_pred_score(score)
            >>> validated_score
            tensor([0.8000, 0.2000, 0.6000])
        """
        return ImageValidator.validate_pred_score(pred_score)

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | np.ndarray | None) -> Mask | None:
        """Validate the anomaly map for video.

        Args:
            anomaly_map (torch.Tensor | None): Input anomaly map.

        Returns:
            Mask | None: Validated anomaly map, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import VideoValidator
            >>> anomaly_map = torch.rand(1, 224, 224)  # [C, H, W]
            >>> validated_map = VideoValidator.validate_anomaly_map(anomaly_map)
            >>> isinstance(validated_map, Mask)
            True
            >>> validated_map.shape
            torch.Size([224, 224])
        """
        if anomaly_map is None:
            return None

        if not isinstance(anomaly_map, torch.Tensor):
            try:
                anomaly_map = torch.tensor(anomaly_map)
            except Exception as e:
                msg = "Anomaly map must be a torch.Tensor. Tried to convert to torch.Tensor but failed."
                raise ValueError(msg) from e

        if anomaly_map.ndim not in {2, 3}:
            msg = f"Anomaly map must have shape [H, W] or [1, H, W], got shape {anomaly_map.shape}."
            raise ValueError(msg)

        if anomaly_map.ndim == 3 and anomaly_map.shape[0] != 1:
            msg = f"Anomaly map must have 1 channel, got {anomaly_map.shape[0]}."
            raise ValueError(msg)

        return Mask(anomaly_map.squeeze(0))

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> Mask | None:
        """Validate the prediction mask for video.

        Args:
            pred_mask (torch.Tensor | None): Input prediction mask.

        Returns:
            Mask | None: Validated prediction mask, or None.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import VideoValidator
            >>> mask = torch.randint(0, 2, (10, 1, 224, 224))  # [T, C, H, W]
            >>> validated_mask = VideoValidator.validate_pred_mask(mask)
            >>> isinstance(validated_mask, Mask)
            True
            >>> validated_mask.shape
            torch.Size([10, 224, 224])
        """
        return VideoValidator.validate_gt_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction label for video.

        Args:
            pred_label (torch.Tensor | None): Input prediction label.

        Returns:
            torch.Tensor | None: Validated prediction label, or None.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import VideoValidator
            >>> label = torch.tensor([1, 0, 1])
            >>> validated_label = VideoValidator.validate_pred_label(label)
            >>> validated_label
            tensor([ True, False,  True])
        """
        return ImageValidator.validate_pred_label(pred_label)


class VideoBatchValidator:
    """Validate torch.Tensor data for batches of videos."""

    @staticmethod
    def validate_image(videos: torch.Tensor) -> Video:
        """Validate and convert a batch of video tensors.

        Args:
            videos (torch.Tensor): Input batch of video tensors.

        Returns:
            Video: Validated and converted batch of videos.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the video shape or number of channels is invalid.

        Examples:
            >>> import torch
            >>> tensor = torch.rand(4, 16, 3, 224, 224)  # Batch of 4 videos, 16 frames each
            >>> validated_videos = VideoBatchValidator.validate_video(tensor)
            >>> isinstance(validated_videos, Video)
            True
            >>> validated_videos.shape
            torch.Size([4, 16, 3, 224, 224])
        """
        if not isinstance(videos, torch.Tensor):
            msg = f"Videos must be a torch.Tensor, got {type(videos)}."
            raise TypeError(msg)

        if videos.ndim != 5:
            msg = f"Videos must have shape [N, T, C, H, W], got shape {videos.shape}."
            raise ValueError(msg)

        if videos.shape[2] not in {1, 3}:
            msg = f"Invalid number of channels: {videos.shape[2]}. Expected 1 or 3."
            raise ValueError(msg)

        return Video(videos.to(torch.float32))

    @staticmethod
    def validate_gt_label(labels: torch.Tensor | list[list[int]] | None) -> torch.Tensor | None:
        """Validate the ground truth labels for a batch of videos.

        Args:
            labels (torch.Tensor | list[list[int]] | None): Input ground truth labels.

        Returns:
            torch.Tensor | None: Validated ground truth labels as a boolean tensor, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor or list[list[int]].
            ValueError: If the labels shape is invalid.

        Examples:
            >>> labels = torch.tensor([[0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0]])
            >>> validated_labels = VideoBatchValidator.validate_gt_label(labels)
            >>> validated_labels
            tensor([[False,  True,  True],
                    [ True, False,  True],
                    [False, False,  True],
                    [ True,  True, False]])
        """
        if labels is None:
            return None
        if isinstance(labels, list):
            labels = torch.tensor(labels)
        if not isinstance(labels, torch.Tensor):
            msg = f"Ground truth labels must be a torch.Tensor or list[list[int]], got {type(labels)}."
            raise TypeError(msg)
        if labels.ndim != 2:
            msg = f"Ground truth labels must be 2-dimensional [N, T], got shape {labels.shape}."
            raise ValueError(msg)
        return labels.to(torch.bool)

    @staticmethod
    def validate_gt_mask(masks: torch.Tensor | None) -> Mask | None:
        """Validate a batch of ground truth masks for videos.

        Args:
            masks (torch.Tensor | None): Input batch of ground truth masks.

        Returns:
            Mask | None: Validated batch of ground truth masks, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> masks = torch.randint(0, 2, (4, 16, 1, 224, 224))  # Batch of 4 video masks, 16 frames each
            >>> validated_masks = VideoBatchValidator.validate_gt_mask(masks)
            >>> isinstance(validated_masks, Mask)
            True
            >>> validated_masks.shape
            torch.Size([4, 16, 224, 224])
        """
        if masks is None:
            return None
        if not isinstance(masks, torch.Tensor):
            msg = f"Ground truth masks must be a torch.Tensor, got {type(masks)}."
            raise TypeError(msg)
        if masks.ndim not in {4, 5}:
            msg = f"Ground truth masks must have shape [N, T, H, W] or [N, T, 1, H, W], got shape {masks.shape}."
            raise ValueError(msg)
        if masks.ndim == 5 and masks.shape[2] != 1:
            msg = f"Ground truth masks must have 1 channel, got {masks.shape[2]}."
            raise ValueError(msg)

        return Mask(masks.squeeze(2) if masks.ndim == 5 else masks, dtype=torch.bool)

    @staticmethod
    def validate_anomaly_map(anomaly_maps: torch.Tensor | None) -> Mask | None:
        """Validate a batch of anomaly maps for videos.

        Args:
            anomaly_maps (torch.Tensor | None): Input batch of anomaly maps.

        Returns:
            Mask | None: Validated batch of anomaly maps as a Mask, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> anomaly_maps = torch.rand(4, 16, 1, 224, 224)  # Batch of 4 video anomaly maps, 16 frames each
            >>> validated_maps = VideoBatchValidator.validate_anomaly_map(anomaly_maps)
            >>> isinstance(validated_maps, Mask)
            True
            >>> validated_maps.shape
            torch.Size([4, 16, 224, 224])
        """
        if anomaly_maps is None:
            return None
        if not isinstance(anomaly_maps, torch.Tensor):
            msg = f"Anomaly maps must be a torch.Tensor, got {type(anomaly_maps)}."
            raise TypeError(msg)
        if anomaly_maps.ndim not in {4, 5}:
            msg = f"Anomaly maps must have shape [N, T, H, W] or [N, T, 1, H, W], got shape {anomaly_maps.shape}."
            raise ValueError(msg)
        if anomaly_maps.ndim == 5 and anomaly_maps.shape[2] != 1:
            msg = f"Anomaly maps must have 1 channel, got {anomaly_maps.shape[2]}."
            raise ValueError(msg)

        return Mask(anomaly_maps.squeeze(2) if anomaly_maps.ndim == 5 else anomaly_maps)

    @staticmethod
    def validate_video_path(video_paths: list[str] | None) -> list[str] | None:
        """Validate the video paths for a batch.

        Args:
            video_paths (list[str] | None): Input video paths.

        Returns:
            list[str] | None: Validated video paths, or None.

        Examples:
            >>> paths = ["/path/to/video1.mp4", "/path/to/video2.mp4"]
            >>> validated_paths = VideoBatchValidator.validate_video_path(paths)
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
    def validate_pred_score(pred_score: torch.Tensor | list[list[float]]) -> torch.Tensor:
        """Validate the prediction scores for a batch of videos.

        Args:
            pred_score (torch.Tensor | list[list[float]]): Input prediction scores.

        Returns:
            torch.Tensor: Validated prediction scores.

        Raises:
            TypeError: If the input is neither a torch.Tensor nor a list of lists of floats.
            ValueError: If the prediction scores are not 2-dimensional.

        Examples:
            >>> scores = torch.tensor([[0.8, 0.2, 0.6], [0.4, 0.9, 0.1]])
            >>> validated_scores = VideoBatchValidator.validate_pred_score(scores)
            >>> validated_scores
            tensor([[0.8000, 0.2000, 0.6000],
                    [0.4000, 0.9000, 0.1000]])
        """
        if isinstance(pred_score, list):
            pred_score = torch.tensor(pred_score)
        if not isinstance(pred_score, torch.Tensor):
            msg = f"Prediction scores must be a torch.Tensor or list[list[float]], got {type(pred_score)}."
            raise TypeError(msg)
        if pred_score.ndim != 2:
            msg = f"Prediction scores must be 2-dimensional [N, T], got shape {pred_score.shape}."
            raise ValueError(msg)
        return pred_score.to(torch.float32)

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction labels for a batch of videos.

        Args:
            pred_label (torch.Tensor | None): Input prediction labels.

        Returns:
            torch.Tensor | None: Validated prediction labels as a boolean tensor, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the prediction labels are not 2-dimensional.

        Examples:
            >>> labels = torch.tensor([[1, 0, 1], [0, 1, 1]])
            >>> validated_labels = VideoBatchValidator.validate_pred_label(labels)
            >>> validated_labels
            tensor([[ True, False,  True],
                    [False,  True,  True]])
        """
        if pred_label is None:
            return None
        if not isinstance(pred_label, torch.Tensor):
            msg = f"Predicted labels must be a torch.Tensor, got {type(pred_label)}."
            raise TypeError(msg)
        if pred_label.ndim != 2:
            msg = f"Predicted labels must be 2-dimensional [N, T], got shape {pred_label.shape}."
            raise ValueError(msg)
        return pred_label.to(torch.bool)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> Mask | None:
        """Validate a batch of prediction masks for videos.

        Args:
            pred_mask (torch.Tensor | None): Input batch of prediction masks.

        Returns:
            Mask | None: Validated batch of prediction masks, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> # Batch of 4 video prediction masks, 16 frames each
            >>> pred_mask = torch.randint(0, 2, (4, 16, 1, 224, 224))
            >>> validated_mask = VideoBatchValidator.validate_pred_mask(pred_mask)
            >>> isinstance(validated_mask, Mask)
            True
            >>> validated_mask.shape
            torch.Size([4, 16, 224, 224])
        """
        return VideoBatchValidator.validate_gt_mask(pred_mask)  # We can reuse the gt_mask validation

    @staticmethod
    def validate_original_image(image: torch.Tensor) -> Video:
        """Validate and convert the original input batch of video tensors.

        Args:
            image (torch.Tensor): Original input batch of video tensors.

        Returns:
            Video: Validated and converted original batch of videos.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the video shape or number of channels is invalid.

        Examples:
            >>> import torch
            >>> tensor = torch.rand(4, 16, 3, 224, 224)  # [N, T, C, H, W]
            >>> validated_videos = VideoBatchValidator.validate_original_image(tensor)
            >>> isinstance(validated_videos, Video)
            True
            >>> validated_videos.shape
            torch.Size([4, 16, 3, 224, 224])
        """
        return VideoBatchValidator.validate_image(image)

    @staticmethod
    def validate_target_frame(frames: torch.Tensor) -> Image:
        """Validate the batch of target frames.

        Args:
            frames (torch.Tensor): Input batch of target frames.

        Returns:
            Image: Validated batch of target frames.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the frame shape or number of channels is invalid.

        Examples:
            >>> frames = torch.rand(4, 3, 224, 224)  # [N, C, H, W]
            >>> validated_frames = VideoBatchValidator.validate_target_frame(frames)
            >>> isinstance(validated_frames, Image)
            True
            >>> validated_frames.shape
            torch.Size([4, 3, 224, 224])
        """
        return ImageBatchValidator.validate_image(frames)

    @staticmethod
    def validate_frames(frames: torch.Tensor) -> Video:
        """Validate the batch of video frames.

        Args:
            frames (torch.Tensor): Input batch of video frames.

        Returns:
            Video: Validated batch of video frames.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the frames shape or number of channels is invalid.

        Examples:
            >>> frames = torch.rand(4, 16, 3, 224, 224)  # [N, T, C, H, W]
            >>> validated_frames = VideoBatchValidator.validate_frames(frames)
            >>> isinstance(validated_frames, Video)
            True
            >>> validated_frames.shape
            torch.Size([4, 16, 3, 224, 224])
        """
        return VideoBatchValidator.validate_image(frames)  # Same validation as video

    @staticmethod
    def validate_last_frame(frames: torch.Tensor) -> Image:
        """Validate the batch of last frames.

        Args:
            frames (torch.Tensor): Input batch of last frames.

        Returns:
            Image: Validated batch of last frames.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the frame shape or number of channels is invalid.

        Examples:
            >>> frames = torch.rand(4, 3, 224, 224)  # [N, C, H, W]
            >>> validated_frames = VideoBatchValidator.validate_last_frame(frames)
            >>> isinstance(validated_frames, Image)
            True
            >>> validated_frames.shape
            torch.Size([4, 3, 224, 224])
        """
        return VideoBatchValidator.validate_target_frame(frames)  # Same validation as target frame
