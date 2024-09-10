"""Validate torch video data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch
from torchvision.tv_tensors import Video

from anomalib.dataclasses.validate.path import validate_path
from anomalib.dataclasses.validate.torch.image import validate_image


def validate_video(video: torch.Tensor) -> Video:
    """Validate and convert the input PyTorch video tensor.

    Args:
        video: The input video to validate. Must be a PyTorch tensor.

    Returns:
        The validated video as a torchvision Video.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the video dimensions are invalid.

    Examples:
        >>> import torch
        >>> video = torch.rand(10, 3, 224, 224)  # 10 frames, 3 channels, 224x224 resolution
        >>> result = validate_video(video)
        >>> isinstance(result, Video)
        True
        >>> result.shape
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


def validate_target_frame(target_frame: torch.Tensor) -> torch.Tensor:
    """Validate the target frame tensor.

    Args:
        target_frame: The input target frame to validate. Must be a PyTorch tensor.

    Returns:
        The validated target frame as a PyTorch tensor.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the target frame dimensions are invalid.

    Examples:
        >>> import torch
        >>> target_frame = torch.rand(3, 224, 224)
        >>> result = validate_target_frame(target_frame)
        >>> result.shape
        torch.Size([3, 224, 224])
    """
    return validate_image(target_frame)


def validate_frames(frames: torch.Tensor) -> torch.Tensor:
    """Validate the frames tensor.

    Args:
        frames: The input frames to validate. Must be a PyTorch tensor.

    Returns:
        The validated frames as a PyTorch tensor.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the frames dimensions are invalid.

    Examples:
        >>> import torch
        >>> frames = torch.rand(5, 3, 224, 224)  # 5 frames
        >>> result = validate_frames(frames)
        >>> result.shape
        torch.Size([5, 3, 224, 224])
    """
    return validate_video(frames)


def validate_last_frame(last_frame: torch.Tensor) -> torch.Tensor:
    """Validate the last frame tensor.

    Args:
        last_frame: The input last frame to validate. Must be a PyTorch tensor.

    Returns:
        The validated last frame as a PyTorch tensor.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the last frame dimensions are invalid.

    Examples:
        >>> import torch
        >>> last_frame = torch.rand(3, 224, 224)
        >>> result = validate_last_frame(last_frame)
        >>> result.shape
        torch.Size([3, 224, 224])
    """
    return validate_image(last_frame)


def validate_video_gt_label(gt_label: int | torch.Tensor | Sequence[int] | None) -> torch.Tensor | None:
    """Validate and convert the input video ground truth label to a boolean PyTorch tensor.

    Args:
        gt_label: The input label to validate. Can be an integer, a PyTorch tensor,
                  a sequence of integers, or None.

    Returns:
        The validated label as a boolean PyTorch tensor, or None if the input was None.

    Raises:
        TypeError: If the input is not an int, PyTorch tensor, or sequence of integers.
        ValueError: If the tensor is not 1D or is a floating point type.

    Examples:
        >>> import torch
        >>> validate_video_gt_label(1)
        tensor([True])

        >>> validate_video_gt_label(torch.tensor([0, 1, 1, 0]))
        tensor([False,  True,  True, False])

        >>> validate_video_gt_label([0, 1, 1, 0])
        tensor([False,  True,  True, False])

        >>> validate_video_gt_label(None)
        None

        >>> validate_video_gt_label(torch.tensor([0.5, 1.5]))
        Traceback (most recent call last):
            ...
        ValueError: Ground truth label must be boolean or integer, got torch.float32.
    """
    if gt_label is None:
        return None
    if isinstance(gt_label, int):
        return torch.tensor([gt_label], dtype=torch.bool)
    if isinstance(gt_label, Sequence):
        gt_label = torch.tensor(gt_label)
    if not isinstance(gt_label, torch.Tensor):
        msg = f"Ground truth label must be an integer, a sequence of integers, or a torch.Tensor, got {type(gt_label)}."
        raise TypeError(msg)
    if gt_label.ndim != 1:
        msg = f"Ground truth label must be a 1-dimensional vector, got shape {gt_label.shape}."
        raise ValueError(msg)
    if torch.is_floating_point(gt_label):
        msg = f"Ground truth label must be boolean or integer, got {gt_label.dtype}."
        raise ValueError(msg)
    return gt_label.bool()


def validate_video_gt_mask(gt_mask: torch.Tensor | None) -> Mask | None:
    """Validate and convert the input video ground truth mask.

    Args:
        gt_mask: The input ground truth mask to validate. Can be a PyTorch tensor or None.

    Returns:
        The validated mask as a torchvision Mask, or None if the input was None.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the mask dimensions are invalid.

    Examples:
        >>> import torch
        >>> from torchvision.tv_tensors import Mask
        >>> # 2D input (single mask for all frames)
        >>> torch_mask_2d = torch.randint(0, 2, (100, 100))
        >>> result = validate_video_gt_mask(torch_mask_2d)
        >>> isinstance(result, Mask)
        True
        >>> result.shape
        torch.Size([100, 100])

        >>> # 3D input (mask per frame)
        >>> torch_mask_3d = torch.randint(0, 2, (10, 100, 100))
        >>> result = validate_video_gt_mask(torch_mask_3d)
        >>> result.shape
        torch.Size([10, 100, 100])

        >>> # 3D input (single mask with channel dimension)
        >>> torch_mask_3d_channel = torch.randint(0, 2, (1, 100, 100))
        >>> result = validate_video_gt_mask(torch_mask_3d_channel)
        >>> result.shape
        torch.Size([100, 100])

        >>> # 4D input (mask per frame with channel dimension)
        >>> torch_mask_4d = torch.randint(0, 2, (10, 1, 100, 100))
        >>> result = validate_video_gt_mask(torch_mask_4d)
        >>> result.shape
        torch.Size([10, 100, 100])

        >>> # Invalid input
        >>> validate_video_gt_mask(torch.rand(10, 3, 100, 100))
        Traceback (most recent call last):
            ...
        ValueError: Ground truth mask must have shape [H, W], [T, H, W], [1, H, W], or [T, 1, H, W],
        got shape torch.Size([10, 3, 100, 100]).

        >>> validate_video_gt_mask(None)
        None
    """
    if gt_mask is None:
        return None
    if not isinstance(gt_mask, torch.Tensor):
        msg = f"Ground truth mask must be a torch.Tensor, got {type(gt_mask)}."
        raise TypeError(msg)

    if gt_mask.ndim not in {2, 3, 4}:
        msg = (
            "Ground truth mask must have shape [H, W], [T, H, W], [1, H, W], or [T, 1, H, W],"
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


# We can reuse existing validation functions for other fields
validate_video_path = validate_path
