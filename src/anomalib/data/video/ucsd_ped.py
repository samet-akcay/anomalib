"""UCSD Pedestrian dataset."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from pathlib import Path
from shutil import move
from typing import TYPE_CHECKING, Any

import albumentations as A  # noqa: N812
import cv2
import numpy as np
import torch
from pandas import DataFrame

from anomalib.data.base import AnomalibVideoDataModule, AnomalibVideoDataset
from anomalib.data.base.video import VideoTargetFrame
from anomalib.data.utils import (
    DownloadInfo,
    InputNormalizationMethod,
    Split,
    ValSplitMode,
    download_and_extract,
    get_transforms,
    read_image,
)
from anomalib.data.utils.video import ClipsIndexer
from anomalib.utils.types import TaskType

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="UCSD Pedestrian",
    url="http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz",
    checksum="5006421b89885f45a6f93b041145f2eb",
)

CATEGORIES = ("UCSDped1", "UCSDped2")


def make_ucsd_dataset(path: Path, split: str | Split | None = None) -> DataFrame:
    """Create UCSD Pedestrian dataset by parsing the file structure.

    The files are expected to follow the structure:
        path/to/dataset/category/split/video_id/image_filename.tif
        path/to/dataset/category/split/video_id_gt/mask_filename.bmp

    Args:
        path (Path): Path to dataset
        split (str | Split | None, optional): Dataset split (ie., either train or test). Defaults to None.

    Example:
        The following example shows how to get testing samples from UCSDped2 category:

        >>> root = Path('./UCSDped')
        >>> category = 'UCSDped2'
        >>> path = root / category
        >>> path
        PosixPath('UCSDped/UCSDped2')

        >>> samples = make_ucsd_dataset(path, split='test')
        >>> samples.head()
                       root folder                     image_path                         mask_path split
        0  UCSDped/UCSDped2   Test  UCSDped/UCSDped2/Test/Test001  UCSDped/UCSDped2/Test/Test001_gt  test
        1  UCSDped/UCSDped2   Test  UCSDped/UCSDped2/Test/Test002  UCSDped/UCSDped2/Test/Test002_gt  test
        ...

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)


    """
    folders = [filename for filename in sorted(path.glob("*/*")) if filename.is_dir()]
    folders = [folder for folder in folders if list(folder.glob("*.tif"))]

    samples_list = [(str(path),) + folder.parts[-2:] for folder in folders]
    samples = DataFrame(samples_list, columns=["root", "folder", "image_path"])

    samples.loc[samples.folder == "Test", "mask_path"] = samples.image_path.str.split(".").str[0] + "_gt"
    samples.loc[samples.folder == "Test", "mask_path"] = samples.root + "/" + samples.folder + "/" + samples.mask_path
    samples.loc[samples.folder == "Train", "mask_path"] = ""

    samples["image_path"] = samples.root + "/" + samples.folder + "/" + samples.image_path

    samples.loc[samples.folder == "Train", "split"] = "train"
    samples.loc[samples.folder == "Test", "split"] = "test"

    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class UCSDpedClipsIndexer(ClipsIndexer):
    """Clips indexer for UCSDped dataset."""

    def get_mask(self, idx: int) -> np.ndarray | None:
        """Retrieve the masks from the file system.

        Args:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            torch.Tensor | None: GT mask for the subclip.
        """
        video_idx, frames_idx = self.get_clip_location(idx)
        mask_folder = self.mask_paths[video_idx]
        if mask_folder == "":  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]

        mask_frames = sorted(Path(mask_folder).glob("*.bmp"))
        mask_paths = [mask_frames[idx] for idx in frames.int()]

        return np.stack([cv2.imread(str(mask_path), flags=0) / 255.0 for mask_path in mask_paths])

    def _compute_frame_pts(self) -> None:
        """Retrieve the number of frames in each video."""
        self.video_pts = []
        for video_path in self.video_paths:
            n_frames = len(list(Path(video_path).glob("*.tif")))
            self.video_pts.append(torch.Tensor(range(n_frames)))

        self.video_fps = [None] * len(self.video_paths)  # fps information cannot be inferred from folder structure

    def get_clip(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], int]:
        """Get a subclip from a list of videos.

        Args:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            Tensor: Video clip of shape (T, H, W, C)
            Tensor: Audio placeholder
            dict: metadata dictionary placeholder
            int: index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            msg = f"Index {idx} out of range ({self.num_clips()} number of clips)"
            raise IndexError(msg)
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        frames = sorted(Path(video_path).glob("*.tif"))

        frame_paths = [frames[pt] for pt in clip_pts.int()]
        video = torch.stack([torch.Tensor(read_image(str(frame_path))) for frame_path in frame_paths])

        return video, torch.empty((1, 0)), {}, video_idx


class UCSDpedDataset(AnomalibVideoDataset):
    """UCSDped Dataset class.

    Args:
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
        category (str): Sub-category of the dataset, e.g. "UCSDped1" or "UCSDped2"
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split (Split): Split of the dataset, usually Split.TRAIN or Split.TEST
        root (Path | str): Path to the root of the dataset.
            Defaults to ``./datasets/ucsd``.
        clip_length_in_frames (int, optional): Number of video frames in each clip.
            Defaults to ``2``.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
            Defaults to ``1``.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval
            Defaults to ``VideoTargetFrame.LAST``.

    Examples:
        To create a UCSDped dataset to train a classification model:

        .. code-block:: python

            transform = A.Compose([A.Resize(256, 256), A.pytorch.ToTensorV2()])
            dataset = UCSDpedDataset(
                task="classification",
                category="UCSDped2",
                transform=transform,
                split="train",
                root="./datasets/ucsd/",
            )

            dataset.setup()
            dataset[0].keys()

            # Output: dict_keys(['image', 'video_path', 'frames', 'last_frame', 'original_image'])

        If you would like to test a segmentation model, you can use the following code:

        .. code-block:: python

            dataset = UCSDpedDataset(
                task="segmentation",
                category="UCSDped2",
                transform=transform,
                split="test",
                root="./datasets/ucsd/",
            )

            dataset.setup()
            dataset[0].keys()

            # Output: dict_keys(['image', 'mask', 'video_path', 'frames', 'last_frame', 'original_image', 'label'])

        UCSDped video dataset can also be used as an image dataset if you set the clip length to 1. This means that
        each video frame will be treated as a separate sample. This is useful for training a classification model on the
        UCSDped dataset. The following code shows how to create an image dataset for classification:

        .. code-block:: python

            dataset = UCSDpedDataset(
                task="classification",
                category="UCSDped2",
                transform=transform,
                split="test",
                root="./datasets/ucsd/",
                clip_length_in_frames=1,
            )

            dataset.setup()
            dataset[0].keys()
            # Output: dict_keys(['image', 'video_path', 'frames', 'last_frame', 'original_image', 'label'])

            dataset[0]["image"].shape
            # Output: torch.Size([3, 256, 256])
    """

    def __init__(
        self,
        task: TaskType,
        category: str,
        transform: A.Compose,
        split: Split,
        root: str | Path = "./datasets/ucsd",
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
    ) -> None:
        super().__init__(task, transform, clip_length_in_frames, frames_between_clips, target_frame)

        self.root_category = Path(root) / category
        self.split = split
        self.indexer_cls: Callable = UCSDpedClipsIndexer

    def _setup(self) -> None:
        """Create and assign samples."""
        self.samples = make_ucsd_dataset(self.root_category, self.split)


class UCSDped(AnomalibVideoDataModule):
    """UCSDped DataModule class.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``./datasets/ucsd``.
        category (str): Sub-category of the dataset, e.g. "UCSDped1" or "UCSDped2".
            Defaults to ``UCSDped2``.
        clip_length_in_frames (int, optional): Number of video frames in each clip.
            Defaults to ``2``.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
            Defaults to ``1``.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval
            Defaults to ``VideoTargetFrame.LAST``.
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'.
            Defaults to ``TaskType.SEGMENTATION``.
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to ``(256, 256)``.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
            Defaults to ``None``.
        normalization (InputNormalizationMethod | str): Normalization method to be applied to the input images.
            Defaults to ``InputNormalizationMethod.IMAGENET``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing
            during training.
            Defaults to ``None``.
        transform_config_val (str | A.Compose | None, optional): Config for pre-processing
            during validation.
            Defaults to ``None``.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.FROM_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defaults to ``None``.

    Examples:
        To create a UCSDped DataModule for training a classification model:

        .. code-block:: python

            datamodule = UCSDped()
            datamodule.setup()

            i, data = next(enumerate(datamodule.train_dataloader()))
            data.keys()
            # Output: dict_keys(['image', 'video_path', 'frames', 'last_frame', 'original_image'])

            i, data = next(enumerate(datamodule.test_dataloader()))
            data.keys()
            # Output: dict_keys(['image', 'mask', 'video_path', 'frames', 'last_frame', 'original_image', 'label'])

            data["image"].shape
            # Output: torch.Size([32, 2, 3, 256, 256])

        Note that the default task type is segmentation and the dataloader returns a mask in addition to the input.
        Also, it is important to note that the dataloader returns a batch of clips, where each clip is a sequence of
        frames. The number of frames in each clip is determined by the ``clip_length_in_frames`` parameter. The
        ``frames_between_clips`` parameter determines the number of frames between each consecutive clip. The
        ``target_frame`` parameter determines which frame in the clip is used for ground truth retrieval. For example,
        if ``clip_length_in_frames=2``, ``frames_between_clips=1`` and ``target_frame=VideoTargetFrame.LAST``, then the
        dataloader will return a batch of clips where each clip contains two consecutive frames from the video. The
        second frame in each clip will be used as the ground truth for the first frame in the clip. The following code
        shows how to create a dataloader for classification:

        .. code-block:: python

            datamodule = UCSDped(
                task="classification",
                clip_length_in_frames=2,
                frames_between_clips=1,
                target_frame=VideoTargetFrame.LAST
            )
            datamodule.setup()

            i, data = next(enumerate(datamodule.train_dataloader()))
            data.keys()
            # Output: dict_keys(['image', 'video_path', 'frames', 'last_frame', 'original_image'])

            data["image"].shape
            # Output: torch.Size([32, 2, 3, 256, 256])

        .. code-block:: python
    """

    def __init__(
        self,
        root: Path | str = "./datasets/ucsd",
        category: str = "UCSDped2",
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        task: TaskType = TaskType.SEGMENTATION,
        image_size: int | tuple[int, int] = (256, 256),
        center_crop: int | tuple[int, int] | None = None,
        normalization: InputNormalizationMethod | str = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        val_split_mode: ValSplitMode = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = category

        transform_train = get_transforms(
            config=transform_config_train,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )
        transform_eval = get_transforms(
            config=transform_config_eval,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )

        self.train_data = UCSDpedDataset(
            task=task,
            transform=transform_train,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            target_frame=target_frame,
            root=root,
            category=category,
            split=Split.TRAIN,
        )

        self.test_data = UCSDpedDataset(
            task=task,
            transform=transform_eval,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            target_frame=target_frame,
            root=root,
            category=category,
            split=Split.TEST,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file system.
        If not, it downloads and extracts the dataset into the appropriate directory.

        Example:
            Assume the dataset is not available on the file system.
            Here's how the directory structure looks before and after calling the
            `prepare_data` method:

            Before:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                └── dataset2

            Calling the method:

            .. code-block:: python

                >> datamodule = UCSDped()
                >> datamodule.prepare_data()

            After:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                ├── dataset2
                └── ucsd
                    ├── UCSDped1
                    |   └── ...
                    └── UCSDped2
                        ├── Test
                        │   ├── Test012
                        │   │   ├── ...
                        │   │   └── 180.tif
                        │   └── Test012_gt
                        │       ├── ...
                        │       └── frame180.bmp
                        └── Train
                            ├── ...
                            └── Train016
                                ├── ...
                                └── 150.tif
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)

            # move contents to root
            extracted_folder = self.root / "UCSD_Anomaly_Dataset.v1p2"
            for filename in extracted_folder.glob("*"):
                move(str(filename), str(self.root / filename.name))
            extracted_folder.rmdir()
