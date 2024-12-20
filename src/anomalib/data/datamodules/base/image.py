"""Base Anomalib data module.

This module provides the base data module class used across Anomalib. It handles
dataset splitting, validation set creation, and dataloader configuration.

The module contains:
    - :class:`AnomalibDataModule`: Base class for all Anomalib data modules

Example:
    Create a datamodule from a config file::

        >>> from anomalib.data import AnomalibDataModule
        >>> data_config = "examples/configs/data/mvtec.yaml"
        >>> datamodule = AnomalibDataModule.from_config(config_path=data_config)

    Override config with additional arguments::

        >>> override_kwargs = {"data.train_batch_size": 8}
        >>> datamodule = AnomalibDataModule.from_config(
        ...     config_path=data_config,
        ...     **override_kwargs
        ... )
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from lightning.pytorch import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader

from anomalib import TaskType
from anomalib.data.utils import TestSplitMode, ValSplitMode, random_split, split_by_label
from anomalib.data.utils.synthetic import SyntheticAnomalyDataset

if TYPE_CHECKING:
    from pandas import DataFrame

    from anomalib.data.datasets.base.image import AnomalibDataset

logger = logging.getLogger(__name__)


class AnomalibDataModule(LightningDataModule, ABC):
    """Base Anomalib data module.

    This class extends PyTorch Lightning's ``LightningDataModule`` to provide
    common functionality for anomaly detection datasets.

    Args:
        train_batch_size (int): Batch size for training dataloader
        eval_batch_size (int): Batch size for validation/test dataloaders
        num_workers (int): Number of workers for all dataloaders
        val_split_mode (ValSplitMode | str): Method to obtain validation set.
            Options:
                - ``none``: No validation set
                - ``same_as_test``: Use test set as validation
                - ``from_test``: Sample from test set
                - ``synthetic``: Generate synthetic anomalies
        val_split_ratio (float): Fraction of data to use for validation
        test_split_mode (TestSplitMode | str | None): Method to obtain test set.
            Options:
                - ``none``: No test split
                - ``from_dir``: Use separate test directory
                - ``synthetic``: Generate synthetic anomalies
            Defaults to ``None``.
        test_split_ratio (float | None): Fraction of data to use for testing.
            Defaults to ``None``.
        seed (int | None): Random seed for reproducible splitting.
            Defaults to ``None``.
    """

    def __init__(
        self,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        val_split_mode: ValSplitMode | str,
        val_split_ratio: float,
        test_split_mode: TestSplitMode | str | None = None,
        test_split_ratio: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.test_split_mode = TestSplitMode(test_split_mode) if test_split_mode else TestSplitMode.NONE
        self.test_split_ratio = test_split_ratio
        self.val_split_mode = ValSplitMode(val_split_mode)
        self.val_split_ratio = val_split_ratio
        self.seed = seed

        self.train_data: AnomalibDataset
        self.val_data: AnomalibDataset
        self.test_data: AnomalibDataset

        self._samples: DataFrame | None = None
        self._category: str = ""

        self._is_setup = False  # flag to track if setup has been called

    @property
    def name(self) -> str:
        """Name of the datamodule.

        Returns:
            str: Class name of the datamodule
        """
        return self.__class__.__name__

    def setup(self, stage: str | None = None) -> None:
        """Set up train, validation and test data.

        This method handles the data splitting logic based on the configured
        modes.

        Args:
            stage (str | None): Current stage (fit/validate/test/predict).
                Defaults to ``None``.
        """
        has_subset = any(hasattr(self, subset) for subset in ["train_data", "val_data", "test_data"])
        if not has_subset or not self._is_setup:
            self._setup(stage)
            self._create_test_split()
            self._create_val_split()
            if isinstance(stage, TrainerFn):
                # only set flag if called from trainer
                self._is_setup = True

    @abstractmethod
    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method should be implemented by subclasses to define dataset-specific
        setup logic.

        Note:
            The ``stage`` argument is not used since all subsets are created on
            first call to accommodate validation set extraction from test set.

        Args:
            _stage (str | None): Current stage (unused).
                Defaults to ``None``.

        Raises:
            NotImplementedError: When not implemented by subclass
        """
        raise NotImplementedError

    @property
    def category(self) -> str:
        """Get dataset category name.

        Returns:
            str: Name of the current category
        """
        return self._category

    @category.setter
    def category(self, category: str) -> None:
        """Set dataset category name.

        Args:
            category (str): Category name to set
        """
        self._category = category

    @property
    def task(self) -> TaskType:
        """Get the task type.

        Returns:
            TaskType: Type of anomaly task (classification/segmentation)

        Raises:
            AttributeError: If no datasets have been set up yet
        """
        if hasattr(self, "train_data"):
            return self.train_data.task
        if hasattr(self, "val_data"):
            return self.val_data.task
        if hasattr(self, "test_data"):
            return self.test_data.task
        msg = "This datamodule does not have any datasets. Did you call setup?"
        raise AttributeError(msg)

    def _create_test_split(self) -> None:
        """Create the test split based on configured mode.

        This handles splitting normal/anomalous samples and optionally creating
        synthetic anomalies.
        """
        if self.test_data.has_normal:
            # split test data into normal and anomalous
            normal_test_data, self.test_data = split_by_label(self.test_data)
        elif self.test_split_mode != TestSplitMode.NONE:
            # sample normal images from training set if none provided
            logger.info(
                "No normal test images found. Sampling from training set using ratio of %0.2f",
                self.test_split_ratio,
            )
            if self.test_split_ratio is not None:
                self.train_data, normal_test_data = random_split(
                    self.train_data,
                    self.test_split_ratio,
                    seed=self.seed,
                )

        if self.test_split_mode == TestSplitMode.FROM_DIR:
            self.test_data += normal_test_data
        elif self.test_split_mode == TestSplitMode.SYNTHETIC:
            self.test_data = SyntheticAnomalyDataset.from_dataset(normal_test_data)
        elif self.test_split_mode != TestSplitMode.NONE:
            msg = f"Unsupported Test Split Mode: {self.test_split_mode}"
            raise ValueError(msg)

    def _create_val_split(self) -> None:
        """Create validation split based on configured mode.

        This handles sampling from train/test sets and optionally creating
        synthetic anomalies.
        """
        if self.val_split_mode == ValSplitMode.FROM_TRAIN:
            # randomly sample from train set
            self.train_data, self.val_data = random_split(
                self.train_data,
                self.val_split_ratio,
                label_aware=True,
                seed=self.seed,
            )
        elif self.val_split_mode == ValSplitMode.FROM_TEST:
            # randomly sample from test set
            self.test_data, self.val_data = random_split(
                self.test_data,
                self.val_split_ratio,
                label_aware=True,
                seed=self.seed,
            )
        elif self.val_split_mode == ValSplitMode.SAME_AS_TEST:
            # use test set as validation
            self.val_data = self.test_data
        elif self.val_split_mode == ValSplitMode.SYNTHETIC:
            # create synthetic anomalies from training samples
            self.train_data, normal_val_data = random_split(
                self.train_data,
                self.val_split_ratio,
                seed=self.seed,
            )
            self.val_data = SyntheticAnomalyDataset.from_dataset(normal_val_data)
        elif self.val_split_mode != ValSplitMode.NONE:
            msg = f"Unknown validation split mode: {self.val_split_mode}"
            raise ValueError(msg)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get training dataloader.

        Returns:
            DataLoader: Training dataloader
        """
        return DataLoader(
            dataset=self.train_data,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_data.collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader.

        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            dataset=self.val_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.val_data.collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader.

        Returns:
            DataLoader: Test dataloader
        """
        return DataLoader(
            dataset=self.test_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.test_data.collate_fn,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Get prediction dataloader.

        By default uses the test dataloader.

        Returns:
            DataLoader: Prediction dataloader
        """
        return self.test_dataloader()

    @classmethod
    def from_config(
        cls: type["AnomalibDataModule"],
        config_path: str | Path,
        **kwargs,
    ) -> "AnomalibDataModule":
        """Create datamodule instance from config file.

        Args:
            config_path (str | Path): Path to config file
            **kwargs: Additional args to override config

        Returns:
            AnomalibDataModule: Instantiated datamodule

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If instantiated object is not AnomalibDataModule

        Example:
            Load from config file::

                >>> config_path = "examples/configs/data/mvtec.yaml"
                >>> datamodule = AnomalibDataModule.from_config(config_path)

            Override config values::

                >>> datamodule = AnomalibDataModule.from_config(
                ...     config_path,
                ...     data_train_batch_size=8
                ... )
        """
        from jsonargparse import ArgumentParser

        if not Path(config_path).exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        data_parser = ArgumentParser()
        data_parser.add_subclass_arguments(AnomalibDataModule, "data", required=False, fail_untyped=False)
        args = ["--data", str(config_path)]
        for key, value in kwargs.items():
            args.extend([f"--{key}", str(value)])
        config = data_parser.parse_args(args=args)
        instantiated_classes = data_parser.instantiate_classes(config)
        datamodule = instantiated_classes.get("data")
        if isinstance(datamodule, AnomalibDataModule):
            return datamodule

        msg = f"Datamodule is not an instance of AnomalibDataModule: {datamodule}"
        raise ValueError(msg)
