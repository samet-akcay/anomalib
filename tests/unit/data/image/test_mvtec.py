"""Unit Tests - MVTec Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import MVTec
from tests.unit.data.base.image import _TestAnomalibImageDatamodule


class TestMVTec(_TestAnomalibImageDatamodule):
    """MVTec Datamodule Unit Tests."""

    @pytest.fixture()
    def datamodule(self, dataset_path: Path, task_type: TaskType) -> MVTec:
        """Create and return a MVTec datamodule."""
        _datamodule = MVTec(
            root=dataset_path / "mvtec",
            category="dummy",
            task=task_type,
            train_batch_size=2,
            eval_batch_size=2,
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule

    @pytest.fixture()
    def fxt_data_config_path(self) -> str:
        """Return the path to the test data config."""
        return "configs/data/mvtec.yaml"
