"""Implementation of PRO metric based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch
from torchmetrics import Metric
from torchmetrics.functional import recall
from torchmetrics.utilities.data import dim_zero_cat

from anomalib.utils.cv import connected_components_cpu, connected_components_gpu


class PRO(Metric):
    """Per-Region Overlap (PRO) Score."""

    target: list[torch.Tensor]
    preds: list[torch.Tensor]

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Compute the PRO score for the current batch."""
        self.target.append(targets)
        self.preds.append(predictions)

    def compute(self) -> torch.Tensor:
        """Compute the macro average of the PRO score across all regions in all batches."""
        target = dim_zero_cat(self.target)
        preds = dim_zero_cat(self.preds)

        target = target.unsqueeze(1).type(torch.float)  # kornia expects N1HW and FloatTensor format
        comps = connected_components_gpu(target) if target.is_cuda else connected_components_cpu(target)
        return pro_score(preds, comps, threshold=self.threshold)


def pro_score(predictions: torch.Tensor, comps: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Calculate the PRO score for a batch of predictions.

    Args:
        predictions (torch.Tensor): Predicted anomaly masks (Bx1xHxW)
        comps: (torch.Tensor): Labeled connected components (BxHxW). The components should be labeled from 0 to N
        threshold (float): When predictions are passed as float, the threshold is used to binarize the predictions.

    Returns:
        torch.Tensor: Scalar value representing the average PRO score for the input batch.
    """
    if predictions.dtype == torch.float:
        predictions = predictions > threshold

    n_comps = len(comps.unique())

    preds = comps.clone()
    # match the shapes in case one of the tensors is N1HW
    preds = preds.reshape(predictions.shape)
    preds[~predictions] = 0
    if n_comps == 1:  # only background
        return torch.Tensor([1.0])
    return recall(preds.flatten(), comps.flatten(), num_classes=n_comps, average="macro", ignore_index=0)