import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from typing import Dict, Tuple

class SampledLogisticRegression(nn.Module):
    def __init__(self, num_classes: int, dim: int):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(num_classes, dim) * 0.01
        )

    def forward_sampled(
        self,
        x: torch.Tensor,
        positive_classes: torch.Tensor,
        sampled_negatives: torch.Tensor,
    ):
        """
        x: [B, D]
        positive_classes: [B]
        sampled_negatives: [B, k]

        returns:
            logits: [B, 1+k]
            targets: [B]
        """

        # --------------------------------------------
        # positive logits
        # --------------------------------------------

        pos_w = self.weight[positive_classes]  # [B, D]

        pos_logits = torch.sum(
            x * pos_w,
            dim=1,
            keepdim=True,
        )  # [B,1]

        # --------------------------------------------
        # negative logits
        # --------------------------------------------

        neg_w = self.weight[sampled_negatives]  # [B, k, D]

        neg_logits = torch.einsum(
            "bd,bkd->bk",
            x,
            neg_w,
        )  # [B,k]

        logits = torch.cat(
            [pos_logits, neg_logits],
            dim=1,
        )  # [B,1+k]

        targets = torch.zeros(
            x.shape[0],
            dtype=torch.long,
            device=x.device,
        )

        return logits, targets

    @torch.no_grad()
    def full_logits(self, x: torch.Tensor):
        """
        Полный logits по всем классам.

        x: [B, D]

        returns:
            [B, K]
        """

        return x @ self.weight.T