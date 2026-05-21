import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple

class ClassificationDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        X: [N, D]
        y: [N]
        """
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class InBatchNegativeSampler:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    @torch.no_grad()
    def sample(
        self,
        batch_targets: torch.Tensor,
        k: int,
    ):
        """
        batch_targets: [B]

        returns:
            negatives: [B, k]
        """

        device = batch_targets.device

        B = batch_targets.shape[0]

        unique_batch_classes = torch.unique(batch_targets)

        all_classes = torch.arange(
            self.num_classes,
            device=device,
        )

        negatives = []

        for i in range(B):

            positive = batch_targets[i]

            # ----------------------------------------
            # in-batch negatives
            # ----------------------------------------

            inbatch_negatives = unique_batch_classes[
                unique_batch_classes != positive
            ]

            # ----------------------------------------
            # enough in-batch negatives
            # ----------------------------------------

            if len(inbatch_negatives) >= k:

                perm = torch.randperm(
                    len(inbatch_negatives),
                    device=device,
                )

                selected = inbatch_negatives[perm[:k]]

            else:

                selected = inbatch_negatives

                remain = k - len(selected)

                # ------------------------------------
                # random global negatives
                # ------------------------------------

                forbidden = torch.cat([
                    selected,
                    positive.view(1),
                ])

                mask = torch.ones(
                    self.num_classes,
                    dtype=torch.bool,
                    device=device,
                )

                mask[forbidden] = False

                candidates = all_classes[mask]

                rand_idx = torch.randint(
                    0,
                    len(candidates),
                    (remain,),
                    device=device,
                )

                extra = candidates[rand_idx]

                selected = torch.cat([
                    selected,
                    extra,
                ])

            negatives.append(selected)

        negatives = torch.stack(
            negatives,
            dim=0,
        )

        return negatives