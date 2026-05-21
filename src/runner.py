import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from typing import List, Dict
from src.data import ClassificationDataset, InBatchNegativeSampler
from src.model import SampledLogisticRegression

class Model():
    def __init__(self, train_cfg, data_cfg):
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg

        self.model = SampledLogisticRegression(
            num_classes=data_cfg.K,
            dim=data_cfg.D,
        ).to(train_cfg.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg.lr,
        )

        self.sampler = InBatchNegativeSampler(
            num_classes=data_cfg.K
        )

        self.history: List[Dict] = []
    
    def fit(self, train_loader, val_loader):
        for epoch in tqdm(range(self.train_cfg.epochs)):
            
            # Train loop
            self.model.train()
            total_loss = 0.0

            for x, y in train_loader:

                x = x.to(self.train_cfg.device)
                y = y.to(self.train_cfg.device)

                negatives = self.sampler.sample(
                    batch_targets=y,
                    k=self.train_cfg.negatives_k,
                )

                logits, targets = self.model.forward_sampled(
                    x=x,
                    positive_classes=y,
                    sampled_negatives=negatives,
                )

                loss = F.cross_entropy(
                    logits,
                    targets,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            train_loss = total_loss / len(train_loader)

            # Validation loop
            self.model.eval()
            with torch.no_grad():
                total_loss = 0.0

                total_correct = 0
                total_samples = 0

                for x, y in val_loader:

                    x = x.to(self.train_cfg.device)
                    y = y.to(self.train_cfg.device)

                    logits = self.model.full_logits(x)
                    loss = F.cross_entropy(
                        logits,
                        y,
                    )
                    total_loss += loss.item()

                    preds = logits.argmax(dim=1)
                    correct = (preds == y).sum().item()

                    total_correct += correct
                    total_samples += y.shape[0]

                val_loss = total_loss / len(val_loader)
                accuracy = total_correct / total_samples


            metrics = {
                "epoch": (epoch+1),
                "num_steps": len(train_loader)*(epoch+1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": accuracy,
            }

            self.history.append(metrics)
        pd.DataFrame(self.history).to_csv('output.csv', index=False)
        self.is_fitted = True