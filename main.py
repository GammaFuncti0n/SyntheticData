from torch.utils.data import DataLoader
import pandas as pd
import torch
from dataclasses import dataclass, asdict
import logging
import os

from src.data import ClassificationDataset
from src.runner import Model
from src.util import set_seed, setup_logging


setup_logging('log/')

@dataclass
class TrainConfig:

    batch_size: int = 1024
    negatives_k: int = 64

    lr: float = 5e-4
    epochs: int = 10

    device: str = "cuda:2"

@dataclass
class DataConfig:
    N_train: int = 100_000
    N_val: int = 10_000

    D: int = 128
    K: int = 50_000

@dataclass
class Config:
    train = TrainConfig()
    data = DataConfig()

config = Config()


def main():
    pairs = [(256, 64), (512, 32)]#[(128, 128), (64, 256), (32, 512), (16, 1024), (8, 2048)]
    seeds = [42, 43, 44]
    for pair in pairs:
        config.train.epochs = int(16*pair[0]//128)
        for seed in seeds:
            config.train.batch_size = pair[0]
            config.train.negatives_k = pair[1]
            logging.info(asdict(config))
            set_seed(seed)
            logging.info(f"seed: {seed}")

            N_total = config.data.N_train + config.data.N_val

            class_centers = torch.randn(config.data.K, config.data.D) * 3.0
            y_total = torch.randint(0, config.data.K, (N_total,))
            X_total = class_centers[y_total] + torch.randn(N_total, config.data.D) * 0.5

            shuffled_indices = torch.randperm(N_total)
            X_total = X_total[shuffled_indices]
            y_total = y_total[shuffled_indices]

            X_train = X_total[:config.data.N_train]
            y_train = y_total[:config.data.N_train]

            X_val = X_total[config.data.N_train:]
            y_val = y_total[config.data.N_train:]

            train_dataset = ClassificationDataset(
                X_train,
                y_train,
            )

            val_dataset = ClassificationDataset(
                X_val,
                y_val,
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.train.batch_size,
                shuffle=True,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=config.train.batch_size,
                shuffle=False,
            )
            
            runner = Model(config.train, config.data)
            runner.fit(train_loader, val_loader)

            path = f'results/n={config.train.batch_size}_k={config.train.negatives_k}_seed={seed}/'
            os.makedirs(path, exist_ok=True)
            pd.DataFrame(runner.history).to_csv(
                path+"history.csv",
                index=False,
            )
            logging.info(f"Saved results to {path}")

if __name__ == "__main__":
    main()