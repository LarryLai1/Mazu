from datetime import datetime

import lightning.pytorch as L
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch.utils.data import DataLoader

from .datasets import ERA5TWDataset


class ERA5TWDataModule(L.LightningDataModule):
    def __init__(self,
                 root_dir: str,
                 n_workers: int,
                 train_start: str = "2020-07-01 00:00:00",
                 train_end: str = "2020-07-21 00:00:00",
                 val_start: str = "2020-07-21 00:00:00",
                 val_end: str = "2020-07-26 00:00:00",
                 test_start: str = "2020-07-26 00:00:00",
                 test_end: str = "2020-08-01 00:00:00",
                 upper_variables: list = ["u", "v", "t", "q", "z", "w"],
                 surface_variables: list = ["u10", "v10", "t2m", "msl", "sp", "tcwv", "tp", "d2m"],
                 batch_size: int = 32,) -> None:
        """
        Data module for ERA5TW dataset.

        Args:
            root_dir (str): Root directory of the dataset.
            train_start (str, optional): Start time of the training set. Defaults to "2020-07-01 00:00:00".
            train_end (str, optional): End time of the training set. Defaults to "2020-07-21 00:00:00".
            val_start (str, optional): Start time of the validation set. Defaults to "2020-07-21 00:00:00".
            val_end (str, optional): End time of the validation set. Defaults to "2020-07-26 00:00:00".
            test_start (str, optional): Start time of the test set. Defaults to "2020-07-26 00:00:00".
            test_end (str, optional): End time of the test set. Defaults to "2020-08-01 00:00:00".
            upper_variables (list, optional): Upper level variables. Defaults to ["u", "v", "t", "q", "z", "w"].
            surface_variables (list, optional): Surface level variables. Defaults to ["u10", "v10", "t2m", "msl", "sp", "tcwv", "tp", "d2m"].
            batch_size (int, optional): Batch size. Defaults to 32.
            n_workers (int, optional): Number of workers for dataloader. Defaults to 4.

        Note:
            The time format is "%Y-%m-%d %H:%M:%S".
            End time is exclusive.
            Time step is 1 hour.
        """
        super().__init__()
        self.root_dir: str = root_dir
        self.train_start = datetime.strptime(train_start, r"%Y-%m-%d %H:%M:%S")
        self.train_end = datetime.strptime(train_end, r"%Y-%m-%d %H:%M:%S")
        self.val_start = datetime.strptime(val_start, r"%Y-%m-%d %H:%M:%S")
        self.val_end = datetime.strptime(val_end, r"%Y-%m-%d %H:%M:%S")
        self.test_start = datetime.strptime(test_start, r"%Y-%m-%d %H:%M:%S")
        self.test_end = datetime.strptime(test_end, r"%Y-%m-%d %H:%M:%S")

        self.upper_variables = upper_variables
        self.surface_variables = surface_variables
        self.batch_size = batch_size
        self.n_workers = n_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        self.train_dataset = ERA5TWDataset(self.root_dir,
                                           self.train_start, self.train_end,
                                           self.upper_variables, self.surface_variables,
                                           standardize=True,
                                           get_stat=False)
        self.val_dataset = ERA5TWDataset(self.root_dir,
                                         self.val_start, self.val_end,
                                         self.upper_variables, self.surface_variables,
                                         standardize=True,
                                         get_stat=True)
        self.test_dataset = ERA5TWDataset(self.root_dir,
                                          self.test_start, self.test_end,
                                          self.upper_variables, self.surface_variables,
                                          standardize=True,
                                          get_stat=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_workers,
                          drop_last=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.n_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.n_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError
