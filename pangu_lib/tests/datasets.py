from datetime import datetime

import torch
import torch.utils.data

from ..utils.datasets import ERA5TWDataset


def test_dataset(data_dir) -> None:
    """
    Test the dataset.
    """
    dataset = ERA5TWDataset(root_dir=data_dir,
                            start_date_hour=datetime(2017, 1, 8, 0),
                            end_date_hour=datetime(2017, 1, 10, 23),
                            upper_variables=["u", "v", "t", "q", "z", "w"],
                            surface_variables=["u10", "v10", "t2m", "msl", "sp", "tcwv", "tp", "d2m"],
                            standardize=True,
                            get_stat=True)
    x, y, stat = dataset[0]  # type: ignore
    print(f"{x[0].shape=}, {x[1].shape=}")
    print(f"{y[0].shape=}, {y[1].shape=}")
    print(f"{stat[0].shape=}, {stat[1].shape=}, {stat[2].shape=}, {stat[3].shape=}")


def test_dataloader(data_dir) -> None:
    """
    Test the dataloader.
    """
    dataset = ERA5TWDataset(root_dir=data_dir,
                            start_date_hour=datetime(2017, 1, 8, 0),
                            end_date_hour=datetime(2017, 1, 10, 23),
                            upper_variables=["u", "v", "t", "q", "z", "w"],
                            surface_variables=["u10", "v10", "t2m", "msl", "sp", "tcwv", "tp", "d2m"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    for i, (x, y) in enumerate(dataloader):
        print(i)
        print(x[0].shape, x[1].shape)
        print(y[0].shape, y[1].shape)
        break
