from dataclasses import dataclass

import torch


@dataclass
class MSEAggregator:
    r"""
    Class to aggregate mean squared errors.

    Attributes:
        mse_mean (torch.Tensor): The current mean squared error. Error is computed element-wise.
        mse_count (int): The number of mean squared errors that have been aggregated.
    """
    se_mean: torch.Tensor
    se_count: int

    def update(self, se: torch.Tensor) -> None:
        self.se_mean = (self.se_mean * self.se_count + se) / (self.se_count + 1)
        self.se_count += 1

    def get_value(self) -> torch.Tensor:
        return self.se_mean
