from numpy import ndarray
from torch import Tensor

__all__ = ["WeatherData", "WeatherDataNumpy", "Shape_T", "Stat_T"]

WeatherData = tuple[Tensor, Tensor]
WeatherDataNumpy = tuple[ndarray, ndarray]
Shape_T = tuple[int, int, int]
Stat_T = tuple[Tensor, Tensor, Tensor, Tensor]
