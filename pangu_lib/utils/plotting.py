from os import listdir, path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from torch import Tensor


class EvalLoader:
    def __init__(self, main_dir: str, ckpt: str, color: Any, fmt: str, label: str):
        self.main_dir = main_dir
        self.ckpt = ckpt
        self.color = color
        self.fmt = fmt
        self.label = label
        self.npz: Any = None
        self._load_npz()

    def __del__(self):
        self._close_npz()

    def plot_args(self) -> dict:
        return {
            "label": self.label,
            "color": self.color,
        }

    def _load_npz(self):
        ckpt_dir = path.join(self.main_dir, self.ckpt)
        if path.exists(path.join(ckpt_dir, "mse.npz")):
            self.npz = np.load(path.join(ckpt_dir, "mse.npz"))
        elif path.exists(path.join(ckpt_dir, "part_0")):
            self.npz = []
            for d in listdir(path.join(ckpt_dir)):
                if d.startswith("part_"):
                    self.npz.append(np.load(path.join(ckpt_dir, d, "mse.npz")))
        else:
            raise FileNotFoundError

    def _close_npz(self):
        if isinstance(self.npz, list):
            for f in self.npz:
                f.close()
        else:
            self.npz.close()

    def get_mse_map(self, var: str, lead_time: int) -> Any:
        assert self.npz is not None
        key = f"{var}_+{lead_time}"
        if isinstance(self.npz, list):
            avg, count = None, 0
            for f in self.npz:
                if key not in f:
                    return None
                mse = f[key]
                avg = mse if avg is None else avg + mse
                count += 1
            assert isinstance(avg, np.ndarray)
            return avg / count
        else:
            if key not in self.npz:
                return None
            return self.npz[key]


class EvalLoaderAgg:
    def __init__(self, main_dir: str, ckpts: list[str], color: Any, fmt: str, label: str):
        self.main_dir = main_dir
        self.ckpts = ckpts
        self.color = color
        self.fmt = fmt
        self.label = label
        self.eval_loaders = [EvalLoader(main_dir, ckpt, color, fmt, label) for ckpt in ckpts]

    def plot_args(self) -> dict:
        return {
            "label": self.label,
            "color": self.color,
        }

    def get_mse_maps(self, var: str, lead_time: int) -> Any:
        return [loader.get_mse_map(var, lead_time) for loader in self.eval_loaders]


def visualize_weather_var(
    weather_var: Tensor,
    v_range: tuple[float, float],
) -> Figure:
    """
    Visualize a 2D tensor of weather variable
    Args:
        forecast (Tensor): A 2D tensor to visualize.
        v_range (tuple[float, float]): vmin and vmax for cmap range.
    Return:
        Figure: matplotlib figure containing the plot.
    """
    fig, ax = plt.subplots(constrained_layout=True, dpi=120)
    norm = Normalize(vmin=v_range[0], vmax=v_range[1])
    ax_im = ax.imshow(weather_var, norm=norm, cmap=sns.color_palette("mako", as_cmap=True))
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.colorbar(ax_im)

    return fig
