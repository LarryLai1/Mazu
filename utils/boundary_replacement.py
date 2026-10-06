import torch
import torch.nn.functional as F


def replace_input_boundary(main_tensor, bc_tensor, boundary_width, smooth_mode = "no"):
    """
    Replace the outer ring (width `boundary_width` cells, over the last two dims H, W) of
    `main_tensor` with `bc_tensor` in physical space. Leading dims are arbitrary.
    smooth_mode: no (hard replace) / linear (linear blend ramp) /
    mean, gaussian (hard replace, then 3x3 smoothing written back where d <= boundary_width).
    """
    bw = int(boundary_width)
    if bw <= 0:
        return main_tensor
    if main_tensor.shape != bc_tensor.shape:
        raise ValueError(
            f"Input boundary replacement shape mismatch: main {tuple(main_tensor.shape)} vs boundary {tuple(bc_tensor.shape)}."
        )
    H, W = main_tensor.shape[-2:]
    h_coords = torch.arange(H, device = main_tensor.device)
    w_coords = torch.arange(W, device = main_tensor.device)
    dist_h = torch.minimum(h_coords, H - 1 - h_coords)
    dist_w = torch.minimum(w_coords, W - 1 - w_coords)
    dist = torch.minimum(dist_h.unsqueeze(1), dist_w.unsqueeze(0))  # (H, W), 0 on the edge

    if smooth_mode == "linear":
        mask = torch.clamp(1.0 - dist.to(main_tensor.dtype) / bw, min = 0.0, max = 1.0)
        return mask * bc_tensor + (1.0 - mask) * main_tensor

    if smooth_mode not in ("no", "mean", "gaussian"):
        raise ValueError(f"Unsupported smoothing mode: {smooth_mode}")

    ring = dist < bw
    replaced = torch.where(ring, bc_tensor, main_tensor)
    if smooth_mode == "no":
        return replaced

    if smooth_mode == "mean":
        kernel = torch.ones((1, 1, 3, 3), dtype = main_tensor.dtype, device = main_tensor.device) / 9.0
    else:
        kernel = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]
        ], dtype = main_tensor.dtype, device = main_tensor.device)
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)

    flat = replaced.reshape(-1, 1, H, W)
    padded = F.pad(flat, (1, 1, 1, 1), mode = "replicate")
    smoothed = F.conv2d(padded, kernel).reshape(replaced.shape)
    return torch.where(dist <= bw, smoothed, main_tensor)
