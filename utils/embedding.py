"""Extract Aurora latent embeddings for a :class:`Batch`.

The embedding is the output of the **last encoder layer of the Aurora backbone**
(the deepest / bottleneck latent of the Swin3D transformer, taken *before* any
decoder layer runs). This is the representation just after `backbone.encoder_layers`
in `Swin3DTransformerBackbone.forward` (see aurora/model/swin3d.py).

Import `encode_batch` wherever an embedding is needed:

    from utils.embedding import encode_batch
    emb = encode_batch(model, batch)  # (B, L', D)
"""

import contextlib
import dataclasses
from datetime import timedelta

import torch

from aurora.batch import Batch
from aurora.model.fourier import lead_time_expansion

__all__ = ["encode_batch"]


def _prepare_and_encode(model, batch: Batch):
    """Run the pre-encoder pipeline and the Perceiver encoder.

    Mirrors the `prepare_and_encode` sequence of the real forward pass
    (`Aurora.forward` in aurora/model/aurora.py). Keep it in sync with that.

    Returns:
        tuple: `(tokens, patch_res)` where `tokens` has shape `(B, L, D)` and
        `patch_res` is the `(C, H, W)` latent patch resolution.
    """
    p = next(model.parameters())

    batch = model.batch_transform_hook(batch)
    batch = batch.type(p.dtype)
    batch = batch.normalise(surf_stats=model.surf_stats)
    batch = batch.crop(patch_size=model.patch_size)
    batch = batch.to(p.device)

    H, W = batch.spatial_shape
    patch_res = (
        model.encoder.latent_levels,
        H // model.encoder.patch_size,
        W // model.encoder.patch_size,
    )

    B, T = next(iter(batch.surf_vars.values())).shape[:2]
    static_vars = {}
    for k, v in batch.static_vars.items():
        static_vars[k] = v[None, None].repeat(B, T, 1, 1) if v.ndim == 2 else v
    batch = dataclasses.replace(batch, static_vars=static_vars)

    transformed = batch
    if model.positive_surf_vars:
        transformed = dataclasses.replace(transformed, surf_vars={
            k: v.clamp(min=0) if k in model.positive_surf_vars else v
            for k, v in transformed.surf_vars.items()
        })
    if model.positive_atmos_vars:
        transformed = dataclasses.replace(transformed, atmos_vars={
            k: v.clamp(min=0) if k in model.positive_atmos_vars else v
            for k, v in transformed.atmos_vars.items()
        })
    transformed = model._pre_encoder_hook(transformed)

    # `model.timestep` (1h), not the forecast lead: the lead-time embedding must be a constant
    # offset shared by both batches, not a trend injected into the lead-time curve.
    tokens = model.encoder(transformed, lead_time=model.timestep)
    return tokens, patch_res, transformed.metadata.rollout_step


@torch.no_grad()
def encode_batch(model, batch: Batch) -> torch.Tensor:
    """Embed a batch as the last backbone-encoder-layer output, shape `(B, L', D)`.

    The tokens produced by the Perceiver encoder are pushed through the backbone's
    `encoder_layers` (the Swin3D downsampling stages); the output of the final encoder
    layer -- the bottleneck latent, before any decoder layer runs -- is returned.
    """
    tokens, patch_res, rollout_step = _prepare_and_encode(model, batch)

    backbone = model.backbone
    lead_time = model.timestep

    if model.autocast:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.xpu.is_available():
            device_type = "xpu"
        else:
            device_type = "cpu"
        context = torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    else:
        context = contextlib.nullcontext()

    with context:
        all_enc_res, _ = backbone.get_encoder_specs(patch_res)

        lead_hours = lead_time / timedelta(hours=1)
        lead_times = lead_hours * torch.ones(
            tokens.shape[0], dtype=torch.float32, device=tokens.device
        )
        c_all = backbone.time_mlp(
            lead_time_expansion(lead_times, backbone.embed_dim).to(dtype=tokens.dtype)
        )

        x = tokens
        for i, layer in enumerate(backbone.encoder_layers):
            # The last encoder layer has `downsample=None`, so `x` is its output and
            # the returned skip is `None`.
            x, _ = layer(x, c_all, all_enc_res[i], rollout_step=rollout_step)

    return x
