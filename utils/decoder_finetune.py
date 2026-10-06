"""The inference forward pass (model_forward_with_latent_boundary in
AuroraSmallTW_gen_eval_pipeline_custom_rollout.py) split into a frozen front half
(encoder + Swin3D backbone, incl. the backbone-level boundary replacement) and the
Perceiver decoder half, for decoder-only finetuning. Keep both halves in sync with that function.
"""
import contextlib
import dataclasses

import torch

from aurora import Batch
from utils.boundary_replacement import replace_input_boundary

REPLACE_BOUNDARY_POSITIONS = ("input", "backbone")


def _autocast_context(model):
    if not model.autocast:
        return contextlib.nullcontext()
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.xpu.is_available():
        device_type = "xpu"
    else:
        device_type = "cpu"
    return torch.autocast(device_type = device_type, dtype = torch.bfloat16)


def _patch_res(model, batch):
    H, W = batch.spatial_shape
    return (
        model.encoder.latent_levels,
        H // model.encoder.patch_size,
        W // model.encoder.patch_size,
    )


def _prepare_batch(model, batch):
    p = next(model.parameters())
    batch = model.batch_transform_hook(batch)
    batch = batch.type(p.dtype)
    batch = batch.crop(patch_size = model.patch_size)
    return batch.to(p.device)


def _encode_prepared(model, batch):
    batch = batch.normalise(surf_stats = model.surf_stats)

    B, T = next(iter(batch.surf_vars.values())).shape[:2]
    static_vars = {}
    for k, v in batch.static_vars.items():
        if v.ndim == 2:
            static_vars[k] = v[None, None].repeat(B, T, 1, 1)
        else:
            static_vars[k] = v
    batch = dataclasses.replace(batch, static_vars = static_vars)

    transformed_batch = batch
    if model.positive_surf_vars:
        transformed_batch = dataclasses.replace(
            transformed_batch,
            surf_vars = {
                k: v.clamp(min = 0) if k in model.positive_surf_vars else v
                for k, v in batch.surf_vars.items()
            },
        )
    if model.positive_atmos_vars:
        transformed_batch = dataclasses.replace(
            transformed_batch,
            atmos_vars = {
                k: v.clamp(min = 0) if k in model.positive_atmos_vars else v
                for k, v in batch.atmos_vars.items()
            },
        )
    transformed_batch = model._pre_encoder_hook(transformed_batch)

    x = model.encoder(transformed_batch, lead_time = model.timestep)
    return x, batch


@torch.no_grad()
def backbone_forward_with_boundary(
    model,
    batch_main: Batch,
    batch_bc: Batch,
    replace_boundary_position,
    boundary_width: int,
    boundary_smooth_mode: str = "no",
):
    """Frozen encoder + backbone of the inference forward pass. Returns the backbone output
    (B, L, 2 * embed_dim) fed to the decoder, and the prepared main batch the decoder decodes for."""
    prepped_main = _prepare_batch(model, batch_main)
    if "input" in replace_boundary_position:
        prepped_bc = _prepare_batch(model, batch_bc)
        prepped_main = dataclasses.replace(
            prepped_main,
            surf_vars = {
                k: replace_input_boundary(v, prepped_bc.surf_vars[k], boundary_width, boundary_smooth_mode)
                for k, v in prepped_main.surf_vars.items()
            },
            atmos_vars = {
                k: replace_input_boundary(v, prepped_bc.atmos_vars[k], boundary_width, boundary_smooth_mode)
                for k, v in prepped_main.atmos_vars.items()
            },
        )

    x_main, prepped_batch_main = _encode_prepared(model, prepped_main)

    x_bc = None
    if "backbone" in replace_boundary_position:
        x_bc, _ = _encode_prepared(model, _prepare_batch(model, batch_bc))

    with _autocast_context(model):
        x = model.backbone(
            x_main,
            lead_time = model.timestep,
            patch_res = _patch_res(model, prepped_batch_main),
            rollout_step = prepped_batch_main.metadata.rollout_step,
            x_bc = x_bc,
            replace_boundary_position = list(replace_boundary_position),
            boundary_width = boundary_width,
            patch_size = model.encoder.patch_size,
            boundary_smooth_mode = boundary_smooth_mode,
        )
    return x, prepped_batch_main


def decode_from_backbone(model, x: torch.Tensor, prepped_batch: Batch, decoder = None) -> Batch:
    """Decoder half of the inference forward pass, stopping before `unnormalise`: returns the
    prediction in normalised space with [B, 1, ...] tensors.

    `decoder` defaults to `model.decoder`; pass the DDP-wrapped decoder when training so that
    gradients are synchronised."""
    decoder = model.decoder if decoder is None else decoder
    pred = decoder(
        x,
        prepped_batch,
        lead_time = model.timestep,
        patch_res = _patch_res(model, prepped_batch),
    )
    pred = dataclasses.replace(
        pred,
        static_vars = {k: v[0, 0] for k, v in prepped_batch.static_vars.items()},
        surf_vars = {k: v[:, None] for k, v in pred.surf_vars.items()},
        atmos_vars = {k: v[:, None] for k, v in pred.atmos_vars.items()},
    )
    pred = model._post_decoder_hook(prepped_batch, pred)

    clamp_at_rollout_step = (
        pred.metadata.rollout_step >= 1
        if model.clamp_at_first_step
        else pred.metadata.rollout_step > 1
    )
    if model.positive_surf_vars and clamp_at_rollout_step:
        pred = dataclasses.replace(
            pred,
            surf_vars = {
                k: v.clamp(min = 0) if k in model.positive_surf_vars else v
                for k, v in pred.surf_vars.items()
            },
        )
    if model.positive_atmos_vars and clamp_at_rollout_step:
        pred = dataclasses.replace(
            pred,
            atmos_vars = {
                k: v.clamp(min = 0) if k in model.positive_atmos_vars else v
                for k, v in pred.atmos_vars.items()
            },
        )
    return pred
