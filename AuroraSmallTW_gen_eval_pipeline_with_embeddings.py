#!/usr/bin/env python
# coding=utf-8
"""
Same rollout/eval pipeline as AuroraSmallTW_gen_eval_pipeline_custom_rollout.py, with an
ADDITIVE, opt-in extension: while the rollout runs, optionally extract Swin3D backbone
bottleneck embeddings (prediction trajectory AND matching ERA5 ground truth) and save them
as flattened per-sample .npz files for downstream t-SNE comparison (see
plot_embedding_tsne_hooked.py).

- Hook mechanism: ported from SOFT's AuroraSmall_FID_extract_embeddings_bottleneck.py
  (register_forward_hook on the Swin3D backbone's deepest encoder_layers[-1]).
- Pooling: NOT mean-pooled like SOFT -- flattened per-sample like Mazu's
  utils/embedding.py + plot_embedding_tsne.py (emb.reshape(-1)), preserving per-token
  spatial structure.
- Ground truth frames are fetched independently via plot_embedding_distance.py's
  SnapshotLoader/build_batch, decoupled from the live (possibly error-accumulating)
  rollout trajectory.

There are two, independently selectable consumers of those embeddings:

- --embedding_output_dir: save them as flattened .npz files (one per sample per save-step).
- --embedding_metrics_output_dir: never store them at all -- compare each rollout step's
  prediction embedding against the matching ERA5 ground-truth embedding on the spot (cosine
  similarity + L2, token-wise and pooled, via plot_embedding_distance.embedding_metrics),
  accumulate running sums per lead time, and at the end write one CSV averaged over init times
  plus the same plots plot_embedding_distance.py draws. This is the mode to use when neither
  the embeddings (~3 MB/sample/step) nor the .nc predictions fit on disk.

  NOTE: embeddings here come from the model's real forward pass (the hook), whereas
  plot_embedding_distance.py re-derives them through utils.embedding.encode_batch, which runs
  a hand-rolled copy of the pre-encoder with lead_time=model.timestep. The numbers are
  therefore close but not bit-identical between the two scripts; this one is the one that
  reflects actual inference, including latent boundary replacement. The metric mode also
  keeps the ground truth's metadata time as pd.Timestamp, matching the rollout's own batches
  (see the timezone note at the ground-truth extraction site) -- the .npz path does not.

Every change here is gated behind those two flags: leave them unset and this script
is behaviourally identical to the original AuroraSmallTW_gen_eval_pipeline_custom_rollout.py.
All new code is marked with "=== EMBEDDING EXTRACTION ===" so it stays easy to diff against
upstream after this file's original inevitably evolves.
"""

import argparse
import contextlib
import dataclasses
import json
import pandas as pd
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import random
import numpy as np
import sys
import os
from types import SimpleNamespace

from aurora import Batch, Metadata
# from aurora import rollout
# from utils.custom_rollout import rollout_with_gpu
from aurora.model.aurora import AuroraSmall
from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora, NETCDF_IO_LOCK
from datasets.BoundaryConditionDataset import BoundaryConditionDataset_Aurora, BoundaryConditionDataset_HRES, BoundaryConditionDataset_GroundTruth
from utils.metrics import AuroraMAELoss, AuroraMSELoss
from utils.metrics import prepare_each_lead_time_agg

# === EMBEDDING EXTRACTION === reuse Mazu's own ground-truth snapshot loader/batch builder.
# NOTE: build_batch() hardcodes atmos_levels=tuple(LEVELS) from this module's own globals
# (not from our --levels arg) -- evaluate() asserts args.levels matches LEVELS before using
# it, so a future edit to either file's level list fails loudly instead of silently building
# a wrong ground-truth Batch.
#
# The on-the-fly metric mode (--embedding_metrics_output_dir) additionally reuses that script's
# metric definitions and plotting, so the two paths cannot drift apart.
from plot_embedding_distance import (
    SnapshotLoader,
    build_batch,
    embedding_metrics,
    make_plots,
    LEVELS as _EMBED_GT_LEVELS,
)

from pathlib import Path

import xarray as xr
from safetensors.torch import load_file

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

# =====================================================================================
# === EMBEDDING EXTRACTION ===
# Swin3D bottleneck hook (ported from SOFT's attach_swin_output_hook / extract_embedding)
# + per-sample flatten (Mazu-style, see utils/embedding.py + plot_embedding_tsne.py)
# instead of SOFT's mean-pool.
# =====================================================================================

_EMBED_DTYPES = {"float32": torch.float32, "float16": torch.float16}


def attach_swin_output_hook(model):
    """Hook the deepest Swin3D backbone encoder layer (the bottleneck, before any decoder
    layer runs). Ported near-verbatim from SOFT's AuroraSmall_FID_extract_embeddings_bottleneck.py.
    """
    swin = next(m for m in model.modules() if m.__class__.__name__ == "Swin3DTransformerBackbone")
    target = swin.encoder_layers[-1]
    buf = {}

    def hook(module, inputs, output):
        # encoder layers return (x, x_unscaled); keep the forwarded tokens x.
        out = output[0] if isinstance(output, tuple) else output
        buf["tokens"] = out.detach()

    handle = target.register_forward_hook(hook)
    return handle, buf


def hook_tokens(buf, expected_batch=None):
    """Consume the hook buffer and return the bottleneck tokens, shape (B, L, D).

    When boundary replacement is applied at the backbone level (--replace_boundary_position
    backbone), Swin3DTransformerBackbone.forward concatenates the main state and the
    boundary-condition state ALONG THE BATCH DIM before running them through encoder_layers
    (see aurora/model/swin3d.py: `x_all = torch.cat([x, x_bc], dim=0)`), splitting them apart
    only after our hooked layer. So the hook sees 2*B rows, not B, whenever that path is taken.
    `expected_batch`, when given, keeps only the leading rows (the real rollout samples --
    x_bc's own encoding is not something we want treated as a prediction embedding).
    """
    tokens = buf.pop("tokens")
    if expected_batch is not None and tokens.shape[0] > expected_batch:
        tokens = tokens[:expected_batch]
    return tokens


def flatten_hook_tokens(buf, dtype, expected_batch=None):
    """(B, L, D) bottleneck tokens -> (B, L*D) flattened per-sample vectors (Mazu-style:
    emb.reshape(-1) per sample, NOT SOFT's mean-pool).
    """
    tokens = hook_tokens(buf, expected_batch)
    return tokens.to(dtype).reshape(tokens.shape[0], -1).cpu().numpy()


def _embed_window(model, batch, hook_buf, dtype):
    """Forward a 2-frame Batch through the real model to trigger the hook, then read+flatten
    the bottleneck tokens it captured. Mirrors _prepare_batch_for_rollout's pre-forward
    sequence (batch_transform_hook -> type -> crop -> to(device), no manual normalise --
    model(...) normalises internally), since that's confirmed to be the exact sequence a
    freshly-built Batch needs before a real forward call in this pipeline.
    """
    b = model.batch_transform_hook(batch)
    p = next(model.parameters())
    b = b.type(p.dtype).crop(model.patch_size).to(p.device)
    hook_buf.clear()
    with torch.no_grad():
        _ = model(b)
    if "tokens" not in hook_buf:
        raise RuntimeError("Swin3D bottleneck hook did not fire during embedding extraction.")
    return flatten_hook_tokens(hook_buf, dtype)


def _embed_window_batched(model, batch, hook_buf):
    """Like _embed_window, but forwards the WHOLE batch in one call and returns the raw
    (B, L, D) bottleneck tokens instead of flattened per-sample numpy vectors.

    Used by the on-the-fly metric mode, where every rollout step is evaluated: one extra
    forward per step (instead of one per sample per step) keeps the added cost at ~2x the
    plain rollout regardless of --batch_size.
    """
    b = model.batch_transform_hook(batch)
    p = next(model.parameters())
    b = b.type(p.dtype).crop(model.patch_size).to(p.device)
    hook_buf.clear()
    with torch.no_grad():
        _ = model(b)
    if "tokens" not in hook_buf:
        raise RuntimeError("Swin3D bottleneck hook did not fire during embedding extraction.")
    return hook_tokens(hook_buf, expected_batch = next(iter(batch.surf_vars.values())).shape[0])


# --- On-the-fly embedding-distance metrics -------------------------------------------
# Same six numbers plot_embedding_distance.embedding_metrics() returns, accumulated as
# running sums per lead time so nothing per-sample is ever held (or written) -- the whole
# point of this mode is that neither embeddings nor predictions fit on disk.

_EMBED_METRIC_KEYS = [
    "cos_token", "cos_pooled", "l2_token", "l2_pooled", "rel_l2_token", "rel_l2_pooled",
]


def _new_embed_metric_agg(rollout_step, lead_time):
    return {
        t * lead_time: {"count": 0, **{k: 0.0 for k in _EMBED_METRIC_KEYS}}
        for t in range(1, rollout_step + 1)
    }


def _update_embed_metric_agg(agg, lead_hours, metrics):
    entry = agg[lead_hours]
    entry["count"] += 1
    for k in _EMBED_METRIC_KEYS:
        entry[k] += float(metrics[k])


def _embed_metrics_to_state(agg):
    return {str(t): dict(entry) for t, entry in agg.items()}


def _merge_embed_metrics_state(agg, state):
    for t, entry in state.items():
        ti = int(t)
        agg[ti]["count"] += int(entry["count"])
        for k in _EMBED_METRIC_KEYS:
            agg[ti][k] += float(entry[k])


def _save_embedding_npz(out_dir, init_time, lead_hours, rollout_step, pred_vec, gt_vec):
    path = Path(out_dir) / f"{init_time.strftime('%Y%m%d_%H%M%S')}+{lead_hours}hr.npz"
    np.savez(
        path,
        pred_embedding=pred_vec,
        gt_embedding=gt_vec,
        init_time=str(init_time),
        lead_time=lead_hours,
        rollout_step=rollout_step,
    )

def _consume_pending_embedding(args, hook_buf, dtype, lead_step, pending, embed_metric_agg):
    """Read the prediction-side tokens the hook just captured and hand them to whichever
    consumers asked for this step: the .npz dump (`pending["gt_vecs"]`) and/or the on-the-fly
    distance metrics (`pending["gt_tokens"]`).

    Called once per pending step, so the hook buffer is popped exactly once no matter how many
    consumers are active.
    """
    dates = pending["dates"]
    pred_tokens = hook_tokens(hook_buf, expected_batch = len(dates))
    assert pred_tokens.shape[0] == len(dates), \
        f"hook returned {pred_tokens.shape[0]} rows for a batch of {len(dates)}"
    lead_hours = lead_step * args.lead_time

    if "gt_tokens" in pending:
        gt_tokens = pending["gt_tokens"]
        assert pred_tokens.shape == gt_tokens.shape, \
            f"{tuple(pred_tokens.shape)} != {tuple(gt_tokens.shape)}"
        for i in range(len(dates)):
            _update_embed_metric_agg(
                embed_metric_agg,
                lead_hours,
                embedding_metrics(pred_tokens[i : i + 1], gt_tokens[i : i + 1]),
            )

    if "gt_vecs" in pending:
        pred_vecs = pred_tokens.to(dtype).reshape(pred_tokens.shape[0], -1).cpu().numpy()
        for i in range(pred_vecs.shape[0]):
            _save_embedding_npz(
                args.embedding_output_dir,
                pd.Timestamp(dates[i]),
                lead_hours,
                lead_step,
                pred_vecs[i],
                pending["gt_vecs"][i],
            )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description = "Aurora Evaluation Script (Single GPU).")
    parser.add_argument('--data_root_dir', type = str, required = True)
    parser.add_argument('--boundary_root_dir', type = str, default = None)
    parser.add_argument('--boundary_width', type = int, default = 0)
    parser.add_argument(
        "--boundary_prediction_timedeltas",
        type = int,
        nargs = "+",
        default = None,
    )
    parser.add_argument(
        "--boundary_source",
        type = str,
        default = "aurora",
        choices = ["aurora", "hres", "ground_truth"],
        help = "Select boundary dataset source format.",
    )
    parser.add_argument(
        "--boundary_smooth_mode",
        type = str,
        default = "no",
        choices = ["no", "mean", "gaussian", "linear"],
        help = "Apply 3x3 smoothing/pooling after boundary replacement.",
    )
    parser.add_argument(
        "--boundary_smooth_width_adjustment",
        type = int,
        default = 0,
        help = "Adjustment to sliced boundary width when smoothing is enabled.",
    )
    parser.add_argument(
        "--boundary_time_interp_mode",
        type = str,
        default = "interpolation",
        choices = ["interpolation", "nearest", "exact"],
        help = "Method of time interpolation for boundary dataset.",
    )
    parser.add_argument(
        "--boundary_use_cache",
        action = "store_true",
        help = "Preload all boundary files into memory and serve boundary data from cache.",
    )
    parser.add_argument(
        "--replace_boundary_position",
        type = str,
        nargs = "+",
        choices = ["encoder", "backbone"],
        default = [],
        help = "Select where to replace the boundary latents.",
    )
    parser.add_argument(
        "--boundary_resolution",
        type = float,
        default = 0.25,
        choices = [0.25, 0.5, 1.5],
        help = "Boundary source resolution. 0.25=native baseline; 0.5=pool the 0.25deg source by "
               "2 on the fly; 1.5=native low-res dir (point --boundary_root_dir at "
               "hres_tw_forecast_1.5deg). Only used for --boundary_source hres.",
    )
    parser.add_argument(
        "--boundary_lowres_apply_mode",
        type = str,
        default = "interp",
        choices = ["direct", "interp"],
        help = "How a low-res boundary is mapped onto the model 0.25deg grid: direct=block/nearest "
               "footprint (each low-res pixel fills its NxN model cells); interp=bilinear/linear. "
               "Ignored at --boundary_resolution 0.25.",
    )
    parser.add_argument(
        "--gpu_cache",
        action = "store_true",
        help = "Enable GPU boundary cache and preload boundary files into memory.",
    )
    parser.add_argument("--use_pretrained_weight", action = "store_true")
    parser.add_argument('--checkpoint_path', type = str, default = None)
    parser.add_argument(
        '--lazy_mode',
        action = 'store_true',
        help = 'Stream rollout targets to GPU one step at a time instead of moving the whole batch of targets to GPU upfront. Reduces peak GPU/CPU memory usage for long rollouts.',
    )
    parser.add_argument(
        '--lazy_prefetch_steps',
        type = int,
        default = 2,
        help = 'Lazy mode only: window size (n) of GPU-resident rollout targets. At step t the '
               'target for step t+n-1 is read on a background thread and staged onto the GPU via a '
               'dedicated copy stream, so targets t..t+n-1 are already on the GPU when needed and '
               'H2D transfers overlap with compute. n=1 disables look-ahead (stage target t at step t).',
    )
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--start_date_hour', type = str, required = True)
    parser.add_argument('--end_date_hour', type = str, required = True)
    parser.add_argument('--upper_variables', type = str, nargs = '+', required = True)
    parser.add_argument('--surface_variables', type = str, nargs = '+', required = True)
    parser.add_argument('--static_variables', type = str, nargs = '+', required = True)
    parser.add_argument('--levels', type = int, nargs = '+', required = True)
    parser.add_argument('--latitude', type = float, nargs = 2, required = True)
    parser.add_argument('--longitude', type = float, nargs = 2, required = True)
    parser.add_argument('--lead_time', type = int, default = 0)
    parser.add_argument('--input_time_window', type = int, default = 2)
    parser.add_argument('--rollout_step', type = int, default = 1)

    parser.add_argument("--timestep_hours", type = int, default = 6)
    parser.add_argument('--use_lora', action = 'store_true')
    parser.add_argument('--bf16_mode', action = 'store_true')
    parser.add_argument('--stabilise_level_agg', action = 'store_true')

    parser.add_argument("--gen_result_folder", type = str, default = './gen_result',)
    parser.add_argument("--save_rollout_step", type = int, nargs = "+", default = None)
    parser.add_argument("--eval_metric", type = str, nargs = "+", default = ["MSE"], choices = ["MSE", "MAE"])

    parser.add_argument("--csv_output_folder", type = str, default = "./errs")
    parser.add_argument(
        '--resume_inference',
        action = 'store_true',
        help = 'Resume from the inference-progress checkpoint (if present) written by a previous '
               'run, restoring accumulated errors and skipping already-completed batches. '
               'Inference-progress checkpoints are ALWAYS written regardless of this flag; this '
               'option only controls whether an existing one is loaded. Note: this is unrelated '
               'to --checkpoint_path, which loads model weights.',
    )
    parser.add_argument('--mixed_precision', type = str, default = None, choices = ["no", "fp16", "bf16"])
    parser.add_argument('--mp_world_size', type = int, default = 1)
    parser.add_argument(
        '--gpus',
        type = str,
        default = None,
        help = 'Comma-separated list of GPU ids to use, e.g. "0,1,2". If provided, spawns one process per GPU and binds each process to the corresponding GPU.',
    )

    # === EMBEDDING EXTRACTION ===
    parser.add_argument(
        '--embedding_output_dir',
        type = str,
        default = None,
        help = 'If set, extract Swin3D backbone bottleneck embeddings (flattened, pred + ground '
               'truth) while the rollout runs, and save one .npz per (sample, save-step) here. '
               'Unset (default) disables embedding extraction entirely -- behaviour is then '
               'identical to AuroraSmallTW_gen_eval_pipeline_custom_rollout.py.',
    )
    parser.add_argument(
        '--embedding_save_steps',
        type = int,
        nargs = '+',
        default = None,
        help = 'Which rollout steps to extract embeddings for. Defaults to --save_rollout_step '
               'if omitted. Keep this small: flattened bottleneck vectors are ~3 MB/sample each.',
    )
    parser.add_argument(
        '--embedding_dtype',
        type = str,
        default = 'float32',
        choices = list(_EMBED_DTYPES.keys()),
        help = 'Storage dtype for the flattened embedding vectors.',
    )
    parser.add_argument(
        '--embedding_metrics_output_dir',
        type = str,
        default = None,
        help = 'If set, compute embedding distance/similarity (cosine + L2, token-wise and '
               'pooled) between the rollout prediction and the matching ERA5 ground truth at '
               'EVERY rollout step while the rollout runs, average over init times, and write '
               'embedding_distance.csv + plots here. Nothing per-sample is kept, so this needs no '
               'stored embeddings and no stored predictions. Independent of '
               '--embedding_output_dir: either, both or neither may be set.',
    )
    parser.add_argument(
        '--embedding_metrics_label',
        type = str,
        default = None,
        help = 'Legend label for the embedding-metric plots. Defaults to the basename of '
               '--embedding_metrics_output_dir.',
    )

    return parser.parse_args()


def _embedding_features_enabled(args):
    """True if anything needs the Swin3D bottleneck hook attached."""
    return bool(args.embedding_output_dir) or bool(args.embedding_metrics_output_dir)

def _resolve_mp_world_size(args):
    # If user explicitly provided GPU ids, use that
    if getattr(args, 'gpus', None):
        gpus = [x for x in args.gpus.split(",") if x.strip() != ""]
        return max(1, len(gpus))
    if args.mp_world_size > 1:
        return args.mp_world_size
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        n = len([x for x in visible.split(",") if x.strip() != ""])
        return max(1, n)
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1

def _manual_split_dataset(dataset, rank: int, world_size: int):
    if world_size <= 1:
        return dataset
    indices = list(range(rank, len(dataset), world_size))
    return Subset(dataset, indices)

def _build_metric_lists(args, total_count = None):
    criterion_list = []
    err_agg_list = []
    for metric in args.eval_metric:
        if metric == "MSE":
            criterion_list.append(AuroraMSELoss)
        elif metric == "MAE":
            criterion_list.append(AuroraMAELoss)
        else:
            raise Exception(f"Unsupported eval metric: {metric}")

        err_agg_list.append(
            prepare_each_lead_time_agg(
                rollout_step = args.rollout_step,
                lead_time = args.lead_time,
                surface_variables = args.surface_variables,
                upper_variables = args.upper_variables,
                levels = args.levels,
                err_type = metric,
                total_count = total_count,
            )
        )
    return criterion_list, err_agg_list

def _err_agg_to_state(err_agg):
    state = {}
    for t, t_dict in err_agg.items():
        state[str(t)] = {"surf_vars": {}, "atmos_vars": {}}
        for var, agg in t_dict["surf_vars"].items():
            state[str(t)]["surf_vars"][var] = {
                "error_sum": float(agg.error_sum),
                "count": int(agg.count),
            }
        for var, lev_dict in t_dict["atmos_vars"].items():
            state[str(t)]["atmos_vars"][var] = {}
            for lev, agg in lev_dict.items():
                state[str(t)]["atmos_vars"][var][str(lev)] = {
                    "error_sum": float(agg.error_sum),
                    "count": int(agg.count),
                }
    return state

def _merge_state_into_err_agg(err_agg, state):
    for t, t_dict in state.items():
        ti = int(t)
        for var, s in t_dict["surf_vars"].items():
            err_agg[ti]["surf_vars"][var].error_sum += float(s["error_sum"])
            err_agg[ti]["surf_vars"][var].count += int(s["count"])
        for var, lev_dict in t_dict["atmos_vars"].items():
            for lev, s in lev_dict.items():
                li = int(lev)
                err_agg[ti]["atmos_vars"][var][li].error_sum += float(s["error_sum"])
                err_agg[ti]["atmos_vars"][var][li].count += int(s["count"])

# --- Inference-progress checkpointing -------------------------------------------------
# This is unrelated to model-weight checkpoints (--checkpoint_path). It periodically
# persists how far the rollout evaluation has progressed (per rank) so that a run killed
# mid-inference (e.g. server failure) can be resumed instead of restarted from scratch.
#
# Each rank stores one JSON file holding:
#   - a signature describing the run configuration (guards against resuming into an
#     incompatible run, which would silently corrupt the aggregated metrics),
#   - completed_batches: how many leading dataloader batches were fully processed,
#   - metrics: the accumulated per-lead-time error state (same schema as the MP handoff).
# The dataloader iterates deterministically (shuffle=False), so on resume we restore the
# accumulated errors and simply skip the first `completed_batches` batches.

def _inference_ckpt_path(args, rank):
    root = Path(args.csv_output_folder) if args.csv_output_folder is not None else Path(args.gen_result_folder)
    return root / ".inference_ckpt" / f"rank_{rank}.json"

def _inference_ckpt_signature(args, rank, world_size, num_samples):
    # Fields that, if changed, would make a restored accumulator or a batch-skip count
    # meaningless (or wrong) for the new run.
    return {
        "rank": rank,
        "world_size": world_size,
        "num_samples": num_samples,
        "batch_size": args.batch_size,
        "eval_metric": list(args.eval_metric),
        "rollout_step": args.rollout_step,
        "lead_time": args.lead_time,
        "input_time_window": args.input_time_window,
        "start_date_hour": args.start_date_hour,
        "end_date_hour": args.end_date_hour,
        # === EMBEDDING EXTRACTION === restoring embedding sums into a run that no longer
        # computes them (or computes them over different steps) would silently mix two runs.
        "embedding_metrics_enabled": bool(args.embedding_metrics_output_dir),
        "embedding_save_steps": sorted(args.embedding_save_steps) if args.embedding_save_steps else None,
    }

def _save_inference_ckpt(args, rank, world_size, num_samples, completed_batches, err_agg_list,
                         embed_metric_agg = None):
    path = _inference_ckpt_path(args, rank)
    path.parent.mkdir(parents = True, exist_ok = True)
    payload = {
        "signature": _inference_ckpt_signature(args, rank, world_size, num_samples),
        "completed_batches": int(completed_batches),
        "metrics": {
            metric: _err_agg_to_state(err_agg)
            for metric, err_agg in zip(args.eval_metric, err_agg_list)
        },
        # === EMBEDDING EXTRACTION ===
        "embedding_metrics": _embed_metrics_to_state(embed_metric_agg) if embed_metric_agg is not None else None,
    }
    # Write to a temp file then atomically rename, so a crash mid-write can never leave a
    # truncated/corrupt checkpoint behind: a reader always sees either the complete old
    # file or the complete new one (rename(2) is atomic). fsync of the file (before the
    # rename) and of the directory (after) make this durable even across a hard machine
    # crash / power loss, not just a process kill. Cost is negligible next to a rollout.
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding = "utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

def _load_inference_ckpt(args, rank, world_size, num_samples, err_agg_list,
                         embed_metric_agg = None):
    """Restore accumulated errors from a previous run and return the number of already
    completed batches (0 if there is no usable checkpoint)."""
    path = _inference_ckpt_path(args, rank)
    if not path.exists():
        logger.info("[rank %s] No inference checkpoint at %s; starting from scratch.", rank, path)
        return 0
    with path.open("r", encoding = "utf-8") as f:
        payload = json.load(f)
    expected = _inference_ckpt_signature(args, rank, world_size, num_samples)
    if payload.get("signature") != expected:
        raise RuntimeError(
            f"Inference checkpoint at {path} does not match the current run configuration; "
            f"refusing to resume (this would corrupt the aggregated metrics). Delete the file "
            f"to start fresh, or drop --resume_inference.\n"
            f"  checkpoint: {payload.get('signature')}\n"
            f"  current:    {expected}"
        )
    for metric_idx, metric in enumerate(args.eval_metric):
        _merge_state_into_err_agg(err_agg_list[metric_idx], payload["metrics"][metric])
    # === EMBEDDING EXTRACTION ===
    if embed_metric_agg is not None:
        embed_state = payload.get("embedding_metrics")
        if embed_state is None:
            raise RuntimeError(
                f"Inference checkpoint at {path} holds no embedding metrics, but this run "
                f"computes them; resuming would drop the already-processed batches from the "
                f"average. Delete the file to start fresh, or drop --resume_inference."
            )
        _merge_embed_metrics_state(embed_metric_agg, embed_state)
    completed = int(payload.get("completed_batches", 0))
    logger.info(
        "[rank %s] Resuming inference from %s: %s batch(es) already completed.",
        rank, path, completed,
    )
    return completed

def load_Aurora_weight(
    Aurora_model,
    checkpoint_path,
):
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
        Aurora_model.load_state_dict(state_dict)

def create_model(args, device):
    model = AuroraSmall(
        use_lora = args.use_lora,
        bf16_mode = args.bf16_mode,
        timestep = pd.Timedelta(hours = args.timestep_hours),
        stabilise_level_agg = args.stabilise_level_agg,
    )
    if args.use_pretrained_weight:
        logger.info("Loading pretrained weights provided by Microsoft Aurora...")
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt", strict = True)
    elif args.checkpoint_path:
        logger.info(f"Loading checkpoint: {args.checkpoint_path}")

        load_Aurora_weight(
            model,
            args.checkpoint_path,
        )

    model.to(device)
    model.eval()
    return model

def create_dataset(args):
    logger.info("Creating Aurora dataset%s...", " (lazy target loading)" if args.lazy_mode else "")
    # Calculate the actual end_date_hour needed to load targets during rollout
    start_dt = pd.Timestamp(args.start_date_hour)
    end_dt = pd.Timestamp(args.end_date_hour)
    rollout_duration = pd.Timedelta(hours = (args.input_time_window - 1 + args.rollout_step) * args.lead_time)
    dataset_end_date_hour = end_dt + rollout_duration

    ds = ERA5TWDatasetforAurora(
        data_root_dir = args.data_root_dir,
        start_date_hour = args.start_date_hour,
        end_date_hour = dataset_end_date_hour,
        upper_variables = args.upper_variables,
        surface_variables = args.surface_variables,
        static_variables = args.static_variables,
        levels = args.levels,
        latitude = args.latitude,
        longitude = args.longitude,
        lead_time = args.lead_time,
        input_time_window = args.input_time_window,
        rollout_step = args.rollout_step,
        sample_stride_hours=args.timestep_hours,  # Align dataset sampling with model rollout timestep
        lazy = args.lazy_mode,
    )
    return ds

def create_boundary_dataset(args, target_latitude = None, target_longitude = None):
    if not args.boundary_root_dir:
        return None
    logger.info("Creating Boundary Condition dataset...")
    boundary_ds_width = 0
    prediction_timedeltas = args.boundary_prediction_timedeltas
    if prediction_timedeltas is None:
        if args.boundary_source == "hres":
            prediction_timedeltas = [0, 12]
        elif args.boundary_source == "ground_truth":
            prediction_timedeltas = [k * args.lead_time for k in range(args.rollout_step + 1)]
        else:
            prediction_timedeltas = [0, 6, 12]

    # Use "nearest" time interpolation internally when "exact" mode is requested 
    # so that data loading never fails with None, allowing us to perform exact 
    # matching selectively during evaluation steps.
    internal_time_interp_mode = "nearest" if args.boundary_time_interp_mode == "exact" else args.boundary_time_interp_mode

    # Calculate the actual end_date_hour needed to load targets during rollout
    start_dt = pd.Timestamp(args.start_date_hour)
    end_dt = pd.Timestamp(args.end_date_hour)
    rollout_duration = pd.Timedelta(hours = (args.input_time_window - 1 + args.rollout_step) * args.lead_time)
    dataset_end_date_hour = end_dt + rollout_duration

    if args.boundary_source == "hres":
        return BoundaryConditionDataset_HRES(
            boundary_root_dir = args.boundary_root_dir,
            start_date_hour = args.start_date_hour,
            end_date_hour = dataset_end_date_hour,
            upper_variables = args.upper_variables,
            surface_variables = args.surface_variables,
            levels = args.levels,
            latitude = args.latitude,
            longitude = args.longitude,
            boundary_width = boundary_ds_width,
            prediction_timedeltas = prediction_timedeltas,
            use_cache = args.boundary_use_cache,
            time_interp_mode = internal_time_interp_mode,
            target_latitude = target_latitude,
            target_longitude = target_longitude,
            boundary_resolution = args.boundary_resolution,
            lowres_apply_mode = args.boundary_lowres_apply_mode,
        )
    elif args.boundary_source == "ground_truth":
        return BoundaryConditionDataset_GroundTruth(
            boundary_root_dir = args.boundary_root_dir,
            start_date_hour = args.start_date_hour,
            end_date_hour = dataset_end_date_hour,
            upper_variables = args.upper_variables,
            surface_variables = args.surface_variables,
            levels = args.levels,
            latitude = args.latitude,
            longitude = args.longitude,
            boundary_width = boundary_ds_width,
            prediction_timedeltas = prediction_timedeltas,
            use_cache = args.boundary_use_cache,
            time_interp_mode = internal_time_interp_mode,
        )
    return BoundaryConditionDataset_Aurora(
        boundary_root_dir = args.boundary_root_dir,
        start_date_hour = args.start_date_hour,
        end_date_hour = dataset_end_date_hour,
        upper_variables = args.upper_variables,
        surface_variables = args.surface_variables,
        levels = args.levels,
        latitude = args.latitude,
        longitude = args.longitude,
        boundary_width = boundary_ds_width,
        prediction_timedeltas = prediction_timedeltas,
        use_cache = args.boundary_use_cache,
        time_interp_mode = internal_time_interp_mode,
    )

def log_weather_variable_error_with_lead_time(loss_dict, t, lead_time_agg, rank):
    for v in loss_dict["surf_vars"]:
        lead_time_agg[t]["surf_vars"][v].update( loss_dict["surf_vars"][v] )
    for v in loss_dict["atmos_vars"]:
        for l in loss_dict["atmos_vars"][v]:
            lead_time_agg[t]["atmos_vars"][v][l].update( loss_dict["atmos_vars"][v][l] )
            if v == "z" and l == 50 and t == 60 and rank == 0:
                # print(f"{v}_{l}, {t}: {lead_time_agg[t]["atmos_vars"][v][l]}")
                # print(f"loss_dict: {loss_dict["atmos_vars"][v][l].sum().item()}")
                if str(loss_dict["atmos_vars"][v][l].sum().item()) == "nan":
                    print(loss_dict["atmos_vars"][v][l], lead_time_agg[t]["atmos_vars"][v][l])

def slice_timeaxis(labels):
    timeaxis_length = next(iter(next(iter(labels.values())).values())).shape[1]
    n_g = {}
    for i in range(timeaxis_length):
        n_g[i] = {}
        for var_type, var_dict in labels.items():
            n_g[i][var_type] = {}
            for var_name, tensor in var_dict.items():
                n_g[i][var_type][var_name] = tensor[:, i : i + 1]
    return n_g

def _build_boundary_batch(
    boundary_dataset,
    base_times,
    target_times,
):
    surf_vars = {}
    atmos_vars = {}

    for base_time, target_time in zip(base_times, target_times):
        # netCDF/HDF5 is not thread-safe: serialise this file read against the lazy
        # target-prefetch thread (see NETCDF_IO_LOCK in datasets/ERA5TWDatasetforAurora.py).
        with NETCDF_IO_LOCK:
            data = boundary_dataset.get_boundary_at_time(base_time, target_time)
        if data is None:
            return None
        for var_name, tensor in data["surf_vars"].items():
            surf_vars.setdefault(var_name, []).append(tensor)
        for var_name, tensor in data["atmos_vars"].items():
            atmos_vars.setdefault(var_name, []).append(tensor)

    surf_vars = {k: torch.stack(v, dim = 0) for k, v in surf_vars.items()}
    atmos_vars = {k: torch.stack(v, dim = 0) for k, v in atmos_vars.items()}
    return {"surf_vars": surf_vars, "atmos_vars": atmos_vars}

def _get_boundary_source_on_device(
    boundary_dataset,
    base_time,
    gpu_cache,
    device,
):
    if base_time in gpu_cache:
        return gpu_cache[base_time]
    # netCDF/HDF5 is not thread-safe: serialise this file read against the lazy
    # target-prefetch thread (see NETCDF_IO_LOCK in datasets/ERA5TWDatasetforAurora.py).
    with NETCDF_IO_LOCK:
        source = boundary_dataset.get_boundary_source(base_time)
    gpu_source = {
        "time_values": source["time_values"],
        "surf_vars": {},
        "atmos_vars": {},
    }
    if "prediction_timedelta_hours" in source:
        gpu_source["prediction_timedelta_hours"] = source["prediction_timedelta_hours"].to(device)
    for var_name, tensor in source["surf_vars"].items():
        gpu_source["surf_vars"][var_name] = tensor.to(device)
    for var_name, tensor in source["atmos_vars"].items():
        gpu_source["atmos_vars"][var_name] = tensor.to(device)
    gpu_cache[base_time] = gpu_source
    return gpu_source

def _build_boundary_batch_from_hres_source(
    boundary_dataset,
    source_cache,
    base_times,
    target_times,
):
    surf_vars = {}
    atmos_vars = {}

    for base_time, target_time in zip(base_times, target_times):
        if target_time < base_time:
            hist_cycle = getattr(boundary_dataset, "forecast_cycle_hours", 12)
            effective_base_time = base_time - pd.Timedelta(hours = hist_cycle)
        else:
            effective_base_time = base_time
        source = source_cache[effective_base_time]
        data = boundary_dataset.get_boundary_at_time_from_source(source, effective_base_time, target_time)
        if data is None:
            return None
        for var_name, tensor in data["surf_vars"].items():
            surf_vars.setdefault(var_name, []).append(tensor)
        for var_name, tensor in data["atmos_vars"].items():
            atmos_vars.setdefault(var_name, []).append(tensor)

    surf_vars = {k: torch.stack(v, dim = 0) for k, v in surf_vars.items()}
    atmos_vars = {k: torch.stack(v, dim = 0) for k, v in atmos_vars.items()}
    return {"surf_vars": surf_vars, "atmos_vars": atmos_vars}

def _build_boundary_batch_from_gpu_cache(
    boundary_dataset,
    gpu_cache,
    base_times,
    target_times,
):
    surf_vars = {}
    atmos_vars = {}

    for base_time, target_time in zip(base_times, target_times):
        if target_time < base_time:
            hist_cycle = getattr(boundary_dataset, "forecast_cycle_hours", 12)
            effective_base_time = base_time - pd.Timedelta(hours = hist_cycle)
        else:
            effective_base_time = base_time
        source = gpu_cache[effective_base_time]
        time_values = source["time_values"]

        # Check exact mode for cached boundary
        if getattr(boundary_dataset, "time_interp_mode", "interpolation") == "exact":
            if "prediction_timedelta_hours" in source:
                prediction_timedelta_hours = source["prediction_timedelta_hours"]
                target_prediction_timedelta_hours = float((target_time - effective_base_time) / pd.Timedelta(hours = 1))
                diffs = torch.abs(prediction_timedelta_hours - target_prediction_timedelta_hours)
                min_diff = torch.min(diffs).item()
                if min_diff > 1e-4:
                    return None
            else:
                if target_time not in time_values:
                    return None

        for var_name, tensor in source["surf_vars"].items():
            selected = boundary_dataset._select_from_source(time_values, tensor, target_time)
            if selected is None:
                return None
            surf_vars.setdefault(var_name, []).append(selected)
        for var_name, tensor in source["atmos_vars"].items():
            selected = boundary_dataset._select_from_source(time_values, tensor, target_time)
            if selected is None:
                return None
            atmos_vars.setdefault(var_name, []).append(selected)

    surf_vars = {k: torch.stack(v, dim = 0) for k, v in surf_vars.items()}
    atmos_vars = {k: torch.stack(v, dim = 0) for k, v in atmos_vars.items()}
    return {"surf_vars": surf_vars, "atmos_vars": atmos_vars}

def _is_increasing(coord: torch.Tensor) -> bool:
    if coord.numel() < 2:
        return False
    return coord[0].item() < coord[-1].item()

def _align_boundary_batch(boundary_batch, flip_lat: bool, flip_lon: bool, print_debug = False):
    if boundary_batch is None:
        return None
    if not (flip_lat or flip_lon):
        return boundary_batch
    lat_dim = -2
    lon_dim = -1
    # flip_lat = (not flip_lat)
    if print_debug:
        print(f"Before alignment {next(iter(boundary_batch['surf_vars']))}: {boundary_batch['surf_vars'][next(iter(boundary_batch['surf_vars']))][..., lat_dim]}")
    for var_dict in (boundary_batch["surf_vars"], boundary_batch["atmos_vars"]):
        for k, tensor in var_dict.items():
            if flip_lat:
                tensor = torch.flip(tensor, dims = (lat_dim,))
            if flip_lon:
                tensor = torch.flip(tensor, dims = (lon_dim,))
            var_dict[k] = tensor
    if print_debug:
        print(f"After alignment {next(iter(boundary_batch['surf_vars']))}: {boundary_batch['surf_vars'][next(iter(boundary_batch['surf_vars']))][..., lat_dim]}")
    return boundary_batch

def _stack_boundary_time_window(single_steps):
    if not single_steps or any(s is None for s in single_steps):
        return None
    surf_vars = {}
    for var in single_steps[0]["surf_vars"]:
        surf_vars[var] = torch.stack([s["surf_vars"][var] for s in single_steps], dim = 1)
    atmos_vars = {}
    for var in single_steps[0]["atmos_vars"]:
        atmos_vars[var] = torch.stack([s["atmos_vars"][var] for s in single_steps], dim = 1)
    return {"surf_vars": surf_vars, "atmos_vars": atmos_vars}

def _center_crop_boundary(tensor, boundary_width):
    if boundary_width <= 0:
        return tensor
    return tensor[..., boundary_width:-boundary_width, boundary_width:-boundary_width]

def _pad_interior_with_boundary(interior_tensor, boundary_tensor, boundary_width):
    if boundary_width <= 0:
        return interior_tensor
    h_int, w_int = interior_tensor.shape[-2:]
    h_b, w_b = boundary_tensor.shape[-2:]
    if h_b != h_int + 2 * boundary_width or w_b != w_int + 2 * boundary_width:
        raise ValueError("Boundary tensor shape does not match interior tensor + boundary_width.")
    padded = boundary_tensor.clone()
    padded[..., boundary_width:-boundary_width, boundary_width:-boundary_width] = interior_tensor
    return padded

def _pad_static_vars(static_vars, boundary_width):
    if boundary_width <= 0:
        return static_vars
    padded = {}
    for var_name, tensor in static_vars.items():
        if tensor.dim() == 2:
            padded_tensor = F.pad(
                tensor.unsqueeze(0).unsqueeze(0),
                (boundary_width, boundary_width, boundary_width, boundary_width),
                mode = "replicate",
            ).squeeze(0).squeeze(0)
        elif tensor.dim() == 3:
            padded_tensor = F.pad(
                tensor.unsqueeze(0),
                (boundary_width, boundary_width, boundary_width, boundary_width),
                mode = "replicate",
            ).squeeze(0)
        else:
            padded_tensor = F.pad(
                tensor,
                (boundary_width, boundary_width, boundary_width, boundary_width),
                mode = "replicate",
            )
        padded[var_name] = padded_tensor
    return padded

def _replace_boundary_inside(pred_tensor, boundary_tensor, boundary_width):
    if boundary_width <= 0:
        return pred_tensor
    if pred_tensor.dim() == boundary_tensor.dim() + 1:
        boundary_tensor = boundary_tensor.unsqueeze(1)
    if pred_tensor.dim() != boundary_tensor.dim():
        raise ValueError("Boundary tensor rank does not match prediction tensor.")
    updated = pred_tensor.clone()
    bw = boundary_width
    updated[..., :bw, :] = boundary_tensor[..., :bw, :]
    updated[..., -bw:, :] = boundary_tensor[..., -bw:, :]
    updated[..., :, :bw] = boundary_tensor[..., :, :bw]
    updated[..., :, -bw:] = boundary_tensor[..., :, -bw:]
    return updated

def _slice_interior(tensor, boundary_width):
    if boundary_width <= 0:
        return tensor
    return tensor[..., boundary_width:-boundary_width, boundary_width:-boundary_width]

def _prepare_batch_for_rollout(model, batch):
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    return batch.to(p.device)

def AuroraBatch_2_nc_files(
    batch,
    args,
):
    surf_vars = batch.surf_vars.keys()
    atmos_vars = batch.atmos_vars.keys()
    static_vars = batch.static_vars.keys()

    def _np(d):
        return d.detach().cpu().numpy()

    _s = set(
        [batch.surf_vars[var].shape[0] for var in surf_vars] +
        [batch.atmos_vars[var].shape[0] for var in atmos_vars]
    )

    assert len(_s) == 1

    batch_dim = next(iter(_s))

    for i in range(batch_dim):
        data_vars = {}

        for k, v in batch.surf_vars.items():
            arr = _np(v)[i]
            data_vars[f"surf_{k}"] = (("history", "latitude", "longitude"), arr)

        for k, v in batch.atmos_vars.items():
            arr = _np(v)[i]
            data_vars[f"atmos_{k}"] = (("history", "level", "latitude", "longitude"), arr)

        for k, v in batch.static_vars.items():
            arr = _np(v)
            data_vars[f"static_{k}"] = (("latitude", "longitude"), arr)

        coords = {
            "latitude": _np(batch.metadata.lat),
            "longitude": _np(batch.metadata.lon),
            "time": [batch.metadata.time[i]],
            "level": list(batch.metadata.atmos_levels),
            "rollout_step": batch.metadata.rollout_step,
        }

        ds = xr.Dataset(data_vars, coords = coords)
        rs = int(batch.metadata.rollout_step)
        # output_file_name = f"{(batch.metadata.time[i] - pd.Timedelta(hours = hours + args.lead_time - 1)).strftime('%Y%m%d_%H%M%S')}+{hours + args.lead_time - 1}hr.nc"
        output_file_name = f"{(batch.metadata.time[i] - pd.Timedelta(hours = rs * args.lead_time)).strftime('%Y%m%d_%H%M%S')}+{rs * args.lead_time}hr.nc"
        
        gen_result_folder = Path(args.gen_result_folder)
        output_path = gen_result_folder / output_file_name

        # netCDF/HDF5 is not thread-safe: this write runs on the main thread inside the
        # rollout loop, potentially while the lazy target-prefetch thread is reading target
        # files. Serialise via the shared lock.
        with NETCDF_IO_LOCK:
            ds.to_netcdf( output_path )

def model_forward_with_latent_boundary(model, batch_main, batch_bc, args):
    """
    Custom forward pass of Aurora model that integrates latent boundary replacement.
    """
    import dataclasses
    import contextlib

    p = next(model.parameters())

    def prepare_and_encode(batch):
        batch = model.batch_transform_hook(batch)
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=model.surf_stats)
        batch = batch.crop(patch_size=model.patch_size)
        batch = batch.to(p.device)

        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        static_vars = {}
        for k, v in batch.static_vars.items():
            if v.ndim == 2:
                static_vars[k] = v[None, None].repeat(B, T, 1, 1)
            else:
                static_vars[k] = v
        batch = dataclasses.replace(batch, static_vars=static_vars)

        transformed_batch = batch

        if model.positive_surf_vars:
            transformed_batch = dataclasses.replace(
                transformed_batch,
                surf_vars={
                    k: v.clamp(min=0) if k in model.positive_surf_vars else v
                    for k, v in batch.surf_vars.items()
                },
            )
        if model.positive_atmos_vars:
            transformed_batch = dataclasses.replace(
                transformed_batch,
                atmos_vars={
                    k: v.clamp(min=0) if k in model.positive_atmos_vars else v
                    for k, v in batch.atmos_vars.items()
                },
            )

        transformed_batch = model._pre_encoder_hook(transformed_batch)

        x = model.encoder(
            transformed_batch,
            lead_time=model.timestep,
        )
        return x, batch

    x_main, prepped_batch_main = prepare_and_encode(batch_main)
    
    x_bc = None
    if batch_bc is not None and any(pos in args.replace_boundary_position for pos in ["encoder", "backbone"]):
        x_bc, _ = prepare_and_encode(batch_bc)
        
    if x_bc is not None and "encoder" in args.replace_boundary_position:
        B, L_tokens, D = x_main.shape
        latent_levels = model.encoder.latent_levels
        patch_size = model.encoder.patch_size
        H, W = prepped_batch_main.spatial_shape
        H_latents = H // patch_size
        W_latents = W // patch_size

        x_main_grid = x_main.view(B, latent_levels, H_latents, W_latents, D)
        x_bc_grid = x_bc.view(B, latent_levels, H_latents, W_latents, D)

        latent_boundary_width = args.boundary_width // patch_size
        x_combined_grid = x_main_grid.clone()
        if latent_boundary_width > 0:
            if args.boundary_smooth_mode == "linear":
                h_coords = torch.arange(H_latents, device=x_main_grid.device)
                w_coords = torch.arange(W_latents, device=x_main_grid.device)
                dist_h = torch.minimum(h_coords, H_latents - 1 - h_coords)
                dist_w = torch.minimum(w_coords, W_latents - 1 - w_coords)
                dist_grid = torch.minimum(dist_h.unsqueeze(1), dist_w.unsqueeze(0))
                
                mask = 1.0 - dist_grid.float() / latent_boundary_width
                mask = torch.clamp(mask, min=0.0, max=1.0)
                mask_expanded = mask.view(1, 1, H_latents, W_latents, 1)
                x_combined_grid = mask_expanded * x_bc_grid + (1.0 - mask_expanded) * x_main_grid
            else:
                x_combined_grid[:, :, :latent_boundary_width, :, :] = x_bc_grid[:, :, :latent_boundary_width, :, :]
                x_combined_grid[:, :, -latent_boundary_width:, :, :] = x_bc_grid[:, :, -latent_boundary_width:, :, :]
                x_combined_grid[:, :, :, :latent_boundary_width, :] = x_bc_grid[:, :, :, :latent_boundary_width, :]
                x_combined_grid[:, :, :, -latent_boundary_width:, :] = x_bc_grid[:, :, :, -latent_boundary_width:, :]

                # Smooth the replacement result over H and W dimensions in the latent grid
                if args.boundary_smooth_mode != "no":
                    import torch.nn.functional as F
                    orig_shape = x_combined_grid.shape
                    # Permute to (B, latent_levels, D, H_latents, W_latents) so H, W are the last dimensions
                    permuted = x_combined_grid.permute(0, 1, 4, 2, 3)
                    # Reshape to flatten all dimensions except H_latents and W_latents
                    flat_tensor = permuted.reshape(-1, 1, H_latents, W_latents)
                    
                    if args.boundary_smooth_mode == "mean":
                        kernel = torch.ones((1, 1, 3, 3), dtype = x_combined_grid.dtype, device = x_combined_grid.device) / 9.0
                    elif args.boundary_smooth_mode == "gaussian":
                        kernel = torch.tensor([
                            [1.0, 2.0, 1.0],
                            [2.0, 4.0, 2.0],
                            [1.0, 2.0, 1.0]
                        ], dtype = x_combined_grid.dtype, device = x_combined_grid.device)
                        kernel = kernel / kernel.sum()
                        kernel = kernel.view(1, 1, 3, 3)
                    else:
                        raise ValueError(f"Unsupported smoothing mode: {args.boundary_smooth_mode}")
                    
                    # Smooth only on H and W dimensions
                    padded = F.pad(flat_tensor, (1, 1, 1, 1), mode = "replicate")
                    smoothed = F.conv2d(padded, kernel)
                    
                    # Reshape and permute back to (B, latent_levels, H_latents, W_latents, D)
                    restored = smoothed.reshape(B, latent_levels, D, H_latents, W_latents)
                    x_combined_grid = restored.permute(0, 1, 3, 4, 2)

        x_combined = x_combined_grid.reshape(B, L_tokens, D)
    else:
        x_combined = x_main
        H, W = prepped_batch_main.spatial_shape
        H_latents = H // model.encoder.patch_size
        W_latents = W // model.encoder.patch_size
        latent_levels = model.encoder.latent_levels

    patch_res = (
        latent_levels,
        H_latents,
        W_latents,
    )

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
        x_combined = model.backbone(
            x_combined,
            lead_time=model.timestep,
            patch_res=patch_res,
            rollout_step=prepped_batch_main.metadata.rollout_step,
            x_bc=x_bc if (x_bc is not None and "backbone" in args.replace_boundary_position) else None,
            replace_boundary_position=args.replace_boundary_position,
            boundary_width=args.boundary_width,
            patch_size=model.encoder.patch_size,
            boundary_smooth_mode=args.boundary_smooth_mode,
        )

    pred = model.decoder(
        x_combined,
        prepped_batch_main,
        lead_time=model.timestep,
        patch_res=patch_res,
    )

    pred = dataclasses.replace(
        pred,
        static_vars={k: v[0, 0] for k, v in prepped_batch_main.static_vars.items()},
    )

    pred = dataclasses.replace(
        pred,
        surf_vars={k: v[:, None] for k, v in pred.surf_vars.items()},
        atmos_vars={k: v[:, None] for k, v in pred.atmos_vars.items()},
    )

    pred = model._post_decoder_hook(prepped_batch_main, pred)

    clamp_at_rollout_step = (
        pred.metadata.rollout_step >= 1
        if model.clamp_at_first_step
        else pred.metadata.rollout_step > 1
    )
    if model.positive_surf_vars and clamp_at_rollout_step:
        pred = dataclasses.replace(
            pred,
            surf_vars={
                k: v.clamp(min=0) if k in model.positive_surf_vars else v
                for k, v in pred.surf_vars.items()
            },
        )
    if model.positive_atmos_vars and clamp_at_rollout_step:
        pred = dataclasses.replace(
            pred,
            atmos_vars={
                k: v.clamp(min=0) if k in model.positive_atmos_vars else v
                for k, v in pred.atmos_vars.items()
            },
        )

    pred = pred.unnormalise(surf_stats=model.surf_stats)
    return pred

def _lazy_produce_target(ds_ref, dates, step, device, copy_stream):
    """
    Runs on the single lazy-prefetch worker thread.

    1. Read one rollout-step target from disk (releases the GIL during netCDF I/O,
       so it overlaps with GPU compute on the main thread).
    2. Asynchronously stage it onto the GPU on a dedicated copy stream, so the target
       becomes GPU-resident ahead of time (targets t..t+n-1 all live on the GPU at once)
       while the H2D DMA overlaps with model compute on the default stream.

    Returns (gpu_data_dict, copy_done_event). The event is recorded on copy_stream and
    the main thread must wait on it (on the compute stream) before using the tensors.
    For a CPU device there is no stream/event; tensors are returned already on-device
    and the event is None.
    """
    cpu_dict = ds_ref.load_single_target(
        base_datetime_strs = dates,
        rollout_step_index = step,
    )

    if copy_stream is None:
        # CPU device (or no async path): move directly, no stream/event needed.
        out = {
            "surf_vars": {k: v.to(device) for k, v in cpu_dict["surf_vars"].items()},
            "atmos_vars": {k: v.to(device) for k, v in cpu_dict["atmos_vars"].items()},
        }
        return out, None

    out = {"surf_vars": {}, "atmos_vars": {}}
    # Enqueue the H2D copies on the copy stream. Source tensors are pinned so the copy
    # is a true async DMA; the pinned staging buffers are handed to PyTorch's caching
    # host allocator, which defers their reuse until the copy-stream event completes,
    # so the CPU keeps only ~1 batch alive at a time rather than an n-deep buffer.
    with torch.cuda.stream(copy_stream):
        for section in ("surf_vars", "atmos_vars"):
            for k, cpu_t in cpu_dict[section].items():
                out[section][k] = cpu_t.pin_memory().to(device, non_blocking = True)
        event = copy_stream.record_event()
    return out, event

def evaluate(
    args,
    model,
    dataloader,
    criterion_list,
    err_agg_list,
    device,
    boundary_dataset = None,
    rank = 0,
    metadata_dataset = None,
    world_size = 1,
    embed_hook_buf = None,       # === EMBEDDING EXTRACTION ===
    embed_metric_agg = None,     # === EMBEDDING EXTRACTION ===
):
    model.eval()
    ds_ref = metadata_dataset if metadata_dataset is not None else dataloader.dataset
    latitudes, longitude = ds_ref.get_latitude_longitude()
    levels = ds_ref.get_levels()
    static_data = ds_ref.get_static_vars_ds()

    # === EMBEDDING EXTRACTION ===
    # Two independent consumers of the hooked bottleneck tokens: the .npz dump (npz_enabled)
    # and the on-the-fly distance metrics (metrics_enabled). `embedding_enabled` means "capture
    # tokens at all"; `embedding_save_steps` is the union of the steps either one asks for.
    npz_enabled = embed_hook_buf is not None and bool(args.embedding_output_dir)
    metrics_enabled = (
        embed_hook_buf is not None
        and bool(args.embedding_metrics_output_dir)
        and embed_metric_agg is not None
    )
    embedding_enabled = npz_enabled or metrics_enabled
    embed_dtype = None
    embed_gt_loader = None
    embedding_save_steps = set()

    if npz_enabled:
        # Only the .npz path builds ground-truth batches through plot_embedding_distance's
        # build_batch(), which hardcodes atmos_levels from that module's own LEVELS. The metric
        # path builds its ground-truth window from the rollout's own ERA5 targets and tags it
        # with `levels` from the dataset, so it is not exposed to this mismatch.
        if list(args.levels) != list(_EMBED_GT_LEVELS):
            raise ValueError(
                f"--levels {list(args.levels)} does not match plot_embedding_distance.py's "
                f"hardcoded LEVELS {list(_EMBED_GT_LEVELS)}; build_batch() would silently tag "
                f"ground-truth embeddings with the wrong atmos_levels. Update LEVELS in "
                f"plot_embedding_distance.py to match, or drop --embedding_output_dir."
            )
        npz_save_steps = set(args.embedding_save_steps or args.save_rollout_step or [])
        embedding_save_steps |= npz_save_steps
        embed_gt_loader = SnapshotLoader(ds_ref)
        embed_dtype = _EMBED_DTYPES[args.embedding_dtype]
        Path(args.embedding_output_dir).mkdir(parents = True, exist_ok = True)
        if not npz_save_steps:
            logger.warning(
                "[embedding] --embedding_output_dir is set but no save steps were resolved "
                "(neither --embedding_save_steps nor --save_rollout_step given); no embeddings "
                "will be extracted."
            )
    else:
        npz_save_steps = set()

    if metrics_enabled:
        # Every rollout step, unless the user narrowed it explicitly. Nothing is stored per
        # step beyond a running sum, so "all steps" is the useful default here.
        metric_save_steps = set(args.embedding_save_steps or range(1, args.rollout_step + 1))
        embedding_save_steps |= metric_save_steps
        Path(args.embedding_metrics_output_dir).mkdir(parents = True, exist_ok = True)
        logger.info(
            "[embedding] on-the-fly distance metrics enabled for %d rollout step(s) -> %s",
            len(metric_save_steps), args.embedding_metrics_output_dir,
        )
    else:
        metric_save_steps = set()

    boundary_enabled = boundary_dataset is not None and args.boundary_width > 0
    gpu_boundary_cache = {}
    # Both hres and aurora expose a per-cycle forecast source (prediction_timedelta indexing +
    # get_boundary_at_time_from_source), so they share the source-cache fast path; ground_truth does not.
    boundary_uses_forecast_source = getattr(boundary_dataset, "uses_forecast_source", False)

    if boundary_enabled:
        boundary_latitudes, boundary_longitude = boundary_dataset.get_latitude_longitude()
        flip_lat = _is_increasing(boundary_latitudes) != _is_increasing(latitudes)
        flip_lon = _is_increasing(boundary_longitude) != _is_increasing(longitude)
        if flip_lat:
            boundary_latitudes = torch.flip(boundary_latitudes, dims = (0,))
        if flip_lon:
            boundary_longitude = torch.flip(boundary_longitude, dims = (0,))
    else:
        flip_lat = False
        flip_lon = False

    # Lazy-mode GPU prefetch machinery:
    #  - a single background thread reads target t+n-1 from disk and enqueues its H2D copy,
    #  - a dedicated CUDA copy stream runs those transfers so they overlap with model
    #    compute on the default stream and targets t..t+n-1 stay GPU-resident,
    #  - CUDA events synchronise the compute stream against each target's copy completion.
    lazy_prefetch_executor = None
    lazy_copy_stream = None
    if args.lazy_mode:
        from concurrent.futures import ThreadPoolExecutor
        # One worker is enough: it only needs to produce a single new target (t+n-1) per step.
        lazy_prefetch_executor = ThreadPoolExecutor(max_workers = 1)
        if device.type == "cuda":
            lazy_copy_stream = torch.cuda.Stream(device = device)

    # --- Inference-progress checkpoint (always-on) ---
    # Restore accumulated errors and the completed-batch count so a killed run can resume.
    # Saving happens after every batch; loading only when --resume_inference is set.
    num_samples = len(dataloader.dataset)
    start_batch = 0
    if getattr(args, "resume_inference", False):
        start_batch = _load_inference_ckpt(
            args, rank, world_size, num_samples, err_agg_list,
            embed_metric_agg = embed_metric_agg if metrics_enabled else None,
        )

    # Optimization: Use inference_mode to reduce memory for gradients
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc = f"Evaluating(rank={rank})", disable = (rank != 0))):
        # for batch in dataloader:
            # if (rank == 0):
            #     print("----------------------------------------")
            # Skip batches already processed in a previous (resumed) run. The dataloader
            # iterates deterministically (shuffle=False), so batch_idx maps to the same data.
            if batch_idx < start_batch:
                continue
            if args.lazy_mode:
                # --- Lazy version: batch is (inputs, dates), no pre-loaded labels ---
                inputs, dates = batch
                labels = None
            else:
                inputs, labels, dates = batch

            # --- Data moving to device ---
            for _k_var_type in inputs:
                for _k_var in inputs[_k_var_type]:
                    inputs[_k_var_type][_k_var] = inputs[_k_var_type][_k_var].to(device)
            if not args.lazy_mode:
                for _k_var_type in labels:
                    for _k_var in labels[_k_var_type]:
                        labels[_k_var_type][_k_var] = labels[_k_var_type][_k_var].to(device)
            if isinstance(static_data["static_vars"], torch.Tensor):
                static_data["static_vars"] = static_data["static_vars"].to(device)

            _label_list = None
            if not args.lazy_mode:
                # Pre-slice labels (this is okay to keep in list if it fits in memory,
                # usually labels are smaller than the computation graph)
                _label_list = slice_timeaxis(labels)

            batch_times = tuple(map(lambda d: pd.Timestamp(d), dates))
            base_times = None
            if boundary_enabled:
                base_times = tuple(boundary_dataset.get_base_time(t) for t in batch_times)

            # Prefetch boundary data for all autoregressive steps and transfer to device.
            # For input_time_window > 1, each entry prefetched_boundary[k] is a multi-timestep
            # boundary with tensors of shape [B, T, ...], where T = input_time_window.
            # The T time-steps correspond to offsets (in hours from date):
            #   [ k*lead_time - (W-1)*timestep_hours,  ...,  k*lead_time ]
            # where W = input_time_window.  For historical steps (k*lead_time - n*timestep_hours < 0)
            # the target time can be negative relative to base_time — boundary_dataset is
            # expected to handle this gracefully (analysis data that pre-dates the rollout).
            prefetched_boundary = None
            if boundary_enabled and args.replace_boundary_position != []:
                prefetched_boundary = {}
                _input_tw = getattr(args, 'input_time_window', 1)
                _ts_hours = getattr(args, 'timestep_hours', 6)

                if boundary_uses_forecast_source:
                    source_cache = gpu_boundary_cache if args.gpu_cache else {}
                    hist_cycle = getattr(boundary_dataset, "forecast_cycle_hours", 12)
                    for base_time in set(base_times):
                        _get_boundary_source_on_device(
                            boundary_dataset,
                            base_time,
                            source_cache,
                            device,
                        )
                        _get_boundary_source_on_device(
                            boundary_dataset,
                            base_time - pd.Timedelta(hours = hist_cycle),
                            source_cache,
                            device,
                        )
                    for k in range(0, args.rollout_step + 1):
                        # Collect W single-step batches (oldest → newest)
                        single_steps = []
                        for tw in range(_input_tw - 1, -1, -1):  # W-1 down to 0
                            offset_hours = k * args.lead_time - tw * _ts_hours
                            target_times_k = tuple(
                                pd.Timestamp(d) + pd.Timedelta(hours=offset_hours)
                                for d in dates
                            )
                            b_single = _build_boundary_batch_from_hres_source(
                                boundary_dataset,
                                source_cache,
                                base_times,
                                target_times_k,
                            )
                            b_single = _align_boundary_batch(b_single, flip_lat, flip_lon)
                            single_steps.append(b_single)
                        prefetched_boundary[k] = _stack_boundary_time_window(single_steps)

                elif args.gpu_cache:
                    hist_cycle = getattr(boundary_dataset, "forecast_cycle_hours", 12)
                    for base_time in set(base_times):
                        _get_boundary_source_on_device(
                            boundary_dataset,
                            base_time,
                            gpu_boundary_cache,
                            device,
                        )
                        _get_boundary_source_on_device(
                            boundary_dataset,
                            base_time - pd.Timedelta(hours = hist_cycle),
                            gpu_boundary_cache,
                            device,
                        )
                    # include initial step k=0 (current time) and future steps 1..rollout_step
                    for k in range(0, args.rollout_step + 1):
                        single_steps = []
                        for tw in range(_input_tw - 1, -1, -1):
                            offset_hours = k * args.lead_time - tw * _ts_hours
                            target_times_k = tuple(
                                pd.Timestamp(d) + pd.Timedelta(hours=offset_hours)
                                for d in dates
                            )
                            b_single = _build_boundary_batch_from_gpu_cache(
                                boundary_dataset,
                                gpu_boundary_cache,
                                base_times,
                                target_times_k,
                            )
                            b_single = _align_boundary_batch(b_single, flip_lat, flip_lon)
                            single_steps.append(b_single)
                        prefetched_boundary[k] = _stack_boundary_time_window(single_steps)

                else:
                    # include initial step k=0 (current time) and future steps 1..rollout_step
                    for k in range(0, args.rollout_step + 1):
                        single_steps = []
                        for tw in range(_input_tw - 1, -1, -1):
                            offset_hours = k * args.lead_time - tw * _ts_hours
                            target_times_k = tuple(
                                pd.Timestamp(d) + pd.Timedelta(hours=offset_hours)
                                for d in dates
                            )
                            b_single = _build_boundary_batch(boundary_dataset, base_times, target_times_k)
                            b_single = _align_boundary_batch(b_single, flip_lat, flip_lon)
                            # Move to device to avoid host->device during rollout
                            if b_single is not None:
                                for var_name, tensor in b_single["surf_vars"].items():
                                    b_single["surf_vars"][var_name] = tensor.to(device)
                                for var_name, tensor in b_single["atmos_vars"].items():
                                    b_single["atmos_vars"][var_name] = tensor.to(device)
                            single_steps.append(b_single)
                        prefetched_boundary[k] = _stack_boundary_time_window(single_steps)

            # --- Lazy GPU-prefetch window setup (per batch) ---
            # Keep up to n targets in flight (t..t+n-1), each read on the worker thread and
            # staged onto the GPU via the copy stream. Priming submits steps 1..n-1 so that
            # inside the rollout loop step t only submits step t+n-1, holding the window at n.
            # NOTE: this must stay AFTER the boundary prefetch above — boundary loading reads
            # netCDF files on the main thread, and the worker is guaranteed idle here (its
            # window fully drains at the end of each batch's rollout loop). Every remaining
            # netCDF touchpoint (worker target reads, boundary reads, prediction writes) is
            # additionally serialised via NETCDF_IO_LOCK, since the HDF5 C library is not
            # thread-safe even across different files.
            lazy_target_futures = None
            _lazy_n = 1
            if args.lazy_mode:
                _lazy_n = max(1, getattr(args, "lazy_prefetch_steps", 1))
                lazy_target_futures = {}
                for _s in range(1, min(_lazy_n - 1, args.rollout_step) + 1):
                    lazy_target_futures[_s] = lazy_prefetch_executor.submit(
                        _lazy_produce_target, ds_ref, dates, _s, device, lazy_copy_stream
                    )

            metadata_lat = latitudes
            metadata_lon = longitude

            _input = Batch(
                surf_vars = inputs["surf_vars"],
                atmos_vars = inputs["atmos_vars"],
                static_vars = static_data["static_vars"],
                metadata = Metadata(
                    lat = metadata_lat,
                    lon = metadata_lon,
                    time = batch_times,
                    atmos_levels = levels,
                ),
            )
            

            assert model.training is False

            # --- Setup Mixed Precision ---
            use_amp = (args.mixed_precision in ("fp16", "bf16")) and (device.type == "cuda")
            dtype = torch.float32  # Default
            if use_amp:
                if args.mixed_precision == "fp16":
                    dtype = torch.float16
                elif args.mixed_precision == "bf16":
                    dtype = torch.bfloat16

            # --- THE OPTIMIZED LOOP ---
            # We create a dummy context manager if AMP is not used
            context_manager = torch.amp.autocast(dtype = dtype) if use_amp else contextlib.nullcontext()
            
            with context_manager:
                rollout_batch = _prepare_batch_for_rollout(model, _input)

                # === EMBEDDING EXTRACTION ===
                # t -> {"gt_vecs"/"gt_tokens": ..., "dates": dates} until its pred-side embedding
                # (captured for free on the FOLLOWING iteration's forward call, see below)
                # arrives too.
                pending_embeddings = {} if embedding_enabled else None

                # === EMBEDDING EXTRACTION ===
                # Ground-truth window for the metric path: [ERA5(t-1), ERA5(t)]. ERA5(t) is the
                # loss label the rollout loop already stages on the GPU each step, and ERA5(0) is
                # the last input frame, so the whole window comes for free -- no extra netCDF
                # reads, unlike the .npz path's SnapshotLoader (which was written for a handful
                # of save steps, not for every step of every init time).
                prev_label_data = None
                if metrics_enabled:
                    prev_label_data = {
                        section: {k: v[:, -1:] for k, v in inputs[section].items()}
                        for section in ("surf_vars", "atmos_vars")
                    }

                for step_index in range(args.rollout_step):
                    # step_index starts at 0, so lead time t is step_index + 1
                    t = step_index + 1

                    if boundary_enabled:
                        b_curr = prefetched_boundary[step_index]

                        # b_curr tensors already have shape [B, T, ...] where T = input_time_window
                        # (stacked by _stack_boundary_time_window during prefetch).
                        # For T=1 (input_time_window=1) this is equivalent to the old .unsqueeze(1) path.
                        boundary_batch = Batch(
                            surf_vars = b_curr["surf_vars"],
                            atmos_vars = b_curr["atmos_vars"],
                            static_vars = static_data["static_vars"],
                            metadata = Metadata(
                                lat = latitudes,
                                lon = longitude,
                                time = rollout_batch.metadata.time,
                                atmos_levels = levels,
                                rollout_step = rollout_batch.metadata.rollout_step,
                            ),
                        )
                        _pred = model_forward_with_latent_boundary(model, rollout_batch, boundary_batch, args)
                        prefetched_boundary[step_index] = None
                    else:
                        _pred = model(rollout_batch)

                    # === EMBEDDING EXTRACTION ===
                    # The forward call just above consumed `rollout_batch` = the 2-frame window
                    # [state(t-2), state(t-1)] (before this iteration's own end-of-loop update),
                    # i.e. the window ENDING at lead `t-1` -- exactly the temporally-aligned
                    # prediction-side embedding for a save-step `t-1` requested earlier. Reading
                    # the hook buffer here is free: no extra forward pass, and it's automatically
                    # boundary-consistent since it's the same call the rollout already needed.
                    if embedding_enabled:
                        capture_lead = t - 1
                        pending = pending_embeddings.pop(capture_lead, None)
                        if pending is not None:
                            _consume_pending_embedding(
                                args, embed_hook_buf, embed_dtype, capture_lead, pending,
                                embed_metric_agg,
                            )

                    # 1. Get the corresponding label for this specific step
                    if args.lazy_mode:
                        # --- LAZY GPU PREFETCH ---
                        # Kick off producing the newest target in the window (t+n-1): the worker
                        # reads it from disk and the copy stream stages it onto the GPU, overlapping
                        # with this step's compute. Then consume target t, which was submitted n-1
                        # steps ago and is (already / almost) GPU-resident.
                        _prefetch_step = t + _lazy_n - 1
                        if _prefetch_step <= args.rollout_step and _prefetch_step not in lazy_target_futures:
                            lazy_target_futures[_prefetch_step] = lazy_prefetch_executor.submit(
                                _lazy_produce_target, ds_ref, dates, _prefetch_step, device, lazy_copy_stream
                            )
                        # .result() only blocks until the disk read + copy enqueue finished; the
                        # actual H2D transfer is synchronised on the GPU via the copy event below.
                        _label_data, _copy_event = lazy_target_futures.pop(t).result()
                        if _copy_event is not None:
                            # Make the compute (default) stream wait for the copy to complete, and
                            # tell the caching allocator the default stream now uses these tensors
                            # (they were allocated on the copy stream) so they are not freed early.
                            torch.cuda.current_stream().wait_event(_copy_event)
                            for _section in ("surf_vars", "atmos_vars"):
                                for _k_var in _label_data[_section]:
                                    _label_data[_section][_k_var].record_stream(torch.cuda.current_stream())
                    else:
                        _label_data = _label_list[step_index]

                    _label = Batch(
                        surf_vars = _label_data["surf_vars"],
                        atmos_vars = _label_data["atmos_vars"],
                        static_vars = static_data["static_vars"],
                        metadata = Metadata(
                            lat = latitudes,
                            lon = longitude,
                            time = tuple(
                                map(
                                    lambda d: pd.Timestamp(d) + pd.Timedelta(hours = t * args.lead_time),
                                    dates,
                                )
                            ),
                            atmos_levels = levels,
                        ),
                    )

                    # === EMBEDDING EXTRACTION ===
                    # Ground-truth side of this save-step's embedding: fetch [ERA5(t-1), ERA5(t)]
                    # directly from the dataset (via SnapshotLoader, same helper
                    # plot_embedding_distance.py/plot_embedding_tsne.py already use), independent
                    # of the model's own (possibly error-accumulating) rollout trajectory. Stash it;
                    # the matching pred-side embedding for this same `t` is captured for free on the
                    # NEXT loop iteration's forward call above (or, if `t` is the last rollout step,
                    # by the one-off fallback forward pass after the loop).
                    #
                    # The metric path takes its ground truth from `_label_data` instead (see
                    # `prev_label_data` above): same ERA5 states, already on the GPU, and embedded
                    # for the whole batch in a single forward rather than one per sample.
                    #
                    # It also reuses `_label.metadata.time`, i.e. pd.Timestamp, exactly like every
                    # batch the rollout itself feeds the model. That matters: the encoder turns the
                    # metadata time into an absolute-time feature via `t.timestamp()`
                    # (aurora/model/encoder.py), and a naive datetime.datetime -- which
                    # plot_embedding_distance.build_batch() produces via .to_pydatetime() -- is
                    # interpreted in the machine's LOCAL timezone, while pd.Timestamp is treated as
                    # UTC. Mixing the two would encode the ground truth at a different absolute
                    # time than the prediction it is compared against (8h apart on this host).
                    if metrics_enabled and t in metric_save_steps:
                        gt_batch = Batch(
                            surf_vars = {
                                k: torch.cat([prev_label_data["surf_vars"][k], v], dim = 1)
                                for k, v in _label_data["surf_vars"].items()
                            },
                            atmos_vars = {
                                k: torch.cat([prev_label_data["atmos_vars"][k], v], dim = 1)
                                for k, v in _label_data["atmos_vars"].items()
                            },
                            static_vars = static_data["static_vars"],
                            metadata = Metadata(
                                lat = latitudes,
                                lon = longitude,
                                time = _label.metadata.time,
                                atmos_levels = levels,
                            ),
                        )
                        pending_embeddings.setdefault(t, {"dates": dates})["gt_tokens"] = (
                            _embed_window_batched(model, gt_batch, embed_hook_buf)
                        )

                    if npz_enabled and t in npz_save_steps:
                        gt_vecs = []
                        for i in range(len(dates)):
                            prev_time_i = pd.Timestamp(dates[i]) + pd.Timedelta(hours = (t - 1) * args.lead_time)
                            curr_time_i = pd.Timestamp(dates[i]) + pd.Timedelta(hours = t * args.lead_time)
                            # SnapshotLoader.gt() opens netCDF files via xarray with no locking of
                            # its own (plot_embedding_distance.py was written for single-threaded
                            # use). In --lazy_mode a background thread concurrently reads netCDF
                            # targets through load_single_target(), which DOES take
                            # NETCDF_IO_LOCK (see datasets/ERA5TWDatasetforAurora.py) because the
                            # HDF5 C library is not thread-safe; without also taking that lock
                            # here, this raced with the background thread and segfaulted (native
                            # crash, no Python traceback).
                            with NETCDF_IO_LOCK:
                                snap_prev = embed_gt_loader.gt(prev_time_i)
                                snap_curr = embed_gt_loader.gt(curr_time_i)
                            gt_batch_i = build_batch(
                                snap_prev,
                                snap_curr,
                                static_data["static_vars"],
                                latitudes,
                                longitude,
                                curr_time_i,
                            )
                            gt_vecs.append(_embed_window(model, gt_batch_i, embed_hook_buf, embed_dtype)[0])
                        pending_embeddings.setdefault(t, {"dates": dates})["gt_vecs"] = gt_vecs

                    # === EMBEDDING EXTRACTION ===
                    # This step's ERA5 target becomes the older half of the next step's
                    # ground-truth window.
                    if metrics_enabled:
                        prev_label_data = _label_data

                    # Determine slice width for error calculation
                    # slice_width = args.boundary_width
                    # if boundary_enabled and args.boundary_smooth_mode != "no":
                    #     slice_width = args.boundary_width + args.boundary_smooth_width_adjustment
                    slice_width = 8

                    # 2. Calculate Loss immediately
                    if boundary_enabled and slice_width > 0:
                        pred_interior = Batch(
                            surf_vars = {
                                k: _slice_interior(v, slice_width)
                                for k, v in _pred.surf_vars.items()
                            },
                            atmos_vars = {
                                k: _slice_interior(v, slice_width)
                                for k, v in _pred.atmos_vars.items()
                            },
                            static_vars = static_data["static_vars"],
                            metadata = Metadata(
                                lat = latitudes,
                                lon = longitude,
                                time = _label.metadata.time,
                                atmos_levels = levels,
                                rollout_step = _pred.metadata.rollout_step,
                            ),
                        )
                        label_interior = Batch(
                            surf_vars = {
                                k: _slice_interior(v, slice_width)
                                for k, v in _label.surf_vars.items()
                            },
                            atmos_vars = {
                                k: _slice_interior(v, slice_width)
                                for k, v in _label.atmos_vars.items()
                            },
                            static_vars = static_data["static_vars"],
                            metadata = Metadata(
                                lat = latitudes,
                                lon = longitude,
                                time = _label.metadata.time,
                                atmos_levels = levels,
                            ),
                        )
                        loss_pred = pred_interior
                        loss_label = label_interior
                    else:
                        loss_pred = _pred
                        loss_label = _label

                    for (criterion, err_agg) in zip(criterion_list, err_agg_list):
                        loss_dict = criterion(loss_pred, loss_label)
                        log_weather_variable_error_with_lead_time(
                            loss_dict,
                            t * args.lead_time,
                            err_agg,
                            rank
                        )

                    # 3. Save to disk if needed (then discard from memory)
                    if args.save_rollout_step and t in args.save_rollout_step:
                        AuroraBatch_2_nc_files(
                            batch = _pred,
                            args = args,
                        )

                    pred_for_next = _pred

                    rollout_batch = dataclasses.replace(
                        pred_for_next,
                        surf_vars = {
                            k: torch.cat([rollout_batch.surf_vars[k][:, 1:], v], dim = 1)
                            for k, v in pred_for_next.surf_vars.items()
                        },
                        atmos_vars = {
                            k: torch.cat([rollout_batch.atmos_vars[k][:, 1:], v], dim = 1)
                            for k, v in pred_for_next.atmos_vars.items()
                        },
                        metadata = Metadata(
                            lat = latitudes,
                            lon = longitude,
                            time = tuple(
                                map(
                                    lambda d: pd.Timestamp(d) + pd.Timedelta(hours = t * args.lead_time),
                                    dates,
                                )
                            ),
                            atmos_levels = levels,
                            rollout_step = t,
                        ),
                    )

                # === EMBEDDING EXTRACTION ===
                # Any save-step still pending here requested `t == args.rollout_step`: there was
                # no following iteration to capture its pred-side embedding for free, so do one
                # explicit extra forward pass now. Plain (non-boundary) since there is no "next"
                # boundary-conditioned state to fall back to.
                if embedding_enabled and pending_embeddings:
                    for t_left, pending in pending_embeddings.items():
                        # logger.warning(
                        #     "[embedding] lead %sh: final rollout step has no next-iteration "
                        #     "forward pass; doing one extra plain (non-boundary-conditioned) "
                        #     "forward to capture its prediction embedding.",
                        #     t_left * args.lead_time,
                        # )
                        # See the matching comment above (ground-truth extraction): avoid
                        # overlapping this extra forward with any in-flight lazy-mode background
                        # copy-stream work.
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        embed_hook_buf.clear()
                        with torch.no_grad():
                            _ = model(rollout_batch)
                        _consume_pending_embedding(
                            args, embed_hook_buf, embed_dtype, t_left, pending, embed_metric_agg,
                        )
                    pending_embeddings.clear()

            if boundary_enabled:
                gpu_boundary_cache.clear()

            # Free CPU and GPU memory at the end of the batch
            inputs = None
            labels = None
            dates = None
            _label_list = None
            _input = None
            rollout_batch = None
            prev_label_data = None  # === EMBEDDING EXTRACTION ===
            if prefetched_boundary is not None:
                prefetched_boundary.clear()
            if lazy_target_futures is not None:
                lazy_target_futures.clear()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Persist inference progress: this batch's errors are now folded into
            # err_agg_list, so record it as completed. Atomic write (see helper).
            _save_inference_ckpt(
                args, rank, world_size, num_samples, batch_idx + 1, err_agg_list,
                embed_metric_agg = embed_metric_agg if metrics_enabled else None,
            )

    if lazy_prefetch_executor is not None:
        lazy_prefetch_executor.shutdown(wait = True)

def export_agg_to_csv(
        args,
        lead_time_err_agg,
        out_path,
    ):

    lead_times = sorted(lead_time_err_agg.keys())
    # lead_time_labels = [f"{t + args.lead_time - 1}h" for t in lead_times]
    lead_time_labels = [f"{t}h" for t in lead_times]

    surf_vars = set()
    atmos_vars_levels = dict()
    for t in lead_time_err_agg:
        for var in lead_time_err_agg[t]["surf_vars"]:
            surf_vars.add(var)
        for var in lead_time_err_agg[t]["atmos_vars"]:
            if var not in atmos_vars_levels:
                atmos_vars_levels[var] = set()
            for lev in lead_time_err_agg[t]["atmos_vars"][var]:
                atmos_vars_levels[var].add(lev)
    surf_vars = sorted(list(surf_vars))

    atmos_rows = []
    for var in sorted(atmos_vars_levels.keys()):
        levels = sorted(list(atmos_vars_levels[var]), reverse = True)
        for lev in levels:
            atmos_rows.append((var, lev))

    rows = []
    row_names = []

    for var in surf_vars:
        row = []
        for t in lead_times:
            agg = lead_time_err_agg[t]["surf_vars"].get(var)
            row.append( agg.mean() if agg is not None else None)
        rows.append(row)
        row_names.append(var)

    for var, lev in atmos_rows:
        row = []
        for t in lead_times:
            agg = lead_time_err_agg[t]["atmos_vars"].get(var, {}).get(lev)
            row.append( agg.mean() if agg is not None else None)
        rows.append(row)
        row_names.append(f"{var}_{lev}")

    df = pd.DataFrame(rows, index = row_names, columns = lead_time_labels)
    df.to_csv(out_path)
    return df

def export_embedding_metrics(args, embed_metric_agg, out_dir, label, init_time_desc):
    """=== EMBEDDING EXTRACTION ===
    Average the running sums over init times, write embedding_distance.csv, and draw the same
    figures plot_embedding_distance.py draws (reusing its make_plots, so the two stay in sync).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    rows = []
    for lead_hours in sorted(embed_metric_agg.keys()):
        entry = embed_metric_agg[lead_hours]
        n = entry["count"]
        if n == 0:
            continue
        rows.append({
            "label": label,
            "lead_time": lead_hours,
            "n": n,
            **{k: entry[k] / n for k in _EMBED_METRIC_KEYS},
        })

    if not rows:
        logger.warning(
            "[embedding] no embedding metrics were accumulated; nothing written to %s", out_dir,
        )
        return None

    df = pd.DataFrame(rows)
    csv_path = out_dir / "embedding_distance.csv"
    df.drop(columns = ["label"]).to_csv(csv_path, index = False)
    logger.info("[embedding] wrote %s (%d lead times)", csv_path, len(df))

    # make_plots() reads only these four attributes off `args`; mirror
    # plot_embedding_distance.py's own argparse defaults rather than adding CLI flags here.
    plot_args = SimpleNamespace(width = 8.0, height = 5.0, ext = "png", dpi = 150)
    make_plots(df, [label], plot_args, out_dir, init_time_desc)
    logger.info("[embedding] wrote plots to %s", out_dir)
    return df


def _embedding_metrics_label(args):
    return args.embedding_metrics_label or Path(args.embedding_metrics_output_dir).name


def _embedding_init_time_desc(args, num_init_times):
    return f"{args.start_date_hour} .. {args.end_date_hour} ({num_init_times} init times)"


def _mp_worker_entry(rank, world_size, args, cuda_available):
    # Restrict this process to its single assigned physical GPU BEFORE any CUDA API
    # call. CUDA_VISIBLE_DEVICES is inherited from the parent shell (all GPUs), so
    # without this every rank sees every GPU; CUDA's lazy runtime init always does its
    # (heavy) first-touch initialization against "current device" = index 0, regardless
    # of which device this rank later calls set_device() on. That leaves a phantom
    # context on physical GPU 0 from every other rank, inflating its memory usage.
    # Narrowing visibility here makes "device 0" inside this process BE this rank's GPU,
    # so no rank can ever leak a context onto another rank's GPU. `cuda_available` is
    # computed once in the parent so nothing here calls torch.cuda.* before narrowing.
    #
    # Bind to the physical GPU id the user actually selected (args.gpu_list[rank]), NOT
    # str(rank): with --gpus 0,1,4,6 the four ranks must land on physical GPUs 0,1,4,6,
    # not 0,1,2,3. This value overrides the parent's CUDA_VISIBLE_DEVICES entirely, and
    # since CUDA_DEVICE_ORDER=PCI_BUS_ID is exported by the launch script the id maps to
    # the same physical device nvidia-smi shows. Fall back to str(rank) when no explicit
    # GPU list was given (e.g. --mp_world_size / CUDA_VISIBLE_DEVICES-derived world size).
    if cuda_available:
        gpu_list = getattr(args, "gpu_list", None)
        physical_gpu = str(gpu_list[rank]) if gpu_list else str(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu

    set_seed(args.seed + rank)
    if cuda_available:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = create_model(args, device)

    # === EMBEDDING EXTRACTION ===
    embed_hook_handle, embed_hook_buf = (None, None)
    if _embedding_features_enabled(args):
        embed_hook_handle, embed_hook_buf = attach_swin_output_hook(model)
    embed_metric_agg = (
        _new_embed_metric_agg(args.rollout_step, args.lead_time)
        if args.embedding_metrics_output_dir else None
    )

    full_dataset = create_dataset(args)
    eval_dataset = _manual_split_dataset(full_dataset, rank, world_size)
    # Pass the model's 0.25deg grid so a low-res boundary is regridded onto it in the loader.
    _model_lat, _model_lon = full_dataset.get_latitude_longitude()
    boundary_dataset = create_boundary_dataset(args, target_latitude = _model_lat, target_longitude = _model_lon)
    dataloader = DataLoader(
        eval_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = True,
    )
    criterion_list, err_agg_list = _build_metric_lists(args, total_count = len(full_dataset))

    evaluate(
        args,
        model,
        dataloader,
        criterion_list,
        err_agg_list,
        device,
        boundary_dataset = boundary_dataset,
        rank = rank,
        metadata_dataset = full_dataset,
        world_size = world_size,
        embed_hook_buf = embed_hook_buf,      # === EMBEDDING EXTRACTION ===
        embed_metric_agg = embed_metric_agg,  # === EMBEDDING EXTRACTION ===
    )

    # === EMBEDDING EXTRACTION ===
    if embed_hook_handle is not None:
        embed_hook_handle.remove()

    tmp_root = Path(args.csv_output_folder) if args.csv_output_folder is not None else Path(args.gen_result_folder)
    tmp_root.mkdir(parents = True, exist_ok = True)
    out_path = tmp_root / f".mp_rank_{rank}_metrics.json"
    payload = {
        "rank": rank,
        "metrics": {
            metric: _err_agg_to_state(err_agg)
            for metric, err_agg in zip(args.eval_metric, err_agg_list)
        },
        # === EMBEDDING EXTRACTION === this rank's share of the init times; main() merges the
        # per-rank sums before averaging, so the result does not depend on the world size.
        "embedding_metrics": _embed_metrics_to_state(embed_metric_agg) if embed_metric_agg is not None else None,
    }
    with out_path.open("w", encoding = "utf-8") as f:
        json.dump(payload, f)

def main():
    args = parse_args()
    print(args)
    # print(args.csv_output_folder)
    # If user passed --gpus, parse into list and attach to args for worker mapping
    gpu_list = None
    if getattr(args, 'gpus', None):
        gpu_list = [x.strip() for x in args.gpus.split(",") if x.strip() != ""]
        args.gpu_list = gpu_list
    else:
        args.gpu_list = None

    world_size = _resolve_mp_world_size(args)

    if args.save_rollout_step is not None:
        gen_result_folder = Path(args.gen_result_folder)
        gen_result_folder.mkdir(parents = True, exist_ok = True)
        logger.info(f"Saving lead time outputs to {args.gen_result_folder}")

    if args.csv_output_folder is not None:
        Path(args.csv_output_folder).mkdir(parents = True, exist_ok = True)

    if world_size <= 1:
        set_seed(args.seed)
        logger.info("Running single-process evaluation.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(args, device)

        # === EMBEDDING EXTRACTION ===
        embed_hook_handle, embed_hook_buf = (None, None)
        if _embedding_features_enabled(args):
            embed_hook_handle, embed_hook_buf = attach_swin_output_hook(model)
        embed_metric_agg = (
            _new_embed_metric_agg(args.rollout_step, args.lead_time)
            if args.embedding_metrics_output_dir else None
        )

        dataset = create_dataset(args)
        # Pass the model's 0.25deg grid so a low-res boundary is regridded onto it in the loader.
        _model_lat, _model_lon = dataset.get_latitude_longitude()
        boundary_dataset = create_boundary_dataset(args, target_latitude = _model_lat, target_longitude = _model_lon)
        dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, pin_memory = True)
        criterion_list, err_agg_list = _build_metric_lists(args, total_count = len(dataset))

        evaluate(
            args,
            model,
            dataloader,
            criterion_list,
            err_agg_list,
            device,
            boundary_dataset = boundary_dataset,
            rank = 0,
            metadata_dataset = dataset,
            world_size = 1,
            embed_hook_buf = embed_hook_buf,      # === EMBEDDING EXTRACTION ===
            embed_metric_agg = embed_metric_agg,  # === EMBEDDING EXTRACTION ===
        )

        # === EMBEDDING EXTRACTION ===
        if embed_hook_handle is not None:
            embed_hook_handle.remove()
        if embed_metric_agg is not None:
            export_embedding_metrics(
                args,
                embed_metric_agg,
                args.embedding_metrics_output_dir,
                _embedding_metrics_label(args),
                _embedding_init_time_desc(args, len(dataset)),
            )

        for metric, err_agg in zip(args.eval_metric, err_agg_list):
            if args.csv_output_folder is not None:
                csv_folder = Path(args.csv_output_folder)
                csv_folder.mkdir(parents = True, exist_ok = True)
                csv_output_path = csv_folder / f"{metric}.csv"
                logger.info(f"Exporting results to CSV: {csv_output_path}")
                export_agg_to_csv(args, err_agg, out_path = csv_output_path)
        return

    logger.info("Running multiprocessing evaluation with world_size=%s", world_size)

    # Computed once in the parent (harmless here) and handed to each child so that no
    # child ever calls torch.cuda.* before it has narrowed CUDA_VISIBLE_DEVICES to its
    # own assigned GPU (see _mp_worker_entry).
    cuda_available = torch.cuda.is_available()

    mp_ctx = mp.get_context("spawn")

    processes = []
    for rank in range(world_size):
        p = mp_ctx.Process(target = _mp_worker_entry, args = (rank, world_size, args, cuda_available))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}.")

    full_dataset = create_dataset(args)
    _, merged_err_agg_list = _build_metric_lists(args, total_count = len(full_dataset))
    # === EMBEDDING EXTRACTION ===
    merged_embed_metric_agg = (
        _new_embed_metric_agg(args.rollout_step, args.lead_time)
        if args.embedding_metrics_output_dir else None
    )
    tmp_root = Path(args.csv_output_folder) if args.csv_output_folder is not None else Path(args.gen_result_folder)
    for rank in range(world_size):
        p = tmp_root / f".mp_rank_{rank}_metrics.json"
        with p.open("r", encoding = "utf-8") as f:
            payload = json.load(f)
        for metric_idx, metric in enumerate(args.eval_metric):
            _merge_state_into_err_agg(merged_err_agg_list[metric_idx], payload["metrics"][metric])
        if merged_embed_metric_agg is not None:
            _merge_embed_metrics_state(merged_embed_metric_agg, payload["embedding_metrics"])
        p.unlink(missing_ok = True)

    # === EMBEDDING EXTRACTION ===
    if merged_embed_metric_agg is not None:
        export_embedding_metrics(
            args,
            merged_embed_metric_agg,
            args.embedding_metrics_output_dir,
            _embedding_metrics_label(args),
            _embedding_init_time_desc(args, len(full_dataset)),
        )

    for metric, err_agg in zip(args.eval_metric, merged_err_agg_list):
        if args.csv_output_folder is not None:
            csv_folder = Path(args.csv_output_folder)
            csv_folder.mkdir(parents = True, exist_ok = True)
            csv_output_path = csv_folder / f"{metric}.csv"
            logger.info(f"Exporting results to CSV: {csv_output_path}")
            export_agg_to_csv(args, err_agg, out_path = csv_output_path)

if __name__ == "__main__":
    main()
