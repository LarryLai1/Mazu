#!/usr/bin/env python3
"""
Extract Swin3D bottleneck embeddings of the ECMWF HRES 0.25deg FORECAST, in the same format
AuroraSmallTW_gen_eval_pipeline_with_embeddings.py writes, so HRES can be dropped into
plot_embedding_tsne_hooked.py as one more series alongside the model rollouts.

HRES is not a rollout of our model, so there is nothing to hook during inference -- its states
already exist on disk as per-cycle forecast files. This script therefore re-embeds them after
the fact: it reads the HRES state valid at each (init time, lead time) through Mazu's existing
BoundaryConditionDataset_HRES (which regrids it onto the ERA5 0.25deg model grid), assembles
the same 2-frame [state(t-1), state(t)] window the rest of the pipeline uses, and pushes it
through the SAME frozen model with the SAME hook and the SAME per-sample flatten -- all
imported from the pipeline script, so the vectors are comparable by construction rather than
by coincidence.

The (init time, lead time) pairs are read off the FILENAMES in --match_embeddings_dir, so the
output lines up exactly with an existing run's embeddings.

Caveat when reading the resulting t-SNE: HRES is a different forecast system, not our model on
a different boundary. Its embedding sits where it does partly because its fields differ
systematically from ERA5/our model, not only because of forecast skill.

Example:
  cd Mazu
  python extract_hres_embeddings.py \
    --match_embeddings_dir /tmp3/b12902101/mazu_embedding_output/embeddings/hres_boundary0_backbone_res0.25_direct \
    --output_dir /tmp3/b12902101/mazu_embedding_output/embeddings/hres_forecast_res0.25 \
    --init_days 10
"""

import argparse
import glob
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora, NETCDF_IO_LOCK
from datasets.BoundaryConditionDataset import BoundaryConditionDataset_HRES

# The hook, the flatten and the .npz layout all come from the pipeline that produced the
# embeddings we are lining up with -- same tap point, same pooling, same file format.
from AuroraSmallTW_gen_eval_pipeline_with_embeddings import (
    _EMBED_DTYPES,
    _embed_window,
    _save_embedding_npz,
    attach_swin_output_hook,
    create_model,
)
from plot_embedding_distance import (
    LATITUDE,
    LEVELS,
    LONGITUDE,
    STATIC_VARIABLES,
    SURFACE_VARIABLES,
    UPPER_VARIABLES,
    SnapshotLoader,
    build_batch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# <INIT>+<LEAD>hr.npz, as written by the pipeline's _save_embedding_npz().
_NAME_RE = re.compile(r"^(\d{8}_\d{6})\+(\d+)hr\.npz$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--match_embeddings_dir", required=True,
                   help="Existing embeddings directory whose <INIT>+<LEAD>hr.npz filenames define "
                        "which (init time, lead time) pairs to reproduce for HRES.")
    p.add_argument("--output_dir", required=True, help="Where to write the HRES .npz files.")
    p.add_argument("--init_days", type=float, default=None,
                   help="Only process initialisation times within this many days of the earliest "
                        "one found (matches plot_embedding_tsne_hooked.py's --init_days).")
    p.add_argument("--lead_times", nargs="*", type=int, default=None,
                   help="Only process these lead times (hours). Default: all found.")
    p.add_argument("--history_step", type=int, default=1,
                   help="Hours between the two history frames fed to the encoder. Must match the "
                        "rollout that produced --match_embeddings_dir (its --lead_time).")
    p.add_argument("--skip_gt", action="store_true",
                   help="Do not re-embed ERA5 ground truth into these files; the gt_embedding key "
                        "is then omitted entirely (never a zero placeholder that could be mistaken "
                        "for real data). Safe when another embeddings directory in the same plot "
                        "supplies ground truth -- the plotter de-duplicates it anyway -- and "
                        "roughly halves the runtime.")

    p.add_argument("--data_root_dir", default="/work/yunye0121/era5_tw")
    p.add_argument("--boundary_root_dir", default="/tmp3/b12902101/hres_tw_forecast_0.25deg",
                   help="HRES forecast root. Point at the 1.5deg tree for --boundary_resolution 1.5.")
    p.add_argument("--boundary_resolution", type=float, default=0.25, choices=[0.25, 0.5, 1.5])
    p.add_argument("--boundary_lowres_apply_mode", default="direct", choices=["direct", "interp"])
    p.add_argument("--forecast_cycle_hours", type=int, default=12)
    p.add_argument("--hres_time_interp_mode", default="interpolation",
                   choices=["interpolation", "nearest", "exact"],
                   help="HRES stores 6-hourly steps but we query hourly leads; 'interpolation' is "
                        "required, since 'nearest' collapses the two history frames onto one step.")

    p.add_argument("--checkpoint_path",
                   default="/tmp2/yuanlim0919/lateral_smooth/model_weights/Aurora/model.safetensors")
    p.add_argument("--use_pretrained_weight", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--stabilise_level_agg", action="store_true")
    p.add_argument("--timestep_hours", type=int, default=1)
    p.add_argument("--bf16_mode", action="store_true",
                   help="Run the model in bf16. Off by default, matching the pipeline runs these "
                        "embeddings are compared against (quantisation would shift the vectors).")
    p.add_argument("--embedding_dtype", default="float32", choices=list(_EMBED_DTYPES.keys()))
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract files that already exist in --output_dir (default: skip them, "
                        "so an interrupted run can simply be re-launched).")
    return p.parse_args()


def scan_pairs(args):
    """(init timestamp, lead hours) pairs from --match_embeddings_dir's filenames."""
    pairs, unnamed = [], 0
    for path in sorted(glob.glob(os.path.join(args.match_embeddings_dir, "*.npz"))):
        m = _NAME_RE.match(os.path.basename(path))
        if not m:
            unnamed += 1
            continue
        pairs.append((pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")), int(m.group(2))))
    if unnamed:
        logger.warning("Ignored %d file(s) not named <INIT>+<LEAD>hr.npz", unnamed)
    if not pairs:
        raise RuntimeError(f"No usable .npz filenames in {args.match_embeddings_dir}")

    if args.lead_times:
        pairs = [pr for pr in pairs if pr[1] in args.lead_times]
    if args.init_days is not None:
        cutoff = min(i for i, _ in pairs) + pd.Timedelta(days=args.init_days)
        pairs = [pr for pr in pairs if pr[0] < cutoff]
        logger.info("Initialisation window: < %s (first %g day(s))", cutoff, args.init_days)

    pairs.sort()
    inits = sorted({i for i, _ in pairs})
    leads = sorted({l for _, l in pairs})
    logger.info("%d (init, lead) pair(s): %d initialisation times %s .. %s, lead times %s",
                len(pairs), len(inits), inits[0], inits[-1], leads)
    return pairs, inits, leads


def _prune_caches(loader, keep_sources=2):
    """Bound SnapshotLoader's memory.

    Its snapshot cache holds a full ERA5/HRES state per timestamp (~4 MB) and its HRES source
    cache holds an entire forecast cycle (all prediction_timedeltas, far larger). Left alone
    over hundreds of initialisation times both grow without limit. Snapshots are only reused
    within one init, so drop them each time; HRES sources are shared by every init inside a
    12 h cycle, so keep the most recent couple.
    """
    loader._cache.clear()
    while len(loader._hres_source_cache) > keep_sources:
        loader._hres_source_cache.pop(next(iter(loader._hres_source_cache)))


def main():
    args = parse_args()
    pairs, inits, leads = scan_pairs(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    span_end = max(i + pd.Timedelta(hours=l) for i, l in pairs)
    ds_gt = ERA5TWDatasetforAurora(
        data_root_dir=args.data_root_dir,
        start_date_hour=min(inits),
        end_date_hour=span_end,
        upper_variables=UPPER_VARIABLES,
        surface_variables=SURFACE_VARIABLES,
        static_variables=STATIC_VARIABLES,
        levels=LEVELS,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        lead_time=1,
        input_time_window=1,
        rollout_step=1,
        sample_stride_hours=1,
    )
    lat, lon = ds_gt.get_latitude_longitude()
    static_vars = ds_gt.get_static_vars_ds()["static_vars"]
    logger.info("Model grid: %d lat x %d lon", len(lat), len(lon))

    base_times = [i.floor(f"{args.forecast_cycle_hours}h") for i in inits]
    hres_ds = BoundaryConditionDataset_HRES(
        boundary_root_dir=args.boundary_root_dir,
        start_date_hour=min(base_times),
        end_date_hour=max(base_times),
        upper_variables=UPPER_VARIABLES,
        surface_variables=SURFACE_VARIABLES,
        levels=LEVELS,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        boundary_width=0,
        # HRES reads its own prediction_timedelta axis from file; this only sizes the unused
        # time_axis, so the native 0..240h/6h range is a safe placeholder.
        prediction_timedeltas=list(range(0, 241, 6)),
        forecast_cycle_hours=args.forecast_cycle_hours,
        use_cache=False,
        time_interp_mode=args.hres_time_interp_mode,
        # Bring the forecast onto the ERA5 model grid so its embedding is comparable.
        target_latitude=lat,
        target_longitude=lon,
        boundary_resolution=args.boundary_resolution,
        lowres_apply_mode=args.boundary_lowres_apply_mode,
    )
    hres_key = args.boundary_root_dir
    loader = SnapshotLoader(ds_gt, hres_datasets={hres_key: hres_ds},
                            forecast_cycle_hours=args.forecast_cycle_hours)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = create_model(args, device)
    hook_handle, hook_buf = attach_swin_output_hook(model)
    dtype = _EMBED_DTYPES[args.embedding_dtype]

    by_init = {}
    for init, lead in pairs:
        by_init.setdefault(init, []).append(lead)

    written = skipped = failed = 0
    for n, init in enumerate(sorted(by_init), start=1):
        for lead in sorted(by_init[init]):
            out_path = out_dir / f"{init.strftime('%Y%m%d_%H%M%S')}+{lead}hr.npz"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            prev_lead = lead - args.history_step
            if prev_lead < 0:
                logger.warning("init %s lead %dh: history_step %dh reaches before the init time; "
                               "skipping.", init, lead, args.history_step)
                failed += 1
                continue
            valid_time = init + pd.Timedelta(hours=lead)

            try:
                # HDF5 is not thread-safe and these datasets do unguarded xarray opens; this
                # script is single-threaded, but hold the shared lock anyway so it stays safe
                # if it is ever run alongside the pipeline's prefetch threads.
                with NETCDF_IO_LOCK:
                    hres_prev = loader.pred(hres_key, init, prev_lead)
                    hres_curr = loader.pred(hres_key, init, lead)
                    gt_prev = None if args.skip_gt else loader.gt(init + pd.Timedelta(hours=prev_lead))
                    gt_curr = None if args.skip_gt else loader.gt(valid_time)
            except (FileNotFoundError, KeyError) as e:
                logger.warning("init %s lead %dh: missing input (%s); skipping.", init, lead, e)
                failed += 1
                continue

            hres_vec = _embed_window(
                model, build_batch(hres_prev, hres_curr, static_vars, lat, lon, valid_time),
                hook_buf, dtype,
            )[0]

            if args.skip_gt:
                # Omit gt_embedding rather than writing a placeholder: a zero vector would be
                # indistinguishable from real data downstream, and the plotter is written to
                # treat the missing key as "this directory carries no ground truth".
                np.savez(out_path, pred_embedding=hres_vec, init_time=str(init),
                         lead_time=lead, rollout_step=lead)
            else:
                gt_vec = _embed_window(
                    model, build_batch(gt_prev, gt_curr, static_vars, lat, lon, valid_time),
                    hook_buf, dtype,
                )[0]
                _save_embedding_npz(out_dir, init, lead, lead, hres_vec, gt_vec)
            written += 1

        _prune_caches(loader)
        if n % 10 == 0 or n == len(by_init):
            logger.info("[%d/%d init times] written=%d skipped=%d failed=%d",
                        n, len(by_init), written, skipped, failed)

    hook_handle.remove()
    logger.info("Done: %d written, %d already present, %d skipped on missing input -> %s",
                written, skipped, failed, out_dir)
    if args.skip_gt:
        logger.info("--skip_gt was set: these files carry no gt_embedding key. Plot this directory "
                    "together with one that does if you want the ground-truth series.")


if __name__ == "__main__":
    main()
