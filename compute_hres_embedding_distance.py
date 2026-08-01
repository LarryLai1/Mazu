#!/usr/bin/env python3
"""
Embedding distance between the ECMWF HRES forecast trajectory and ERA5 ground truth, vs. lead
time, averaged over start times -- the baseline line that the rollout-based metric mode cannot
produce (HRES is not a model rollout, so there is no inference pass to hook).

Output is the same `embedding_distance.csv` + PNG set that
AuroraSmallTW_gen_eval_pipeline_with_embeddings.py writes in its --embedding_metrics_output_dir
mode, so combine_embedding_distance_csv.py can overlay this on the rollout curves.

Both sides are embedded with utils.embedding.encode_batch, which runs the pre-encoder, the
Perceiver encoder and the backbone's encoder layers and stops at the bottleneck -- a faithful
mirror of Aurora.forward up to that point (same sequence, same lead_time=model.timestep), but
without the decoder half of the forward pass. No rollout is involved: for lead `t` the two
history slots are simply the states at `t - 1` and `t`, on both sides.

Everything is batched over init times and runs on the GPU: per (batch, lead) there are exactly
two encoder passes (HRES window + ERA5 window), and each state is read once and reused as the
next lead's older history slot.

Example:
  cd Mazu
  python compute_hres_embedding_distance.py \
    --data_root_dir /work/yunye0121/era5_tw \
    --boundary_root_dir /tmp3/b12902101/hres_tw_forecast_0.25deg \
    --start_date_hour '2020-03-01 00:00:00' --end_date_hour '2020-03-01 23:00:00' \
    --rollout_step 168 --batch_size 8 \
    --output_dir /tmp3/b12902101/mazu_embedding_output/embedding_distance/hres_forecast
"""

import argparse
import logging

import pandas as pd
import torch
from tqdm.auto import tqdm

from aurora import Batch, Metadata
from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora
from plot_embedding_distance import embedding_metrics
from utils.embedding import encode_batch

# Reuse the rollout pipeline's dataset construction, boundary plumbing, metric accumulator and
# CSV/plot export verbatim, so this baseline is configured and reported exactly like the rollout
# runs it is plotted next to.
from AuroraSmallTW_gen_eval_pipeline_with_embeddings import (
    create_boundary_dataset,
    create_model,
    export_embedding_metrics,
    _align_boundary_batch,
    _build_boundary_batch_from_hres_source,
    _get_boundary_source_on_device,
    _is_increasing,
    _new_embed_metric_agg,
    _stack_boundary_time_window,
    _update_embed_metric_agg,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--data_root_dir', type=str, required=True)
    p.add_argument('--boundary_root_dir', type=str, required=True,
                   help='HRES forecast trajectory root, e.g. hres_tw_forecast_0.25deg.')
    p.add_argument('--start_date_hour', type=str, required=True)
    p.add_argument('--end_date_hour', type=str, required=True,
                   help='Last init time; results are averaged over the whole range.')
    p.add_argument('--sample_stride_hours', type=int, default=1,
                   help='Spacing between init times.')
    p.add_argument('--rollout_step', type=int, default=168,
                   help='Number of lead times; leads 1..N * --lead_time hours are evaluated.')
    p.add_argument('--lead_time', type=int, default=1)
    p.add_argument('--input_time_window', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=8,
                   help='Init times encoded together.')

    p.add_argument('--surface_variables', nargs='+', default=['t2m', 'u10', 'v10', 'msl'])
    p.add_argument('--upper_variables', nargs='+', default=['u', 'v', 't', 'q', 'z'])
    p.add_argument('--static_variables', nargs='+', default=['lsm', 'slt', 'z'])
    p.add_argument('--levels', nargs='+', type=int, default=[1000, 925, 850, 700, 500, 300, 150, 50])
    p.add_argument('--latitude', nargs=2, type=float, default=[39.75, 5])
    p.add_argument('--longitude', nargs=2, type=float, default=[100, 144.75])

    # Names match the rollout pipeline's, so create_boundary_dataset() can be reused as-is.
    p.add_argument('--boundary_source', type=str, default='hres', choices=['hres'])
    p.add_argument('--boundary_prediction_timedeltas', nargs='*', type=int, default=None)
    p.add_argument('--boundary_time_interp_mode', type=str, default='nearest',
                   choices=['interpolation', 'nearest', 'exact'],
                   help="How HRES's 6-hourly forecast steps are sampled at the hourly leads asked "
                        "for here. Match the rollout runs this is plotted against.")
    p.add_argument('--boundary_resolution', type=float, default=0.25, choices=[0.25, 0.5, 1.5])
    p.add_argument('--boundary_lowres_apply_mode', type=str, default='direct',
                   choices=['direct', 'interp'])
    p.add_argument('--boundary_use_cache', action='store_true')
    p.add_argument('--forecast_cycle_hours', type=int, default=12,
                   help='Each init time is floored to this to pick the HRES forecast base time.')

    p.add_argument('--checkpoint_path', type=str,
                   default='/tmp2/yuanlim0919/lateral_smooth/model_weights/Aurora/model.safetensors')
    p.add_argument('--use_pretrained_weight', action='store_true')
    p.add_argument('--use_lora', action='store_true')
    p.add_argument('--stabilise_level_agg', action='store_true')
    p.add_argument('--bf16_mode', action='store_true')
    p.add_argument('--timestep_hours', type=int, default=1)

    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--label', type=str, default='HRES forecast')

    return p.parse_args()


def build_window(prev_state, curr_state, static_vars, lat, lon, valid_times, levels):
    """Two single-step states -> a Batch with a length-2 history axis.

    `valid_times` are pd.Timestamps, matching what the rollout pipeline puts in its own batches:
    the encoder turns metadata time into an absolute-time feature via `t.timestamp()`, and a
    naive datetime.datetime would be read in the machine's local timezone instead of UTC.
    """
    window = _stack_boundary_time_window([prev_state, curr_state])
    return Batch(
        surf_vars=window["surf_vars"],
        atmos_vars=window["atmos_vars"],
        static_vars=static_vars,
        metadata=Metadata(lat=lat, lon=lon, time=tuple(valid_times), atmos_levels=levels),
    )


def era5_state(ds, dates, step, device):
    """ERA5 target at `init + step * lead_time` for every init time in `dates`, on device."""
    data = ds.load_single_target(base_datetime_strs=list(dates), rollout_step_index=step)
    return {
        section: {k: v[:, 0].to(device) for k, v in data[section].items()}
        for section in ("surf_vars", "atmos_vars")
    }


def hres_state(boundary_dataset, source_cache, base_times, target_times, flip_lat, flip_lon):
    """HRES forecast state valid at `target_times`, on the model grid and on device."""
    state = _build_boundary_batch_from_hres_source(
        boundary_dataset, source_cache, base_times, target_times,
    )
    if state is None:
        return None
    return _align_boundary_batch(state, flip_lat, flip_lon)


def main():
    args = parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_times = pd.date_range(
        pd.Timestamp(args.start_date_hour),
        pd.Timestamp(args.end_date_hour),
        freq=f"{args.sample_stride_hours}h",
    )
    leads = list(range(1, args.rollout_step + 1))
    logger.info("%d init time(s) x %d lead time(s)", len(init_times), len(leads))

    # end_date_hour must cover the last valid time so the ERA5 files are in range.
    ds = ERA5TWDatasetforAurora(
        data_root_dir=args.data_root_dir,
        start_date_hour=init_times[0],
        end_date_hour=init_times[-1] + pd.Timedelta(hours=args.rollout_step * args.lead_time),
        upper_variables=args.upper_variables,
        surface_variables=args.surface_variables,
        static_variables=args.static_variables,
        levels=args.levels,
        latitude=args.latitude,
        longitude=args.longitude,
        lead_time=args.lead_time,
        input_time_window=1,
        rollout_step=1,
        sample_stride_hours=args.sample_stride_hours,
    )
    lat, lon = ds.get_latitude_longitude()
    levels = ds.get_levels()
    static_vars = ds.get_static_vars_ds()["static_vars"]
    if isinstance(static_vars, torch.Tensor):
        static_vars = static_vars.to(device)
    else:
        static_vars = {k: v.to(device) for k, v in static_vars.items()}

    boundary_dataset = create_boundary_dataset(args, target_latitude=lat, target_longitude=lon)
    if boundary_dataset is None:
        raise SystemExit("--boundary_root_dir did not yield an HRES dataset.")

    # The HRES grid may run the other way round from the model grid; align once.
    b_lat, b_lon = boundary_dataset.get_latitude_longitude()
    flip_lat = _is_increasing(b_lat) != _is_increasing(lat)
    flip_lon = _is_increasing(b_lon) != _is_increasing(lon)

    model = create_model(args, device)
    agg = _new_embed_metric_agg(args.rollout_step, args.lead_time)

    n_batches = (len(init_times) + args.batch_size - 1) // args.batch_size
    for b in tqdm(range(n_batches), desc="HRES embedding distance"):
        batch_inits = init_times[b * args.batch_size : (b + 1) * args.batch_size]
        dates = [t.strftime("%Y-%m-%d %H:%M:%S") for t in batch_inits]
        base_times = tuple(t.floor(f"{args.forecast_cycle_hours}h") for t in batch_inits)

        # One GPU-resident source per HRES cycle in this batch (plus the preceding cycle, which
        # _build_boundary_batch_from_hres_source falls back to for target times before the base
        # time). Cleared per batch so a month-long run does not accumulate them.
        source_cache = {}
        for base_time in set(base_times):
            _get_boundary_source_on_device(boundary_dataset, base_time, source_cache, device)
            _get_boundary_source_on_device(
                boundary_dataset,
                base_time - pd.Timedelta(hours=args.forecast_cycle_hours),
                source_cache,
                device,
            )

        # Lead 0 is the older history slot of lead 1. The HRES side takes its own state at lead 0
        # (not ERA5): the whole line is then the HRES trajectory, matching how the old
        # plot_embedding_distance.py served HRES entries.
        prev_times = tuple(batch_inits)
        prev_gt = era5_state(ds, dates, 0, device)
        prev_hres = hres_state(boundary_dataset, source_cache, base_times, prev_times,
                               flip_lat, flip_lon)

        for t in leads:
            curr_times = tuple(ti + pd.Timedelta(hours=t * args.lead_time) for ti in batch_inits)
            curr_gt = era5_state(ds, dates, t, device)
            curr_hres = hres_state(boundary_dataset, source_cache, base_times, curr_times,
                                   flip_lat, flip_lon)
            if curr_hres is None or prev_hres is None:
                logger.warning("Skipping lead %dh: no HRES state (interp mode %s).",
                               t * args.lead_time, args.boundary_time_interp_mode)
                prev_gt, prev_hres, prev_times = curr_gt, curr_hres, curr_times
                continue

            emb_gt = encode_batch(
                model, build_window(prev_gt, curr_gt, static_vars, lat, lon, curr_times, levels))
            emb_hres = encode_batch(
                model, build_window(prev_hres, curr_hres, static_vars, lat, lon, curr_times, levels))
            assert emb_hres.shape == emb_gt.shape, f"{emb_hres.shape} != {emb_gt.shape}"

            for i in range(len(dates)):
                _update_embed_metric_agg(
                    agg,
                    t * args.lead_time,
                    embedding_metrics(emb_hres[i : i + 1], emb_gt[i : i + 1]),
                )

            prev_gt, prev_hres, prev_times = curr_gt, curr_hres, curr_times

        source_cache.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    export_embedding_metrics(
        args,
        agg,
        args.output_dir,
        args.label,
        f"{args.start_date_hour} .. {args.end_date_hour} ({len(init_times)} init times)",
    )


if __name__ == "__main__":
    main()
