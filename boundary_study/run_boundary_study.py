#!/usr/bin/env python3
"""Boundary-only study of the 6-hourly error burst.

Three factors, each swept against a common baseline (hres, width 8, 0.25deg, direct, nearest,
backbone, no pooling):

    resolution   0.25 / 0.5 / 1.5   (apply mode always `direct`)
    width        4 / 8 / 12 / 16
    source       hres / ground_truth hourly / ground_truth 6-hourly

Two time scales, run separately:

    day      2020-03-01, every init hour 00Z..23Z       (24 rollouts per config)
    month    2020-03, every hour of every day           (744 rollouts per config)

Per (config, init, lead) it records
  * MAE / MSE per variable, over the full domain and over the COMMON interior (16 cells removed
    on every side, so the width sweep is not just measuring how much of the domain each config
    overwrote with boundary data),
  * embedding distance between the predicted and the ERA5 state window,
  * the mean-pooled bottleneck feature, for FID,
  * the zonal wavenumber spectrum (meridionally averaged, repo convention, computed on the same
    common interior) of the prediction and of ERA5.
"""

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from aurora import Batch, Metadata
from datasets.ERA5TWDatasetforAurora import ERA5TWDatasetforAurora
from plot_embedding_distance import embedding_metrics
from utils.embedding import encode_batch

from datasets.BoundaryConditionDataset import BoundaryConditionDataset_GroundTruth
from AuroraSmallTW_gen_eval_pipeline_with_embeddings import (
    create_boundary_dataset,
    create_model,
    model_forward_with_latent_boundary,
    _align_boundary_batch,
    _build_boundary_batch_from_hres_source,
    _get_boundary_source_on_device,
    _is_increasing,
    _prepare_batch_for_rollout,
    _stack_boundary_time_window,
)

SURF = ["t2m", "u10", "v10", "msl"]
UPPER = ["u", "v", "t", "q", "z"]
LEVELS = [1000, 925, 850, 700, 500, 300, 150, 50]
ERA5_ROOT = "/work/yunye0121/era5_tw"
HRES_ROOT = {0.25: "/tmp3/b12902101/hres_tw_forecast_0.25deg",
             0.5: "/tmp3/b12902101/hres_tw_forecast_0.25deg",   # derived by coarsen+replicate
             1.5: "/tmp3/b12902101/hres_tw_forecast_1.5deg"}
GT_ROOT = "/tmp3/yunye0121/era5_tw"

COMMON_RIM = 16          # widest boundary in the sweep; the fair interior for every config
EMB_MAX_LEAD = 48        # embeddings/spectra are only analysed over this window
SPECTRUM_VARS = [("z", 4), ("msl", None)]     # z at 500 hPa (index 4 of LEVELS), and msl

# The nine panels draw_error_plots.py puts in its 3x3 summary grid, so the error-growth figure
# here is the same plot the production draw_figure.sh produces.
KEY_VARS = [("2t", "surf", None), ("msl", "surf", None),
            ("10u", "surf", None), ("10v", "surf", None),
            ("z", "atmos", 500), ("t", "atmos", 850), ("q", "atmos", 700),
            ("u", "atmos", 850), ("v", "atmos", 850)]

# name -> (source, width, resolution, factor this config belongs to)
CONFIGS = {
    "base":   ("hres",         8,  0.25, "baseline"),
    "res0.5": ("hres",         8,  0.5,  "resolution"),
    "res1.5": ("hres",         8,  1.5,  "resolution"),
    "w4":     ("hres",         4,  0.25, "width"),
    "w12":    ("hres",        12,  0.25, "width"),
    "w16":    ("hres",        16,  0.25, "width"),
    "gt":     ("ground_truth", 8,  0.25, "source"),
    "gt6":    ("ground_truth", 8,  0.25, "source"),
    # No boundary replacement at all. Nothing clock-locked can reach the model through the
    # boundary, so whatever periodicity survives comes from the ERA5 truth the run is scored
    # against -- the clean control for "what does the reference field alone do?".
    "none":   ("none",         0,  0.25, "baseline"),
}


class GroundTruth6h(BoundaryConditionDataset_GroundTruth):
    """ERA5 analysis served on HRES's own sampling grid: 6-hourly, anchored to 00Z / 12Z.

    The point is to separate the two things that differ between the hres and ground-truth
    boundaries. hres is both a *forecast* (so it carries forecast error) and *6-hourly* (so
    `nearest` turns it into a staircase). This class keeps the perfect analysis fields but
    throws away the hourly resolution, on exactly the clock-anchored grid hres uses -- so any
    burst it shows is caused by the sampling alone.
    """

    def get_base_time(self, target_time: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(target_time).floor("12h")


def make_boundary_dataset(name, a, lat, lon, rollout_step):
    if name != "gt6":
        return create_boundary_dataset(a, target_latitude=lat, target_longitude=lon)
    # base is floor(12h), so an init can sit up to 11 h past it; cover that plus the rollout,
    # and go negative so the first step's history slot resolves without changing cycle.
    hi = 6 * (((12 + rollout_step) // 6) + 2)
    return GroundTruth6h(
        boundary_root_dir=a.boundary_root_dir,
        start_date_hour=a.start_date_hour, end_date_hour=a.end_date_hour,
        upper_variables=a.upper_variables, surface_variables=a.surface_variables,
        levels=a.levels, latitude=a.latitude, longitude=a.longitude,
        boundary_width=0, prediction_timedeltas=list(range(-12, hi + 1, 6)),
        forecast_cycle_hours=12, use_cache=False,
        time_interp_mode=a.boundary_time_interp_mode,
    )


class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def build_args(source, width, resolution, rollout_step, start, end):
    if source == "ground_truth":
        root = GT_ROOT
    elif source == "none":
        root = ""
    else:
        root = HRES_ROOT[resolution]
    return Cfg(
        data_root_dir=ERA5_ROOT,
        boundary_root_dir=root,
        checkpoint_path="/tmp2/yuanlim0919/lateral_smooth/model_weights/Aurora/model.safetensors",
        use_pretrained_weight=False, use_lora=False, bf16_mode=False,
        stabilise_level_agg=False,
        timestep_hours=1, lead_time=1, input_time_window=2,
        rollout_step=rollout_step,
        start_date_hour=str(start), end_date_hour=str(end),
        surface_variables=SURF, upper_variables=UPPER,
        static_variables=["lsm", "slt", "z"], levels=LEVELS,
        latitude=[39.75, 5], longitude=[100, 144.75],
        boundary_source=source,
        boundary_prediction_timedeltas=None,
        boundary_time_interp_mode="nearest",
        boundary_resolution=resolution,
        boundary_lowres_apply_mode="direct",
        boundary_use_cache=False,
        boundary_width=width,
        boundary_smooth_mode="no",
        boundary_smooth_width_adjustment=0,
        replace_boundary_position="backbone",
    )


# ---------------------------------------------------------------------------------------------

def era5_state(ds, dates, step, device):
    data = ds.load_single_target(base_datetime_strs=list(dates), rollout_step_index=step)
    return {sec: {k: v[:, 0].to(device) for k, v in data[sec].items()}
            for sec in ("surf_vars", "atmos_vars")}


def make_batch(prev, curr, static_vars, lat, lon, times, levels, rollout_step=None):
    w = _stack_boundary_time_window([prev, curr])
    md = dict(lat=lat, lon=lon, time=tuple(times), atmos_levels=levels)
    if rollout_step is not None:
        md["rollout_step"] = rollout_step
    return Batch(surf_vars=w["surf_vars"], atmos_vars=w["atmos_vars"],
                 static_vars=static_vars, metadata=Metadata(**md))


def zonal_spectrum(x, rim):
    """Meridionally averaged zonal amplitude spectrum -- plot_wavenumber.py's convention.

    x: (B, H, W). Crop the rim, remove the spatial mean, 2D FFT, average |FFT| along latitude,
    keep the positive zonal wavenumbers. Returns (B, W'//2).
    """
    if rim > 0:
        x = x[..., rim:-rim, rim:-rim]
    x = x.float()
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    mag = torch.fft.fft2(x).abs()
    zonal = mag.mean(dim=-2)                       # meridional average
    return zonal[..., 1:zonal.shape[-1] // 2 + 1]  # positive wavenumbers only


def field_errors(pred, gt, rim):
    out = {}
    for sec in ("surf_vars", "atmos_vars"):
        for k, p in pred[sec].items():
            d = p.float() - gt[sec][k].float()
            out[f"mae_{k}"] = d.abs().mean().item()
            out[f"mse_{k}"] = d.pow(2).mean().item()
            di = d[..., rim:-rim, rim:-rim]
            out[f"mae_int_{k}"] = di.abs().mean().item()
            out[f"mse_int_{k}"] = di.pow(2).mean().item()

    # Per-level MAE for the nine error-growth panels; `mae_z` above is a level average and
    # cannot stand in for z_500.
    for name, sec, lev in KEY_VARS:
        if sec == "surf":
            d = pred["surf_vars"][name].float() - gt["surf_vars"][name].float()
            col = name
        else:
            i = LEVELS.index(lev)
            d = pred["atmos_vars"][name][:, i].float() - gt["atmos_vars"][name][:, i].float()
            col = f"{name}_{lev}"
        out[f"mae_kv_{col}"] = d.abs().mean().item()
        out[f"mae_kvint_{col}"] = d[..., rim:-rim, rim:-rim].abs().mean().item()
    return out


def run_config(name, init_times, rollout_step, batch_size, device, out_dir, tag):
    source, width, resolution, factor = CONFIGS[name]
    a = build_args(source, width, resolution, rollout_step,
                   init_times[0], init_times[-1] + pd.Timedelta(hours=rollout_step))

    ds = ERA5TWDatasetforAurora(
        data_root_dir=ERA5_ROOT,
        start_date_hour=init_times[0] - pd.Timedelta(hours=1),
        end_date_hour=init_times[-1] + pd.Timedelta(hours=rollout_step + 2),
        upper_variables=UPPER, surface_variables=SURF,
        static_variables=a.static_variables, levels=LEVELS,
        latitude=a.latitude, longitude=a.longitude,
        lead_time=1, input_time_window=1, rollout_step=1, sample_stride_hours=1,
    )
    lat, lon = ds.get_latitude_longitude()
    levels = ds.get_levels()
    static_vars = {k: v.to(device) for k, v in ds.get_static_vars_ds()["static_vars"].items()}

    use_boundary = width > 0
    bd = None
    flip_lat = flip_lon = False
    if use_boundary:
        bd = make_boundary_dataset(name, a, lat, lon, rollout_step)
        b_lat, b_lon = bd.get_latitude_longitude()
        flip_lat = _is_increasing(b_lat) != _is_increasing(lat)
        flip_lon = _is_increasing(b_lon) != _is_increasing(lon)

    model = create_model(a, device)

    rows, emb_p_all, emb_g_all, idx_all = [], [], [], []
    spec_p, spec_g = [], []
    n_batches = (len(init_times) + batch_size - 1) // batch_size

    for bi in tqdm(range(n_batches), desc=f"{tag}/{name}", disable=False):
        inits = init_times[bi * batch_size:(bi + 1) * batch_size]
        dates = [t.strftime("%Y-%m-%d %H:%M:%S") for t in inits]
        base_times = tuple(bd.get_base_time(t) for t in inits) if use_boundary else ()

        # Preload exactly the cycles this batch will touch (ground truth: just the base).
        cache = {}
        needed = set()
        for i, bt in enumerate(base_times if use_boundary else ()):
            for k in range(rollout_step):
                for tw in (1, 0):
                    tt = inits[i] + pd.Timedelta(hours=k - tw)
                    needed.add(bd.effective_base_time(bt, tt))
        for bt in needed:
            _get_boundary_source_on_device(bd, bt, cache, device)

        s_prev = era5_state(ds, dates, -1, device)
        s_curr = era5_state(ds, dates, 0, device)
        rollout_batch = _prepare_batch_for_rollout(
            model, make_batch(s_prev, s_curr, static_vars, lat, lon, list(inits), levels))

        gt_curr, pred_prev = s_curr, s_curr

        with torch.no_grad():
            for k in range(rollout_step):
                t = k + 1
                valid_times = [ti + pd.Timedelta(hours=t) for ti in inits]

                if use_boundary:
                    singles = []
                    for tw in (1, 0):
                        tt = tuple(ti + pd.Timedelta(hours=k - tw) for ti in inits)
                        b = _build_boundary_batch_from_hres_source(bd, cache, base_times, tt)
                        singles.append(_align_boundary_batch(b, flip_lat, flip_lon))
                    b_curr = _stack_boundary_time_window(singles)
                    boundary_batch = Batch(
                        surf_vars=b_curr["surf_vars"], atmos_vars=b_curr["atmos_vars"],
                        static_vars=static_vars,
                        metadata=Metadata(lat=lat, lon=lon, time=rollout_batch.metadata.time,
                                          atmos_levels=levels,
                                          rollout_step=rollout_batch.metadata.rollout_step))
                    pred = model_forward_with_latent_boundary(model, rollout_batch,
                                                              boundary_batch, a)
                else:
                    pred = model(rollout_batch)

                gt_next = era5_state(ds, dates, t, device)
                pred_state = {sec: {kk: v[:, 0] for kk, v in getattr(pred, sec).items()}
                              for sec in ("surf_vars", "atmos_vars")}

                emb_p = encode_batch(model, make_batch(pred_prev, pred_state, static_vars,
                                                       lat, lon, valid_times, levels))
                emb_g = encode_batch(model, make_batch(gt_curr, gt_next, static_vars,
                                                       lat, lon, valid_times, levels))
                pooled_p = emb_p.float().mean(dim=1).cpu().numpy()
                pooled_g = emb_g.float().mean(dim=1).cpu().numpy()

                sp, sg = [], []
                for var, lev in SPECTRUM_VARS:
                    if lev is None:
                        fp, fg = pred_state["surf_vars"][var if var != "msl" else "msl"], \
                                 gt_next["surf_vars"]["msl"]
                    else:
                        fp, fg = pred_state["atmos_vars"][var][:, lev], gt_next["atmos_vars"][var][:, lev]
                    sp.append(zonal_spectrum(fp, COMMON_RIM).cpu().numpy())
                    sg.append(zonal_spectrum(fg, COMMON_RIM).cpu().numpy())
                sp = np.stack(sp, axis=1)      # (B, nvar, K)
                sg = np.stack(sg, axis=1)

                # FID and the spectra are only ever analysed over the fixed 1..EMB_MAX_LEAD
                # window, so storing them for every one of 168 leads would just burn disk.
                keep_emb = t <= EMB_MAX_LEAD

                for i in range(len(inits)):
                    row = {"config": name, "factor": factor, "source": source,
                           "width": width, "resolution": resolution,
                           "init_time": str(inits[i]), "init_hour": inits[i].hour,
                           "lead": t, "valid_time": str(valid_times[i]),
                           "clock_hour": valid_times[i].hour}
                    row.update(field_errors(
                        {s: {kk: v[i:i + 1] for kk, v in pred_state[s].items()}
                         for s in ("surf_vars", "atmos_vars")},
                        {s: {kk: v[i:i + 1] for kk, v in gt_next[s].items()}
                         for s in ("surf_vars", "atmos_vars")}, COMMON_RIM))
                    row.update(embedding_metrics(emb_p[i:i + 1], emb_g[i:i + 1]))
                    rows.append(row)
                    if keep_emb:
                        emb_p_all.append(pooled_p[i]); emb_g_all.append(pooled_g[i])
                        idx_all.append((str(inits[i]), t, valid_times[i].hour))
                        spec_p.append(sp[i]); spec_g.append(sg[i])

                pred_prev, gt_curr = pred_state, gt_next
                rollout_batch = dataclasses.replace(
                    pred,
                    surf_vars={kk: torch.cat([rollout_batch.surf_vars[kk][:, 1:], v], dim=1)
                               for kk, v in pred.surf_vars.items()},
                    atmos_vars={kk: torch.cat([rollout_batch.atmos_vars[kk][:, 1:], v], dim=1)
                                for kk, v in pred.atmos_vars.items()},
                    metadata=Metadata(lat=lat, lon=lon, time=tuple(valid_times),
                                      atmos_levels=levels, rollout_step=t))

        cache.clear()
        torch.cuda.empty_cache()

    out = Path(out_dir)
    pd.DataFrame(rows).to_csv(out / f"metrics_{tag}_{name}.csv", index=False)
    np.savez_compressed(
        out / f"emb_{tag}_{name}.npz",
        pred=np.asarray(emb_p_all, dtype=np.float32), gt=np.asarray(emb_g_all, dtype=np.float32),
        init_time=np.asarray([r[0] for r in idx_all]),
        lead=np.asarray([r[1] for r in idx_all], dtype=np.int32),
        clock_hour=np.asarray([r[2] for r in idx_all], dtype=np.int32))
    np.savez_compressed(
        out / f"spec_{tag}_{name}.npz",
        pred=np.asarray(spec_p, dtype=np.float32), gt=np.asarray(spec_g, dtype=np.float32),
        vars=np.asarray([v for v, _ in SPECTRUM_VARS]),
        lead=np.asarray([r[1] for r in idx_all], dtype=np.int32),
        clock_hour=np.asarray([r[2] for r in idx_all], dtype=np.int32),
        init_time=np.asarray([r[0] for r in idx_all]))
    print(f"[{tag}/{name}] wrote {len(rows)} rows")


def init_times_for(scale):
    if scale == "day":
        return list(pd.date_range("2020-03-01 00:00:00", periods=24, freq="1h"))
    # month: every hour of every day in March -- 31 x 24 = 744 inits. Averaging then groups by
    # clock hour, and every clock-hour bin holds exactly the same set of lead times.
    return list(pd.date_range("2020-03-01 00:00:00", "2020-03-31 23:00:00", freq="1h"))


def worker(rank, world, args, cuda):
    if cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_list[rank])
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    inits = init_times_for(args.scale)[rank::world]
    for name in args.config_list:
        run_config(name, inits, args.rollout_step, args.batch_size, device,
                   args.out, f"{args.scale}_r{rank}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["day", "month"], required=True)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--rollout_step", type=int, default=168)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--out", default="/tmp3/b12902101/Mazu/boundary_study/data")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    args.gpu_list = [x.strip() for x in args.gpus.split(",") if x.strip()]
    args.config_list = [c.strip() for c in args.configs.split(",") if c.strip()]
    world = len(args.gpu_list)
    cuda = torch.cuda.is_available()

    if world <= 1:
        worker(0, 1, args, cuda)
    else:
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        ps = []
        for r in range(world):
            p = ctx.Process(target=worker, args=(r, world, args, cuda))
            p.start(); ps.append(p)
        for p in ps:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker {p.exitcode}")

    # merge shards
    for name in args.config_list:
        for kind in ("metrics", "emb", "spec"):
            parts = sorted(Path(args.out).glob(f"{kind}_{args.scale}_r*_{name}.*"))
            if not parts:
                continue
            if kind == "metrics":
                df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
                df.sort_values(["init_time", "lead"]).to_csv(
                    Path(args.out) / f"metrics_{args.scale}_{name}.csv", index=False)
            else:
                zs = [np.load(p, allow_pickle=False) for p in parts]
                keys = [k for k in zs[0].files if k != "vars"]
                merged = {k: np.concatenate([z[k] for z in zs], axis=0) for k in keys}
                if "vars" in zs[0].files:
                    merged["vars"] = zs[0]["vars"]
                np.savez_compressed(Path(args.out) / f"{kind}_{args.scale}_{name}.npz", **merged)
            for p in parts:
                p.unlink()
        print(f"merged {args.scale}/{name}")


if __name__ == "__main__":
    main()
