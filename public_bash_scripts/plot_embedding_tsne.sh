#!/bin/bash
set -euo pipefail

# t-SNE of Aurora backbone embeddings for the two LAM_output_preds predictions plus ERA5 ground
# truth, over the 240h forecast starting 2020-03-01 00:00. Each hourly snapshot becomes a backbone
# embedding, flattened to a 1-D vector; all vectors go through one shared t-SNE. Series are drawn in
# three colour families (baseline / boundary replacement / ground truth) with colour depth fading as
# the forecast advances (dark near init, light at later leads). See plot_embedding_tsne.py.

python plot_embedding_tsne.py \
    --preds_dirs /tmp3/b12902101/LAM_output_preds/hres_boundary0_no_nearest_backbone_res0.25_direct/preds \
    /tmp3/b12902101/LAM_output_preds/hres_boundary8_no_nearest_backbone_res0.25_direct/preds \
    --labels baseline "boundary replacement" \
    --init_time '2020-03-01 00:00:00' \
    --lead_times $(seq 1 240)