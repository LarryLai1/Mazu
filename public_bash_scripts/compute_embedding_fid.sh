#!/bin/bash
set -euo pipefail

# Frechet distance (FID) from ERA5 per lead time, over the Swin3D bottleneck embeddings that
# AuroraSmallTW_gen_eval_pipeline_with_embeddings.py and extract_hres_embeddings.py already wrote.
# Produces embedding_fid.csv plus two line charts -- forecast lead time on x, one line per series:
# embedding_fid.png (the FID itself) and embedding_fid_terms.png (the mean term ||dmu||^2 and the
# trace term side by side, i.e. drifting away from ERA5 vs. dispersing around it; the CSV carries
# both as their own columns). This is the numeric counterpart to plot_embedding_tsne_hooked.sh:
# t-SNE shows that the series separate, this says by how much in a number you can tabulate.
#
# Downstream-only: no model, no GPU. Runs in seconds.
#
# The two defaults worth knowing about, both deliberate:
#
#   * The saved vectors are flattened (432 tokens x 1024 channels). FID needs a sample covariance,
#     and 442368 dims against a few hundred samples has none, so the script mean-pools back to the
#     1024-d feature -- which is exactly what SOFT's FID stage and Mazu's own
#     reference_artifact/AuroraSmallTW_calc_backbone_embedding_FID.py score -- and then projects to
#     --pca_dim 64 so the covariance is full rank. --pca_dim 0 gives the unreduced house definition,
#     with a warning, for comparison.
#
#   * Every series is scored on the initialisation times they ALL share (--no_common_inits turns
#     this off). FID's small-sample bias grows as d/n, so a series scored on more inits would score
#     lower for that reason alone. All three directories currently cover the same 216 inits, so this
#     costs nothing today -- it is a guard, not a sacrifice.
#
# The chart also carries an ERA5-self-split noise floor, since a FID value has no absolute scale:
# anything sitting at that line is indistinguishable from ERA5 at this sample size.
# --no_noise_floor removes it.
#
# CAVEAT ON THE HRES SERIES: hres_forecast_res0.25 was extracted with
# --hres_time_interp_mode nearest. HRES stores 6-hourly steps while we query hourly leads, so
# under 'nearest' the two frames of the [t-1, t] history window round to the same HRES step and
# the model sees zero tendency -- which is why extract_hres_embeddings.py's own help for that
# flag says 'interpolation' is required. Measured over 30 (init, lead) pairs, EVERY 'interpolation'
# vector is closer to ERA5 than its 'nearest' counterpart, and the choice changes the shape of the
# HRES line: flat at ~0.84 across all leads under interpolation, drifting 0.80 -> 0.92 under
# nearest. To score the corrected series instead, extract it and point --embeddings_dirs at it:
#   python extract_hres_embeddings.py \
#     --match_embeddings_dir "${EMBED_ROOT}/hres_boundary0_backbone_res0.25_direct" \
#     --output_dir "${EMBED_ROOT}/hres_forecast_res0.25_interp" \
#     --hres_time_interp_mode interpolation --skip_gt

EMBED_ROOT="/tmp3/b12902101/mazu_embedding_output/embeddings"
OUTPUT_DIR="/tmp3/b12902101/mazu_embedding_output/fid"

python compute_embedding_fid.py \
    --embeddings_dirs \
        "${EMBED_ROOT}/hres_boundary0_backbone_res0.25_direct" \
        "${EMBED_ROOT}/hres_boundary8_backbone_res0.25_direct" \
        "${EMBED_ROOT}/hres_forecast_res0.25" \
    --labels baseline "boundary replacement" "hres boundary 0.25" \
    --pca_dim 64 \
    --output_dir "${OUTPUT_DIR}"
