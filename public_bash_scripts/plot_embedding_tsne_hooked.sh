#!/bin/bash
set -euo pipefail

# Per-lead-time t-SNE of the Swin3D bottleneck embeddings extracted by
# AuroraSmallTW_gen_eval_pipeline_with_embeddings.py (hook-based, flattened -- see
# plot_embedding_tsne_hooked.py). Produces ONE FIGURE PER LEAD TIME: with the default
# --embedding_save_steps "1 6 24 72" from val_script_with_embeddings.sh you get four files,
# embedding_tsne_lead{1,6,24,72}h.png, each with its own independent PCA->t-SNE fit.
#
# Compares three forecast series -- our baseline rollout, our boundary-replacement rollout, and
# the raw ECMWF HRES 0.25deg forecast -- against the shared ERA5 ground truth, which is drawn as
# a neutral reference rather than a fourth coloured series (three is the most a scatter can keep
# colour-distinguishable). Downstream-only: reads the .npz files those runs already wrote, so no
# model or GPU is needed here.
#
# The HRES series comes from extract_hres_embeddings.py, which re-embeds the HRES forecast
# through the same frozen model and hook; run it first if that directory is missing:
#   python extract_hres_embeddings.py \
#     --match_embeddings_dir "${EMBED_ROOT}/hres_boundary0_backbone_res0.25_direct" \
#     --output_dir "${EMBED_ROOT}/hres_forecast_res0.25" --init_days 10 --skip_gt
#
# --init_days 5 restricts this to the first 5 days of initialisation times: the full month is
# ~744 hourly inits per series, which over-plots into a solid mass. Raise it (or drop the flag
# for everything) if you want the denser view.

EMBED_ROOT="/tmp3/b12902101/mazu_embedding_output/embeddings"

python plot_embedding_tsne_hooked.py \
    --embeddings_dirs \
        "${EMBED_ROOT}/hres_boundary0_backbone_res0.25_direct" \
        "${EMBED_ROOT}/hres_boundary8_backbone_res0.25_direct" \
        "${EMBED_ROOT}/hres_forecast_res0.25" \
    --labels baseline "boundary replacement" "hres boundary 0.25" \
    --init_days 3 \
    --perplexity 60 \
    --output_dir /tmp3/b12902101/mazu_embedding_output/tsne_plots
