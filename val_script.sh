#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/public_bash_scripts/AuroraSmallTW_gen_eval_pipeline_custom_rollout.sh"
BOUNDARY_MODE="inject-inside"

BOUNDARY_WIDTHS=(0 2 4 6 8)
# GPU="0,2,3,5"
# GPU="0,2,3"
GPU="4,5"

pooling="no"
for bw in "${BOUNDARY_WIDTHS[@]}"; do
	log_file="${SCRIPT_DIR}/bash_outputs/val_bw${bw}_${BOUNDARY_MODE}_pooling_${pooling}.log"

	mkdir -p "${SCRIPT_DIR}/bash_outputs"
	echo "Starting boundary_width=${bw}, boundary_pooling=${pooling} on GPU ${GPU}..."
	"${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width "${bw}" --boundary_mode "${BOUNDARY_MODE}" --boundary_pooling "${pooling}"
	break
done

echo "All jobs completed."
