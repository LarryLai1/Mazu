#!/bin/bash
set -euo pipefail

# Run three jobs sequentially on a single GPU with boundary_width 0, 1, 2.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/public_bash_scripts/AuroraSmallTW_gen_eval_pipeline_custom_rollout.sh"
BOUNDARY_MODE="inject-inside"
# BOUNDARY_MODE="pad-outside"

GPU=0
BOUNDARY_WIDTHS=(0 2 4 6 8 10)

# "${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width 1 --boundary_mode "inject-inside" --boundary_pooling "${pooling}" 2>&1 | tee "${log_file}"

for pooling in "no" "yes"; do
	for bw in "${BOUNDARY_WIDTHS[@]}"; do
		log_file="${SCRIPT_DIR}/bash_outputs/val_bw${bw}_${BOUNDARY_MODE}_pooling${pooling}.log"

		mkdir -p "${SCRIPT_DIR}/bash_outputs"
		echo "Starting boundary_width=${bw}, boundary_mode=${BOUNDARY_MODE}, boundary_pooling=${pooling} on GPU ${GPU}..."
		"${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width "${bw}" --boundary_mode "${BOUNDARY_MODE}" --boundary_pooling "${pooling}" 2>&1 | tee "${log_file}"
		break
	done
	break
done

echo "All jobs completed."
