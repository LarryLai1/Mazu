#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/public_bash_scripts/AuroraSmallTW_gen_eval_pipeline_custom_rollout.sh"
BOUNDARY_MODE="inject-inside"

BOUNDARY_WIDTHS=(0 4 8)
GPU="0,1,2,3"


for pooling in "no" "yes"; do
	for bw in "${BOUNDARY_WIDTHS[@]}"; do
		if [[ "${pooling}" == "no" && bw -eq 0 ]]; then
			# Skip the case where boundary_width is 0 and pooling is yes, as it doesn't make sense to apply pooling when there is no boundary.
			continue
		fi
		log_file="${SCRIPT_DIR}/bash_outputs/val_bw${bw}_${BOUNDARY_MODE}_pooling${pooling}.log"

		mkdir -p "${SCRIPT_DIR}/bash_outputs"
		echo "Starting boundary_width=${bw}, boundary_pooling=${pooling} on GPU ${GPU}..."
		"${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width "${bw}" --boundary_mode "${BOUNDARY_MODE}" --boundary_pooling "${pooling}"
		# break
	done
	# break
done

echo "All jobs completed."
