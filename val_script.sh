#!/bin/bash
set -euo pipefail

# Run three jobs sequentially on a single GPU with boundary_width 0, 1, 2.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/public_bash_scripts/AuroraSmallTW_gen_eval_pipeline_custom_rollout.sh"
BOUNDARY_MODE="inject-inside"
# BOUNDARY_MODE="pad-outside"

BOUNDARY_WIDTHS=(0 4 8)
GPU="0,1,2,3"

# "${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width 1 --boundary_mode "inject-inside" --boundary_pooling "${pooling}" 2>&1 | tee "${log_file}"

for pooling in "no" "yes"; do
	for bw in "${BOUNDARY_WIDTHS[@]}"; do
		if [[ "${pooling}" == "no" && bw -eq 0 ]]; then
			# Skip the case where boundary_width is 0 and pooling is yes, as it doesn't make sense to apply pooling when there is no boundary.
			continue
		fi
		log_file="${SCRIPT_DIR}/bash_outputs/val_bw${bw}_${BOUNDARY_MODE}_pooling${pooling}.log"

		mkdir -p "${SCRIPT_DIR}/bash_outputs"
		echo "Starting boundary_width=${bw}, boundary_mode=${BOUNDARY_MODE}, boundary_pooling=${pooling} on GPU ${GPU}..."
		"${RUN_SCRIPT}" --gpus "${GPU}" --boundary_width "${bw}" --boundary_mode "${BOUNDARY_MODE}" --boundary_pooling "${pooling}" --boundary_use_cache 2>&1 | tee "${log_file}"
		# break
	done
	# break
done

echo "All jobs completed."
