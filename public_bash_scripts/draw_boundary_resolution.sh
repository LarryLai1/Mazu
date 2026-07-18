#!/bin/bash
# Plot the 2020-03-01 T00 boundary across 3 resolutions x 2 apply modes.
# "field" shows the boundary itself; "diff" shows the difference against the 0.25deg
# baseline, which is the only way to actually see the 0.5deg effect (the raw fields
# look near-identical at msl's ~99000-103500 Pa scale).

for plot_mode in "field" "diff"; do
    python plot_boundary_resolution.py --var_name msl \
        --output_dir boundary_resolution_plots \
        --init_time "2020-03-01 00:00:00" \
        --plot_mode "${plot_mode}"
done
