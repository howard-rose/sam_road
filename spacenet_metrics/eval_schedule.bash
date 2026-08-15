#!/bin/bash

# Add pixi python to PATH if not already found (needed when called from Windows bash)
if ! command -v python &> /dev/null; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PIXI_ENV="$SCRIPT_DIR/../.pixi/envs/default"
    if [ -d "$PIXI_ENV" ]; then
        export PATH="$PIXI_ENV:$PIXI_ENV/Scripts:$PATH"
    fi
fi

# List of directories to process, add more as needed
output_dirs=(
    "infer_2026-08-10_123456"
)

# Base directory where the output directories are located
base_dir="save"

# Loop through each output_dir in the list
for output_dir in "${output_dirs[@]}"; do
    # Construct the full path to the output directory
    full_path="$base_dir/$output_dir"
    
    # Run the apls.bash script with the full path to the output directory
    bash apls.bash "$full_path"
    
    # Run the topo.bash script with the full path to the output directory
    bash topo.bash "$full_path"
done