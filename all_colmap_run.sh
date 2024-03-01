#!/bin/bash

# Base directory for the dataset
BASE_DATASET_PATH="/mnt/sda/epic-fields/Sparse"

# Read all directories into an array
readarray -t dirs < <(find "$BASE_DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | shuf)

# Number of parallel jobs
parallel_jobs=25

# Function to process directories
process_dirs() {
    for folder in "${@}"; do
        if [ -d "$folder" ]; then
            fused_ply_path="$folder/dense/fused.ply"

            if [ -f "$fused_ply_path" ]; then
                echo "fused.ply already exists in $folder, skipping..."
                continue
            fi

            echo "Processing folder: $folder"

            colmap image_undistorter --image_path "$folder/images" \
                --input_path "$folder/sparse/0" \
                --output_path "$folder/dense" \
                --output_type COLMAP \
                --max_image_size 2000

            echo "Finished image_undistorter for $folder"

            colmap patch_match_stereo --workspace_path "$folder/dense" \
                --workspace_format COLMAP \
                --PatchMatchStereo.geom_consistency true

            echo "Finished patch_match_stereo for $folder"

            colmap stereo_fusion --workspace_path "$folder/dense" \
                --workspace_format COLMAP \
                --input_type geometric \
                --output_path "$folder/dense/fused.ply"

            echo "Finished stereo_fusion for $folder"
        fi
    done
}

# Calculate the number of directories each job should process
dirs_per_job=$(( (${#dirs[@]} + parallel_jobs - 1) / parallel_jobs ))

# Create and run jobs in parallel
for ((i = 0; i < ${#dirs[@]}; i += dirs_per_job)); do
    process_dirs "${dirs[@]:i:dirs_per_job}" &
done

# Wait for all background jobs to finish
wait

echo "All folders processed."

