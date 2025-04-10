#!/bin/bash
# Convert demo.mp4 to demo.gif using two-pass palette generation
ffmpeg -y -i demo.mp4 -vf "fps=10,scale=320:-1:flags=lanczos,palettegen" palette_demo.png
ffmpeg -y -i demo.mp4 -i palette_demo.png -filter_complex "fps=10,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse" demo.gif

# Loop through each Kazam_screencast WebM file
for file in Kazam_screencast_*.webm; do
    # Extract the numeric part from the filename (removing leading zeros)
    num=$(echo "$file" | sed -E 's/Kazam_screencast_0*([0-9]+)\.webm/\1/')
    # Format the output filename as screencastXX.gif (e.g., screencast01.gif)
    output=$(printf "screencast%02d.gif" "$num")
    
    # Generate a palette for the current video file
    ffmpeg -y -i "$file" -vf "fps=10,scale=320:-1:flags=lanczos,palettegen" "palette_$output.png"
    
    # Convert the video to GIF using the generated palette
    ffmpeg -y -i "$file" -i "palette_$output.png" -filter_complex "fps=10,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse" "$output"
done

