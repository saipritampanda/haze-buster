import os
from pathlib import Path

# Path to the 'hazy' folder
hazy_dir = Path(r"C:\Users\SAI PRITAM PANDA\Desktop\SDP Project\DATASETS\Reside\Synthetic Objective Testing Set (SOTS)\outdoor\hazy")

# Get all hazy images
hazy_images = sorted([f for f in hazy_dir.iterdir() if f.is_file()])

for file in hazy_images:
    # Extract filename without extension
    stem = file.stem  # e.g., '0001_0.8_0.2'
    parts = stem.split("_")

    if parts and parts[0].isdigit():
        # Remove leading zeros from the first part
        new_prefix = str(int(parts[0]))  # e.g., '0001' -> '1'
        new_stem = "_".join([new_prefix] + parts[1:])  # Recombine
        new_name = f"{new_stem}{file.suffix}"
        new_path = hazy_dir / new_name

        print(f"Renaming: {file.name} -> {new_name}")
        file.rename(new_path)

print("Outdoor hazy image renaming completed.")
