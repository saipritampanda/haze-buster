import os
from pathlib import Path

# Convert string paths to Path objects
clear_dir = Path(r"C:\Users\SAI PRITAM PANDA\Desktop\SDP Project\DATASETS\Reside\Indoor Training Set (ITS)\clear")
hazy_dir = Path(r"C:\Users\SAI PRITAM PANDA\Desktop\SDP Project\DATASETS\Reside\Indoor Training Set (ITS)\hazy")

# Get all clear image names and sort by number
clear_images = sorted([f for f in clear_dir.iterdir() if f.is_file()], key=lambda x: int(x.stem))

for clear_img in clear_images:
    base_name = clear_img.stem  # e.g., "1", "2", ...

    # Find all hazy and trans files starting with base_name
    hazy_files = sorted(hazy_dir.glob(f"{base_name}_*.*"))

    # Rename hazy images
    for i, file in enumerate(hazy_files, start=1):
        new_name = f"{base_name}_{i}{file.suffix}"
        new_path = hazy_dir / new_name
        print(f"Renaming hazy: {file.name} -> {new_name}")
        file.rename(new_path)

print("Renaming completed.")
