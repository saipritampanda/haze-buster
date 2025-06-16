import os

def rename_images(folder_path):
    # Get all .png files in the folder
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    # Sort the files to have a consistent order
    png_files.sort()

    # Rename each file
    for idx, filename in enumerate(png_files, start=1):
        new_name = f"hazy_{idx}.jpg"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_name}'")

# Example usage
rename_images("images")
