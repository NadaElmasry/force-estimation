import os
from PIL import Image

# Replace with your own directory path
SOURCE_DIR = r"C:/project/dafoes/DaFoEs/data/images"

# Example crop coordinates (left, upper, right, lower)
# Adjust these to match the crop region you want
CROP_BOX = (180, 72, 468, 360)

def crop_images_in_directory(source_dir, crop_box):
    """
    Loops through each subfolder in the source directory,
    crops all images, and saves them to new folders.
    
    :param source_dir: Path to the main directory containing subfolders of images
    :param crop_box:   A tuple (left, upper, right, lower) for the crop area in pixels
    """
    # List all items in the source directory
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        
        # Only process if it's a directory
        if os.path.isdir(folder_path):
            # Create a new folder name with "_cropped" appended
            new_folder_name = folder_name + "_cropped"
            new_folder_path = os.path.join(source_dir, new_folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            
            # Loop through the images in the current folder
            for file_name in os.listdir(folder_path):
                # Check if the file extension looks like an image
                if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                    original_path = os.path.join(folder_path, file_name)
                    new_file_path = os.path.join(new_folder_path, file_name)
                    
                    # Open, crop, and save
                    with Image.open(original_path) as img:
                        cropped_img = img.crop(crop_box)
                        cropped_img.save(new_file_path)
                    
                    print(f"Cropped and saved: {new_file_path}")


# Run the function
crop_images_in_directory(SOURCE_DIR, CROP_BOX)
