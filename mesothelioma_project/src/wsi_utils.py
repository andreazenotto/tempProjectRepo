import os
import numpy as np
import openslide
from skimage.color import rgb2hsv
from PIL import Image
from itertools import product
from tqdm import tqdm


def load_wsi(file_path):
    slide = openslide.OpenSlide(file_path)
    return slide


def segment_tissue(hsv_img, saturation_threshold=0.2):
    # Saturation is located in the second channel of the HSV image
    saturation = hsv_img[:, :, 1]
    # Create a binary mask based on the saturation threshold
    tissue_mask = saturation > saturation_threshold
    return tissue_mask


def extract_patches(slide, save_dir, level, patch_size=(224, 224), tissue_threshold=0.6):
    width, height = slide.level_dimensions[level]
    patch_height, patch_width = patch_size

    x_coords = range(0, width - patch_width, patch_width)
    y_coords = range(0, height - patch_height, patch_height)

    coords = list(product(x_coords, y_coords))
    os.makedirs(save_dir, exist_ok=True)
    
    # Iterate over all possible patch positions
    for x, y in tqdm(coords, desc="Extracting patches"):
        # Extract the image region
        region = slide.read_region((x, y), level, patch_size).convert("RGB")
        region = np.array(region)
        
        # Segment and create the patch mask
        patch_hsv = rgb2hsv(region)
        patch_mask = segment_tissue(patch_hsv)
        
        # Calculate the tissue coverage percentage in the patch
        tissue_coverage = np.sum(patch_mask) / (patch_height * patch_width)
        
        # If the tissue coverage is sufficient, save the patch
        if tissue_coverage >= tissue_threshold:
            patch = Image.fromarray(region)
            # Save the patch as a PNG image
            patch_filename = f"patch_{x}_{y}.png"
            patch.save(os.path.join(save_dir, patch_filename))


def load_patches(directory):
    patches = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            patches.append(img)
    
    return patches