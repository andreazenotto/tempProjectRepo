import os
import numpy as np
import openslide
from skimage.color import rgb2hsv
from PIL import Image
from itertools import product
from tqdm import tqdm
import cv2


def load_wsi(file_path):
    slide = openslide.OpenSlide(file_path)
    return slide


def isWhitePatch(patch, satThresh):
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:,:,1]) < satThresh else False


def extract_patches(slide, save_dir, level=1, threshold=30, patch_size=(224, 224)):
    width, height = slide.level_dimensions[level]
    patch_height, patch_width = patch_size

    x_coords = range(0, width - patch_width, patch_width)
    y_coords = range(0, height - patch_height, patch_height)

    coords = list(product(x_coords, y_coords))
    os.makedirs(save_dir, exist_ok=True)
    
    # Iterate over all possible patch positions
    for x, y in coords:
        # Extract the image region
        region = slide.read_region((x, y), level, patch_size).convert("RGB")
        region = np.array(region)

        if not isWhitePatch(region, threshold):
            patch = Image.fromarray(region)
            # Save the patch as a PNG image
            patch_filename = f"patch_{x}_{y}.png"
            patch.save(os.path.join(save_dir, patch_filename))


def count_patches(slide, level=1, threshold=30, patch_size=(224, 224)):
    width, height = slide.level_dimensions[level]
    patch_height, patch_width = patch_size

    x_coords = range(0, width - patch_width, patch_width)
    y_coords = range(0, height - patch_height, patch_height)
    coords = list(product(x_coords, y_coords))

    count = 0
    
    # Iterate over all possible patch positions
    for x, y in coords:
        # Extract the image region
        region = slide.read_region((x, y), level, patch_size).convert("RGB")
        region = np.array(region)

        if not isWhitePatch(region, threshold):
            count += 1

    return count
