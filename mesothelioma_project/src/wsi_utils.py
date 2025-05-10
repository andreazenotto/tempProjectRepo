import os
import numpy as np
import openslide
import cv2
from skimage import img_as_ubyte
from skimage.color import rgb2hsv
from PIL import Image


def load_wsi(file_path):
    """
    Load a WSI from the .ndpi file using OpenSlide.
    
    Args:
        file_path (str): Path to the .ndpi file.
    
    Returns:
        openslide.OpenSlide: OpenSlide object containing the WSI image.
    """
    slide = openslide.OpenSlide(file_path)
    return slide

def convert_to_hsv(slide, level=0):
    """
    Converts a WSI (Whole Slide Image) to an HSV image at a given zoom level.
    
    Args:
        slide (openslide.OpenSlide): OpenSlide object containing the WSI image.
        level (int): Zoom level to use (0 is the highest detail level).
    
    Returns:
        numpy.ndarray: Image in HSV format.
    """
    width, height = slide.dimensions
    img = slide.read_region((0, 0), level, (width, height)).convert("RGB")
    img = np.array(img)
    hsv_img = rgb2hsv(img)
    
    return hsv_img


def segment_tissue(hsv_img, saturation_threshold=0.05):
    """
    Segments tissue from background areas in an HSV image.
    
    Args:
        hsv_img (numpy.ndarray): Image in HSV format.
        saturation_threshold (float): Saturation threshold for tissue.
    
    Returns:
        numpy.ndarray: Binary mask for tissue segmentation.
    """
    # Saturation is located in the second channel of the HSV image
    saturation = hsv_img[:, :, 1]
    # Create a binary mask based on the saturation threshold
    tissue_mask = saturation > saturation_threshold

    return tissue_mask


def extract_patches(slide, patch_size=(224, 224), tissue_threshold=0.6, level=0):
    """
    Extracts patches from a WSI image based on a tissue mask.
    
    Args:
        slide (openslide.OpenSlide): OpenSlide object containing the WSI image.
        tissue_mask (numpy.ndarray): Binary mask indicating tissue areas.
        patch_size (tuple): Size of the patch (height, width).
        tissue_threshold (float): Minimum tissue coverage threshold to keep the patch.
        level (int): Zoom level to use (0 is the highest detail level).
    
    Returns:
        list: List of patches saved as images.
    """
    width, height = slide.dimensions
    patch_height, patch_width = patch_size
    patches = []
    
    # Iterate over all possible patch positions
    for y in range(0, height - patch_height, patch_height):
        for x in range(0, width - patch_width, patch_width):
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
                patches.append(patch)
                # Save the patch as a PNG image
                patch_filename = f"patch_{x}_{y}.png"
                patch.save(os.path.join('outputs', 'patches', patch_filename))
    
    return patches


def save_patches(directory, patches):
    """
    Save the patches in a specified directory.

    Args:
        directory (str): Directory where the patches will be saved.
        patches (list): List of patches to save as images.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i, patch in enumerate(patches):
        patch.save(os.path.join(directory, f"patch_{i}.png"))


def load_patches(directory):
    """
    Load patches from a specified directory.
    
    Args:
        directory (str): Directory containing the saved patches.
    
    Returns:
        list: List of loaded patches as images.
    """
    patches = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            patches.append(img)
    
    return patches