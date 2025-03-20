#!/usr/bin/env python3
"""
palette swap effect - messes with color palettes in specific regions
"""

import random
import numpy as np
from numba import jit, prange
from effects.base import Effect
from effects.utils import (
    select_region_type, get_region_params, extract_region, apply_to_region,
    apply_hsv_transform, ROW_REGION, RECT_REGION, rgb_to_hsv, hsv_to_rgb
)

# params need to live outside jit because numba HATES them
COMPLEMENTARY_PARAMS = np.array([0.0])  
HUE_SHIFT_BASE = np.array([0.3])
GRAYSCALE_BASE = np.array([0.8])
NEON_BASE = np.array([1.0])
VINTAGE_BASE = np.array([1.0])
VAPORWAVE_BASE = np.array([1.0])
CONTRAST_BASE = np.array([2.0])
COLOR_BLAST_BASE = np.array([1.0])

class PaletteSwap(Effect):
    """swaps color palette in targeted image regions"""
    
    def __init__(self):
        """init the PaletteSwap effect"""
        super().__init__()
        self.name = "palette_swap"
        self.description = "swaps the color palette in defined regions"
    
    def apply(self, data, width, height, row_length, bytes_per_pixel, intensity):
        """apply palette swap to raw image
        
        args:
            data (bytearray): raw image data to mess with
            width (int): image width in pixels
            height (int): image height in pixels
            row_length (int): bytes per row
            bytes_per_pixel (int): bytes per pixel (3 for rgb)
            intensity (float): effect strength 0.0-1.0
            
        returns:
            bytearray: modified image data
        """
        # defensive copy of input data to prevent gc issues
        data_copy = bytearray(data)
        
        # convert to numpy array for faster ops
        img_array = np.frombuffer(data_copy, dtype=np.uint8).reshape((height, width, 3))
        
        # how many regions to hit based on intensity (1-8)
        num_regions = max(1, int(8 * intensity))
        
        # pre-generate random values outside JIT
        palette_types = []
        for _ in range(num_regions):
            if intensity > 0.6 and random.random() < 0.7:
                palette_types.append(random.choice([3, 5, 6, 7]))  # neon, vaporwave, extreme_contrast, color_blast
            else:
                palette_types.append(random.randint(0, 7))  # any random palette
        
        # create a persistent array to hold all parameters
        # this prevents garbage collection from destroying param arrays during jit execution
        all_params = np.zeros((num_regions, 1), dtype=np.float64)
        
        # fill the persistent array 
        for i, palette_type in enumerate(palette_types):
            if palette_type == 0:  # complementary
                all_params[i, 0] = COMPLEMENTARY_PARAMS[0]
            elif palette_type == 1:  # hue_shift
                all_params[i, 0] = HUE_SHIFT_BASE[0] * random.uniform(0.5, 1.0) * intensity
            elif palette_type == 2:  # grayscale
                all_params[i, 0] = GRAYSCALE_BASE[0] * intensity
            elif palette_type == 3:  # neon
                all_params[i, 0] = NEON_BASE[0] * intensity
            elif palette_type == 4:  # vintage
                all_params[i, 0] = VINTAGE_BASE[0] * intensity
            elif palette_type == 5:  # vaporwave
                all_params[i, 0] = VAPORWAVE_BASE[0] * intensity
            elif palette_type == 6:  # extreme_contrast
                all_params[i, 0] = CONTRAST_BASE[0] * (0.25 + 0.75 * intensity)
            else: #palette_type == 7:  # color_blast
                all_params[i, 0] = COLOR_BLAST_BASE[0] * intensity
        
        # pre-generate region types
        region_types = [select_region_type(row_probability=0.4) for _ in range(num_regions)]
        
        # process each region
        for i in range(num_regions):
            # get region parameters based on type
            y_start, region_height, x_start, region_width = get_region_params(
                height, width, region_types[i], intensity
            )
            
            # extract region as copy
            region = extract_region(img_array, y_start, region_height, x_start, region_width)
            
            # extract single param row - this is a view not a copy, so parent won't be gc'd
            params = all_params[i]
            
            # transform using our hsv transform function
            transformed_region = apply_hsv_transform(region, get_transform_function(palette_types[i]), params)
            
            # copy back to main image
            apply_to_region(img_array, transformed_region, y_start, x_start)
        
        # convert back to bytearray and return
        return bytearray(img_array.tobytes())

@jit(nopython=True)
def get_transform_function(transform_type):
    """grab the right transform function for the given type"""
    # needed because numba sucks at passing functions as args directly
    if transform_type == 0:
        return transform_complementary
    elif transform_type == 1:
        return transform_hue_shift
    elif transform_type == 2:
        return transform_grayscale
    elif transform_type == 3:
        return transform_neon
    elif transform_type == 4:
        return transform_vintage
    elif transform_type == 5:
        return transform_vaporwave
    elif transform_type == 6:
        return transform_extreme_contrast
    else:  # transform_type == 7
        return transform_color_blast

@jit(nopython=True, cache=True)
def transform_complementary(h, s, v, params):
    """complementary colors - flips hue 180deg and boosts saturation"""
    h = (h + 0.5) % 1.0
    s = min(1.0, s * 1.3)  # bump sat
    v = min(1.0, v * 1.1)  # small brightness bump
    return h, s, v

@jit(nopython=True, cache=True)
def transform_hue_shift(h, s, v, params):
    """hue shifter - rotates colors around the wheel with saturation boost"""
    h = (h + params[0]) % 1.0
    s = min(1.0, s * (1.0 + params[0]))
    return h, s, v

@jit(nopython=True, cache=True)
def transform_grayscale(h, s, v, params):
    """grayscale with contrast - desaturates with shadow/highlight tweaks"""
    s *= (1 - params[0])
    if v < 0.5:
        v = max(0.0, v * (0.9 - 0.2 * params[0]))  # darken shadows
    else:
        v = min(1.0, v * (1.1 + 0.2 * params[0]))  # brighten highlights
    return h, s, v

@jit(nopython=True, cache=True)
def transform_neon(h, s, v, params):
    """neon glow - makes shit glow with crazy saturation"""
    intensity = params[0]
    s = min(1.0, s * (1.5 + intensity * 0.5))  # up to 2x sat
    v = min(1.0, v * (1.2 + intensity * 0.3))  # up to 1.5x brightness
    
    # fixed hue shift - no calcs needed
    h_shift = 0.15
    h = (h + h_shift) % 1.0
    return h, s, v

@jit(nopython=True, cache=True)
def transform_vintage(h, s, v, params):
    """vintage filter - old-timey look with muted colors and sepia"""
    intensity = params[0]
    # cut sat
    s = s * (0.7 - intensity * 0.3)
    # push hue toward yellow/orange
    h = h * 0.8 + 0.1
    # slight darkening
    v = v * (0.9 - intensity * 0.1)
    return h, s, v

@jit(nopython=True, cache=True)
def transform_vaporwave(h, s, v, params):
    """80s aesthetics with purple/pink/cyan"""
    intensity = params[0]
    
    # push toward purple/blue range
    if h < 0.3 or h > 0.8:
        h = (h * 0.5 + 0.7) % 1.0
    
    # crank sat for pinks and cyans
    if (h > 0.7 and h < 0.9) or (h > 0.4 and h < 0.6):
        s = min(1.0, s * (1.0 + intensity))
    
    # pump brightness
    v = min(1.0, v * (1.0 + intensity * 0.3))
    return h, s, v

@jit(nopython=True, cache=True)
def transform_extreme_contrast(h, s, v, params):
    """harsh contrast with saturation pump"""
    intensity = params[0]
    
    # brutal contrast
    if v < 0.5:
        v = v * (1.0 - intensity * 0.8)
    else:
        v = v * (1.0 + intensity * 0.5)
        v = min(1.0, v)
    
    # crank saturation
    s = min(1.0, s * (1.0 + intensity * 0.5))
    return h, s, v

@jit(nopython=True, cache=True)
def transform_color_blast(h, s, v, params):
    """super saturated trippy color effect"""
    intensity = params[0]
    
    # fixed hue shift
    h = (h * 0.1 + 0.42) % 1.0
    
    # massive sat boost
    s = min(1.0, s * 0.2 + 0.8 * intensity)
    
    # brighten lighter areas
    if v > 0.3:
        v = min(1.0, v * (1.0 + intensity * 0.3))
    
    return h, s, v