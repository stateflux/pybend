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
        # this prevents gc from destroying param arrays during jit execution
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
            
            # transform using our hsv transform function with the palette type directly
            # instead of passing function references that numba can't handle
            transformed_region = apply_palette_transform(region, palette_types[i], params)
            
            # copy back to main image
            apply_to_region(img_array, transformed_region, y_start, x_start)
        
        # convert back to bytearray and return
        return bytearray(img_array.tobytes())

# Replace the existing apply_hsv_transform + get_transform_function approach
# with a single function that handles the transform type directly
@jit(nopython=True, parallel=True, cache=True)
def apply_palette_transform(region, transform_type, params):
    """apply hsv transformation based directly on transform type"""
    # convert rgb to hsv - gotta do this manually for arrays
    height, width, _ = region.shape
    hsv_region = np.empty_like(region, dtype=np.float64)  # use float64 to avoid type conflicts
    
    # convert rgb to hsv pixel by pixel with parallel processing
    for y in prange(height):
        for x in range(width):
            r, g, b = region[y, x]
            h, s, v = rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            hsv_region[y, x, 0] = h
            hsv_region[y, x, 1] = s
            hsv_region[y, x, 2] = v
    
    # grab references to channels for vectorized ops
    h, s, v = hsv_region[:, :, 0], hsv_region[:, :, 1], hsv_region[:, :, 2]
    
    # apply transformation based on type
    if transform_type == 0:  # complementary
        h = (h + 0.5) % 1.0
        s = np.minimum(1.0, s * 1.3)  # bump sat
        v = np.minimum(1.0, v * 1.1)  # small brightness bump
    
    elif transform_type == 1:  # hue_shift
        h = (h + params[0]) % 1.0
        s = np.minimum(1.0, s * (1.0 + params[0]))
    
    elif transform_type == 2:  # grayscale
        s *= (1 - params[0])
        # handle the shadows/highlights with loops since numba hates boolean masks
        for y in prange(height):
            for x in range(width):
                if v[y, x] < 0.5:
                    v[y, x] = max(0.0, v[y, x] * (0.9 - 0.2 * params[0]))  # darken shadows
                else:
                    v[y, x] = min(1.0, v[y, x] * (1.1 + 0.2 * params[0]))  # brighten highlights
    
    elif transform_type == 3:  # neon
        intensity = params[0]
        s = np.minimum(1.0, s * (1.5 + intensity * 0.5))  # up to 2x sat
        v = np.minimum(1.0, v * (1.2 + intensity * 0.3))  # up to 1.5x brightness
        h_shift = 0.15
        h = (h + h_shift) % 1.0
    
    elif transform_type == 4:  # vintage
        intensity = params[0]
        s = s * (0.7 - intensity * 0.3)
        h = h * 0.8 + 0.1
        v = v * (0.9 - intensity * 0.1)
    
    elif transform_type == 5:  # vaporwave
        intensity = params[0]
        # do the conditional transformations with loops since numba hates boolean masks
        for y in prange(height):
            for x in range(width):
                if h[y, x] < 0.3 or h[y, x] > 0.8:
                    h[y, x] = (h[y, x] * 0.5 + 0.7) % 1.0
                
                if (h[y, x] > 0.7 and h[y, x] < 0.9) or (h[y, x] > 0.4 and h[y, x] < 0.6):
                    s[y, x] = min(1.0, s[y, x] * (1.0 + intensity))
        
        v = np.minimum(1.0, v * (1.0 + intensity * 0.3))
    
    elif transform_type == 6:  # extreme_contrast
        intensity = params[0]
        # explicit loops for conditional stuff - numba limitation
        for y in prange(height):
            for x in range(width):
                if v[y, x] < 0.5:
                    v[y, x] = v[y, x] * (1.0 - intensity * 0.8)
                else:
                    v[y, x] = min(1.0, v[y, x] * (1.0 + intensity * 0.5))
                
                s[y, x] = min(1.0, s[y, x] * (1.0 + intensity * 0.5))
    
    else:  # transform_type == 7, color_blast
        intensity = params[0]
        h = (h * 0.1 + 0.42) % 1.0
        s = np.minimum(1.0, s * 0.2 + 0.8 * intensity)
        # more explicit loops because of numba's boolean mask limitations
        for y in prange(height):
            for x in range(width):
                if v[y, x] > 0.3:
                    v[y, x] = min(1.0, v[y, x] * (1.0 + intensity * 0.3))
    
    # put channels back into hsv array
    hsv_region[:, :, 0] = h
    hsv_region[:, :, 1] = s
    hsv_region[:, :, 2] = v
    
    # convert back to rgb with parallel processing
    result = np.empty_like(region)
    for y in prange(height):
        for x in range(width):
            h, s, v = hsv_region[y, x]
            r, g, b = hsv_to_rgb(h, s, v)
            result[y, x, 0] = min(255, int(r * 255))
            result[y, x, 1] = min(255, int(g * 255))
            result[y, x, 2] = min(255, int(b * 255))
            
    return result

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
    s = np.minimum(1.0, s * (1.5 + intensity * 0.5))  # up to 2x sat
    v = np.minimum(1.0, v * (1.2 + intensity * 0.3))  # up to 1.5x brightness
    
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
        s = np.minimum(1.0, s * (1.0 + intensity))
    
    # pump brightness
    v = np.minimum(1.0, v * (1.0 + intensity * 0.3))
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
        v = np.minimum(1.0, v)
    
    # crank saturation
    s = np.minimum(1.0, s * (1.0 + intensity * 0.5))
    return h, s, v

@jit(nopython=True, cache=True)
def transform_color_blast(h, s, v, params):
    """super saturated trippy color effect"""
    intensity = params[0]
    
    # fixed hue shift
    h = (h * 0.1 + 0.42) % 1.0
    
    # massive sat boost
    s = np.minimum(1.0, s * 0.2 + 0.8 * intensity)
    
    # brighten lighter areas
    if v > 0.3:
        v = np.minimum(1.0, v * (1.0 + intensity * 0.3))
    
    return h, s, v