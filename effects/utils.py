#!/usr/bin/env python3
"""
shared utilities for image effects, mostly optimization
contains common numba-accelerated functions used across multiple effects
"""

import random
import numpy as np
from numba import jit, prange, int32, float32, boolean

# constants
LARGE_IMAGE_THRESHOLD = 10000000  # >10MP

# region selection constants
ROW_REGION = 0
RECT_REGION = 1
COLUMN_REGION = 2

# color thresholds for different color types - indexed by int not string for numba compatibility
# (r_thresh, g_thresh, b_thresh, r_comp, g_comp, b_comp)
# where r_comp etc are -1 for < comparison, 1 for > comparison
COLOR_THRESHOLDS = np.array([
    [200, 200, 200, 1, 1, 1],   # 0 = whites (r>200, g>200, b>200)
    [50, 50, 50, -1, -1, -1],   # 1 = blacks (r<50, g<50, b<50)
    [180, 100, 100, 1, -1, -1], # 2 = reds (r>180, g<100, b<100)
    [100, 180, 100, -1, 1, -1], # 3 = greens (r<100, g>180, b<100)
    [100, 100, 180, -1, -1, 1]  # 4 = blues (r<100, g<100, b>180)
])

#
# PERFORMANCE UTILITIES
#

@jit(nopython=True, cache=True)
def is_large_image(width, height):
    """quick check if image is large enough to need optimization"""
    return width * height > LARGE_IMAGE_THRESHOLD

@jit(nopython=True, cache=True)
def get_stride_for_image(width, height, base_stride=1):
    """
    calculates optimal stride for an image based on its size
    
    args:
        width: image width
        height: image height
        base_stride: minimum stride value
        
    returns:
        stride value (larger for bigger images)
    """
    if not is_large_image(width, height):
        return base_stride
    
    # scale stride based on image size
    pixel_count = width * height
    if pixel_count > 100000000:  # 100MP+
        return max(8, base_stride)
    elif pixel_count > 50000000:  # 50MP+
        return max(4, base_stride)
    else:  # 10MP-50MP
        return max(2, base_stride)

#
# REGION SELECTION UTILITIES
#

@jit(nopython=True, cache=True)
def select_region_type(row_probability=0.33, column_probability=0.33):
    """
    select between row, column or rectangular region types with specified probabilities
    
    args:
        row_probability: probability of selecting a row region (0.0-1.0)
        column_probability: probability of selecting a column region (0.0-1.0)
        
    returns:
        region type constant (ROW_REGION, COLUMN_REGION, or RECT_REGION)
    """
    r = random.random()
    if r < row_probability:
        return ROW_REGION
    elif r < (row_probability + column_probability):
        return COLUMN_REGION
    return RECT_REGION

@jit(nopython=True, cache=True)
def get_column_region_params(height, width, min_width_pct=0.02, max_width_pct=0.3, intensity=1.0):
    """
    get parameters for a column region
    
    args:
        height: image height in pixels
        width: image width in pixels
        min_width_pct: minimum width as percentage of image width
        max_width_pct: maximum width as percentage of image width (default 30%)
        intensity: effect intensity (scales max_width_pct)
        
    returns:
        (y_start, height, x_start, width) tuple defining region
    """
    # calculate region width based on intensity and image width
    region_width = int(width * random.uniform(min_width_pct, min(1.0, max_width_pct * intensity)))
    region_width = max(5, min(width, region_width))  # ensure valid width (5px minimum)
    
    # random horizontal position
    x_start = random.randint(0, max(0, width - region_width))
    
    # for column regions, we use the full height
    return 0, height, x_start, region_width

@jit(nopython=True, cache=True)
def get_row_region_params(height, width, min_height_pct=0.02, max_height_pct=0.3, intensity=1.0):
    """
    get parameters for a row region
    
    args:
        height: image height in pixels
        width: image width in pixels
        min_height_pct: minimum height as percentage of image height
        max_height_pct: maximum height as percentage of image height (default 30%)
        intensity: effect intensity (scales max_height_pct)
        
    returns:
        (y_start, height, x_start, width) tuple defining region
    """
    # clamp the max_height_pct * intensity to 1.0 to ensure the region never exceeds image height
    clamped_max_height = min(1.0, max_height_pct * intensity)
    
    region_height = int(height * random.uniform(min_height_pct, clamped_max_height))
    region_height = max(5, region_height)  # ensure at least 5 pixels high
    
    # make sure region height doesn't exceed image height
    region_height = min(height, region_height)
    
    # random vertical position
    y_start = random.randint(0, max(0, height - region_height))
    
    # for row regions, we use the full width
    return y_start, region_height, 0, width

@jit(nopython=True, cache=True)
def get_rect_region_params(height, width, min_size_pct=0.1, max_size_pct=0.5, intensity=1.0):
    """
    get parameters for a rectangular region
    
    args:
        height: image height in pixels
        width: image width in pixels
        min_size_pct: minimum size as percentage of image dimensions
        max_size_pct: maximum size as percentage of image dimensions (default 50%)
        intensity: effect intensity (scales max_size_pct)
        
    returns:
        (y_start, height, x_start, width) tuple defining region
    """
    # clamp the max_size_pct * intensity to 1.0 to ensure the region never exceeds image dimensions
    clamped_max_size = min(1.0, max_size_pct * intensity)
    
    region_width = int(width * random.uniform(min_size_pct, clamped_max_size))
    region_height = int(height * random.uniform(min_size_pct, clamped_max_size))
    
    # ensure region is at least 10 pixels in each dimension
    region_width = max(10, region_width)
    region_height = max(10, region_height)
    
    # make sure region dimensions don't exceed image dimensions
    region_width = min(width, region_width)
    region_height = min(height, region_height)
    
    # make sure the region fits within the image
    x_start = random.randint(0, max(0, width - region_width))
    y_start = random.randint(0, max(0, height - region_height))
    
    return y_start, region_height, x_start, region_width

@jit(nopython=True, cache=True)
def get_region_params(height, width, region_type, intensity=1.0):
    """
    get region parameters based on region type
    
    args:
        height: image height in pixels
        width: image width in pixels
        region_type: ROW_REGION, COLUMN_REGION, or RECT_REGION constant
        intensity: effect intensity (0.0-1.0)
        
    returns:
        (y_start, height, x_start, width) tuple defining region
    """
    if region_type == ROW_REGION:
        return get_row_region_params(height, width, intensity=intensity)
    elif region_type == COLUMN_REGION:
        return get_column_region_params(height, width, intensity=intensity)
    else:
        return get_rect_region_params(height, width, intensity=intensity)

@jit(nopython=True, cache=True)
def extract_region(img_array, y_start, region_height, x_start, region_width):
    """
    extract a region from an image array with proper copy semantics for numba
    
    args:
        img_array: 3d numpy array of image data
        y_start, region_height, x_start, region_width: region parameters
        
    returns:
        copy of the image region as a numpy array
    """
    return img_array[y_start:y_start+region_height, x_start:x_start+region_width, :].copy()

@jit(nopython=True, cache=True)
def apply_to_region(img_array, processed_region, y_start, x_start):
    """
    apply a processed region back to the original image
    
    args:
        img_array: 3d numpy array of image data to modify
        processed_region: the processed region data to apply
        y_start: starting y coordinate
        x_start: starting x coordinate
    """
    region_height, region_width = processed_region.shape[:2]
    img_array[y_start:y_start+region_height, x_start:x_start+region_width, :] = processed_region

#
# COLOR-MATCHING UTILITIES
#

@jit(nopython=True, cache=True)
def check_color_match(r, g, b, target_color, channel, threshold):
    """
    unified color matching function
    
    args:
        r, g, b: pixel color values
        target_color: color type to match (0-5)
        channel: channel to check if using threshold
        threshold: threshold value
    """
    if target_color < 5:  # use predefined color thresholds
        thresh = COLOR_THRESHOLDS[target_color]
        r_match = (r > thresh[0]) if thresh[3] > 0 else (r < thresh[0])
        g_match = (g > thresh[1]) if thresh[4] > 0 else (g < thresh[1])
        b_match = (b > thresh[2]) if thresh[5] > 0 else (b < thresh[2])
        return r_match and g_match and b_match
    else:  # threshold based (target_color == 5)
        if channel == 0:
            return r > threshold
        elif channel == 1:
            return g > threshold
        else:
            return b > threshold

#
# BLENDING UTILITIES
#

@jit(nopython=True, parallel=True, cache=True)
def blend_chunks_with_stride(src_chunk, dst_chunk, blend_factor, stride):
    """
    blend two chunks using numba with stride for improved performance
    vectorized implementation
    
    args:
        src_chunk (numpy.ndarray): source chunk
        dst_chunk (numpy.ndarray): destination chunk to be modified
        blend_factor (float): blending factor (0.0-1.0)
        stride (int): stride for processing (skip pixels)
    """
    inv_blend = 1.0 - blend_factor
    indices = np.arange(0, len(src_chunk), stride)
    
    # process reference pixels
    for i in prange(len(indices)):
        idx = indices[i]
        dst_chunk[idx] = min(255, int(dst_chunk[idx] * inv_blend + src_chunk[idx] * blend_factor))
        
        # fill gaps
        for offset in range(1, stride):
            if idx + offset < len(src_chunk):
                dst_chunk[idx + offset] = dst_chunk[idx]

@jit(nopython=True, parallel=True, cache=True)
def blend_2d_chunks(src_chunk, dst_chunk, blend_factor, stride):
    """
    blend two 2d chunks using numba with stride
    vectorized implementation
    
    args:
        src_chunk (numpy.ndarray): source chunk (2d)
        dst_chunk (numpy.ndarray): destination chunk to be modified (2d)
        blend_factor (float): blending factor (0.0-1.0)
        stride (int): stride for processing (skip rows)
    """
    inv_blend = 1.0 - blend_factor
    height, width = src_chunk.shape
    
    # get row indices at stride intervals
    row_indices = np.arange(0, height, stride)
    
    # process each stride row
    for y_idx in prange(len(row_indices)):
        y = row_indices[y_idx]
        
        # process all pixels in this row
        for x in range(width):
            dst_chunk[y, x] = min(255, int(dst_chunk[y, x] * inv_blend + src_chunk[y, x] * blend_factor))
        
        # fill in following rows up to next stride
        for offset in range(1, stride):
            if y + offset < height:
                # copy the entire row using slice assignment
                dst_chunk[y + offset, :] = dst_chunk[y, :]

#
# COLORSPACE CONVERSION UTILITIES
#

@jit(nopython=True, cache=True)
def rgb_to_hsv(r, g, b):
    """
    fast rgb->hsv conversion
    """
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    v = max_val
    
    if max_val == min_val:
        return 0.0, 0.0, v
    
    d = max_val - min_val
    s = d / max_val if max_val > 0 else 0
    
    if d == 0: #  avoid destroying the foundations
        return 0.0, s, v
        
    if max_val == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif max_val == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    
    return h/6.0, s, v

@jit(nopython=True, cache=True)
def hsv_to_rgb(h, s, v):
    """
    fast hsv->rgb conversion
    """
    if s == 0:
        return v, v, v
        
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    if h_i == 0:
        return v, t, p
    elif h_i == 1:
        return q, v, p
    elif h_i == 2:
        return p, v, t
    elif h_i == 3:
        return p, q, v
    elif h_i == 4:
        return t, p, v
    else:
        return v, p, q

@jit(nopython=True, parallel=True, cache=False)
def apply_hsv_transform(region, transform_func, params=None):
    """
    apply hsv transformation to an image region
    
    args:
        region: numpy array of shape (height, width, 3) where 3 is r, g, b
        transform_func: function to transform each pixel
        params: additional parameters for the transform function
        
    returns:
        transformed region
    """
    height, width, _ = region.shape
    result = np.empty_like(region)
    
    if params is None:
        params = np.array([0.0])
    
    for y in prange(height):
        for x in range(width):
            r, g, b = region[y, x]
            h, s, v = rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            
            # apply the transformation
            h, s, v = transform_func(h, s, v, params)
            
            # convert back to rgb
            r_new, g_new, b_new = hsv_to_rgb(h, s, v)
            result[y, x, 0] = min(255, int(r_new * 255))
            result[y, x, 1] = min(255, int(g_new * 255))
            result[y, x, 2] = min(255, int(b_new * 255))
    
    return result

#
# BITDEPTH UTILITIES
#

@jit(nopython=True, parallel=True, cache=True)
def process_bit_depth(block, mask):
    """
    apply bit depth reduction to a block
    
    args:
        block (numpy.ndarray): block data to process
        mask (int): bit mask for reduction
    """
    height, width, channels = block.shape
    
    for y in prange(height):
        for x in prange(width):
            for c in range(channels):
                block[y, x, c] = block[y, x, c] & mask

@jit(nopython=True, cache=True)
def process_bit_depth_with_stride(block, mask, stride):
    """
    apply bitdepth reduction with stride for better performance
    
    args:
        block (numpy.ndarray): block data to process
        mask (int): bit mask for reduction
        stride (int): stride for processing
    """
    height, width, channels = block.shape
    
    # get y and x indices at stride intervals
    y_indices = np.arange(0, height, stride)
    x_indices = np.arange(0, width, stride)
    
    # process with stride for better performance
    for y_idx in range(len(y_indices)):
        y = y_indices[y_idx]
        for x_idx in range(len(x_indices)):
            x = x_indices[x_idx]
            
            # process this reference pixel
            for c in range(channels):
                block[y, x, c] = block[y, x, c] & mask
            
            # fill surrounding pixels in the miniblock
            for dy in range(stride):
                if y + dy < height:
                    for dx in range(stride):
                        if x + dx < width and (dy > 0 or dx > 0):
                            for c in range(channels):
                                block[y + dy, x + dx, c] = block[y, x, c] 

@jit(nopython=True, cache=True)
def generate_chunk_indices(height, width, chunk_size=256):
    """
    generate chunk indices for parallel processing.
    
    args:
        height: image height
        width: image width
        chunk_size: size of chunks
        
    returns:
        list of (y_start, y_end, x_start, x_end) tuples
    """
    # calculate how many chunks in each dimension
    num_chunks_y = (height + chunk_size - 1) // chunk_size  
    num_chunks_x = (width + chunk_size - 1) // chunk_size
    
    # create all chunk indices
    chunk_indices = []
    for chunk_y in range(num_chunks_y):
        for chunk_x in range(num_chunks_x):
            y_start = chunk_y * chunk_size
            y_end = min(y_start + chunk_size, height)
            x_start = chunk_x * chunk_size
            x_end = min(x_start + chunk_size, width)
            chunk_indices.append((y_start, y_end, x_start, x_end))
            
    return chunk_indices 