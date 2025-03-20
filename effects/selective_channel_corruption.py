#!/usr/bin/env python3
"""
selective channel corruption effect - messes with rgb channels for color-channel-specific glitch artifacts
"""

import random
import numpy as np
from numba import jit, prange, int32, float32, boolean
from effects.base import Effect
from effects.utils import (
    is_large_image, check_color_match, process_bit_depth_with_stride,
    get_stride_for_image, get_region_params, extract_region, select_region_type,
    RECT_REGION, ROW_REGION, COLUMN_REGION
)

class SelectiveChannelCorruption(Effect):
    """messes with rgb channels for color-channel-specific glitch artifacts"""
    
    def __init__(self):
        """initialize the SelectiveChannelCorruption effect"""
        super().__init__()
        self.name = "selective_channel_corruption"
        self.description = "corrupts specific color channels in defined regions"
    
    def apply(self, data, width, height, row_length, bytes_per_pixel, intensity):
        """apply targeted color-channel corruption to image data.
        
        args:
            data (bytearray): raw image data
            width (int): image width
            height (int): image height
            row_length (int): bytes per row
            bytes_per_pixel (int): bytes per pixel (3 for rgb)
            intensity (float): effect strength 0.0-1.0
            
        returns:
            bytearray: corrupted image data with targeted glitches
        """
        # convert to numpy array for vectorized operations
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        
        # scale corruption amount dramatically with intensity
        # exponential scaling for more dramatic results at high intensity
        if intensity < 0.3:
            base_corruption = int(200 * intensity) # max 60 - subtle shifts
        elif intensity < 0.7:
            base_corruption = int(60 + 200 * intensity) # 80-140 - medium distortion
        else:
            base_corruption = int(150 + 400 * (intensity - 0.7)) # 150-255 - full-on data mangling
            
        # more intensity = more corruption passes
        num_passes = max(1, int(4 * intensity))
        
        # unlock additional corruption modes based on intensity and randomness
        use_inversion = intensity > 0.4 and random.random() < (intensity * 0.8)           # scales from 0-80% chance based on intensity
        multi_channel_corruption = intensity > 0.2 and random.random() < (intensity * 0.9) # scales from 0-90% chance based on intensity
        random_per_pixel = intensity > 0.1 and random.random() < (intensity * 0.7)        # scales from 0-70% chance based on intensity
        use_bit_reduction = intensity > 0.3 and random.random() < (intensity * 0.6)       # scales from 0-60% chance based on intensity
        
        # big images get subsampling because perf was suffering
        is_large = is_large_image(width, height)
        
        # pre-create channel arrays because numpy gets cranky when created in loops
        all_channels = np.array([0, 1, 2], dtype=np.int32)
        
        for _ in range(num_passes):
            # randomly select region type - rectangle, row, or column
            region_type = select_region_type()
            
            # get region parameters based on selected type
            y_start, region_height, x_start, region_width = get_region_params(
                height, width, region_type, intensity
            )
            
            # we're going to corrupt in place for speed
            # (copying large image regions is expensive)
            
            # decide which channels to mess with
            if multi_channel_corruption and random.random() < 0.4:
                # corrupt multiple channels with different values
                num_channels = random.randint(2, 3)
                
                # choose random channels without duplicates
                channels = np.random.choice(all_channels, size=num_channels, replace=False)
                
                # add some variation to corruption values
                variations = np.random.uniform(0.7, 1.3, size=num_channels)
                corruption_values = (base_corruption * variations).astype(np.int32)
            else:
                # just corrupt one lonely channel
                channel = random.randint(0, 2)
                channels = np.array([channel], dtype=np.int32)
                corruption_values = np.array([base_corruption], dtype=np.int32)
            
            # target a specific color range - more surgical corruption
            target_color = random.randint(0, 5)  # matches various color ranges
            threshold = random.randint(100, 200)  # how strict the color matching is
            
            # determine per-pixel randomness
            rand_per_pixel = random_per_pixel and random.random() < 0.5
            rand_amount = random.randint(10, 50) if rand_per_pixel else 0
            
            # bit depth reduction - the 8-bit retro look
            if use_bit_reduction and random.random() < 0.3:
                # create a bit mask (higher intensity = fewer bits = chunkier look)
                bit_mask = ~((1 << random.randint(3, 6)) - 1)  # bitwise wizardry
                apply_bit_reduction = True
            else:
                bit_mask = 0xFF  # full 8 bits preserved
                apply_bit_reduction = False
            
            # choose inversion sometimes for that negative film look
            use_invert_for_this = use_inversion and random.random() < 0.3
            
            # optimize with stride for large images - process blocks instead of pixels
            stride = get_stride_for_image(region_width, region_height) if is_large else 1
            
            # package region parameters
            region_params = (y_start, region_height, x_start, region_width)
            
            # call the optimized function with all our chaos parameters
            # (this took forever to tune without crashing numba)
            corrupt_image(
                img_array, region_params, channels, corruption_values, target_color, 
                threshold, stride, use_invert_for_this, rand_per_pixel, 
                rand_amount, apply_bit_reduction, bit_mask
            )
        
        # convert back to bytearray
        return bytearray(img_array.tobytes())

@jit(nopython=True, cache=True)
def corrupt_image(img_array, region_params, channels, corruption_values, target_color, threshold, 
                 stride, use_inversion, rand_per_pixel, rand_amount, apply_bit_reduction, bit_mask):
    """
    optimized corruption engine - does the actual pixel manipulation
    had to rebuild this like 5 times to make numba happy
    """
    y_start, region_height, x_start, region_width = region_params
    height, width, _ = img_array.shape
    
    # pre-calculate parameters
    num_channels = len(channels)
    y_max = height - 1
    
    # process with stride for better performance
    for y in range(y_start, y_start + region_height):
        # calculate row-based randomness once per row for consistent patterns
        # prime numbers create more visually interesting patterns, i think
        row_offset = (y * 17) % 30 if rand_per_pixel else 0
        
        for x in range(x_start, x_start + region_width):
            # get pixel values
            r, g, b = img_array[y, x, 0], img_array[y, x, 1], img_array[y, x, 2]
            
            # check if this pixel's color should be corrupted
            if check_color_match(r, g, b, target_color, channels[0], threshold):
                # calculate pixel random component if needed
                # again with the primes for that sweet chaotic look
                pixel_rand = ((x * 13 + y * 7) % rand_amount) - (rand_amount // 2) if rand_per_pixel else 0
                
                # process all channels we're targeting
                for i in range(num_channels):
                    channel = channels[i]
                    corruption = corruption_values[i]
                    curr_val = img_array[y, x, channel]
                    
                    # apply corruption method
                    if use_inversion:
                        new_val = 255 - curr_val  # photographic negative effect
                    else:
                        offset = corruption + pixel_rand + row_offset
                        new_val = (curr_val + offset) % 256  # wrap around for clean corruption
                    
                    # apply bit reduction if enabled (quantization effect)
                    if apply_bit_reduction:
                        new_val = new_val & bit_mask  # bitwise AND for bit reduction
                    
                    # set value and propagate to stride area for consistency
                    img_array[y, x, channel] = new_val
                    
                    # fill stride block efficiently
                    for dy in range(1, min(stride, y_max - y + 1)):
                        img_array[y + dy, x, channel] = new_val