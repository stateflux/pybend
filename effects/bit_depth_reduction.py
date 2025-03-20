#!/usr/bin/env python3
"""
bitdepth-reduction effect that reduces color depth resulting in noisy posterization artifacts
"""

import random
import numpy as np
from numba import jit, prange
from effects.base import Effect
from effects.utils import (
    get_region_params, extract_region, apply_to_region, RECT_REGION
)

class BitDepthReduction(Effect):
    """bitdepth-reduction effect that reduces color depth resulting in posterization effect"""
    
    def __init__(self):
        """initialize the BitDepthReduction effect"""
        super().__init__()
        self.name = "bit_depth_reduction"
        self.description = "reduces color bitdepth in blocks"
    
    def apply(self, data, width, height, row_length, bytes_per_pixel, intensity):
        """apply the bitdepth-reduction effect to the raw image data
        
        args:
            data (bytearray): the raw image data to modify
            width (int): image width in pixels
            height (int): image height in pixels
            row_length (int): number of bytes per row
            bytes_per_pixel (int): number of bytes per pixel (3 for rgb)
            intensity (float): effect intensity
            
        returns:
            bytearray: the modified image data
        """
        # convert to numpy array for faster processing
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        
        # 2-8 bits (lower = more reduced)
        min_bits = 2  # minimum 2 bits at max intensity
        levels = max(min_bits, int(8 - (6 * intensity)))
        # kill potential overflow: ensure mask is within 0-255 range
        mask = min(255, (256 - (1 << levels)))
        
        # 1-5 blocks
        num_blocks = max(1, int(5 * intensity))
        
        for _ in range(num_blocks):
            # get a rectangular region for bitdepth reduction
            y_start, region_height, x_start, region_width = get_region_params(
                height, width, RECT_REGION, intensity
            )
            
            # extract the block
            block = extract_region(img_array, y_start, region_height, x_start, region_width)
            
            # apply bitdepth reduction
            block &= mask
                
        # convert back to bytearray
        return bytearray(img_array.tobytes()) 