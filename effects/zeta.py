#!/usr/bin/env python3
"""
zeta_invert effect: uses riemann zeta to create stylized color inversions 
"""

import random
import numpy as np
from numba import jit, prange, float32, int32, boolean
from effects.base import Effect
from effects.utils import (
    is_large_image, select_region_type, get_region_params, extract_region, 
    apply_to_region, ROW_REGION, RECT_REGION, COLUMN_REGION, get_stride_for_image
)

# perf constants
MIN_STRIDE = 2             # min stride for large imgs
MAX_STRIDE = 16            # max stride for huge imgs  

@jit(nopython=True, cache=True)
def custom_zeta(x, max_terms=20, offset=0.1):
    """
    zeta function implementation with normalization
    
    args:
        x: input val
        max_terms: series terms to compute
        offset: pole avoidance value
        
    returns:
        normalized zeta output [0-255]
    """
    # avoid poles and add some wiggle room
    x = abs(x) + offset
    
    sum_val = 0.0
    for k in range(1, max_terms + 1):
        sum_val += 1.0 / (k ** x)
    
    # normalize the output to avoid extreme values
    normalized = min(max(sum_val, 0.0), 10.0) / 10.0
    
    # add some non-linearity to make the visual effects more pronounced
    return (normalized ** 0.8) * 255

@jit(nopython=True, parallel=True, cache=True)
def apply_zeta_simple(block, stride, intensity, channel_offsets):
    """
    apply zeta transform to pixel block
    
    args:
        block: input pixels (h, w, 3)
        stride: processing gap
        intensity: effect strength [0-1]
        channel_offsets: per-channel variation values
        
    returns:
        transformed pixel block
    """
    height, width, channels = block.shape
    output = np.zeros_like(block, dtype=np.float32)
    
    # vary max_terms based on intensity for more dramatic effects at high intensity
    max_terms = int(5 + (intensity * 20))  # 5-25 terms
    
    # vary offset based on intensity - closer to pole = more extreme
    base_offset = max(0.05, 0.2 - (intensity * 0.15))  # 0.05-0.2 range
    
    # fixed prange stride but skip based on var stride
    for y in prange(height):
        if y % stride != 0:
            continue
            
        y_end = min(y + stride, height)
        
        for x in range(0, width, stride):
            x_end = min(x + stride, width)
            
            # per-channel processing
            for c in range(channels):
                # get normalized input (0-1 range)
                val = block[y, x, c] / 255.0
                
                # apply channel-specific offset
                channel_offset = base_offset * channel_offsets[c]
                
                # apply zeta with slight variation per channel
                zeta_val = custom_zeta(val, max_terms, channel_offset)
                
                # fill stride block
                for dy in range(y_end - y):
                    for dx in range(x_end - x):
                        output[y + dy, x + dx, c] = zeta_val
    
    return output

class ZetaInvert(Effect):
    """riemann zeta-based inversion with mathematical style"""
    
    def __init__(self):
        """init zeta_invert effect"""
        super().__init__()
        self.name = "zeta_invert"
        self.description = "zeta-powered color inversion with style"
    
    def apply(self, data, width, height, row_length, bytes_per_pixel, intensity):
        """
        apply zeta inversion to image data
        
        args:
            data: raw image data
            width: px width
            height: px height
            row_length: bytes per row
            bytes_per_pixel: channel depth (3 for rgb)
            intensity: effect strength [0.0-1.0+]
            
        returns:
            modified image data
        """
        # non-linear intensity curve
        if intensity > 0:
            if intensity <= 1.0:
                log_intensity = np.power(intensity, 2.0) * 0.85  # squared
            else:
                # log scale for >1.0 intensity
                base_intensity = 0.85
                log_intensity = base_intensity + (np.log10(intensity) * 0.9)
        else:
            log_intensity = 0
        
        # debug info
        print(f"intensity: {intensity:.2f}, log_intensity: {log_intensity:.2f}")
        
        # slight per-channel variation for more interesting color effects
        # different offsets create subtly different transformations per channel
        channel_offsets = np.array([
            random.uniform(0.9, 1.1),  # R offset multiplier
            random.uniform(0.9, 1.1),  # G offset multiplier
            random.uniform(0.9, 1.1)   # B offset multiplier
        ], dtype=np.float32)
        
        # numpy for speed
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        
        # copy to preserve orig
        output_array = img_array.copy()
        
        # region debug
        print(f"\n--- ZETA_INVERT REGION INFO ---")
        print(f"img: {width}x{height} px")
        print(f"intensity: {intensity:.2f}")
        
        # random region type
        region_type = select_region_type(row_probability=0.33, column_probability=0.33)
        
        # region debug
        if region_type == ROW_REGION:
            region_type_str = "ROW"
        elif region_type == COLUMN_REGION:
            region_type_str = "COLUMN"
        else:
            region_type_str = "RECT"
        print(f"region type: {region_type_str}")
        
        # get region params
        y_start, region_height, x_start, region_width = get_region_params(
            height, width, region_type, intensity=log_intensity
        )
        
        # safety bounds check
        region_width = min(width, region_width)
        region_height = min(height, region_height)
        
        # region stats
        height_percent = (region_height / height) * 100
        width_percent = (region_width / width) * 100
        print(f"region: {region_width}x{region_height} px")
        print(f"pct: {width_percent:.1f}% w, {height_percent:.1f}% h")
        print(f"pos: ({x_start}, {y_start})")
        
        # grab region for processing
        block = extract_region(output_array, y_start, region_height, x_start, region_width)
        
        if region_type == ROW_REGION:
            print(f"zeta_invert on rows: {region_height} rows")
        elif region_type == COLUMN_REGION:
            print(f"zeta_invert on col: {region_width} px")
        else:
            print(f"zeta_invert on rect: {region_width}x{region_height}")
        
        try:
            # calc stride based on size
            is_large = is_large_image(region_width, region_height)
            
            # intensity→stride mapping
            base_stride = max(MIN_STRIDE, min(MAX_STRIDE, int(16 - 14 * log_intensity)))
            
            # actual stride
            stride = 1
            if is_large:
                stride = get_stride_for_image(region_width, region_height, base_stride)
                print(f"stride {stride} for {region_width}x{region_height}")
            
            # f32 > f64 for speed
            float_block = block.astype(np.float32)
            
            # apply simplified zeta transformation
            transformed = apply_zeta_simple(float_block, stride, log_intensity, channel_offsets)
            
            # back to uint8
            block[:] = np.clip(transformed, 0, 255).astype(np.uint8)
            
            # copy back to output
            apply_to_region(output_array, block, y_start, x_start)
            
        except Exception as e:
            print(f"zeta_invert error: {str(e)}")
            # bail and return original data on error
            return data
        
        # back to bytearray
        return bytearray(output_array.tobytes()) 