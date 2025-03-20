#!/usr/bin/env python3
"""
chunk-duplication effect that duplicates chunks of image data, creating repeating patterns
"""

import random
import numpy as np
from numba import jit, prange
from effects.base import Effect
from effects.utils import is_large_image, blend_chunks_with_stride, blend_2d_chunks, get_stride_for_image

class ChunkDuplication(Effect):
    """chunk-duplication effect that duplicates chunks of image data, creating repeating patterns"""
    
    def __init__(self):
        """initialize the ChunkDuplication effect."""
        super().__init__()
        self.name = "chunk_duplication"
        self.description = "duplicates chunks of image data"
    
    def apply(self, data, width, height, row_length, bytes_per_pixel, intensity):
        """Apply the chunk duplication effect to the raw image data.
        
        args:
            data (bytearray): raw image data
            width (int): image width in pixels
            height (int): image height in pixels
            row_length (int): number of bytes per row
            bytes_per_pixel (int): number of bytes per pixel (3 for rgb)
            intensity (float): effect intensity from 0.0 to 1.0
            
        returns:
            bytearray: the modified image data
        """
        # convert to numpy for faster operations
        data_array = np.frombuffer(data, dtype=np.uint8).copy()
        
        # safe starting position (avoid headers)
        start_pos = 100
        
        # row-based chunk duplication
        min_chunk_size = row_length
        
        # max chunk size (0.5% to 3% of image - more can look bad)
        max_chunk_pct = 0.005 + (0.025 * intensity)
        max_chunk_size = int(row_length * height * max_chunk_pct)
        max_chunk_size = max(min_chunk_size, max_chunk_size)
        
        # 1-2 duplications based on intensity
        num_duplications = max(1, int(2 * intensity))
        
        # for large images, limit chunk size
        if is_large_image(width, height):
            max_chunk_size = min(max_chunk_size, row_length * 50)  # limit to 50 rows
        
        for _ in range(num_duplications):
            # chunk size multiple of row length
            chunk_size = min_chunk_size * random.randint(1, max(1, max_chunk_size // min_chunk_size))
            
            if len(data_array) - start_pos - chunk_size > 0:
                src_pos = start_pos + (random.randint(0, (len(data_array) - start_pos - chunk_size) // min_chunk_size) * min_chunk_size)
                dst_pos = start_pos + (random.randint(0, (len(data_array) - start_pos - chunk_size) // min_chunk_size) * min_chunk_size)
                
                # blend or replace
                blend_prob = 0.5 * intensity
                
                if random.random() < blend_prob:
                    # extract chunks
                    src_chunk = data_array[src_pos:src_pos+chunk_size]
                    dst_chunk = data_array[dst_pos:dst_pos+chunk_size]
                    
                    # determine if we should use normal blend or numba-accelerated blend
                    large_chunk = chunk_size > 1000000
                    
                    if large_chunk:
                        # calculate adaptive stride based on chunk size
                        chunk_pixels = chunk_size // bytes_per_pixel
                        stride = get_stride_for_image(
                            int(chunk_pixels**0.5), int(chunk_pixels**0.5),
                            base_stride=2
                        )
                        
                        blend_factor = 0.2 + (0.6 * intensity)
                        
                        # for oversized chunks, reshape to 2d for better processing
                        if chunk_size > 5000000:
                            # process as 2d array with subsampling
                            chunk_height = chunk_size // row_length
                            src_reshaped = src_chunk.reshape(chunk_height, row_length)
                            dst_reshaped = dst_chunk.reshape(chunk_height, row_length)
                            
                            # use numba-accelerated 2d blending from shared utils
                            blend_2d_chunks(src_reshaped, dst_reshaped, blend_factor, stride)
                        else:
                            # use numba-accelerated 1d blending from shared utils
                            blend_chunks_with_stride(src_chunk, dst_chunk, blend_factor, stride)
                        
                        # update data
                        data_array[dst_pos:dst_pos+chunk_size] = dst_chunk
                    else:
                        # for smaller chunks, use vectorized operations
                        blend_factor = 0.2 + (0.6 * intensity)
                        inv_blend = 1.0 - blend_factor
                        
                        # blend using vectorized operations
                        blended = np.clip(
                            (dst_chunk.astype(np.float32) * inv_blend + 
                             src_chunk.astype(np.float32) * blend_factor),
                            0, 255).astype(np.uint8)
                        
                        # update data
                        data_array[dst_pos:dst_pos+chunk_size] = blended
                else:
                    # direct replacement (already fast with numpy views)
                    data_array[dst_pos:dst_pos+chunk_size] = data_array[src_pos:src_pos+chunk_size]
        
        # convert back to bytearray
        return bytearray(data_array) 