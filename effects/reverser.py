#!/usr/bin/env python3
"""
reverser effect - reverses chunks of image data, flipping them
"""

import random
import numpy as np
from effects.base import Effect
from effects.utils import select_region_type, get_region_params, ROW_REGION

class Reverser(Effect):
    """reverses chunks of image data, flipping them"""
    
    def __init__(self):
        """initialize the Reverser effect"""
        super().__init__()
        self.name = "reverser"
        self.description = "reverses byte patterns in rows or blocks"
    
    def apply(self, data, width, height, row_length, bytes_per_pixel, intensity):
        """apply byte reversal to parts of the image
        
        args:
            data (bytearray): raw image data
            width (int): image width
            height (int): image height
            row_length (int): bytes per row
            bytes_per_pixel (int): bytes per pixel (3 for rgb)
            intensity (float): effect strength 0.0-1.0
            
        returns:
            bytearray: corrupted image data
        """
        # convert to 1d numpy array for faster manipulation
        data_array = np.frombuffer(data, dtype=np.uint8).copy()
        
        # skip headers because 1d
        start_pos = 100
        
        # bias towards row-based reversing because it's cooler
        region_type = select_region_type(row_probability=0.7)  # 70% chance of row-based reversing
        
        if region_type == ROW_REGION:
            # get parameters for the row region
            y_start, region_height, _, _ = get_region_params(height, width, region_type, intensity)
            
            # calculate byte range to flip
            region_start = start_pos + (y_start * row_length)
            region_size = region_height * row_length
            region_end = region_start + region_size
            
            if region_end <= len(data_array):
                #reverse this chunk using numpy array
                data_array[region_start:region_end] = data_array[region_start:region_end][::-1]
        else:
            # need to work with the 3d structure for rectangles
            img_array = data_array.reshape((height, width, bytes_per_pixel))
            
            # get parameters for the rectangular region
            y_start, region_height, x_start, region_width = get_region_params(height, width, region_type, intensity)
            
            # process each row in the block one by one
            for y in range(y_start, min(y_start + region_height, height)):
                # extract the segment and reverse it
                segment = img_array[y, x_start:x_start+region_width, :]
                # flatten, reverse, and reshape back - the numpy shuffle
                flat_segment = segment.flatten()
                reversed_segment = flat_segment[::-1]
                # put it back where we found it but backwards
                img_array[y, x_start:x_start+region_width, :] = reversed_segment.reshape(segment.shape)
        
        # convert back to bytearray
        return bytearray(data_array) 