#!/usr/bin/env python3
"""
mirror effect: selects image regions and flips them to create mirror-like glitches
"""

import random
import numpy as np
from numba import jit, prange
from effects.base import Effect
from effects.utils import (
    is_large_image, get_stride_for_image, select_region_type,
    get_region_params, extract_region, apply_to_region,
    ROW_REGION, COLUMN_REGION, RECT_REGION
)

class Mirror(Effect):
    """creates mirror-like effects by flipping regions of the image along random axes"""
    
    def __init__(self):
        """initialize the Mirror effect"""
        super().__init__()
        self.name = "mirror"
        self.description = "flips regions of the image to create mirror-like effects"
    
    def apply(self, data, width, height, row_length, bytes_per_pixel, intensity):
        """apply the mirror effect to the raw image data."""
        # convert to numpy array for easier manipulation
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        
        # use a fixed base number of regions (1) 
        # note: the "strength" parameter in bend.py will act as a multiplier for this
        # so strength=1.0 gives 1 region, strength=2.0 gives 2 regions
        num_regions = 1
        
        # determine maximum region size based on intensity
        # higher intensity = larger potential regions
        max_size_pct = min(0.8, 0.3 + (intensity * 0.5))
        
        # check if this is a large image for performance optimization
        is_large = is_large_image(width, height)
        
        print(f"applying mirror effect with {num_regions} region, max size: {max_size_pct:.2f}")
        
        # process each region
        for i in range(num_regions):
            # select region type with equal probability
            region_type = select_region_type()
            
            # get region parameters
            y_start, region_height, x_start, region_width = get_region_params(
                height, width, region_type, intensity
            )
            
            # extract the region
            region = extract_region(img_array, y_start, region_height, x_start, region_width)
            
            # choose flip axis based on region type to ensure proper mirroring effect
            # ROW_REGION: vertical flip only (1) - creates mirror images above/below
            # COLUMN_REGION: horizontal flip only (0) - creates mirror images left/right
            # RECT_REGION: either horizontal or vertical flip (random)
            if region_type == ROW_REGION:
                # rows should be flipped vertically to create mirror effect above/below
                flip_axis = 1
            elif region_type == COLUMN_REGION:
                # columns should be flipped horizontally to create mirror effect left/right
                flip_axis = 0
            else:
                # rectangular regions can be flipped either way
                flip_axis = random.randint(0, 1)
            
            # determine destination coordinates for the mirrored region
            dest_x, dest_y = get_mirror_destination(
                x_start, y_start, region_width, region_height, 
                width, height, flip_axis
            )
            
            # create the mirrored region
            mirrored_region = create_mirrored_region(region, flip_axis)
            
            # apply the mirrored region to the image
            apply_to_region(img_array, mirrored_region, dest_y, dest_x)
            
        # convert back to bytearray
        return bytearray(img_array.tobytes())


@jit(nopython=True, cache=True)
def get_mirror_destination(x, y, region_width, region_height, img_width, img_height, flip_axis):
    """
    determine destination coordinates for the mirrored region.
    
    args:
        x, y: origin coordinates of the source region
        region_width, region_height: dimensions of the region
        img_width, img_height: dimensions of the full image
        flip_axis: axis to flip (0=horizontal flip, 1=vertical flip)
        
    returns:
        (dest_x, dest_y): coordinates where the mirrored region should be placed
    """
    if flip_axis == 0:  # horizontal flip (place to the right or left)
        if x + region_width * 2 <= img_width:
            # there's room to the right
            dest_x = x + region_width
            dest_y = y
        elif x - region_width >= 0:
            # there's room to the left
            dest_x = x - region_width
            dest_y = y
        else:
            # not enough room, just overlay
            dest_x = max(0, min(x, img_width - region_width))
            dest_y = y
    else:  # vertical flip (place below or above)
        if y + region_height * 2 <= img_height:
            # there's room below
            dest_x = x
            dest_y = y + region_height
        elif y - region_height >= 0:
            # there's room above
            dest_x = x
            dest_y = y - region_height
        else:
            # not enough room, just overlay
            dest_x = x
            dest_y = max(0, min(y, img_height - region_height))
    
    return dest_x, dest_y


@jit(nopython=True, cache=True)
def create_mirrored_region(region, flip_axis):
    """
    create a mirrored copy of the region
    
    args:
        region: source region as numpy array
        flip_axis: axis to flip (0=horizontal, 1=vertical)
        
    returns:
        mirrored region as numpy array
    """
    # create a copy to avoid modifying the original
    mirrored = region.copy()
    
    height, width, _ = region.shape
    
    if flip_axis == 0:  # horizontal flip
        # mirror each row
        for y in range(height):
            for x in range(width // 2):
                # swap pixels
                mirrored[y, x, :] = region[y, width - 1 - x, :]
                mirrored[y, width - 1 - x, :] = region[y, x, :]
    else:  # vertical flip
        # mirror each column
        for x in range(width):
            for y in range(height // 2):
                # swap pixels
                mirrored[y, x, :] = region[height - 1 - y, x, :]
                mirrored[height - 1 - y, x, :] = region[y, x, :]
    
    return mirrored 