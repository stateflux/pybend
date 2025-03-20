#!/usr/bin/env python3
"""
phantom regions effect - creates color-channel separations for the classic glitchart "ghost object" effect
"""

import random
import math
import numpy as np
from numba import jit, prange, float32, int32, boolean
from effects.base import Effect
from effects.utils import (
    is_large_image, select_region_type, get_region_params, extract_region, 
    apply_to_region, ROW_REGION, RECT_REGION, COLUMN_REGION, get_stride_for_image,
    generate_chunk_indices, rgb_to_hsv, hsv_to_rgb, apply_hsv_transform,
    get_column_region_params, get_row_region_params, get_rect_region_params
)

# different ways we can mess with the channels
OFFSET_ONLY = 0
MIRROR_HORIZONTAL = 1
MIRROR_VERTICAL = 2
ROTATE_90 = 3
ROTATE_180 = 4
SCALE = 5

@jit(nopython=True, cache=True)
def calculate_region_brightness(region):
    """
    calculates average brightness of a region - brighter regions get stronger phantom effects
    
    args:
        region: numpy array of region data (h, w, 3)
        
    returns:
        float: average brightness (0.0-1.0)
    """
    # bail with default value if region is invalid
    if region.size == 0:
        return 0.5
        
    # average all color channels and normalize to 0-1
    return np.mean(region) / 255.0

@jit(nopython=True, cache=True)
def get_channel_offset(region_brightness, max_offset_pct, image_dim, intensity):
    """
    calculates how far to shift phantom channels based on brightness and intensity
    brighter regions get more extreme displacement via logarithmic scaling
    
    args:
        region_brightness: brightness of region (0.0-1.0)
        max_offset_pct: maximum offset as percentage of image dimension
        image_dim: relevant image dimension (width or height)
        intensity: effect intensity for logarithmic scaling
        
    returns:
        int: pixel offset amount (can be negative)
    """
    # scale offset based on brightness (more brightness = more offset)
    # add some randomness because deterministic effects are boring
    offset_factor = region_brightness * (0.5 + random.random())
    
    # apply logarithmic scaling based on intensity
    # higher intensity = exponentially larger offsets
    log_scale = 1.0 + math.log(1.0 + intensity * 10.0) / math.log(11.0)
    max_offset = int(image_dim * max_offset_pct * log_scale)
    
    # ensure minimum offset so it's actually visible
    min_offset = max(2, int(max_offset * 0.1))
    
    # calculate final offset
    offset = int(min_offset + (max_offset - min_offset) * offset_factor)
    
    # randomly go left/right or up/down
    if random.random() < 0.5:
        offset = -offset
        
    return offset

@jit(nopython=True, cache=True)
def extract_channel(region, channel):
    """
    isolates a single rgb channel from a region
    
    args:
        region: numpy array of region data (h, w, 3)
        channel: channel to extract (0=r, 1=g, 2=b)
        
    returns:
        numpy array with only the specified channel active
    """
    height, width, _ = region.shape
    result = np.zeros_like(region)
    
    # copy only the selected channel, zero out the others
    result[:, :, channel] = region[:, :, channel]
    
    return result

@jit(nopython=True, cache=True)
def apply_phantom_transform(region, channel, transform_type, h_offset, v_offset, alpha, width, height, intensity):
    """
    applies one of several transformations to create phantom effect on a single color channel
    can offset, mirror, rotate or scale the channel
    
    args:
        region: numpy array of region data (h, w, 3)
        channel: color channel to isolate (0=r, 1=g, 2=b)
        transform_type: type of transform to apply (see constants)
        h_offset: horizontal offset in pixels
        v_offset: vertical offset in pixels
        alpha: transparency level (0.0-1.0)
        width: region width
        height: region height
        intensity: effect intensity (0.0-1.0)
        
    returns:
        transformed region as numpy array
    """
    # extract just one channel
    channel_data = extract_channel(region, channel)
    
    # create output buffer (same size as input to start)
    result = np.zeros_like(region)
    result_height, result_width, _ = result.shape
    
    # determine threshold for this channel (only apply to pixels with enough color)
    # higher intensity = lower threshold (more pixels affected)
    # much higher thresholds to create more interesting shapes
    base_threshold = 180 - int(intensity * 80)  # 180 at intensity 0, 100 at intensity 1.0
    
    # each channel needs different thresholds or they look bad
    if channel == 0:  # red - slightly lower threshold (red is visually dominant)
        threshold = max(90, base_threshold - 10)
    elif channel == 1:  # green - medium threshold
        threshold = max(100, base_threshold)
    else:  # blue - higher threshold (blue is less perceptually strong)
        threshold = max(110, base_threshold + 10)
    
    # high intensity = extra contrast boost
    if intensity > 0.7:
        # make only the brightest parts of the channel visible
        # exponential contrast boost based on intensity
        contrast_boost = 1.0 + (intensity - 0.7) * 3.0  # 1.0 at intensity 0.7, up to 1.9 at intensity 1.0
        channel_data = np.copy(channel_data)
        
        # boost values above threshold, reduce values below
        for y in range(height):
            for x in range(width):
                val = channel_data[y, x, channel]
                if val > threshold:
                    # boost values above threshold
                    channel_data[y, x, channel] = min(255, int(val * contrast_boost))
                else:
                    # reduce values below threshold even more
                    channel_data[y, x, channel] = int(val * 0.7)
    
    # apply different transformations based on type
    # there are 6 different transformations - now i hate numba
    if transform_type == OFFSET_ONLY:
        # simple offset without other transformations
        for y in range(height):
            for x in range(width):
                # check if the pixel has enough color to be worth phantoming
                if channel_data[y, x, channel] < threshold:
                    continue
                    
                # calculate destination coordinates with offset
                dest_y = y + v_offset
                dest_x = x + h_offset
                
                # check if destination is within bounds
                if 0 <= dest_y < result_height and 0 <= dest_x < result_width:
                    # apply a stronger alpha to make it more visible (1.2-1.5x)
                    # we'll clamp to 255 later when blending
                    boosted_alpha = min(1.0, alpha * 1.2)
                    result[dest_y, dest_x, channel] = int(channel_data[y, x, channel] * boosted_alpha)
    
    elif transform_type == MIRROR_HORIZONTAL:
        # mirror horizontally and offset
        for y in range(height):
            for x in range(width):
                # mirror x coordinate
                src_x = width - 1 - x
                
                # check if the pixel has enough color to be worth phantoming
                if channel_data[y, src_x, channel] < threshold:
                    continue
                
                # calculate destination coordinates with offset
                dest_y = y + v_offset
                dest_x = x + h_offset
                
                # check if source and destination are within bounds
                if 0 <= src_x < width and 0 <= dest_y < result_height and 0 <= dest_x < result_width:
                    # apply a stronger alpha to make it more visible (1.2-1.5x)
                    # we'll clamp to 255 later when blending
                    boosted_alpha = min(1.0, alpha * 1.2)
                    result[dest_y, dest_x, channel] = int(channel_data[y, src_x, channel] * boosted_alpha)
    
    elif transform_type == MIRROR_VERTICAL:
        # mirror vertically and offset
        for y in range(height):
            for x in range(width):
                # mirror y coordinate
                src_y = height - 1 - y
                
                # check if the pixel has enough color to be worth phantoming
                if channel_data[src_y, x, channel] < threshold:
                    continue
                
                # calculate destination coordinates with offset
                dest_y = y + v_offset
                dest_x = x + h_offset
                
                # check if source and destination are within bounds
                if 0 <= src_y < height and 0 <= dest_y < result_height and 0 <= dest_x < result_width:
                    # apply a stronger alpha to make it more visible (1.2-1.5x)
                    # we'll clamp to 255 later when blending
                    boosted_alpha = min(1.0, alpha * 1.2)
                    result[dest_y, dest_x, channel] = int(channel_data[src_y, x, channel] * boosted_alpha)
    
    elif transform_type == ROTATE_90:
        # rotate 90 degrees and offset - the forbidden geometry
        for y in range(height):
            for x in range(width):
                # rotate coordinates 90 degrees clockwise
                src_y = x
                src_x = height - 1 - y
                
                # check if the pixel has enough color to be worth phantoming
                if 0 <= src_y < height and 0 <= src_x < width:
                    if channel_data[src_y, src_x, channel] < threshold:
                        continue
                
                # calculate destination coordinates with offset
                dest_y = y + v_offset
                dest_x = x + h_offset
                
                # check if source and destination are within bounds
                if (0 <= src_y < height and 0 <= src_x < width and 
                    0 <= dest_y < result_height and 0 <= dest_x < result_width):
                    # apply a stronger alpha to make it more visible (1.2-1.5x)
                    # we'll clamp to 255 later when blending
                    boosted_alpha = min(1.0, alpha * 1.2)
                    result[dest_y, dest_x, channel] = int(channel_data[src_y, src_x, channel] * boosted_alpha)
    
    elif transform_type == ROTATE_180:
        # rotate 180 degrees and offset - the ol' upsidedown
        for y in range(height):
            for x in range(width):
                # rotate coordinates 180 degrees
                src_y = height - 1 - y
                src_x = width - 1 - x
                
                # check if the pixel has enough color to be worth phantoming
                if 0 <= src_y < height and 0 <= src_x < width:
                    if channel_data[src_y, src_x, channel] < threshold:
                        continue
                
                # calculate destination coordinates with offset
                dest_y = y + v_offset
                dest_x = x + h_offset
                
                # check if source and destination are within bounds
                if (0 <= src_y < height and 0 <= src_x < width and 
                    0 <= dest_y < result_height and 0 <= dest_x < result_width):
                    # apply a stronger alpha to make it more visible (1.2-1.5x)
                    # we'll clamp to 255 later when blending
                    boosted_alpha = min(1.0, alpha * 1.2)
                    result[dest_y, dest_x, channel] = int(channel_data[src_y, src_x, channel] * boosted_alpha)
    
    elif transform_type == SCALE:
        # scale the region - the funhouse mirror approach
        scale = 0.7 + random.random() * 0.6  # 0.7-1.3
        
        # calculate dimensions after scaling
        scaled_height = int(height * scale)
        scaled_width = int(width * scale)
        
        # center offset for scaling
        center_y_offset = (height - scaled_height) // 2
        center_x_offset = (width - scaled_width) // 2
        
        # apply scaling transformation
        for y in range(scaled_height):
            for x in range(scaled_width):
                # map to original coordinates (prevent divide by zero)
                if scaled_height > 0 and scaled_width > 0:
                    src_y = int(y * height / scaled_height)
                    src_x = int(x * width / scaled_width)
                else:
                    # fallback to simple identity mapping if scaled dimensions are zero
                    src_y = min(y, height - 1)
                    src_x = min(x, width - 1)
                
                # check if the pixel has enough color to be worth phantoming
                if 0 <= src_y < height and 0 <= src_x < width:
                    if channel_data[src_y, src_x, channel] < threshold:
                        continue
                
                # calculate destination coordinates with offset
                dest_y = y + center_y_offset + v_offset
                dest_x = x + center_x_offset + h_offset
                
                # check if source and destination are within bounds
                if (0 <= src_y < height and 0 <= src_x < width and 
                    0 <= dest_y < result_height and 0 <= dest_x < result_width):
                    # apply a stronger alpha to make it more visible (1.2-1.5x)
                    # we'll clamp to 255 later when blending
                    boosted_alpha = min(1.0, alpha * 1.2)
                    result[dest_y, dest_x, channel] = int(channel_data[src_y, src_x, channel] * boosted_alpha)
    
    return result

@jit(nopython=True, cache=True)
def create_phantom_region(original_region, img_width, img_height, intensity):
    """
    creates ghost-like effects by isolating and offsetting rgb channels
    the core of the phantom effect - separates colors into transparent overlapping layers
    
    args:
        original_region: numpy array of original region data
        img_width: full image width (for scaling offset)
        img_height: full image height (for scaling offset)
        intensity: effect intensity (0.0-1.0)
        
    returns:
        transformed region with phantom effects
    """
    # get region dimensions
    height, width, _ = original_region.shape
    
    # calculate average brightness of region
    brightness = calculate_region_brightness(original_region)
    
    # create output buffer initialized with original data
    result = original_region.copy()
    
    # roll the dice - will we do multi-channel or single channel?
    rand_val = random.random()
    if rand_val < 0.66:  # 66% chance for multi-channel
        # Multi-channel mode
        # do we use all three channels or just two?
        if random.random() < 0.5:  # 50% chance for all three
            channels = [0, 1, 2]  # R, G, B - the gang's all here
        else:
            # pick two random channels to mess with
            all_channels = [0, 1, 2]
            # kick one random channel out of the club
            to_remove = random.randint(0, 2)
            all_channels.pop(to_remove)
            channels = all_channels  # two channels remain
    else:
        # single-channel mode - one color to rule them all
        # try to find the dominant channel in the region for maximum visual impact
        avg_r = np.mean(original_region[:, :, 0])
        avg_g = np.mean(original_region[:, :, 1])
        avg_b = np.mean(original_region[:, :, 2])
        
        # add some randomness so it's not always the same
        r_score = avg_r * (0.8 + random.random() * 0.4)  # 80-120% of actual value
        g_score = avg_g * (0.8 + random.random() * 0.4)
        b_score = avg_b * (0.8 + random.random() * 0.4)
        
        # get max channel, but if they're all very close, just pick randomly
        max_score = max(r_score, g_score, b_score)
        min_score = min(r_score, g_score, b_score)
        
        # if there's a significant difference (more than 15%), use the dominant channel
        if max_score > 0 and (max_score - min_score) / max_score > 0.15:
            if r_score >= g_score and r_score >= b_score:
                channels = [0]  # red is the boss
            elif g_score >= r_score and g_score >= b_score:
                channels = [1]  # green machine
            else:
                channels = [2]  # blue crew
        else:
            # no clear dominant channel, pick randomly
            channel_weights = [1/3, 1/3, 1/3]
            rand_val = random.random() 
            
            if rand_val < channel_weights[0]:
                channels = [0]  # r
            elif rand_val < channel_weights[0] + channel_weights[1]:
                channels = [1]  # g
            else:
                channels = [2]  # b
    
    # maximum offset as percentage of dimension (scaled by intensity)
    max_offset_pct = 0.05 + (intensity * 0.15)  # 5-20% of dimension
    
    # for each channel we're processing
    for channel in channels:
        # calculate horizontal and vertical offsets based on brightness and randomness
        h_offset = get_channel_offset(brightness, max_offset_pct, img_width, intensity)
        v_offset = get_channel_offset(brightness, max_offset_pct, img_height, intensity)
        
        # determine alpha value with logarithmic scaling (80-100% opacity)
        # higher intensity = exponentially higher alpha
        base_alpha = 0.8 + (random.random() * 0.2)  # 80-100% base
        
        # apply logarithmic scaling to alpha too
        log_scale = 1.0 + math.log(1.0 + intensity * 8.0) / math.log(9.0)  # steeper scaling
        alpha = min(1.0, base_alpha * log_scale)  # cap at 1.0
        
        # determine transform type based on intensity and channel
        # assign specific transformations to specific channels for a more
        # consistent, designed look rather than random
        if channel == 0:  # red channel
            if intensity > 0.7:
                transform_type = MIRROR_HORIZONTAL
            else:
                transform_type = OFFSET_ONLY
        elif channel == 1:  # green channel
            if intensity > 0.5:
                transform_type = MIRROR_VERTICAL
            else:
                transform_type = OFFSET_ONLY
        else:  # blue channel
            if intensity > 0.6:
                transform_type = SCALE
            else:
                transform_type = OFFSET_ONLY
        
        # apply the transformation
        phantom = apply_phantom_transform(
            original_region, channel, transform_type, 
            h_offset, v_offset, alpha, width, height, intensity
        )
        
        # blend the phantom with the result (simple additive blending for transparency)
        for y in range(height):
            for x in range(width):
                # only affect the current channel
                # add phantom effect (capped at 255)
                # boost the phantom effect by multiplying it
                phantom_value = phantom[y, x, channel]
                boosted_value = min(255, int(phantom_value * 1.3))  # boost by 30%
                result[y, x, channel] = min(255, result[y, x, channel] + boosted_value)
    
    return result

class PhantomRegions(Effect):
    """creates ghost-like separated color channels that make reality look broken"""
    
    def __init__(self):
        """initialize the phantom regions effect"""
        super().__init__()
        self.name = "phantom_regions"
        self.description = "creates ghosting effects with transparent colored shadows"
    
    def apply(self, data, width, height, row_length, bytes_per_pixel, intensity):
        """apply phantom regions effect to image data.
        
        args:
            data (bytearray): raw image data
            width (int): image width
            height (int): image height
            row_length (int): bytes per row
            bytes_per_pixel (int): bytes per pixel (3 for rgb)
            intensity (float): effect strength 0.0-1.0
            
        returns:
            bytearray: modified image data with glitchy phantom regions
        """
        # sanity check dimensions
        if width <= 0 or height <= 0:
            return data
            
        # convert to numpy array for processing
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        result_array = img_array.copy()
        
        # keep it simple - 1 or 2 regions at most, but make them count
        if intensity < 0.5 or random.random() < 0.7:  # 70% chance of single region
            num_regions = 1
        else:
            num_regions = 2  # max of 2 regions
        
        # more aggressive size scaling with intensity
        # low intensity = small ghosts
        # high intensity = THE GHOST DIMENSION COMETH
        size_boost = 1.0 + math.pow(intensity, 0.7) * 2.0  # dramatic scaling
        effective_intensity = min(1.0, intensity * size_boost)
        
        # dynamic size bounds based on intensity
        # at low intensity: small subtle regions
        # at high intensity: huge regions covering most of the image
        min_size_pct_base = 0.1 + (intensity * 0.3)  # 10-40% at min
        max_size_pct_base = 0.4 + (intensity * 0.5)  # 40-90% at max
        
        # for each region
        for i in range(num_regions):
            # select region type with stronger bias toward rectangles
            if random.random() < 0.7:  # 70% chance for rectangles
                region_type = RECT_REGION
            else:
                # rows and columns with equal probability
                region_type = ROW_REGION if random.random() < 0.5 else COLUMN_REGION
            
            # get region parameters with custom sizes - dynamic sizing by intensity
            if region_type == RECT_REGION:
                # for rectangles, use larger size percentage
                y_start, region_height, x_start, region_width = get_rect_region_params(
                    height, width, 
                    min_size_pct=min_size_pct_base,
                    max_size_pct=max_size_pct_base,
                    intensity=effective_intensity
                )
            elif region_type == ROW_REGION:
                # for rows, use larger height
                y_start, region_height, x_start, region_width = get_row_region_params(
                    height, width,
                    min_height_pct=min_size_pct_base,
                    max_height_pct=max_size_pct_base,
                    intensity=effective_intensity
                )
            else:  # COLUMN_REGION
                # for columns, use larger width
                y_start, region_height, x_start, region_width = get_column_region_params(
                    height, width,
                    min_width_pct=min_size_pct_base,
                    max_width_pct=max_size_pct_base,
                    intensity=effective_intensity
                )
            
            # safety check region dimensions
            if region_height <= 0 or region_width <= 0:
                continue
            
            # extract region
            region = extract_region(img_array, y_start, region_height, x_start, region_width)
            
            # summon the phantoms
            phantom_region = create_phantom_region(region, width, height, intensity)
            
            # apply back to result
            apply_to_region(result_array, phantom_region, y_start, x_start)
        
        # convert back to bytearray
        return bytearray(result_array.tobytes()) 