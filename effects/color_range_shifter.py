#!/usr/bin/env python3
"""
color range shifter effect that identifies pixels within a specified color range and applies various transformations to them
"""

import random
import numpy as np
import time
from numba import njit, prange, float32, int32, uint8, boolean
from effects.base import Effect
from effects.utils import (
    rgb_to_hsv, hsv_to_rgb, get_stride_for_image, is_large_image, 
    apply_hsv_transform, generate_chunk_indices
)

# 800 lines of... idk, this needs to be refactored

# parallel functions for large images - this effect will kill perf 
@njit(parallel=True, fastmath=True, cache=True)
def calculate_color_distances(img_array, target_color, height, width):
    """
    calculate squared distances between image pixels and target color using parallel chunking
    much faster for large images due to better cache locality and parallelism
    """
    # preallocate result array
    dist_squared = np.empty((height, width), dtype=np.int32)
    target_r, target_g, target_b = target_color[0], target_color[1], target_color[2]
    
    # use the util function to generate chunk indices
    chunk_indices = generate_chunk_indices(height, width, chunk_size=256)
    
    # process chunks in parallel
    for chunk_idx in prange(len(chunk_indices)):
        y_start, y_end, x_start, x_end = chunk_indices[chunk_idx]
        
        # process all pixels in this chunk
        for i in range(y_start, y_end):
            for j in range(x_start, x_end):
                r_diff = int(img_array[i, j, 0]) - target_r
                g_diff = int(img_array[i, j, 1]) - target_g
                b_diff = int(img_array[i, j, 2]) - target_b
                dist_squared[i, j] = r_diff*r_diff + g_diff*g_diff + b_diff*b_diff
    
    return dist_squared

@njit(parallel=True, fastmath=True, cache=True)
def calculate_channel_differences(img_array, height, width):
    """
    calculate max difference between any two channels in parallel
    optimized for large images (better cache usage patterns)
    """
    # preallocate result array
    channel_diff = np.empty((height, width), dtype=np.int16)
    
    # use the util function to generate chunk indices
    chunk_indices = generate_chunk_indices(height, width, chunk_size=256)
    
    # process chunks in parallel
    for chunk_idx in prange(len(chunk_indices)):
        y_start, y_end, x_start, x_end = chunk_indices[chunk_idx]
        
        # process all pixels in this chunk
        for i in range(y_start, y_end):
            for j in range(x_start, x_end):
                r = int(img_array[i, j, 0])
                g = int(img_array[i, j, 1])
                b = int(img_array[i, j, 2])
                
                rg_diff = abs(r - g)
                rb_diff = abs(r - b)
                gb_diff = abs(g - b)
                
                max_diff = rg_diff
                if rb_diff > max_diff:
                    max_diff = rb_diff
                if gb_diff > max_diff:
                    max_diff = gb_diff
                
                channel_diff[i, j] = max_diff
    
    return channel_diff

@njit(fastmath=True, cache=True)
def process_hsv_shift(img_array, y_coords, x_coords, intensity, hue_shift_value=None):
    """
    jit-compiled function to process hsv shifts efficiently
    
    args:
        img_array: the image array (h, w, 3) 3 = r, g, b
        y_coords, x_coords: coordinates of pixels to modify
        intensity: effect intensity
        hue_shift_value: optional specific hue shift value to use (0.0-1.0)
    """
    # make hue shift more dramatic based on intensity
    if hue_shift_value is None:
        hue_shift = 0.4 + (0.6 * intensity)
    else:
        hue_shift = hue_shift_value
    
    # saturation + value (brightness) boost
    sat_boost = 1.3 + (0.7 * intensity)
    val_boost = 1.0 + (0.3 * intensity)
    
    n_pixels = len(y_coords)
    use_parallel = n_pixels > 1000
    
    # single implementation with conditional parallel processing
    for i in (prange(n_pixels) if use_parallel else range(n_pixels)):
        y, x = y_coords[i], x_coords[i]
        
        # get rgb values
        r = img_array[y, x, 0] / 255.0
        g = img_array[y, x, 1] / 255.0
        b = img_array[y, x, 2] / 255.0
        
        # convert to hsv
        h, s, v = rgb_to_hsv(r, g, b)
            
            # apply stronger modifications
        h = (h + hue_shift) % 1.0
        s = min(1.0, s * sat_boost)  # boosted saturation
        v = min(1.0, v * val_boost)  # boosted brightness
        
        # convert back to rgb
        r, g, b = hsv_to_rgb(h, s, v)
        
        # update pixel
        img_array[y, x, 0] = min(255, int(r * 255))
        img_array[y, x, 1] = min(255, int(g * 255))
        img_array[y, x, 2] = min(255, int(b * 255))

@njit(fastmath=True, cache=True)
def enhance_contrast_batch(img_array, y_coords, x_coords, intensity):
    """
    jit-compiled function to enhance contrast efficiently
    
    args:
        img_array: the image array
        y_coords, x_coords: coordinates of pixels to modify
        intensity: effect intensity
    """
    # increased contrast factor for better effects
    contrast_factor = 1.6 + (1.2 * intensity)
    # add a saturation boost
    saturation_boost = 1.2 + (0.8 * intensity)
    n_pixels = len(y_coords)
    
    # get mean for each channel
    r_mean = 0.0
    g_mean = 0.0
    b_mean = 0.0
    
    for i in range(n_pixels):
        y, x = y_coords[i], x_coords[i]
        r_mean += img_array[y, x, 0]
        g_mean += img_array[y, x, 1]
        b_mean += img_array[y, x, 2]
    
    r_mean /= n_pixels
    g_mean /= n_pixels
    b_mean /= n_pixels
    
    # apply contrast enhancement
    for i in range(n_pixels):
        y, x = y_coords[i], x_coords[i]
        
        # get rgb values for potential saturation adjustment
        r_val = img_array[y, x, 0]
        g_val = img_array[y, x, 1]
        b_val = img_array[y, x, 2]
        
        # apply contrast boost
        r_adjusted = (r_val - r_mean) * contrast_factor + r_mean
        g_adjusted = (g_val - g_mean) * contrast_factor + g_mean
        b_adjusted = (b_val - b_mean) * contrast_factor + b_mean
        
        # clip to valid range
        img_array[y, x, 0] = min(255, max(0, int(r_adjusted)))
        img_array[y, x, 1] = min(255, max(0, int(g_adjusted)))
        img_array[y, x, 2] = min(255, max(0, int(b_adjusted)))

@njit(fastmath=True, cache=True)
def apply_color_replacement(img_array, y_coords, x_coords, target_palette, intensity):
    """
    jit-compiled function to apply complete color replacement with a palette
    
    args:
        img_array: the image array
        y_coords, x_coords: coordinates of pixels to modify
        target_palette: array of target rgb colors to use
        intensity: effect intensity
    """
    n_pixels = len(y_coords)
    n_colors = len(target_palette)
    
    # determine how much of the original color to preserve
    # at intensity 1.0, we replace completely; at 0.0, we keep original
    original_weight = max(0.0, 1.0 - (intensity * 1.5))  # goes negative at high intensity for more dramatic effect
    
    for i in range(n_pixels):
        y, x = y_coords[i], x_coords[i]
        
        # get brightness of the original pixel (0-255)
        brightness = int(0.299 * img_array[y, x, 0] + 0.587 * img_array[y, x, 1] + 0.114 * img_array[y, x, 2])
        
        # scale brightness to palette index
        color_idx = min(n_colors - 1, int(brightness / 256.0 * n_colors))
        
        # get target color
        r_target = target_palette[color_idx][0]
        g_target = target_palette[color_idx][1]
        b_target = target_palette[color_idx][2]
        
        # blend between original and target color based on intensity
        if original_weight > 0:
            r_orig = img_array[y, x, 0]
            g_orig = img_array[y, x, 1]
            b_orig = img_array[y, x, 2]
            
            img_array[y, x, 0] = min(255, max(0, int(r_orig * original_weight + r_target * (1.0 - original_weight))))
            img_array[y, x, 1] = min(255, max(0, int(g_orig * original_weight + g_target * (1.0 - original_weight))))
            img_array[y, x, 2] = min(255, max(0, int(b_orig * original_weight + b_target * (1.0 - original_weight))))
        else:
            # at high intensity, we can even exaggerate beyond the target color
            exaggeration = min(1.5, 1.0 - original_weight)  # cap at 1.5x
            
            # calculate distance from mid gray and amplify
            r_diff = r_target - 128
            g_diff = g_target - 128
            b_diff = b_target - 128
            
            img_array[y, x, 0] = min(255, max(0, int(128 + r_diff * exaggeration)))
            img_array[y, x, 1] = min(255, max(0, int(128 + g_diff * exaggeration)))
            img_array[y, x, 2] = min(255, max(0, int(128 + b_diff * exaggeration)))

@njit(fastmath=True, cache=True)
def amplify_channel_swap(img_array, y_coords, x_coords, amp_factor):
    """
    jit-compiled function to amplify the contrast after channel-swapping
    
    args:
        img_array: the image array
        y_coords, x_coords: coordinates of pixels to modify
        amp_factor: amplification factor (typically 1.0-1.6)
    """
    n_pixels = len(y_coords)
    
    # process in parallel if there are enough pixels
    if n_pixels > 1000:
        for i in prange(n_pixels):
            y, x = y_coords[i], x_coords[i]
            
            # amplify differences from mid-gray for each channel
            for c in range(3):
                diff = int(img_array[y, x, c]) - 128
                img_array[y, x, c] = min(255, max(0, int(128 + (diff * amp_factor))))
    else:
        # regular loop for small number of pixels
        for i in range(n_pixels):
            y, x = y_coords[i], x_coords[i]
            
            # amplify differences from mid-gray for each channel
            for c in range(3):
                diff = int(img_array[y, x, c]) - 128
                img_array[y, x, c] = min(255, max(0, int(128 + (diff * amp_factor))))

@njit(fastmath=True, cache=True)
def apply_harmonious_colorize(img_array, y_coords, x_coords, base_hue, hue_offset, intensity):
    """
    jit-compiled function to apply harmonious colorization to specific pixels
    
    args:
        img_array: the image array
        y_coords, x_coords: coordinates of pixels to modify
        base_hue: base hue value (0.0-1.0)
        hue_offset: variation in hue (+/- from base_hue)
        intensity: effect intensity
    """
    n_pixels = len(y_coords)
    use_parallel = n_pixels > 1000
    
    # single implementation with conditional parallel processing
    for i in (prange(n_pixels) if use_parallel else range(n_pixels)):
        y, x = y_coords[i], x_coords[i]
        
        # get rgb values
        r = img_array[y, x, 0] / 255.0
        g = img_array[y, x, 1] / 255.0
        b = img_array[y, x, 2] / 255.0
        
        # convert to hsv
        h, s, v = rgb_to_hsv(r, g, b)
        
        # apply new harmonious hue while preserving brightness and adjusting saturation
        h = (base_hue + hue_offset) % 1.0
        
        # keep original value/brightness, boost saturation
        s = min(1.0, s * 0.25 + 0.6 + (0.3 * intensity))  # more saturation with higher intensity
        
        # convert back to rgb
        r, g, b = hsv_to_rgb(h, s, v)
        
        # update pixel
        img_array[y, x, 0] = min(255, int(r * 255))
        img_array[y, x, 1] = min(255, int(g * 255))
        img_array[y, x, 2] = min(255, int(b * 255))

class ColorRangeShifter(Effect):
    """ColorRangeShifter effect with detailed performance logging"""
    
    def __init__(self):
        """initialize the ColorRangeShifter effect"""
        super().__init__()
        self.name = "color_range_shifter"
        self.description = "shifts colors within selected ranges with performance logging"
        self.debug = False
    
    def apply(self, data, width, height, row_length, bytes_per_pixel, intensity):
        """apply the ColorRangeShifter effect.
        
        args:
            data (bytearray): the raw image data to modify
            width (int): image width in pixels
            height (int): image height in pixels
            row_length (int): number of bytes per row
            bytes_per_pixel (int): number of bytes per pixel (3 for rgb)
            intensity (float): effect intensity from 0.0 to 1.0

        returns:
            bytearray: the modified image data
        """
        # reset random seed to ensure different selections each time
        random.seed(int(time.time() * 1000))
        
        # timing dict to store step durations
        timings = {}
        total_start = time.time()
        
        # step 1: convert to numpy array and setup
        step_start = time.time()
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        timings["1_array_conversion"] = (time.time() - step_start) * 1000
        
        # step 2: select target color and tolerance
        step_start = time.time()
        # select a target color with more diversity
        if random.random() < 0.65:  # 65% chance to use a more diverse predefined color range
            color_variation = random.randint(0, 3)  # determine the type of color selection
            
            if color_variation == 0:  # primary and secondary colors
                target_type = random.randint(0, 7)
                color_map = {
                    0: (220, 30, 30),    # Reds
                    1: (30, 220, 30),    # Greens
                    2: (30, 30, 220),    # Blues
                    3: (220, 220, 30),   # Yellows
                    4: (30, 220, 220),   # Cyans
                    5: (220, 30, 220),   # Magentas
                    6: (220, 150, 30),   # Orange
                    7: (150, 220, 30)    # Lime
                }
                target_color = np.array(color_map[target_type])
            
            elif color_variation == 1:  # generate pastel colors with all channels in higher range
                target_color = np.array([
                    random.randint(180, 240),
                    random.randint(180, 240),
                    random.randint(180, 240)
                ])
                
            elif color_variation == 2:  # generate darker/muted colors with all channels in lower range
                target_color = np.array([
                    random.randint(30, 120),
                    random.randint(30, 120),
                    random.randint(30, 120)
                ])
                
            else:  # color_variation == 3: two bright channels
                channels = [0, 1, 2]
                random.shuffle(channels)
                
                channel_values = [0, 0, 0]
                # two high channels (180-230 range)
                channel_values[channels[0]] = random.randint(180, 230)
                channel_values[channels[1]] = random.randint(180, 230)
                # one lower channel (30-120 range)
                channel_values[channels[2]] = random.randint(30, 120)
                
                target_color = np.array(channel_values)
                
            # add some randomness to tolerance
            random_factor = 0.7 + (random.random() * 0.6)  # 0.7-1.3 randomness
            tolerance = int((80 + (80 * intensity)) * random_factor)
            
        else:  # 35% chance to use a fully random color range
            # more diverse random ranges
            rand_method = random.randint(0, 1)
            
            if rand_method == 0:
                # normal random distribution across all channels
                target_color = np.array([
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ])
            else:
                # create a color with more-extreme channel differences
                base = random.randint(0, 255)
                spread = random.randint(70, 150)
                
                # create variations from the base value for each channel
                r_adjust = random.choice([-1, 1]) * spread
                g_adjust = random.choice([-1, 1]) * spread
                b_adjust = random.choice([-1, 1]) * spread
                
                target_color = np.array([
                    max(0, min(255, base + r_adjust)),
                    max(0, min(255, base + g_adjust)),
                    max(0, min(255, base + b_adjust))
                ])
            
            # add some randomness to tolerance
            random_factor = 0.7 + (random.random() * 0.6)  # 0.7-1.3 randomness
            tolerance = int((70 + (90 * intensity)) * random_factor)
        
        # cap tolerance
        tolerance = min(tolerance, 210)
        
        ## DEBUG
        print(f"Target color: RGB({target_color[0]}, {target_color[1]}, {target_color[2]}) with tolerance: {tolerance}")
        
        timings["2_color_selection"] = (time.time() - step_start) * 1000
        
        # step 3: calculate distances and create mask
        step_start = time.time()
        
        # use optimized path if >10mp
        use_optimized_path = (width * height > 10000000)
        
        if use_optimized_path:
            # use optimized parallel chunking for large images, use parallel chunked distance calculation 
            dist_squared = calculate_color_distances(img_array, target_color, height, width)
            
            # create mask from distances
            tolerance_squared = tolerance**2
            final_mask = dist_squared <= tolerance_squared
        else:
            # use standard vectorized approach for smaller images
            # calculate squared distances - use full vectorized approach
            r_dist = (img_array[:,:,0].astype(np.int16) - target_color[0])**2
            g_dist = (img_array[:,:,1].astype(np.int16) - target_color[1])**2
            b_dist = (img_array[:,:,2].astype(np.int16) - target_color[2])**2
            
            # combined distance
            dist_squared = r_dist + g_dist + b_dist
            tolerance_squared = tolerance**2
            final_mask = dist_squared <= tolerance_squared
            
        timings["3a_distance_calculation"] = (time.time() - step_start) * 1000
        
        # step 3b: refine mask
        step_start = time.time()
        
        if use_optimized_path:
            # use optimized parallel chunking for channel differences
            channel_diff = calculate_channel_differences(img_array, height, width)
        else:
            # add exclusion for very common colors that might match too broadly
            # create masks for grayscale colors (white, black, grays)
            # calculate the max difference between any two channels
            # this quickfixes (or reduces) finding the entire image rather than a small range
            channel_diff = np.maximum(
                np.maximum(
                    np.abs(img_array[:,:,0].astype(np.int16) - img_array[:,:,1].astype(np.int16)),
                    np.abs(img_array[:,:,0].astype(np.int16) - img_array[:,:,2].astype(np.int16))
                ),
                np.abs(img_array[:,:,1].astype(np.int16) - img_array[:,:,2].astype(np.int16))
            )
        
        # exclude areas with low color difference (grayscale) if they're too common
        gray_threshold = 20 # 20 levels of difference
        gray_mask = channel_diff < gray_threshold
        gray_pixels = np.sum(gray_mask)
        gray_percentage = (gray_pixels / (width * height)) * 100
        
        ## DEBUG
        print(f"Grayscale pixels: {gray_percentage:.2f}% of image")
        
        # if grayscale is common in the image, handle differently
        if gray_percentage > 25:  # grayscale-heavy image
            if gray_percentage > 75:  # mostly grayscale image (like the wolf photo)
                # for predominantly grayscale images, don't exclude gray areas completely
                # instead, randomly select a subset of gray pixels
                ## FIX this at some point
                print("Predominantly grayscale image detected, randomly selecting gray pixels")
                selection_percentage = 0.3  # select 30% of matching gray areas
                if random.random() < 0.5:  # 50% chance to include some grayscale pixels
                    # create a random mask to select some percentage of the grayscale pixels
                    random_gray_selection = np.random.random(gray_mask.shape) < selection_percentage
                    # only exclude gray pixels not selected by the random mask
                    exclude_mask = gray_mask & ~random_gray_selection
                    final_mask = final_mask & ~exclude_mask
                # else: don't exclude any gray pixels
            else:
                # for moderately grayscale images, exclude all gray areas as before
                final_mask = final_mask & ~gray_mask
        
        # limit percentage of affected pixels
        matching_pixels = np.sum(final_mask)
        total_pixels = width * height
        MAX_AFFECTED_PERCENTAGE = 28.0 ## FIX make this scale w strength in mainpy
        matching_percentage = (matching_pixels / total_pixels) * 100
        
        if matching_pixels > 0 and matching_percentage > MAX_AFFECTED_PERCENTAGE:
            # create a random mask using vectorized operations
            random_mask = np.random.random(final_mask.shape) < (MAX_AFFECTED_PERCENTAGE / matching_percentage)
            final_mask = final_mask & random_mask
            matching_pixels = np.sum(final_mask)
        
        # check if we have enough matching pixels
        if matching_pixels < 50:
            ## DEBUG
            print(f"ColorRangeShifter: Not enough matching pixels ({matching_pixels}/{total_pixels}), trying again with increased tolerance")
            
            # try once more with increased tolerance
            tolerance_squared = int(tolerance_squared * 2.7)
            final_mask = dist_squared <= tolerance_squared
            matching_pixels = np.sum(final_mask)
            
            if matching_pixels < 50:
                ## DEBUG
                print(f"ColorRangeShifter: Still not enough matching pixels ({matching_pixels}), skipping effect")
                timings["mask_refinement_failed"] = (time.time() - step_start) * 1000
                
                # calculate total time
                total_time = (time.time() - total_start) * 1000
                
                ## DEBUG
                self._print_performance_report(timings, total_time)
                
                return data
        
        timings["3b_mask_refinement"] = (time.time() - step_start) * 1000
        
        # step 4: get coordinates of matching pixels
        step_start = time.time()
        y_coords, x_coords = np.where(final_mask)
        matching_pixels = len(y_coords)
        
        ## DEBUG
        print(f"ColorRangeShifter: Found {matching_pixels} matching pixels ({matching_percentage:.2f}% of image)")
        timings["4_get_coordinates"] = (time.time() - step_start) * 1000
        
        # step 5: select and apply effect
        step_start = time.time()
        # apply the selected effect
        # added multiple radical transformation types that completely replace colors
        shift_type = random.randint(0, 8) # 8 effects
        
        ## DEBUG nested timer to track specific effect application time
        effect_start = time.time()
        
        if shift_type == 0:
            # enhanced channel offsets - vary by channel and allow multiple channels
            # determine how many channels to shift (1-3)
            num_channels_to_shift = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
            
            # select random channels to shift
            channels = random.sample([0, 1, 2], num_channels_to_shift)
            
            sub_start = time.time()
            for channel in channels:
                # generate different shift amounts for each channel
                shift_strength = random.uniform(0.7, 1.3)  # varied strength
                shift_amount = int(140 * intensity * shift_strength)
                
                # randomize direction
                if random.random() < 0.5:
                    shift_amount = -shift_amount
                
                # apply the shift using vectorized operations
                img_array[y_coords, x_coords, channel] = np.clip(
                    img_array[y_coords, x_coords, channel].astype(np.int16) + shift_amount,
                    0, 255
                ).astype(np.uint8)
            timings["5a_channel_offset"] = (time.time() - sub_start) * 1000
            effect_name = "channel_offsets"
            
        elif shift_type == 1: # hue rotation, shifts range hue by a value
            # randomly choose between complementary shift (0.5) or random shift
            sub_start = time.time()
            if random.random() < 0.45:
                # use defined hue shift
                hue_shift_option = random.choice([0.33, 0.5, 0.66])  # 120, 180, or 240deg shifts
                process_hsv_shift(img_array, y_coords, x_coords, intensity, hue_shift_option)
            else:
                # standard random shift
                process_hsv_shift(img_array, y_coords, x_coords, intensity)
            timings["5a_hue_rotation"] = (time.time() - sub_start) * 1000
            effect_name = "hue_rotation"
            
        elif shift_type == 2: # inversion - can be partial else channel-specific
            inversion_type = random.randint(0, 2)
            sub_start = time.time()
            
            if inversion_type == 0:
                # full inversion using vectorized operations
                img_array[y_coords, x_coords] = 255 - img_array[y_coords, x_coords]
            elif inversion_type == 1:
                # partial inversion (closer to middle gray)
                strength = random.uniform(0.6, 0.9)
                middle = 128
                affected = img_array[y_coords, x_coords].astype(np.float32)
                inverted = 255 - affected
                img_array[y_coords, x_coords] = np.clip(
                    (affected * (1 - strength) + inverted * strength),
                    0, 255
                ).astype(np.uint8)
            else:  # inversion_type == 2
                # channel-specific inversion
                channels_to_invert = random.sample([0, 1, 2], random.randint(1, 2))
                for channel in channels_to_invert:
                    img_array[y_coords, x_coords, channel] = 255 - img_array[y_coords, x_coords, channel]
                    
            timings["5a_inversion"] = (time.time() - sub_start) * 1000
            effect_name = "inversion"
            
        elif shift_type == 3: # channel swapping w/ amplification
            sub_start = time.time()
            permutations = [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
            perm = permutations[random.randint(0, 4)]
            
            # copy the affected region
            affected_pixels = img_array[y_coords, x_coords].copy()
            
            # apply permutation
            img_array[y_coords, x_coords, 0] = affected_pixels[:, perm[0]]
            img_array[y_coords, x_coords, 1] = affected_pixels[:, perm[1]] 
            img_array[y_coords, x_coords, 2] = affected_pixels[:, perm[2]]
            
            # add an intensity-based amplification factor for more dramatic effect
            if intensity > 0.4:  ## FIX artbirary thresholds. should probably just scale with strength in mainpy
                # apply additional contrast to make the swapped colors stand out more
                amp_factor = 1.0 + (intensity * 0.6)  # 1.0 to 1.6 amplification
                
                # use numba-accelerated function for amplification
                amplify_channel_swap(img_array, y_coords, x_coords, amp_factor)
                        
            timings["5a_channel_swapping"] = (time.time() - sub_start) * 1000
            effect_name = "channel_swapping"
            
        elif shift_type == 4: # contrast/saturation enhancement
            sub_start = time.time()
            batch_size = 10000
            for i in range(0, len(y_coords), batch_size):
                batch_end = min(i + batch_size, len(y_coords))
                batch_y = y_coords[i:batch_end]
                batch_x = x_coords[i:batch_end]
                
                # use numba-accelerated function for contrast enhancement
                enhance_contrast_batch(img_array, batch_y, batch_x, intensity)
                
            timings["5a_contrast_enhancement"] = (time.time() - sub_start) * 1000
            effect_name = "contrast_enhancement"
            
        elif shift_type == 5: # harmonious colorization - transforms regions using analogous "harmonious" colors
            sub_start = time.time()
            
            # generate a harmonious hue shift option (analogous colors)
            base_hue = random.random()  # random base hue
            hue_offset = random.uniform(-0.1, 0.1)  # small variation
            
            # use numba-accelerated function for harmonious colorization
            apply_harmonious_colorize(img_array, y_coords, x_coords, base_hue, hue_offset, intensity)
            
            timings["5a_harmonious_colorize"] = (time.time() - sub_start) * 1000
            effect_name = "harmonious_colorize"
                
        elif shift_type == 6: # complete color replacement with synthwave palette - purples, blues, pinks, teals
            sub_start = time.time()
            synthwave_palette = np.array([
                (20, 10, 40),      # dark purple
                (80, 30, 120),     # purple
                (180, 50, 160),    # hot pink
                (240, 100, 200),   # neon pink
                (100, 220, 255),   # cyan
                (50, 120, 230),    # blue
                (30, 50, 120)      # dark blue
            ])
            
            # apply complete color replacement
            apply_color_replacement(img_array, y_coords, x_coords, synthwave_palette, intensity)
            
            timings["5a_synthwave_palette"] = (time.time() - sub_start) * 1000
            effect_name = "synthwave_palette"
                
        elif shift_type == 7: # complete color replacement with retro gaming palette - inspired by old computer systems
            sub_start = time.time()
            retro_palette = np.array([
                (0, 0, 0),        # black
                (255, 255, 255),  # white
                (240, 0, 0),      # red
                (0, 240, 0),      # green
                (0, 0, 240),      # blue
                (240, 240, 0),    # yellow
                (240, 0, 240),    # magenta
                (0, 240, 240)     # cyan
            ])
            
            # apply complete color replacement
            apply_color_replacement(img_array, y_coords, x_coords, retro_palette, intensity)
            
            timings["5a_retro_palette"] = (time.time() - sub_start) * 1000
            effect_name = "retro_palette"
                
        else:  # shift_type == 8 - complete color replacement with custom random palette
            sub_start = time.time()
            # generate a custom palette of 5-8 colors with high contrast
            num_colors = random.randint(5, 8)
            custom_palette = np.zeros((num_colors, 3), dtype=np.int32)
            
            # first color is randomly generated
            custom_palette[0] = [
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ]
            
            # generate remaining colors with high contrast
            for i in range(1, num_colors):
                # strategy: either complementary or analogous with high contrast
                if random.random() < 0.5:
                    # complementary (opposite) color with variation
                    custom_palette[i] = [
                        255 - custom_palette[i-1][0] + random.randint(-40, 40),
                        255 - custom_palette[i-1][1] + random.randint(-40, 40),
                        255 - custom_palette[i-1][2] + random.randint(-40, 40)
                    ]
                else:
                    # high contrast variation
                    custom_palette[i] = [
                        (custom_palette[i-1][0] + random.randint(100, 150)) % 256,
                        (custom_palette[i-1][1] + random.randint(100, 150)) % 256,
                        (custom_palette[i-1][2] + random.randint(100, 150)) % 256
                    ]
                
                # clip to valid range
                custom_palette[i] = np.clip(custom_palette[i], 0, 255)
            
            # apply complete color replacement
            apply_color_replacement(img_array, y_coords, x_coords, custom_palette, intensity)
            
            timings["5a_custom_palette"] = (time.time() - sub_start) * 1000
            effect_name = "custom_palette"
        
        timings["5_effect_application"] = (time.time() - step_start) * 1000
        timings["5z_effect_type"] = effect_name
        
        # step 6: convert back to bytearray
        step_start = time.time()
        result = bytearray(img_array.tobytes())
        timings["6_bytearray_conversion"] = (time.time() - step_start) * 1000
        
        ## DEBUG calculate total time
        total_time = (time.time() - total_start) * 1000
        
        ## DEBUG print performance report
        self._print_performance_report(timings, total_time, matching_pixels, width * height)
        
        return result
    
    ## DEBUG perf reporter
    def _print_performance_report(self, timings, total_time, matching_pixels=0, total_pixels=0):
        """report perf"""
        if not self.debug:
            return
        
        print("colorrange shifter perf")
        print(f"{"STEP":<35} {"TIME (ms)":>10} {"PERCENT":>10}")
        print("-" * 57)
        
        # only include timing keys, not metadata keys
        timing_keys = [k for k in timings.keys() if not k.startswith("5z_")]
        
        # sort timings by key number to get them in the correct order
        for step in sorted(timing_keys):
            duration = timings[step]
            percentage = (duration / total_time) * 100
            step_name = step[2:].replace("_", " ").title()
            print(f"{step_name:<35} {duration:>10.2f} {percentage:>9.1f}%")
        
        # print the effect type used if available
        effect_type = timings.get("5z_effect_type")
        if effect_type:
            print(f"\nEffect type applied: {effect_type.replace('_', ' ').title()}")
        
        print("-" * 57)
        print(f"{"TOTAL":<35} {total_time:>10.2f} {100:>9.1f}%")
        
        # additional info
        if matching_pixels and total_pixels:
            print(f"\nMatching pixels: {matching_pixels:,} ({(matching_pixels/total_pixels)*100:.2f}% of image)")
            
            # approximate memory for coordinates
            coords_memory = (matching_pixels * 2 * 4) / (1024 * 1024)  # 2 int32 per point, converted to MB
            print(f"Coordinate arrays memory usage: {coords_memory:.2f} MB") 