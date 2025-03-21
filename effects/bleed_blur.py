#!/usr/bin/env python3
"""
bleed blur effect - creates smears that look like color bleeding
"""

import random
import math
import numpy as np
from numba import jit, prange
from effects.base import Effect
from effects.utils import (
    is_large_image, select_region_type, get_region_params, extract_region,
    apply_to_region, ROW_REGION, COLUMN_REGION, RECT_REGION
)

class BleedBlur(Effect):
    """creates smears that look like color bleeding"""
    
    def __init__(self):
        """initialize the BleedBlur effect"""
        super().__init__()
        self.name = "bleed_blur"
        self.description = "creates smears that look like color bleeding"
    
    def apply(self, data, width, height, row_length, bytes_per_pixel, intensity):
        """apply bleeding blur to raw image data
        
        args:
            data (bytearray): raw image data
            width (int): image width
            height (int): image height
            row_length (int): bytes per row
            bytes_per_pixel (int): bytes per pixel (3 for rgb)
            intensity (float): effect strength 0.0-1.0
            
        returns:
            bytearray: modified image data
        """
        # convert to numpy
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        
        # figure out how many smear paths based on intensity
        # low intensity = fewer paths
        if random.random() < 0.7:  # 70% chance of scaling with intensity
            # intensity determines max possible paths (1-3)
            max_possible = max(1, min(3, int(1 + intensity * 2)))
            num_paths = random.randint(1, max_possible)
        else:
            # sometimes just use a single path
            num_paths = 1
        
        # randomize size with some variance
        size_variance = random.uniform(0.8, 1.2)
        
        # calc min/max sizes based on intensity
        min_size = max(8, int(12 * intensity))
        max_width = max(min_size + 8, int(width * 0.4 * intensity))
        max_height = max(min_size + 8, int(height * 0.4 * intensity))
        
        print(f"doing bleed blur with {num_paths} smears")
        
        # process each smear path
        for i in range(num_paths):
            # choose horizontal (0) or vertical (1) line
            line_type = random.randint(0, 1)
            
            if line_type == 0:  # horizontal line
                # randomize length
                line_width = random.randint(min_size, max_width)
                line_height = 1  # 1px tall
                
                # pick random spot
                src_x = random.randint(0, width - line_width)
                src_y = random.randint(0, height - line_height)
            else:  # vertical line
                # randomize length
                line_width = 1  # 1px wide
                line_height = random.randint(min_size, max_height)
                
                # pick random spot
                src_x = random.randint(0, width - line_width)
                src_y = random.randint(0, height - line_height)
            
            # create smear path - randomize length
            # shorter paths for low intensity
            min_length = max(5, int(10 * intensity))
            max_length = max(min_length + 5, int(min(width, height) * 0.7 * intensity))
            path_length = random.randint(min_length, max_length)
            
            print(f"  path {i+1}: size {line_width}x{line_height}, length {path_length}")
            
            # grab the source line
            source_line = extract_region(img_array, src_y, line_height, src_x, line_width)
            
            # pick direction - only use cardinal directions
            if line_type == 0:  # horizontal line
                # horizontal lines go up or down
                directions = [math.pi/2, 3*math.pi/2]
                angle = random.choice(directions)
            else:  # vertical line
                # vertical lines go left or right
                directions = [0, math.pi]
                angle = random.choice(directions)
            
            # apply smear along the path
            apply_line_smear(
                img_array, 
                source_line, 
                src_x, 
                src_y, 
                angle,
                path_length, 
                intensity
            )
        
        # back to bytearray
        return bytearray(img_array.tobytes())


@jit(nopython=True, cache=True)
def apply_line_smear(img_array, source_line, src_x, src_y, angle, path_length, intensity):
    """applies a 1px line smear with fading opacity where brighter pixels go further
    
    args:
        img_array: 3d numpy array (h, w, 3)
        source_line: source to smear (1px tall or 1px wide)
        src_x, src_y: source coordinates
        angle: direction in radians
        path_length: steps in smear path
        intensity: effect strength (0.0-1.0)
    """
    height, width, _ = img_array.shape
    line_height, line_width, _ = source_line.shape
    
    # get path coords
    coords = generate_straight_path(
        src_x, src_y, 
        width, height,
        line_width, line_height,
        angle, path_length, 
        intensity
    )
    
    # total path for calcs
    total_path_points = len(coords)
    
    # calc brightness for each pixel in source line
    # these adjust how far each pixel travels (0.5-1.0 multiplier)
    brightness_factors = np.zeros((line_height, line_width), dtype=np.float32)
    
    for y in range(line_height):
        for x in range(line_width):
            # avg brightness (0.0-1.0)
            brightness = 0.0
            for c in range(3):
                brightness += float(source_line[y, x, c]) / 255.0
            brightness /= 3.0
            
            # brighter pixels go further (50-100% of path)
            brightness_factors[y, x] = 0.5 + (brightness * 0.5)
    
    # apply effect for each pixel in the source
    for i, (current_x, current_y) in enumerate(coords):
        # bail if we're outside image
        if (current_x < 0 or current_x + line_width > width or
            current_y < 0 or current_y + line_height > height):
            continue
        
        # global progress (0.0-1.0)
        global_progress = i / max(1, total_path_points - 1)
        
        # calc per-pixel opacities using brightness
        pixel_opacities = np.zeros((line_height, line_width), dtype=np.float32)
        
        # minimal noise, scaled down
        max_noise = 0.05 * intensity  # very minimal
        noise_level = global_progress * max_noise
        
        # process each pixel's opacity
        all_transparent = True
        
        for y in range(line_height):
            for x in range(line_width):
                # adjust progress based on brightness
                # brighter pixels have different "effective path length"
                brightness_factor = brightness_factors[y, x]
                
                # stretch/compress path based on brightness
                adjusted_progress = global_progress / brightness_factor
                
                # if we're past max adjusted path, pixel is transparent
                if adjusted_progress >= 1.0:
                    pixel_opacities[y, x] = 0.0
                    continue
                
                # linear fade from 100% to 0%
                pixel_opacities[y, x] = 1.0 - adjusted_progress
                all_transparent = False
        
        # bail if everything's transparent
        if all_transparent:
            break
        
        # copy with varying opacity
        copy_pixels_with_variable_opacity(
            img_array, 
            source_line,
            current_y, current_x,
            pixel_opacities, 
            noise_level
        )


@jit(nopython=True, cache=True)
def generate_straight_path(src_x, src_y, width, height, block_width, block_height, 
                          angle, path_length, intensity):
    """makes coords for a straight path with no jitter
    
    args:
        src_x, src_y: start coords
        width, height: image dimensions
        block_width, block_height: dimensions of block being smeared
        angle: direction in radians
        path_length: steps in path
        intensity: effect intensity
        
    returns:
        list of (x, y) tuples
    """
    # calc step based on angle
    dx = math.cos(angle)
    dy = math.sin(angle)
    
    # step size based on intensity
    # smaller steps for higher intensity = more detailed smears
    step_size = max(0.5, min(1.0, (1.0 - intensity) * 1.5))
    
    # total distance based on path length
    total_distance = path_length * step_size
    
    # make coords with spacing
    coords = []
    
    # current pos (float for accuracy)
    x, y = float(src_x), float(src_y)
    
    # add starting pos
    coords.append((int(x), int(y)))
    
    # evenly spaced points on path
    # more points for higher intensity
    step_count = max(path_length // 2, int(path_length * intensity))
    
    for i in range(1, step_count + 1):
        # progress (0.0-1.0)
        progress = i / step_count
        
        # simple linear interpolation, no acceleration
        dist = progress * total_distance
        x = src_x + dx * dist
        y = src_y + dy * dist
        
        # add point with no jitter
        current_x = int(x)
        current_y = int(y)
        
        # skip dupes
        if len(coords) == 0 or coords[-1] != (current_x, current_y):
            coords.append((current_x, current_y))
    
    return coords


@jit(nopython=True, cache=True)
def copy_pixels_with_variable_opacity(img_array, source_block, dest_y, dest_x, pixel_opacities, noise_level):
    """copies pixels with different opacity for each one
    
    args:
        img_array: destination image
        source_block: source to copy
        dest_y, dest_x: destination coords
        pixel_opacities: 2D array of opacities (0.0-1.0)
        noise_level: noise amount (0.0-1.0)
    """
    block_height, block_width, _ = source_block.shape
    
    # don't go out of bounds
    max_y = min(dest_y + block_height, img_array.shape[0])
    max_x = min(dest_x + block_width, img_array.shape[1])
    
    # limit source to what fits
    usable_height = max_y - dest_y
    usable_width = max_x - dest_x
    
    # random seed for consistent noise pattern
    random_seed = (int(dest_x * 12347 + dest_y * 4567) % 10000) + 1
    
    # copy with blending and minimal noise
    for y in range(usable_height):
        for x in range(usable_width):
            # skip transparent pixels
            if y >= pixel_opacities.shape[0] or x >= pixel_opacities.shape[1] or pixel_opacities[y, x] <= 0.01:
                continue
                
            # get opacity
            opacity = pixel_opacities[y, x]
            
            for c in range(3):  # RGB
                # blend source and destination
                src_val = float(source_block[y, x, c])
                dst_val = float(img_array[dest_y + y, dest_x + x, c])
                
                # pseudo-random noise
                noise_seed = (random_seed + y * 173 + x * 371 + c * 97) % 1000
                noise = (2.0 * (noise_seed / 1000.0) - 1.0) * noise_level * 10.0  # reduced noise
                
                # add noise to source
                noisy_src_val = max(0.0, min(255.0, src_val + noise))
                
                # blend with dest
                blended = int(noisy_src_val * opacity + dst_val * (1.0 - opacity))
                
                # clamp to valid range
                blended = max(0, min(255, blended))
                
                img_array[dest_y + y, dest_x + x, c] = blended 