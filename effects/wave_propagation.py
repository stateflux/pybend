#!/usr/bin/env python3
"""
wave propagation effect - creates distortion waves in images using sine and triangle waves
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

# wave shape constants
SINE_WAVE = 0
TRIANGLE_WAVE = 1
# removed square and sawtooth waves as they created harsh visual artifacts

@jit(nopython=True, cache=True)
def apply_wave_distortion(img_array, wave_type, direction, wavelength, amplitude, 
                          brightness_modulation, color_modulation, stride=1):
    """
    applies wave distortion to image data
    
    args:
        img_array: 3d numpy array of image data (h, w, 3)
        wave_type: type of wave (0=sine, 1=triangle)
        direction: 0 for horizontal waves, 1 for vertical waves
        wavelength: distance between wave peaks in pixels
        amplitude: wave height in pixels
        brightness_modulation: how much brightness affects wave height (0.0-1.0)
        color_modulation: how much color affects wave position (0.0-1.0)
        stride: pixel increment for performance optimization
        
    returns:
        distorted image array
    """
    height, width, _ = img_array.shape
    result = np.zeros_like(img_array)
    
    # calculate center once for efficiency
    center_y = height // 2
    center_x = width // 2
    
    # create coordinate arrays for vectorized operations
    y_indices = np.arange(0, height, stride)
    x_indices = np.arange(0, width, stride)
    
    # wave calculation - inline to avoid numba function pointer limitations
    def calculate_wave(x, wavelength, amplitude, wave_type):
        # prevent division by zero
        if wavelength <= 0:
            wavelength = 1.0
            
        # normalize x to [0, 1] within wavelength
        x_norm = (x % wavelength) / wavelength
        
        if wave_type == SINE_WAVE:
            # sine wave - smooth transition
            return amplitude * math.sin(2 * math.pi * x_norm)
        else:  # TRIANGLE_WAVE
            # triangle wave - linear transition
            if x_norm < 0.5:
                return amplitude * (4 * x_norm - 1)
            else:
                return amplitude * (3 - 4 * x_norm)
    
    # parallel processing with boundary checking
    for y_idx in prange(len(y_indices)):
        y = y_indices[y_idx]
        y_end = min(y + stride, height)
        
        for x_idx in range(len(x_indices)):
            x = x_indices[x_idx]
            x_end = min(x + stride, width)
            
            # get pixel color values
            pixel = img_array[y, x]
            r, g, b = pixel[0], pixel[1], pixel[2]
            
            # simple brightness calculation
            brightness = (r + g + b) / 3.0 / 255.0
            
            # extract hue for color modulation
            h, s, v = rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            
            # adjust amplitude based on brightness if enabled
            mod_amplitude = amplitude
            if brightness_modulation > 0:
                # brighter pixels get more displacement
                mod_amplitude *= (1.0 + (brightness - 0.5) * 2 * brightness_modulation)
            
            # adjust wave phase based on color if enabled
            phase_offset = 0
            if color_modulation > 0:
                # different hues get different offsets
                phase_offset = h * wavelength * color_modulation
            
            # initialize offsets
            wave_offset_x = 0
            wave_offset_y = 0
            
            # calculate offset based on wave direction
            if direction == 0:  # horizontal waves
                # wave varies with y position
                pos = y + phase_offset
                wave_offset_x = calculate_wave(pos, wavelength, mod_amplitude, wave_type)
                wave_offset_y = 0
            else:  # vertical waves
                # wave varies with x position
                pos = x + phase_offset
                wave_offset_x = 0
                wave_offset_y = calculate_wave(pos, wavelength, mod_amplitude, wave_type)
            
            # convert to integer offsets
            src_y = y + int(wave_offset_y)
            src_x = x + int(wave_offset_x)
            
            # ensure coordinates stay within bounds
            src_y = max(0, min(height - 1, src_y))
            src_x = max(0, min(width - 1, src_x))
            
            # get source pixel
            src_pixel = img_array[src_y, src_x]
            
            # apply to all pixels in this stride block
            for dy in range(y_end - y):
                for dx in range(x_end - x):
                    # calculate destination coordinates
                    dest_y = y + dy
                    dest_x = x + dx
                    
                    # verify bounds to prevent errors
                    if 0 <= dest_y < height and 0 <= dest_x < width:
                        result[dest_y, dest_x] = src_pixel
    
    return result

class WavePropagation(Effect):
    """creates wave distortions that respond to image properties"""
    
    def __init__(self):
        """initialize the wave propagation effect"""
        super().__init__()
        self.name = "wave_propagation"
        self.description = "creates wave distortions that ripple through the image"
    
    def apply(self, data, width, height, row_length, bytes_per_pixel, intensity):
        """apply wave distortion to image data.
        
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
        try:
            # validate dimensions
            if width <= 0 or height <= 0:
                return data
                
            # convert to numpy array
            img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            
            # determine if performance optimization needed
            is_large = is_large_image(width, height)
            stride = get_stride_for_image(width, height, base_stride=1)
            
            # scale parameters based on intensity and size
            # wavelength: higher intensity means more frequent waves
            base_wavelength = max(10, min(width, height) // 10)
            wavelength = int(base_wavelength * (1.0 + (1.0 - intensity) * 2))
            
            # ensure valid wavelength
            wavelength = max(1, wavelength)
            
            # amplitude: higher intensity means larger displacement
            max_amplitude = max(5, min(width, height) // 20)
            amplitude = max_amplitude * intensity
            
            # modulation increases with intensity
            brightness_modulation = intensity * 0.8  # up to 80% modulation
            color_modulation = intensity * 0.6       # up to 60% modulation
            
            # randomly select wave characteristics
            wave_type = random.randint(0, 1)  # sine or triangle
            direction = random.randint(0, 1)  # horizontal or vertical
            
            # focus effect on a specific region
            region_type = select_region_type()
            
            # get region based on intensity
            y_start, region_height, x_start, region_width = get_region_params(
                height, width, region_type, intensity
            )
            
            # validate region dimensions
            if region_height <= 0 or region_width <= 0:
                return data
            
            # extract target region
            region = extract_region(img_array, y_start, region_height, x_start, region_width)
            
            # process the region
            processed_region = apply_wave_distortion(
                region, wave_type, direction, wavelength, amplitude,
                brightness_modulation, color_modulation, stride
            )
            
            # integrate processed region back into image
            result_array = img_array.copy()
            apply_to_region(result_array, processed_region, y_start, x_start)
            
            # convert back to bytearray
            return bytearray(result_array.tobytes())
            
        except Exception as e:
            # bail if issue
            print(f"error in wave_propagation: {str(e)}")
            return data 