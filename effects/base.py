#!/usr/bin/env python3
"""
base effect class that all glitch effects inherit from
"""

class Effect:
    # all effects should inherit from this class and implement the apply() method.
    
    def __init__(self):
        """initialize the BitDepthReduction effect"""
        self.name = "base_effect"
        self.description = "base effect class"
    
    def apply(self, data, width, height, row_length, bytes_per_pixel, intensity):
        """apply the effect to the raw image data.
        
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
        # this method simply takes image data and parameters as input
        # and returns the unmodified data without applying any effects
        # child classes will override this method with actual effect implementations
        return data 