#!/usr/bin/env python3
"""
effects package: contains all glitch effects for the databender.
"""

from effects.base import Effect
from effects.reverser import Reverser
from effects.bit_depth_reduction import BitDepthReduction
from effects.selective_channel_corruption import SelectiveChannelCorruption
from effects.chunk_duplication import ChunkDuplication
from effects.palette_swap import PaletteSwap
from effects.color_range_shifter import ColorRangeShifter
from effects.zeta import ZetaInvert
from effects.wave_propagation import WavePropagation
from effects.phantom_regions import PhantomRegions
from effects.mirror import Mirror
from effects.bleed_blur import BleedBlur
from effects.perf_logger import perf_logger

EFFECTS = {
    "reverser": Reverser(),
    "bit_depth_reduction": BitDepthReduction(),
    "selective_channel_corruption": SelectiveChannelCorruption(),
    "chunk_duplication": ChunkDuplication(),
    "palette_swap": PaletteSwap(),
    "color_range_shifter": ColorRangeShifter(),
    "zeta_invert": ZetaInvert(),
    "wave_propagation": WavePropagation(),
    "phantom_regions": PhantomRegions(),
    "mirror": Mirror(),
    "bleed_blur": BleedBlur()
}

## DEBUG: perf logging
for name, effect in EFFECTS.items():
    EFFECTS[name] = perf_logger.wrap_effect(effect)

EFFECT_SEQUENCE = list(EFFECTS.keys())

__all__ = ['Effect', 'EFFECTS', 'EFFECT_SEQUENCE', 'perf_logger'] 