# Copy of the file in the Figure code folder, unfortunately
import math
from typing import List, Optional

import numpy
from numpy import ndarray

from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData


def get_data_array(position_data: PositionData, position: Position, input_names: List[str]) -> Optional[ndarray]:
    """Extract the data array from the given position."""
    array = numpy.empty(len(input_names), dtype=numpy.float32)
    for i, name in enumerate(input_names):
        value = None
        if name == "sphericity":
            # Special case, we need to calculate
            volume = position_data.get_position_data(position, "volume_um3")
            surface = position_data.get_position_data(position, "surface_um2")
            if volume is not None and surface is not None:
                value = math.pi ** (1/3) * (6 * volume) ** (2/3) / surface
        else:
            # Otherwise, just look up
            value = position_data.get_position_data(position, name)

        if value is None or value == 0:
            return None  # Abort, a value is missing

        if name in {"neighbor_distance_variation", "solidity", "sphericity", "ellipticity", "intensity_factor"}\
                or name.endswith("_local"):
            # Ratios should be an exponential, as the analysis will log-transform the data
            value = math.exp(value)

        array[i] = value
    return array