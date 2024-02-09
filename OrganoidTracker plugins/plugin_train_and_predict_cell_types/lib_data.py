import math
from typing import List, Optional

import numpy
from numpy import ndarray

from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData


STANDARD_METADATA_NAMES = ["neighbor_distance_variation", "neighbor_distance_median_um", "intensity_factor_local", "neighbor_distance_mean_um", "volume_um3", "volume_um3_local", "solidity", "solidity_local", "surface_um2", "surface_um2_local", "feret_diameter_max_um", "feret_diameter_max_um_local", "intensity_factor", "ellipticity", "ellipticity_local", "extent", "extent_local", "minor_axis_length_um", "minor_axis_length_um_local", "intermediate_axis_length_um", "intermediate_axis_length_um_local", "major_axis_length_um", "major_axis_length_um_local"]


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