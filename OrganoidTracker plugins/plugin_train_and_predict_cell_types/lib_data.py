import math
from typing import List, Optional

import anndata
from anndata import AnnData
import numpy
from numpy import ndarray

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position_data import PositionData


STANDARD_METADATA_NAMES = ["neighbor_distance_variation", "neighbor_distance_median_um", "intensity_factor_local", "neighbor_distance_mean_um", "volume_um3", "volume_um3_local", "solidity", "solidity_local", "surface_um2", "surface_um2_local", "feret_diameter_max_um", "feret_diameter_max_um_local", "intensity_factor", "ellipticity", "ellipticity_local", "extent", "extent_local", "minor_axis_length_um", "minor_axis_length_um_local", "intermediate_axis_length_um", "intermediate_axis_length_um_local", "major_axis_length_um", "major_axis_length_um_local"]


def should_be_exponential(input_name: str) -> bool:
    """Returns True if the given input name should be exponential, so that the log-transform during training
    turns it back. Used for all factors."""
    return input_name in {"neighbor_distance_variation", "solidity", "sphericity", "ellipticity", "intensity_factor"}\
           or input_name.endswith("_local")


def get_data_array(positions: PositionCollection, position: Position, input_names: List[str]) -> Optional[ndarray]:
    """Extract the data array from the given position."""
    array = numpy.empty(len(input_names), dtype=numpy.float32)
    for i, name in enumerate(input_names):
        value = None
        if name == "sphericity":
            # Special case, we need to calculate
            volume = positions.get_position_data(position, "volume_um3")
            surface = positions.get_position_data(position, "surface_um2")
            if volume is not None and surface is not None:
                value = math.pi ** (1/3) * (6 * volume) ** (2/3) / surface
        else:
            # Otherwise, just look up
            value = positions.get_position_data(position, name)

        if value is None or value == 0:
            return None  # Abort, a value is missing

        if should_be_exponential(name):
            # Ratios should be an exponential, as the analysis will log-transform the data
            value = math.exp(value)

        array[i] = value
    return array


def get_cell_type_mapping(experiment: Experiment) -> dict[str, str]:
    """Returns a mapping from cell types to their merged names."""
    return_dict = dict()
    for key, value in experiment.global_data.get_all_data().items():
        if key.startswith("ct_mapping_") and isinstance(value, str):
            cell_type_key = key[len("ct_mapping_"):]
            return_dict[cell_type_key] = value

    if len(return_dict) == 0:
        # Nothing specified, return the legacy cell type mapping
        return {
            "ENTEROCYTE": "ENTEROCYTE",
            "PANETH": "PANETH",
            "STEM": "STEM",
            "STEM_PUTATIVE": "STEM",
            "MATURE_GOBLET": "MATURE_GOBLET"
        }
    return return_dict


def set_cell_type_mapping(experiment: Experiment, mapping: dict[str, str]):
    """Updates the cell type mapping in the experiment for training the model."""

    # Remove existing mappings
    for key in list(experiment.global_data.get_all_data().keys()):
        if key.startswith("ct_mapping_"):
            experiment.global_data.set_data(key, None)

    # Set new mappings
    for key, value in mapping.items():
        experiment.global_data.set_data(f"ct_mapping_{key}", value)

