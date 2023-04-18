import math
from typing import Optional

import numpy
import pandas
from anndata import AnnData
from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.imaging import list_io


_INPUT_FILE = r"../Data/Training data.autlist"
_METADATA_NAMES = ["neighbor_distance_variation", "neighbor_distance_median_um", "intensity_factor_local", "neighbor_distance_mean_um", "volume_um3", "volume_um3_local", "solidity", "solidity_local", "surface_um2", "surface_um2_local", "feret_diameter_max_um", "feret_diameter_max_um_local", "intensity_factor", "ellipticity", "ellipticity_local", "extent", "extent_local", "minor_axis_length_um", "minor_axis_length_um_local", "intermediate_axis_length_um", "intermediate_axis_length_um_local", "major_axis_length_um", "major_axis_length_um_local"]
_OUTPUT_FILE = "../Data/all_data.h5ad"


def _convert_cell_type(position_type: Optional[str]) -> str:
    """Converts the cell type to one suitable for training. Returns "NONE" if no such type exists.
    (We're using "NONE" instead of None because that works better with pandas.Categorial.)"""
    if position_type is None:
        return "NONE"
    if position_type in {"ENTEROENDOCRINE", "GOBLET", "TUFT", "SECRETORY"}:
        # Seeing the difference between these is hard for the network
        return "SECRETORY"
    if position_type in {"PANETH", "WGA_PLUS"}:
        return "PANETH"
    if position_type in {"STEM", "STEM_PUTATIVE"}:
        return "STEM"
    if position_type == "ENTEROCYTE":
        return "ENTEROCYTE"
    return "NONE"


def _get_data_array(position_data: PositionData, position: Position) -> Optional[ndarray]:
    array = numpy.empty(len(_METADATA_NAMES), dtype=numpy.float32)
    for i, name in enumerate(_METADATA_NAMES):
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

        if value is None:
            return None  # Abort, a value is missing

        if name in {"neighbor_distance_variation", "solidity", "sphericity", "ellipticity", "intensity_factor"}\
                or name.endswith("_local"):
            # Ratios should be an exponential, as the analysis will log-transform the data
            value = math.exp(value)

        array[i] = value
    return array

def main():
    data_array = list()
    cell_type_list = list()
    cell_type_training_list = list()
    organoid_list = list()
    cell_names = list()

    # Collect position data for last 10 time points of each experiment
    for experiment in list_io.load_experiment_list_file(_INPUT_FILE):
        print("Loading", experiment.name)
        position_data = experiment.position_data

        last_time_point_number = experiment.positions.last_time_point_number()
        time_points = [TimePoint(last_time_point_number), TimePoint(last_time_point_number - 5),
                       TimePoint(last_time_point_number - 10), TimePoint(last_time_point_number - 15)]
        for time_point in time_points:
            for position in experiment.positions.of_time_point(time_point):
                position_data_array = _get_data_array(position_data, position)
                cell_type = position_data.get_position_data(position, "type")
                cell_type_training = _convert_cell_type(cell_type)
                if position_data_array is not None and cell_type is not None:
                    data_array.append(position_data_array)
                    cell_type_list.append(cell_type)
                    cell_type_training_list.append(cell_type_training)
                    organoid_list.append(experiment.name.get_name())
                    cell_names.append(f"{experiment.name}-t{position.time_point_number()}-{int(position.x)}-{int(position.y)}-{int(position.z)}")

    data_array = numpy.array(data_array)
    adata = AnnData(data_array)
    adata.var_names = _METADATA_NAMES
    adata.obs_names = cell_names
    adata.obs["cell_type"] = pandas.Categorical(cell_type_list)
    adata.obs["cell_type_training"] = pandas.Categorical(cell_type_training_list)
    adata.obs["organoid"] = pandas.Categorical(organoid_list)
    adata.write_h5ad(_OUTPUT_FILE, compression="gzip")


if __name__ == "__main__":
    main()
