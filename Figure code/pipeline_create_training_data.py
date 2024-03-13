from typing import Optional

import numpy
import pandas
from anndata import AnnData

import lib_data
from organoid_tracker.core import TimePoint
from organoid_tracker.imaging import list_io

_INPUT_FILE = r"../../Data/Testing data - output - treatments.autlist"
_METADATA_NAMES = lib_data.STANDARD_METADATA_NAMES
_OUTPUT_FILE = "../../Data/treatments_data.h5ad"


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
                position_data_array = lib_data.get_data_array(position_data, position, _METADATA_NAMES)
                cell_type = position_data.get_position_data(position, "type")
                cell_type_training = lib_data.convert_cell_type(cell_type)
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
