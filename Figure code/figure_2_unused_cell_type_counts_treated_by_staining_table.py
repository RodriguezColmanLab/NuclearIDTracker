import os
from enum import Enum, auto
from typing import Dict, List, NamedTuple

import numpy
from matplotlib import pyplot as plt

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers


_DATA_FILE_STAINING = "../../Data/Immunostaining conditions.autlist"
_DATA_FILE_PREDICTIONS = "../../Data/Testing data - predictions - treatments - fixed.autlist"
_OUTPUT_FOLDER = "../../Data/Testing data - predictions - treatments - fixed"


class _Condition(Enum):
    CONTROL = auto()
    DAPT_CHIR = auto()
    NO_RSPONDIN = auto()

    @property
    def display_name(self):
        return self.name.lower().replace("_", " ")


def _get_condition(experiment: Experiment) -> _Condition:
    name = experiment.name.get_name()
    if "control" in name:
        return _Condition.CONTROL
    if "dapt chir" in name:
        return _Condition.DAPT_CHIR
    if "EN" in name:
        return _Condition.NO_RSPONDIN
    raise ValueError("Unknown condition: " + name)


class _CellKey(NamedTuple):
    x_px: float
    y_px: float
    z_px: float
    organoid_name: str


def main():
    # Load the cells
    cells_staining = dict()

    experiments_staining = list_io.load_experiment_list_file(_DATA_FILE_STAINING, load_images=False)
    for experiment_staining in experiments_staining:
        if "secondary only" in experiment_staining.name.get_name():
            continue  # Skip these experiments
        if "chir vpa" in experiment_staining.name.get_name():
            continue  # Also skipped, as KRT20 signal was everywhere, which indicates the experiment failed
        print(experiment_staining.name, len(experiment_staining.positions))
        for position in experiment_staining.positions:
            cell_type = position_markers.get_position_type(experiment_staining.position_data, position)
            if cell_type is None:
                cell_type = "None"  # Double-negative cells
            key = _CellKey(position.x, position.y, position.z, experiment_staining.name.get_name())
            cells_staining[key] = cell_type

    output_file = os.path.abspath(os.path.join(_OUTPUT_FOLDER, "all_cells_prediction_and_staining.csv"))
    os.makedirs(_OUTPUT_FOLDER, exist_ok=True)
    with open(output_file, "w") as handle:
        handle.write("x_px,y_px,z_px,organoid_name,condition,cell_type_staining,cell_type_predicted\n")

        experiments_prediction = list_io.load_experiment_list_file(_DATA_FILE_PREDICTIONS, load_images=False)
        for experiment_prediction in experiments_prediction:
            if "chir vpa" in experiment_prediction.name.get_name():
                continue  # Skipped, as KRT20 signal was everywhere, which indicates the experiment failed
            condition = _get_condition(experiment_prediction)
            for position in experiment_prediction.positions:
                cell_type_prediction = position_markers.get_position_type(experiment_prediction.position_data, position)
                if cell_type_prediction is None:
                    continue
                key = _CellKey(position.x, position.y, position.z, experiment_prediction.name.get_name())
                if key not in cells_staining:
                    continue
                cell_type_staining = cells_staining[key]
                handle.write(
                    f"{position.x},{position.y},{position.z},{experiment_prediction.name.get_name()},{condition.name},{cell_type_staining},{cell_type_prediction}\n")
    print("Output written to", output_file)


if __name__ == "__main__":
    main()
