"""We plot all stem and Paneth cells on a stem-to-Paneth axis, and see how their amount changes over time."""
import csv

from enum import Enum, auto
from typing import List, Optional, NamedTuple

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lib_figures
from organoid_tracker.core import Name
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE_STAINING = "../../Data/Live and immunostaining multiple time points/Immunostaining all.autlist"
_DATA_FILE_LIVE = "../../Data/Live and immunostaining multiple time points/Live all.autlist"


class _Age(Enum):
    YOUNG = auto()
    OLD = auto()

    def get_age_text(self) -> str:
        if self == _Age.YOUNG:
            return "Day 2"
        elif self == _Age.OLD:
            return "Day 5"
        else:
            raise ValueError(f"Unknown age {self}")


class _CellTypePredictions(NamedTuple):
    """Holds the cell type predictions for a single cell."""

    cell_type_names: List[str]
    cell_type_probabilities: List[float]

    def get_probability(self, cell_type_name: str) -> float:
        """Get the probability of a cell being of a certain type. Returns NaN if the type is not present."""
        if cell_type_name not in self.cell_type_names:
            return float("NaN")
        index = self.cell_type_names.index(cell_type_name)
        return self.cell_type_probabilities[index]

# This is a null-object for when no predictions are available for a cell
NO_PREDICTIONS = _CellTypePredictions(cell_type_names=[], cell_type_probabilities=[])


class _CellInfoTable:
    positions: List[Position]
    experiment_names: List[Name]
    cell_types: List[Optional[str]]
    is_live: List[bool]
    ages: List[_Age]
    predictions: List[_CellTypePredictions]

    def __init__(self):
        self.positions = list()
        self.experiment_names = list()
        self.cell_types = list()
        self.is_live = list()
        self.ages = list()
        self.predictions = list()

    def add_cell(self, position: Position, experiment_name: Name, cell_type: Optional[str], is_live: bool, age: _Age,
                 cell_type_predictions: _CellTypePredictions):
        self.positions.append(position)
        self.experiment_names.append(experiment_name)
        self.cell_types.append(cell_type)
        self.is_live.append(is_live)
        self.ages.append(age)
        self.predictions.append(cell_type_predictions)

    def write_csv(self, output_file: str):
        # Find all the cell types that are present in the predictions
        predicted_cell_types = set()
        for predictions in self.predictions:
            for cell_type_name in predictions.cell_type_names:
                predicted_cell_types.add(cell_type_name)
        predicted_cell_types = sorted(predicted_cell_types)
        predicted_cell_types_row_names = [f"{cell_type_name.lower()}_predicted_probability" for cell_type_name in predicted_cell_types]

        with open(output_file, "w", newline="") as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(["organoid", "x_px", "y_px", "z_px", "is_fixed", "organoid_age", "ki67_positive",
                                 "lgr5_positive", "krt20_positive", "wga_positive", *predicted_cell_types_row_names])

            for position, experiment_name, cell_type, is_live, age, predictions in zip(
                    self.positions, self.experiment_names, self.cell_types, self.is_live, self.ages, self.predictions):
                ki67_positive = 0
                lgr5_positive = 0
                krt20_positive = 0
                wga_positive = 0
                if cell_type == "STEM_PUTATIVE":
                    ki67_positive = 1
                elif cell_type == "STEM":
                    lgr5_positive = 1
                elif cell_type == "ENTEROCYTE":
                    krt20_positive = 1
                elif cell_type == "ABSORPTIVE_PRECURSOR":
                    krt20_positive = 1
                    ki67_positive = 1
                elif cell_type == "PANETH":
                    wga_positive = 1

                # Null out values that we can't measure for the given condition (they would be False otherwise, which could
                # be confused with a real negative)
                if is_live:
                    ki67_positive = float("NaN")
                    krt20_positive = float("NaN")
                    wga_positive = float("NaN")
                else:
                    lgr5_positive = float("NaN")
                is_fixed = 1 if not is_live else 0

                cell_type_probabilities = list()
                for cell_type_name in predicted_cell_types:
                    cell_type_probabilities.append(predictions.get_probability(cell_type_name))

                csv_writer.writerow([experiment_name.get_name(), position.x, position.y, position.z,
                                     is_fixed, age.get_age_text(), ki67_positive, lgr5_positive, krt20_positive, wga_positive] + cell_type_probabilities)


def _get_age(experiment_name: Name) -> _Age:
    if "A1 »" in experiment_name.get_name():
        return _Age.YOUNG  # This well contained young organoids
    if "B2 »" in experiment_name.get_name():
        return _Age.OLD  # This well contained old organoids
    raise ValueError(f"Cannot determine age from experiment name {experiment_name}")


def main():
    stainings = _CellInfoTable()
    _count_stainings(stainings)

    # Write the counts to a CSV file
    stainings.write_csv("cell_info_table.csv")


def _get_cell_type_predictions(experiment: Experiment, position: Position) -> _CellTypePredictions:
    cell_types = experiment.global_data.get_data("ct_probabilities")
    if cell_types is None:
        raise ValueError(f"Experiment {experiment.name.get_name()} does not contain cell type predictions")
    cell_type_probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
    if cell_type_probabilities is None:
        return NO_PREDICTIONS
    return _CellTypePredictions(cell_type_names=cell_types, cell_type_probabilities=cell_type_probabilities)


def _count_stainings(stainings: _CellInfoTable):
    experiments_live = list(list_io.load_experiment_list_file(_DATA_FILE_LIVE, load_images=False))
    experiments_staining = list(list_io.load_experiment_list_file(_DATA_FILE_STAINING, load_images=False))
    for experiment_staining in experiments_staining:
        experiment_live = None
        for experiment_live_candidate in experiments_live:
            if experiment_live_candidate.name.get_name() == experiment_staining.name.get_name():
                experiment_live = experiment_live_candidate
                break
        if experiment_live is None:
            continue

        age = _get_age(experiment_staining.name)
        for position in experiment_staining.positions.of_time_point(experiment_staining.first_time_point()):
            cell_type = position_markers.get_position_type(experiment_staining.position_data, position)
            predictions = _get_cell_type_predictions(experiment_staining, position)
            stainings.add_cell(position, experiment_name=experiment_staining.name, cell_type=cell_type, is_live=False,
                               age=age, cell_type_predictions=predictions)
        for position in experiment_live.positions.of_time_point(experiment_live.first_time_point()):
            cell_type = position_markers.get_position_type(experiment_live.position_data, position)
            predictions = _get_cell_type_predictions(experiment_live, position)
            stainings.add_cell(position, experiment_name=experiment_staining.name, cell_type=cell_type, is_live=True,
                               age=age, cell_type_predictions=predictions)


if __name__ == "__main__":
    main()

