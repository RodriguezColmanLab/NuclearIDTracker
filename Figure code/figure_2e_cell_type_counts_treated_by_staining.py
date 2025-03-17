from enum import Enum, auto
from typing import Dict, List

import numpy
from matplotlib import pyplot as plt

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE = "../../Data/Immunostaining conditions.autlist"


class _Condition(Enum):
    CONTROL = auto()
    NO_RSPONDIN = auto()

    @property
    def display_name(self):
        return self.name.lower().replace("_", " ")


class _CountByConditionAndCellType:

    _counts: Dict[_Condition, Dict[str, int]]

    def __init__(self):
        self._counts = dict()

    def add_one(self, condition: _Condition, cell_type: str):
        """Adds one to the count of cells with the given type in the given condition."""
        if condition not in self._counts:
            self._counts[condition] = dict()
        counts_for_condition = self._counts[condition]
        if cell_type not in counts_for_condition:
            counts_for_condition[cell_type] = 1
        else:
            counts_for_condition[cell_type] += 1

    def get_count_per_condition(self, condition: _Condition) -> int:
        """Gets the total amount of cells (of all types) for each condition."""
        if condition not in self._counts:
            return 0
        counts_for_condition = self._counts[condition]
        return sum(counts_for_condition.values())

    def cell_types(self) -> List[str]:
        """Gets all used cell types."""
        cell_types = set()
        for counts_for_condition in self._counts.values():
            cell_types |= counts_for_condition.keys()
        cell_types = list(cell_types)
        cell_types.sort(reverse=True)
        return cell_types

    def get_fraction(self, condition: _Condition, cell_type: str) -> float:
        """Gets the fraction of cells with that type, for the given condition."""
        if condition not in self._counts:
            return 0
        counts_for_condition = self._counts[condition]
        if cell_type not in counts_for_condition:
            return 0
        total_counts = sum(counts_for_condition.values())
        return counts_for_condition[cell_type] / total_counts


def _get_condition(experiment: Experiment) -> _Condition:
    name = experiment.name.get_name()

    if "control" in name:
        return _Condition.CONTROL
    if "EN" in name:
        return _Condition.NO_RSPONDIN
    raise ValueError("Unknown condition: " + name)


def _get_immunostained_names(cell_type: str) -> str:
    if cell_type == "UNLABELED":
        return "double-negative"
    if cell_type == "PANETH":
        return "lysozyme-positive"
    if cell_type == "ENTEROCYTE":
        return "KRT20-positive"
    return cell_type


def main():
    # Load the trajectories
    experiments = list_io.load_experiment_list_file(_DATA_FILE)
    counts = _CountByConditionAndCellType()
    for experiment in experiments:
        if "secondary only" in experiment.name.get_name():
            continue  # Skip these experiments
        if "chir vpa" in experiment.name.get_name():
            continue  # Also skipped, as KRT20 signal was everywhere, which indicates the experiment failed
        if "dapt chir" in experiment.name.get_name():
            continue  # In this panel, we only plot control and No Rspondin
        condition = _get_condition(experiment)
        print(experiment.name, condition)
        for position in experiment.positions:
            cell_type = position_markers.get_position_type(experiment.position_data, position)
            if cell_type is None:
                cell_type = "UNLABELED"
            counts.add_one(condition, cell_type)
    print(counts._counts)


    # Plot the counts
    figure = lib_figures.new_figure()
    ax = figure.gca()

    # Draw the bars, from top to bottom (to be consistent with the legend)
    y_offset_values = numpy.ones(len(_Condition), dtype=numpy.float32)
    for cell_type in counts.cell_types():
        x_values = list()
        height_values = list()
        for i, condition in enumerate(_Condition):
            x_values.append(i)
            height_values.append(-counts.get_fraction(condition, cell_type))
        ax.bar(x_values, height_values, bottom=y_offset_values, color=lib_figures.CELL_TYPE_PALETTE[cell_type],
               label=_get_immunostained_names(cell_type))
        y_offset_values += height_values

    for i, condition in enumerate(_Condition):
        ax.text(i, 1.1, str(counts.get_count_per_condition(condition)), horizontalalignment="center")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("Fraction")

    ax.set_xticks(list(range(len(_Condition))), [condition.display_name for condition in _Condition])
    ax.set_ylim(-0.05, 1.05)
    figure.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
