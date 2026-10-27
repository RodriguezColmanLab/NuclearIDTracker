from enum import Enum, auto

import numpy
from matplotlib import pyplot as plt

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io

_DATA_FILE = "../../Data/Testing data - predictions - treatments - fixed.autlist"


class _Condition(Enum):
    CONTROL = auto()
    # DAPT_CHIR = auto()  # Not plotted
    # CHIR_VPA = auto()  #  Skipped, as KRT20 signal was everywhere, which indicates the experiment failed
    NO_RSPONDIN = auto()

    @property
    def display_name(self):
        return self.name.lower().replace("_", " ")


class _ProbabilitiesByCondition:
    _enterocyte_probabilities: dict[_Condition, dict[str, list[float]]]

    def __init__(self):
        self._enterocyte_probabilities = dict()

    def add_one(self, condition: _Condition, cell_types: list[str], probabilities: list[float]):
        if condition not in self._enterocyte_probabilities:
            self._enterocyte_probabilities[condition] = dict()
        probabilities_all = self._enterocyte_probabilities[condition]

        main_cell_type = cell_types[numpy.argmax(probabilities)]
        if main_cell_type not in probabilities_all:
            probabilities_all[main_cell_type] = list()

        enterocyte_probability = probabilities[cell_types.index("ENTEROCYTE")]
        self._enterocyte_probabilities[condition][main_cell_type].append(enterocyte_probability)

    def get_enterocyte_probabilities(self, condition: _Condition, main_cell_type: str) -> list[float]:
        """Gets the probabilities of being an enterocyte for all cells with the given main cell type, for the given condition."""
        if condition not in self._enterocyte_probabilities:
            return []
        probabilities_by_type = self._enterocyte_probabilities[condition]
        if main_cell_type not in probabilities_by_type:
            return []
        return probabilities_by_type[main_cell_type]

    def get_enterocyte_probabilities_others(self, condition: _Condition, excluding_cell_type: str) -> list[float]:
        """Gets the probabilities of being an enterocyte for all cells with a main cell type other than the given one,
        for the given condition."""
        if condition not in self._enterocyte_probabilities:
            return []
        probabilities_by_type = self._enterocyte_probabilities[condition]
        probabilities = list()
        for cell_type, probabilities_for_type in probabilities_by_type.items():
            if cell_type == excluding_cell_type:
                continue
            probabilities.extend(probabilities_for_type)
        return probabilities


def _get_condition(experiment: Experiment) -> _Condition | None:
    name = experiment.name.get_name()
    if "control" in name:
        return _Condition.CONTROL
    if "dapt chir" in name:
        return None  # _Condition.DAPT_CHIR not plotted
    if "chir vpa" in name:
        return None  # Skipped, as KRT20 signal was everywhere, which indicates the experiment failed
    if "EN" in name:
        return _Condition.NO_RSPONDIN
    raise ValueError("Unknown condition: " + name)


def main():
    # Load the trajectories
    experiments = list_io.load_experiment_list_file(_DATA_FILE)

    counts = _ProbabilitiesByCondition()
    for experiment in experiments:
        condition = _get_condition(experiment)
        if condition is None:
            continue  # Condition that is not plotted
        cell_types: list[str] = experiment.global_data.get_data("ct_probabilities")
        for position in experiment.positions:
            probabilities: list[float] | None = experiment.positions.get_position_data(position, "ct_probabilities")
            if probabilities is None:
                continue
            counts.add_one(condition, cell_types, probabilities)

    # Plot the counts
    figure = lib_figures.new_figure(size=(3.5, 2.5))
    axes = figure.subplots(nrows=len(_Condition), ncols=1, sharex=True)

    bins = numpy.linspace(0, 1, 51)

    for condition, ax in zip(_Condition, axes):
        ax.set_title(condition.display_name)

        enterocyte_probabilities = counts.get_enterocyte_probabilities(condition, "ENTEROCYTE")
        ax.hist(enterocyte_probabilities, bins=bins, color=lib_figures.CELL_TYPE_PALETTE["ENTEROCYTE"], alpha=0.8, label=f"Enterocyte (n={len(enterocyte_probabilities)})")
        mean = numpy.mean(enterocyte_probabilities)
        ax.axvline(mean, color="black", linestyle="dashed")
        ax.text(mean, 0.98, f"mean", ha="left", va="bottom", transform=ax.get_xaxis_transform())

        # enterocyte_probabilities_others = counts.get_enterocyte_probabilities_others(condition, "ENTEROCYTE")
        # ax.hist(enterocyte_probabilities_others, bins=bins, color="black", alpha=0.2, label=f"Other cells (n={len(enterocyte_probabilities_others)})")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 1.0)
        ax.set_xticks(numpy.linspace(0, 1, 6, endpoint=True))

    axes[-1].set_xlabel("Enterocyte probability")
    axes[len(axes) // 2].set_ylabel("Cell count")
    figure.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
