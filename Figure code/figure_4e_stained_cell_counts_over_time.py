"""This plots the number of Paneth cells over time compared to the number of stem cells - during ablation and afterwards."""
from collections import defaultdict
from enum import Enum, auto
from typing import Dict, NamedTuple, List, Optional

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy.random
from matplotlib.figure import Figure

import lib_figures
from organoid_tracker.core import Name, MPLColor
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io, io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE_STAINING = "../../Data/Live and immunostaining multiple time points/Immunostaining all.autlist"
_DATA_FILE_LIVE = "../../Data/Live and immunostaining multiple time points/Live all.autlist"

_REGENERATION_OFFSET_H = 16


class _ExperimentCondition(Enum):
    OLD_CONTROL = auto()
    DIRECTLY_AFTER_DT = auto()
    AFTER_DT_8H = auto()
    AFTER_DT_24H = auto()
    AFTER_DT_48H = auto()

    @staticmethod
    def from_letter(letter: str) -> Optional["_ExperimentCondition"]:
        """Gets the condition from the single uppercase letter."""
        if letter == "A":
            return None
        elif letter == "B":
            return _ExperimentCondition.OLD_CONTROL
        elif letter == "C":
            return _ExperimentCondition.DIRECTLY_AFTER_DT
        elif letter == "D":
            return _ExperimentCondition.AFTER_DT_8H
        elif letter == "E":
            return _ExperimentCondition.AFTER_DT_24H
        elif letter == "F":
            return _ExperimentCondition.AFTER_DT_48H
        else:
            raise ValueError(f"Unknown experiment condition {letter}")

    def display_name(self) -> str:
        if self == self.OLD_CONTROL:
            return "Control"
        elif self == self.DIRECTLY_AFTER_DT:
            return "After DT"
        else:
            return f"+{self.time_h()}h"

    def time_h(self) -> int:
        """Returns the time in hours since the start of the experiment."""
        if self == self.OLD_CONTROL:
            return -_REGENERATION_OFFSET_H
        elif self == self.DIRECTLY_AFTER_DT:
            return 0
        elif self == self.AFTER_DT_8H:
            return 8
        elif self == self.AFTER_DT_24H:
            return 24
        elif self == self.AFTER_DT_48H:
            return 48
        else:
            raise ValueError(f"Unknown experiment condition {self.name}")


class _PanethCellCount(NamedTuple):
    ki67_positive_count: int
    wga_positive_count: int
    krt20_positive_count: int

    def is_zero(self) -> bool:
        """Returns true if all counts are zero."""
        return self.ki67_positive_count == 0 and self.wga_positive_count == 0 and self.krt20_positive_count == 0


def _count_staining(experiment: Experiment) -> _PanethCellCount:
    ki67_positive_count = 0
    wga_positive_count = 0
    krt20_positive_count = 0
    for position in experiment.positions.of_time_point(experiment.positions.first_time_point()):
        cell_type = position_markers.get_position_type(experiment.position_data, position)
        if cell_type is None:
            continue
        if cell_type == "PANETH":
            wga_positive_count += 1
        elif cell_type == "STEM_PUTATIVE":
            ki67_positive_count += 1
        elif cell_type == "ABSORPTIVE_PRECURSOR":
            ki67_positive_count += 1
            krt20_positive_count += 1
        elif cell_type == "ENTEROCYTE":
            krt20_positive_count += 1
        else:
            print(f"Unhandled cell type {cell_type} in {experiment.name} at {position}")

    return _PanethCellCount(ki67_positive_count=ki67_positive_count, wga_positive_count=wga_positive_count,
                            krt20_positive_count=krt20_positive_count)


def _count_lgr5(experiment: Experiment) -> int:
    lgr5_count = 0
    for position in experiment.positions.of_time_point(experiment.positions.first_time_point()):
        cell_type = position_markers.get_position_type(experiment.position_data, position)
        if cell_type is None:
            continue
        if cell_type == "STEM":
            lgr5_count += 1
        else:
            print(f"Unhandled cell type {cell_type} in {experiment.name} at {position}")
    return lgr5_count


def main():
    results_stained = defaultdict(list)
    results_live = defaultdict(list)

    experiments_stained = list(list_io.load_experiment_list_file(_DATA_FILE_STAINING, load_images=False))
    experiments_live = list(list_io.load_experiment_list_file(_DATA_FILE_LIVE, load_images=False))

    # Count the cells in the staining images
    for experiment in experiments_stained:
        name = _parse_experiment_name(experiment.name)
        counts = _count_staining(experiment)
        if counts.is_zero():
            continue  # Not annotated

        condition = _ExperimentCondition.from_letter(name[0])
        if condition is None:
            continue  # Not interested in this condition

        results_stained[condition].append(counts)

        # Count the LGR5 cells in the corresponding live images
        found_live_experiment = False
        for experiment_live in experiments_live:
            if _parse_experiment_name(experiment_live.name) == name:
                lgr5_count = _count_lgr5(experiment_live)
                results_live[condition].append(lgr5_count)
                found_live_experiment = True
                break
        if not found_live_experiment:
            raise ValueError(f"Could not find live experiment for {name}")

    figure = lib_figures.new_figure(size=(3.2, 3))
    _plot_counts(figure, results_stained, results_live)
    figure.tight_layout()
    plt.show()

    figure = lib_figures.new_figure(size=(3.2, 3))
    _plot_ratio(figure, results_stained, results_live)
    figure.tight_layout()
    plt.show()


def _make_darker(color: MPLColor) -> str:
    rgb_color = matplotlib.colors.to_rgb(color)
    h, s, v = matplotlib.colors.rgb_to_hsv(rgb_color)
    v = max(0, v - 0.3)
    return matplotlib.colors.hsv_to_rgb((h, s, v))


def _plot_counts(figure: Figure, results_stained: Dict[_ExperimentCondition, List[_PanethCellCount]],
                 results_lgr5: Dict[_ExperimentCondition, List[int]]):
    ax_ki67, ax_wga, ax_krt20, ax_lgr5 = numpy.array(figure.subplots(ncols=2, nrows=2, sharex=True)).flatten()
    generator = numpy.random.Generator(numpy.random.MT19937(seed=1))

    x_label_pos = list()
    x_label_names = list()
    for condition in _ExperimentCondition:
        x = condition.time_h()
        condition_results = results_stained[condition]

        ax_ki67.scatter(generator.normal(x, 1, len(condition_results)),
                        [result.ki67_positive_count for result in condition_results],
                        color=lib_figures.CELL_TYPE_PALETTE["STEM"], s=25, linewidth=0, alpha=0.8)
        ax_wga.scatter(generator.normal(x, 1, len(condition_results)),
                       [result.wga_positive_count for result in condition_results],
                       color=lib_figures.CELL_TYPE_PALETTE["PANETH"], s=25, linewidth=0, alpha=0.8)
        ax_krt20.scatter(generator.normal(x, 1, len(condition_results)),
                         [result.krt20_positive_count for result in condition_results],
                         color=lib_figures.CELL_TYPE_PALETTE["ENTEROCYTE"], s=25, linewidth=0, alpha=0.8)
        ax_lgr5.scatter(generator.normal(x, 1, len(results_lgr5[condition])), results_lgr5[condition],
                        color=lib_figures.CELL_TYPE_PALETTE["STEM"], s=25, linewidth=0, alpha=0.8)

        x_label_pos.append(x)
        x_label_names.append(condition.display_name())

    # Set the y-axis limits
    max_ki67 = max([max([result.ki67_positive_count for result in condition_results]) for condition_results in
                    results_stained.values()])
    max_wga = max([max([result.wga_positive_count for result in condition_results]) for condition_results in
                   results_stained.values()])
    max_krt20 = max([max([result.krt20_positive_count for result in condition_results]) for condition_results in
                     results_stained.values()])
    max_lgr5 = max([max(condition_results) for condition_results in results_lgr5.values()])
    ax_ki67.set_ylim(0, max_ki67 * 1.1)
    ax_wga.set_ylim(0, max_wga * 1.1)
    ax_krt20.set_ylim(0, max_krt20 * 1.1)
    ax_lgr5.set_ylim(0, max_lgr5 * 1.1)

    # Add plot titles
    ax_ki67.text(1, 1, "Ki67+ cells", ha="right", va="top", transform=ax_ki67.transAxes)
    ax_wga.text(1, 1, "WGA+ cells", ha="right", va="top", transform=ax_wga.transAxes)
    ax_krt20.text(1, 1, "KRT20+ cells", ha="right", va="top", transform=ax_krt20.transAxes)
    ax_lgr5.text(1, 1, "LGR5+ cells", ha="right", va="top", transform=ax_lgr5.transAxes)

    # Add average lines
    x_values = [condition.time_h() for condition in _ExperimentCondition]
    y_values_ki67 = [numpy.mean([result.ki67_positive_count for result in results_stained[condition]]) for condition in
                     _ExperimentCondition]
    y_values_wga = [numpy.mean([result.wga_positive_count for result in results_stained[condition]]) for condition in
                    _ExperimentCondition]
    y_values_krt20 = [numpy.mean([result.krt20_positive_count for result in results_stained[condition]]) for condition
                      in _ExperimentCondition]
    y_values_lgr5 = [numpy.mean(results_lgr5[condition]) for condition in _ExperimentCondition]
    ax_ki67.plot(x_values, y_values_ki67, color=_make_darker(lib_figures.CELL_TYPE_PALETTE["STEM"]), linewidth=1.5,
                 zorder=-10)
    ax_wga.plot(x_values, y_values_wga, color=_make_darker(lib_figures.CELL_TYPE_PALETTE["PANETH"]), linewidth=1.5,
                zorder=-10)
    ax_krt20.plot(x_values, y_values_krt20, color=_make_darker(lib_figures.CELL_TYPE_PALETTE["ENTEROCYTE"]),
                  linewidth=1.5, zorder=-10)
    ax_lgr5.plot(x_values, y_values_lgr5, color=_make_darker(lib_figures.CELL_TYPE_PALETTE["STEM"]), linewidth=1.5,
                 zorder=-10)

    # Set the x-axis limits and y label
    ax_krt20.set_xlim(min([c.time_h() for c in _ExperimentCondition]) - 10,
                      max([c.time_h() for c in _ExperimentCondition]) + 4)
    ax_ki67.set_ylabel("Cell count")

    # Spines and x-ticks
    for ax in [ax_ki67, ax_wga, ax_krt20, ax_lgr5]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(x_label_pos)
        ax.set_xticklabels(x_label_names, rotation=-45, ha="left")


def _plot_ratio(figure: Figure, results_stained: Dict[_ExperimentCondition, List[_PanethCellCount]],
                results_lgr5: Dict[_ExperimentCondition, List[int]]):
    """Plot the ratio of ste"""
    ax = figure.gca()

    generator = numpy.random.Generator(numpy.random.MT19937(seed=1))

    mean_ratios = list()
    times_h = list()
    for condition in _ExperimentCondition:
        condition_results = results_stained[condition]

        wga_positive_counts = numpy.array([result.wga_positive_count for result in condition_results])
        ki67_positive_counts = numpy.array([result.ki67_positive_count for result in condition_results])
        lgr5_positive_counts = numpy.array(results_lgr5[condition])
        ratios = wga_positive_counts / (lgr5_positive_counts + 1)

        ax.scatter(generator.normal(condition.time_h(), 1, len(ratios)), ratios,
                   color=lib_figures.CELL_TYPE_PALETTE["PANETH"], s=25, linewidth=0, alpha=0.8)

        mean_ratios.append(numpy.mean(ratios))
        times_h.append(condition.time_h())

    ax.plot(times_h, mean_ratios, color=_make_darker(lib_figures.CELL_TYPE_PALETTE["PANETH"]), linewidth=1.5)

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("WGA+ / (LGR5+ + 1) cell count")
    ax.set_xticks([condition.time_h() for condition in _ExperimentCondition])
    ax.set_xticklabels([condition.display_name() for condition in _ExperimentCondition], rotation=-45, ha="left")


def _parse_experiment_name(name: Name) -> str:
    """Parses the experiment name into a standardized format, to allow matching experiments."""
    name_str = name.get_name()
    name_str = name_str.replace("Position", "Pos")
    name_str_parts = name_str.split(" » ", maxsplit=1)
    if len(name_str_parts) != 2:
        return name_str  # Cannot parse, return original name
    return name_str_parts[0][0:2] + " » " + name_str_parts[1]


if __name__ == "__main__":
    main()
