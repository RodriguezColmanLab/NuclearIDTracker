"""
Plots the intensities of Lgr5 and the probabilities of stem cells along the crypt axis for each experiment. They
should show some correlation, although Lgr5 is only expected at the bottom of the crypt.
"""

from typing import NamedTuple, Any

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lib_figures
import lib_data
from organoid_tracker.core import Color, MPLColor

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.util.moving_average import MovingAverage

_INPUT_FILE = "../../Data/Testing data - output - CD24 and Lgr5.autlist"


_LGR_COLOR = "#891e1f"
_STEM_CELL_PROBABILITY_COLOR = "#00ff00"
_WINDOW_WIDTH_UM = 20


class _IntensitiesAlongCryptAxis:
    experiment_name: str

    crypt_axis_positions_um: list[float]
    lgr5_intensities: list[float]
    stem_cell_probabilities: list[float]

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.crypt_axis_positions_um = []
        self.lgr5_intensities = []
        self.stem_cell_probabilities = []

    def add_position(self, crypt_axis_position_um: float, lgr5_intensity: float, stem_cell_probability: float):
        self.crypt_axis_positions_um.append(crypt_axis_position_um)
        self.lgr5_intensities.append(lgr5_intensity)
        self.stem_cell_probabilities.append(stem_cell_probability)

    def plot_lgr5_intensity(self, ax: Axes):


        moving_average = MovingAverage(self.crypt_axis_positions_um, self.lgr5_intensities, window_width=_WINDOW_WIDTH_UM)
        moving_average.plot(ax, color=Color.from_matplotlib(_LGR_COLOR))

        ax.scatter(self.crypt_axis_positions_um, self.lgr5_intensities, color=_LGR_COLOR, s=3, lw=0)

    def plot_stem_cell_probability(self, ax: Axes):
        moving_average = MovingAverage(self.crypt_axis_positions_um, self.stem_cell_probabilities, window_width=_WINDOW_WIDTH_UM)
        moving_average.plot(ax, color=Color.from_matplotlib(_STEM_CELL_PROBABILITY_COLOR), error_opacity=0.95)

        ax.scatter(self.crypt_axis_positions_um, self.stem_cell_probabilities, color=_STEM_CELL_PROBABILITY_COLOR, s=3, lw=0)


def main():
    all_crypt_axes = list()

    for experiment in list_io.load_experiment_list_file(_INPUT_FILE):
        if experiment.name.get_name() == "E409-1":
            continue  # EXCLUSION, does not have a villus
        _measure_experiment(experiment, into=all_crypt_axes)

    # EXCLUSION - only include crypt axis that are fully in the view
    all_crypt_axes = [crypt_axis for crypt_axis in all_crypt_axes if max(crypt_axis.crypt_axis_positions_um) > 30]

    _plot(all_crypt_axes)


def _measure_experiment(experiment: Experiment, *, into: list[_IntensitiesAlongCryptAxis]):
    crypt_axes = dict()

    cell_types = experiment.global_data.get_data("ct_probabilities")

    resolution = experiment.images.resolution()

    first_time_point = experiment.positions.first_time_point()
    for position in experiment.positions.of_time_point(first_time_point):
        spline_position = experiment.splines.to_position_on_spline(position)
        if spline_position is None:
            continue
        if spline_position.spline_id not in crypt_axes:
            crypt_axes[spline_position.spline_id] = _IntensitiesAlongCryptAxis(experiment.name.get_name())

        crypt_axis = crypt_axes[spline_position.spline_id]
        lgr5_intensity = intensity_calculator.get_normalized_intensity(experiment, position,
                                                                       intensity_key="intensity_lgr5", per_pixel=True)
        ct_probabilities = experiment.positions.get_position_data(position, "ct_probabilities")
        if ct_probabilities is None or lgr5_intensity is None:
            continue

        stem_to_ec_prob = lib_data.find_stem_to_ec_location(cell_types, ct_probabilities)
        if stem_to_ec_prob is None:
            continue
        crypt_axis.add_position(spline_position.pos * resolution.pixel_size_x_um, lgr5_intensity, stem_to_ec_prob)

    into.extend(crypt_axes.values())


def _plot(all_crypt_axes: list[_IntensitiesAlongCryptAxis]):
    figure = lib_figures.new_figure()
    ax_lgr5 = figure.gca()

    ax_stem_cell_prob = ax_lgr5.twinx()

    for crypt_axis in all_crypt_axes:
        crypt_axis.plot_lgr5_intensity(ax_lgr5)
        crypt_axis.plot_stem_cell_probability(ax_stem_cell_prob)

    ax_stem_cell_prob.set_ylim(0, 0.7)
    ax_stem_cell_prob.set_ylabel("Stem-to-EC probability", color=_STEM_CELL_PROBABILITY_COLOR)
    ax_lgr5.set_ylabel("LGR5 intensity/px", color=_LGR_COLOR)
    ax_lgr5.set_xlabel("Distance from crypt bottom (μm)")

    plt.show()

if __name__ == "__main__":
    main()
