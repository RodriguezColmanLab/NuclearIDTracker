from typing import List

import numpy
from matplotlib import pyplot as plt

from organoid_tracker.core.beacon_collection import BeaconCollection
from organoid_tracker.core.spline import SplinePosition
from organoid_tracker.imaging import list_io
import lib_data
import lib_figures
from numpy import ndarray

# _DATA_FILE = "../../Data/Tracking data as controls/Dataset.autlist"
_DATA_FILE = "../../Data/Stem cell regeneration/Dataset - during DT treatment.autlist"

_CRYPT_BIN_SIZE = 0.1
_STEM_TO_EC_BIN_SIZE = 0.05


class _BinnedData:
    """Stores the counts for each crypt-to-villus bin and each stem-to-ec bin (so 2D count data)."""

    _count_data: ndarray
    max_plotted_crypt_position: float

    def __init__(self):
        self.max_plotted_crypt_position = 2
        self._count_data = numpy.zeros(
            (int(self.max_plotted_crypt_position / _CRYPT_BIN_SIZE), int(1 / _STEM_TO_EC_BIN_SIZE)), dtype=numpy.uint16)

    def add_position(self, crypt_position: float, stem_to_ec_position: float):
        crypt_bin = int(crypt_position / _CRYPT_BIN_SIZE)
        stem_to_ec_bin = int(stem_to_ec_position / _STEM_TO_EC_BIN_SIZE)
        if crypt_bin >= self._count_data.shape[0] or stem_to_ec_bin >= self._count_data.shape[1]:
            return
        self._count_data[crypt_bin, stem_to_ec_bin] += 1

    def crypt_villus_positions(self) -> List[float]:
        """Returns the centers of the crypt bins."""
        return [(i + 0.5) * _CRYPT_BIN_SIZE for i in range(self._count_data.shape[0])]

    def stem_to_ec_positions(self) -> List[float]:
        """Returns the lower bound of the stem-to-ec bins."""
        return [i * _STEM_TO_EC_BIN_SIZE for i in range(self._count_data.shape[1])]

    def get_fraction(self, crypt_bin: int, stem_to_ec_bin: int) -> float:
        """Gets the fraction of cells that fall in the given stem-to-ec bin for the given crypt bin."""
        return self._count_data[crypt_bin, stem_to_ec_bin] / self._count_data[crypt_bin].sum()

    def get_fractions(self, stem_to_ec_bin: int) -> List[float]:
        """Gets the fractions for all crypt bins for the given stem-to-ec bin."""
        return [self.get_fraction(crypt_bin, stem_to_ec_bin) for crypt_bin in range(self._count_data.shape[0])]

    def get_count_for_stem_to_ec_bin(self, stem_to_ec_bin: int) -> int:
        """Gets the total count for the given stem-to-ec bin."""
        return self._count_data[:, stem_to_ec_bin].sum()

    def get_count_for_crypt_villus_bin(self, crypt_villus_bin: int) -> int:
        """Gets the total count for the given crypt-villus bin."""
        return self._count_data[crypt_villus_bin].sum()


def main():
    binned_data = _BinnedData()

    for experiment in list_io.load_experiment_list_file(_DATA_FILE, load_images=False):
        beacons = experiment.beacons
        splines = experiment.splines
        position_data = experiment.position_data
        cell_types = experiment.global_data.get_data("ct_probabilities")
        if cell_types is None:
            raise ValueError(f"Cell types not found in experiment {experiment.name}")

        for time_point in experiment.positions.time_points():
            for position in experiment.positions.of_time_point(time_point):
                spline_position = splines.to_position_on_spline(position, only_axis=True)
                if spline_position is None:
                    continue
                crypt_axis_position = _make_axis_position_relative_to_neck(beacons, spline_position)

                ct_probabilities = position_data.get_position_data(position, "ct_probabilities")
                if ct_probabilities is None:
                    continue
                stem_to_ec_location = lib_data.find_stem_to_ec_location(cell_types, ct_probabilities)
                if stem_to_ec_location is None:
                    continue

                binned_data.add_position(crypt_axis_position, stem_to_ec_location)

    # Plot the data
    figure = lib_figures.new_figure(size=(3.5, 2.5))
    crypt_villus_positions = binned_data.crypt_villus_positions()

    ax = figure.gca()

    # Draw all the lines

    # Make lines extend to the edges of the plot
    x_values = numpy.array(crypt_villus_positions)
    x_values[0] -= _CRYPT_BIN_SIZE / 2
    x_values[-1] += _CRYPT_BIN_SIZE / 2

    bottom_values = numpy.zeros(len(x_values), dtype=numpy.float32)
    for stem_to_ec_bin, stem_to_ec_value in enumerate(binned_data.stem_to_ec_positions()):
        label = f"{stem_to_ec_value:.2f}-{stem_to_ec_value + _STEM_TO_EC_BIN_SIZE:.2f}"
        if binned_data.get_count_for_stem_to_ec_bin(stem_to_ec_bin) < 100:
            label = None  # So small, don't show in legend, it's not visible anyway
        color = lib_figures.get_stem_to_ec_color(stem_to_ec_value)
        y_values = binned_data.get_fractions(stem_to_ec_bin)
        ax.fill_between(x_values, bottom_values, bottom_values + y_values, color=color, label=label)
        bottom_values += y_values

    ax.set_xlabel("Crypt position")
    ax.set_ylabel("Fraction")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Stem-to-EC location", title_fontsize="small")
    plt.show()


def _make_axis_position_relative_to_neck(beacons: BeaconCollection, spline_position: SplinePosition) -> float:
    """Makes the axis position relative, such that the position of the closest beacon to the spline without a defined
     type  is defined as 1.0."""
    closest_beacon_spline_position = None
    for beacon in beacons.of_time_point_with_type(spline_position.time_point):
        if beacon.beacon_type is not None:
            continue  # Some specialized beacon, ignore
        beacon_spline_position = spline_position.spline.to_position_on_axis(beacon.position)
        if closest_beacon_spline_position is None or beacon_spline_position.distance < closest_beacon_spline_position.distance:
            closest_beacon_spline_position = beacon_spline_position
    return spline_position.pos / closest_beacon_spline_position.pos


if __name__ == "__main__":
    main()
