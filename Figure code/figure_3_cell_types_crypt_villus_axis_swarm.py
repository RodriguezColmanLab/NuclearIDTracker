from typing import List, Tuple, Iterable

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from organoid_tracker.core.beacon_collection import BeaconCollection
from organoid_tracker.core.spline import SplinePosition
from organoid_tracker.imaging import list_io
import lib_data
import lib_figures
from numpy import ndarray

# _DATA_FILE = "../../Data/Tracking data as controls/Dataset.autlist"
_DATA_FILE = "../../Data/Stem cell regeneration/Dataset - during DT treatment.autlist"

_CRYPT_BIN_SIZE = 0.1
_MAX_AXIS_POSITION_IN_AVERAGE = 2.0

_EXCLUDED_CRYPTS = {
    # Crypts that are only in the view partially
    "20240130 pos3A-DTadd": [1, 3, 4, 5],
    "20240130 pos7-DTadd": [2, 5, 6, 7],
    "20240130 pos1-DTadd": [2, 5],
    "20240130 pos14-DTadd": []
}


class _CellsInCrypt:
    """Stores the counts for each crypt-to-villus bin and each stem-to-ec bin (so 2D count data)."""

    crypt_positions: List[float]
    colors: List[Tuple[float, float, float]]
    probabilities: List[List[float]]
    cell_types: List[str]
    name: str

    def __init__(self, name: str, cell_types: List[str]):
        self.name = name
        self.crypt_positions = list()
        self.colors = list()
        self.cell_types = cell_types
        self.probabilities = list()

    def add_position(self, crypt_position: float, probabilities: List[float]):
        self.crypt_positions.append(crypt_position)
        color = lib_figures.get_mixed_cell_type_color(self.cell_types, probabilities)
        self.colors.append(color)
        self.probabilities.append(probabilities)

    def paneth_positions(self) -> Iterable[float]:
        """Gets the positions of the cells predicted as Paneth cells in the crypt."""
        paneth_index = self.cell_types.index("PANETH")
        for crypt_position, probabilities in zip(self.crypt_positions, self.probabilities):
            if numpy.argmax(probabilities) == paneth_index:
                yield crypt_position



class _AverageProbabilities:

    _bin_probability_sums: List[ndarray]
    _bin_counts: List[int]
    _cell_types: List[str]

    def __init__(self, cell_types: List[str], max_axis_position: float):
        self._cell_types = cell_types

        bin_count = int(max_axis_position / _CRYPT_BIN_SIZE)

        self._bin_probability_sums = [numpy.zeros(len(cell_types)) for _ in range(bin_count)]
        self._bin_counts = [0 for _ in range(bin_count)]

    def add_position(self, crypt_position: float, probabilities: List[float], cell_types: List[str]):
        bin_index = int(crypt_position / _CRYPT_BIN_SIZE)
        for cell_type, probability in zip(cell_types, probabilities):
            cell_type_index = self._cell_types.index(cell_type)
            self._bin_probability_sums[bin_index][cell_type_index] += probability
        self._bin_counts[bin_index] += 1

    def get_average_probability_image(self) -> ndarray:
        color_image = numpy.zeros((1, len(self._bin_probability_sums), 3), dtype=numpy.float32)
        for i, (sums, count) in enumerate(zip(self._bin_probability_sums, self._bin_counts)):
            if count == 0:
                color_image[0, i] = color_image[0, i - 1]
                continue
            average_probabilities = sums / count
            color = lib_figures.get_mixed_cell_type_color(self._cell_types, average_probabilities)
            color_image[0, i] = color
        return color_image



def main():
    data_by_crypt = list()

    for experiment in list_io.load_experiment_list_file(_DATA_FILE, load_images=False):
        excluded_crypts = _EXCLUDED_CRYPTS.get(experiment.name.get_name(), [])

        beacons = experiment.beacons
        splines = experiment.splines
        position_data = experiment.position_data
        cell_types = experiment.global_data.get_data("ct_probabilities")
        if cell_types is None:
            raise ValueError(f"Cell types not found in experiment {experiment.name}")

        for spline_id, spline in splines.of_time_point(splines.reference_time_point()):
            if spline_id in excluded_crypts:
                continue
            binned_data = _CellsInCrypt(experiment.name.get_name() + " / Crypt" + str(spline_id), cell_types)

            for time_point in experiment.positions.time_points():
                for position in experiment.positions.of_time_point(time_point):
                    spline_position = splines.to_position_on_spline(position, only_axis=True)
                    if spline_position is None:
                        continue
                    if spline_position.spline_id != spline_id:
                        continue
                    crypt_axis_position = _make_axis_position_relative_to_neck(beacons, spline_position)

                    ct_probabilities = position_data.get_position_data(position, "ct_probabilities")
                    if ct_probabilities is None:
                        continue

                    binned_data.add_position(crypt_axis_position, ct_probabilities)

            data_by_crypt.append(binned_data)

    # Plot the data
    figure = lib_figures.new_figure(size=(4.5, 5.5))

    ax_main, ax_average, ax_paneth = figure.subplots(nrows=3, ncols=1, height_ratios=[20, 1, 1], sharex=True, sharey=False)

    generator = numpy.random.Generator(numpy.random.MT19937(seed=1))
    average_probability = _AverageProbabilities(data_by_crypt[0].cell_types, _MAX_AXIS_POSITION_IN_AVERAGE)
    for i, binned_data in enumerate(data_by_crypt):
        random_y_positions = generator.normal(i, 0.07, len(binned_data.crypt_positions))
        ax_main.scatter(binned_data.crypt_positions, random_y_positions, s=10, c=binned_data.colors, linewidth=0)
        for crypt_position, cell_probabilities in zip(binned_data.crypt_positions, binned_data.probabilities):
            average_probability.add_position(crypt_position, cell_probabilities, binned_data.cell_types)
    ax_main.set_yticks(range(len(data_by_crypt)))
    ax_main.set_yticklabels([binned_data.name for binned_data in data_by_crypt])
    figure.tight_layout()
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    ax_average.imshow(average_probability.get_average_probability_image(), extent=(0, _MAX_AXIS_POSITION_IN_AVERAGE, 0, 1))
    ax_average.set_aspect("auto")
    ax_average.set_yticks([0.5])
    ax_average.tick_params(axis="y", which="both", color=(0, 0, 0, 0))  # Hide the tick itself
    ax_average.set_yticklabels(["Average cell type"])
    ax_average.spines["top"].set_visible(False)
    ax_average.spines["right"].set_visible(False)
    ax_average.spines["left"].set_visible(False)

    _draw_paneths(ax_paneth, data_by_crypt)
    ax_paneth.set_yticks([0.5])
    ax_paneth.set_yticklabels(["Paneth cells"])
    ax_paneth.tick_params(axis="y", which="both", color=(0, 0, 0, 0))  # Hide the tick itself
    ax_paneth.set_ylim(0, 1)
    ax_paneth.spines["top"].set_visible(False)
    ax_paneth.spines["right"].set_visible(False)
    ax_paneth.spines["left"].set_visible(False)
    ax_paneth.set_xlabel("Position along crypt-villus axis")

    figure.tight_layout()
    plt.show()


def _draw_paneths(ax: Axes, data_by_crypt: Iterable[_CellsInCrypt]):
    paneth_color = lib_figures.get_mixed_cell_type_color(["PANETH", "STEM", "ENTEROCYTE"], [1, 0, 0])
    x_positions = list()
    for crypt in data_by_crypt:
        x_positions += list(crypt.paneth_positions())
    random = numpy.random.Generator(numpy.random.MT19937(seed=2))
    ax.scatter(x_positions, random.normal(loc=0.5, scale=0.15, size=len(x_positions)),
               s=5, c=[paneth_color], linewidth=0, marker="D")


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
