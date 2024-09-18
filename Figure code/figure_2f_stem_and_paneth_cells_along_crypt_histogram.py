from collections import defaultdict

import math
import matplotlib.cm
import numpy
from matplotlib import pyplot as plt
from typing import NamedTuple, List

import lib_figures
from organoid_tracker.imaging import list_io
from organoid_tracker.util.moving_average import MovingAverage

_EXPERIMENTS_FILE = "../../Data/Testing data - output - CD24 and Lgr5.autlist"


class _CryptAxis(NamedTuple):
    organoid_name: str
    crypt_id: int


class _CryptStemness:
    crypt_positions_um: List[float]
    is_stem_or_paneth: List[bool]
    time_points_h: List[float]

    def __init__(self):
        self.crypt_positions_um = list()
        self.is_stem_or_paneth = list()
        self.time_points_h = list()


def main():
    results = defaultdict(_CryptStemness)
    for experiment in list_io.load_experiment_list_file(_EXPERIMENTS_FILE):
        stem_index = experiment.global_data.get_data("ct_probabilities").index("STEM")
        paneth_index = experiment.global_data.get_data("ct_probabilities").index("PANETH")
        resolution = experiment.images.resolution()
        timings = experiment.images.timings()
        experiment_name = experiment.name.get_name()

        for time_point in experiment.time_points():
            for position in experiment.positions.of_time_point(time_point):
                probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
                if not probabilities:
                    continue
                is_stem_or_paneth = numpy.argmax(probabilities) == stem_index or numpy.argmax(probabilities) == paneth_index

                spline_position = experiment.splines.to_position_on_spline(position, only_axis=True)
                if spline_position is None:
                    continue

                crypt_axis = _CryptAxis(experiment_name, spline_position.spline_id)
                results[crypt_axis].crypt_positions_um.append(spline_position.pos * resolution.pixel_size_x_um)
                results[crypt_axis].is_stem_or_paneth.append(is_stem_or_paneth)
                results[crypt_axis].time_points_h.append(timings.get_time_h_since_start(time_point.time_point_number()))

    # EXCLUSION: Remove this particular crypt, since the bottom part is outside the imaged area
    del results[_CryptAxis("E409-3", 3)]

    figure = lib_figures.new_figure(size=(12, 8))
    axes = figure.subplots(ncols=3, nrows=math.ceil(len(results) / 3), sharex=True, sharey=True).flatten()
    for ax, (crypt_axis, result_of_axis) in zip(axes, results.items()):
        time_points_h = numpy.array(result_of_axis.time_points_h)
        stemness = numpy.array(result_of_axis.is_stem_or_paneth, dtype=numpy.float32)  # Convert from bool to float, for averaging
        crypt_positions_um = numpy.array(result_of_axis.crypt_positions_um)

        time_window_h = 3
        for time_start in range(int(time_points_h.min()), int(time_points_h.max()), time_window_h):
            time_end = time_start + time_window_h
            time_points_in_window = (time_points_h >= time_start) & (time_points_h < time_end)
            average = MovingAverage(crypt_positions_um[time_points_in_window], stemness[time_points_in_window], window_width=25)
            color = matplotlib.cm.bone_r(time_start / 20 + 0.2)
            ax.plot(average.x_values, average.mean_values, label=f"{time_start}-{time_end} h", color=color, linewidth=3, zorder=-time_start)

        ax.set_title(f"{crypt_axis.organoid_name}, crypt {crypt_axis.crypt_id}")
        ax.set_xlabel("Position (μm)")
        ax.set_ylabel("Fraction stem + Paneth cells")
        ax.legend()
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
