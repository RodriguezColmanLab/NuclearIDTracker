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
_TIME_WINDOW_H = 2


class _CryptAxis(NamedTuple):
    organoid_name: str
    crypt_id: int


class _CryptStemness:
    crypt_positions_um: List[float]
    is_stem_or_paneth: List[bool]

    def __init__(self):
        self.crypt_positions_um = list()
        self.is_stem_or_paneth = list()


def main():
    results = defaultdict(_CryptStemness)
    for experiment in list_io.load_experiment_list_file(_EXPERIMENTS_FILE):
        stem_index = experiment.global_data.get_data("ct_probabilities").index("STEM")
        paneth_index = experiment.global_data.get_data("ct_probabilities").index("PANETH")
        resolution = experiment.images.resolution()
        timings = experiment.images.timings()
        experiment_name = experiment.name.get_name()

        last_time_h = timings.get_time_h_since_start(experiment.last_time_point())
        for time_point in experiment.time_points():
            time_h = timings.get_time_h_since_start(time_point)
            if last_time_h - time_h > _TIME_WINDOW_H:
                continue
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

    # EXCLUSION: Remove this particular crypt, since the bottom part is outside the imaged area
    del results[_CryptAxis("E409-3", 3)]

    figure = lib_figures.new_figure(size=(12, 8))
    axes = figure.subplots(ncols=3, nrows=math.ceil(len(results) / 3), sharex=True, sharey=True).flatten()
    for ax, (crypt_axis, result_of_axis) in zip(axes, results.items()):
        stemness = numpy.array(result_of_axis.is_stem_or_paneth, dtype=numpy.float32)  # Convert from bool to float, for averaging
        crypt_positions_um = numpy.array(result_of_axis.crypt_positions_um)
        average = MovingAverage(crypt_positions_um, stemness, window_width=25)
        ax.plot(average.x_values, average.mean_values, linewidth=3, color="black")
        ax.set_title(f"{crypt_axis.organoid_name}, crypt {crypt_axis.crypt_id}")
        ax.set_xlabel("Position (μm)")
        ax.set_ylabel("Fraction stem + Paneth cells")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
