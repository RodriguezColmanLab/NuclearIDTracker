import math
import scipy

from typing import List

from matplotlib import pyplot as plt

import lib_figures
from organoid_tracker.core import Color
from organoid_tracker.imaging import list_io
from organoid_tracker.util.moving_average import MovingAverage

_EXPERIMENTS_FILE = "../../Data/Testing data - output - CD24 and Lgr5.autlist"
_CD24_AVERAGING_TIME_H = 1.0
_MIN_PLOT_LENGTH_H = 5.0


class _Cd24Track:
    times_h: List[float]
    cd24_values: List[float]
    panethness_values: List[float]
    organoid_name: str
    track_id: int

    def __init__(self, organoid_name: str, track_id: int):
        self.organoid_name = organoid_name
        self.track_id = track_id
        self.times_h = list()
        self.cd24_values = list()
        self.panethness_values = list()


def main():
    cd24_tracks = _collect_tracks()

    figure = lib_figures.new_figure(size=(12, 8))
    axes = figure.subplots(math.ceil(len(cd24_tracks) / 4), 4).flatten()
    for ax, track in zip(axes, cd24_tracks):
        cd24_color = "#2d3436"
        cd24_average = MovingAverage(track.times_h, track.cd24_values, window_width=_CD24_AVERAGING_TIME_H,
                                     x_step_size=1 / 5)
        cd24_average.plot(ax, color=Color.from_matplotlib(cd24_color))
        ax.set_ylabel("CD24 intensity", color=cd24_color)
        ax.tick_params(axis="y", labelcolor=cd24_color)
        ax.set_ylim(0, 35000)

        pearson = scipy.stats.pearsonr([cd24_average.get_mean_at(t) for t in track.times_h], track.panethness_values)

        # Add organoid name to top left corner of plot
        ax.text(0.05, 0.95, f"{track.organoid_name}, track {track.track_id}\nr={pearson.statistic:.2f}",
                transform=ax.transAxes,
                verticalalignment="top", horizontalalignment="left", color="black")

        panethness_color = lib_figures.CELL_TYPE_PALETTE["PANETH"]
        ax_panethness = ax.twinx()
        ax_panethness.plot(track.times_h, track.panethness_values, color=panethness_color, linewidth=2)
        ax_panethness.set_ylabel("Panethness", color=panethness_color)
        ax_panethness.tick_params(axis="y", labelcolor=panethness_color)
        ax_panethness.set_ylim(0.3, 0.7)
        ax.set_xlabel("Time (h)")
    figure.tight_layout()
    plt.show()


def _collect_tracks() -> List[_Cd24Track]:
    cd24_tracks = list()

    for experiment in list_io.load_experiment_list_file(_EXPERIMENTS_FILE):
        experiment.links.sort_tracks_by_x()
        paneth_index = experiment.global_data.get_data("ct_probabilities").index("PANETH")
        timings = experiment.images.timings()

        for track in experiment.links.find_all_tracks():
            last_position = track.find_last_position()
            cd24_positive = experiment.position_data.get_position_data(last_position, "cd24_positive")
            if not cd24_positive:
                continue

            cd24_track = _Cd24Track(str(experiment.name), experiment.links.get_track_id(track))
            for position in track.positions():
                probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
                if not probabilities:
                    continue
                panethness = probabilities[paneth_index]
                cd24_intensity = experiment.position_data.get_position_data(position, "intensity_cd24")
                if cd24_intensity is None:
                    continue

                time_h = timings.get_time_h_since_start(position.time_point())
                cd24_track.times_h.append(time_h)
                cd24_track.cd24_values.append(cd24_intensity)
                cd24_track.panethness_values.append(panethness)

            if len(cd24_track.times_h) > 0 and max(cd24_track.times_h) - min(cd24_track.times_h) > _MIN_PLOT_LENGTH_H:
                cd24_tracks.append(cd24_track)
    return cd24_tracks


if __name__ == "__main__":
    main()
