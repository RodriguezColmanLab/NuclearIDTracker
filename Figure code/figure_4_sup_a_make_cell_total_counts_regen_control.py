from typing import List, NamedTuple

from matplotlib import pyplot as plt

import lib_figures

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io

_DATASET_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_DATASET_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_MAX_PLOTTED_TIME_POINT = 118  # Makes all experiments have the same length


class _CellCountsOverTime(NamedTuple):
    experiment_name: str
    time_hours: List[float]
    cell_counts: List[int]


def _analyze_experiment(experiment: Experiment) -> _CellCountsOverTime:
    timings = experiment.images.timings()
    time_hours = []
    cell_counts = []
    for time_point in experiment.positions.time_points():
        time_hours.append(timings.get_time_h_since_start(time_point))
        cell_counts.append(len(experiment.positions.of_time_point(time_point)))
    return _CellCountsOverTime(experiment_name=str(experiment.name), time_hours=time_hours, cell_counts=cell_counts)


def main():
    # Analyze the controls
    control_data = list()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_CONTROL, load_images=False,
                                                        max_time_point=_MAX_PLOTTED_TIME_POINT):
        control_data.append(_analyze_experiment(experiment))

    # Analyze the regeneration data
    regeneration_data = list()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_REGENERATION, load_images=False,
                                                        max_time_point=_MAX_PLOTTED_TIME_POINT):
        regeneration_data.append(_analyze_experiment(experiment))

    # Plot everything over time
    figure = lib_figures.new_figure()
    ax = figure.gca()
    for data in control_data:
        ax.plot(data.time_hours, data.cell_counts, label=f"{data.experiment_name} (control)", linestyle="--", linewidth=2)
    for data in regeneration_data:
        ax.plot(data.time_hours, data.cell_counts, label=f"{data.experiment_name} (regeneration)", linewidth=2)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cell count")
    ax.set_ylim(0, 500)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
