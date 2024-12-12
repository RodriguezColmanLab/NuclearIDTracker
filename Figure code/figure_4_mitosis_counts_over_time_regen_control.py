from typing import List, NamedTuple

from matplotlib import pyplot as plt

import lib_figures

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.linking import cell_division_finder

_DATASET_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_DATASET_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_MAX_PLOTTED_TIME_POINT = 118  # Makes all experiments have the same length


class _MitosisCountsOverTime(NamedTuple):
    experiment_name: str
    time_hours: List[float]
    mitosis_counts: List[int]


def _analyze_experiment(experiment: Experiment) -> _MitosisCountsOverTime:
    timings = experiment.images.timings()
    mitotic_cells = cell_division_finder.find_mothers(experiment.links, exclude_multipolar=False)

    time_hours = []
    mitosis_counts = []
    for time_point in experiment.positions.time_points():
        time_hours.append(timings.get_time_h_since_start(time_point))
        mitosis_counts.append(sum(1 for position in mitotic_cells if position.time_point() == time_point))
    return _MitosisCountsOverTime(experiment_name=str(experiment.name), time_hours=time_hours, mitosis_counts=mitosis_counts)


def main():
    bar_width_h = 12/60

    # Analyze the controls
    control_data = list()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_CONTROL):
        control_data.append(_analyze_experiment(experiment))

    # Analyze the regeneration data
    regeneration_data = list()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_REGENERATION):
        regeneration_data.append(_analyze_experiment(experiment))

    # Plot everything over time
    figure = lib_figures.new_figure()
    axes = list(figure.subplots(nrows=len(control_data) + len(regeneration_data), sharex=True, sharey=True))
    for data in control_data:
        ax = axes.pop(0)
        ax.bar(data.time_hours, data.mitosis_counts, width=bar_width_h, color="gray")
        ax.set_title(f"{data.experiment_name} (control)")
    for data in regeneration_data:
        ax = axes.pop(0)
        ax.bar(data.time_hours, data.mitosis_counts, width=bar_width_h)
        ax.set_title(f"{data.experiment_name} (regeneration)")
    plt.show()


if __name__ == "__main__":
    main()
