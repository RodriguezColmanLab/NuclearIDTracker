import math
from typing import List, NamedTuple, Optional, Tuple

import numpy
from matplotlib import pyplot as plt
from numpy import ndarray

import lib_data
import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io
from organoid_tracker.imaging.cropper import crop_2d

_TRACKING_INPUT_FILE = "../../Data/H3K9ac reporter/Tracking data.autlist"
_NUCLEUS_CHANNEL = ImageChannel(index_one=3)  # For the displayed crops

_FRET_SIGNAL_KEY = "intensity"
_ENTEROCYTENESS_MEASUREMENT_PERIOD_H = 5


class _PlottedTrack(NamedTuple):
    # The following lists are all the same length, and share the same index
    times_h: List[float]
    fret_values: List[float]
    enterocyteness_values: List[float]
    stemness_values: List[float]
    position_names: List[str]
    sphericity_values: List[float]

    # The indices of the next list don't correspond to the above lists. This list is simply a list of all the
    # division time points
    division_times_h: List[float]

    # Just for show, to judge the quality of the nuclei
    first_image: ndarray
    last_image: ndarray

    def get_min_max_enterocyteness(self) -> Tuple[float, float]:
        """Get the minimum and maximum fret noise in the track, measured as the standard deviation over windows of 5h.
        """
        min_time_h = min(self.times_h)
        max_time_h = max(self.times_h)

        times_h = numpy.array(self.times_h)
        enterocyteness_values = numpy.array(self.enterocyteness_values)

        # Loop over windows of 5h
        enterocyteness_in_time_window = []
        time_h = min_time_h
        while time_h < max_time_h:
            window_mask = (times_h >= time_h) & (times_h < time_h + _ENTEROCYTENESS_MEASUREMENT_PERIOD_H)
            if numpy.any(window_mask):
                enterocyteness_in_time_window.append(numpy.mean(enterocyteness_values[window_mask]))
            time_h += _ENTEROCYTENESS_MEASUREMENT_PERIOD_H
        return min(enterocyteness_in_time_window), max(enterocyteness_in_time_window)


def _find_first_time_point_number(track: LinkingTrack) -> int:
    """Find the first time point number of a track, considering all its parent tracks."""
    first_time_point_number = track.first_time_point_number()
    for some_track in track.find_all_previous_tracks(include_self=False):
        if some_track.first_time_point_number() < first_time_point_number:
            first_time_point_number = some_track.first_time_point_number()
    return first_time_point_number


def main():
    # Load and analyze the tracking data
    plotted_tracks = list()
    for experiment in list_io.load_experiment_list_file(_TRACKING_INPUT_FILE):
        for track in experiment.links.find_ending_tracks():
            if track.last_time_point_number() != experiment.positions.last_time_point_number():
                continue  # Only analyze tracks that end at the last time point
            first_time_point_number = _find_first_time_point_number(track)
            if first_time_point_number != experiment.positions.first_time_point_number():
                continue  # Only analyze tracks that start at the first time point

            analyzed_track = _analyze_track(experiment, track.find_last_position())
            if analyzed_track is not None:
                plotted_tracks.append(analyzed_track)

    # Find min and max enterocyteness values
    enterocyteness_differences = list()
    for analyzed_track in plotted_tracks:
        min_enterocyteness, max_enterocyteness = analyzed_track.get_min_max_enterocyteness()
        print(min_enterocyteness, max_enterocyteness)
        enterocyteness_differences.append(max_enterocyteness - min_enterocyteness)

    # Find the four least and most differentiating tracks
    enterocyteness_differences = numpy.array(enterocyteness_differences)
    max_indices = numpy.argsort(enterocyteness_differences)
    plotted_tracks = [plotted_tracks[i] for i in max_indices[:4]] + [plotted_tracks[i] for i in max_indices[-4:]]

    # Plot the selected tracks
    figure = lib_figures.new_figure(size=(12, 6))
    axes = figure.subplots(nrows=4, ncols=len(plotted_tracks), sharex=True, sharey="row", squeeze=False)
    axes[1, 0].set_ylabel("FRET signal")
    axes[2, 0].set_ylabel("Predicted likelihood")
    axes[3, 0].set_ylabel("Sphericity")
    for i, plotted_track in enumerate(plotted_tracks):
        ax_images = axes[0, i]
        ax_fret = axes[1, i]
        ax_cell_types = axes[2, i]
        ax_sphericity = axes[3, i]

        max_x = max(plotted_track.times_h) * 1.1
        ax_images.imshow(plotted_track.first_image, cmap="gray", extent=[0, max_x * 0.48, 0, max_x * 0.48])
        ax_images.imshow(plotted_track.last_image, cmap="gray", extent=[max_x * 0.52, max_x, 0, max_x * 0.48])

        ax_fret.scatter(plotted_track.times_h, plotted_track.fret_values, color="black", s=5, linewidth=0)
        ax_fret.plot(plotted_track.times_h, plotted_track.fret_values, linewidth=0.5, color="black")
        ax_fret.set_ylim(0.5, 1.0)
        ax_cell_types.plot(plotted_track.times_h, plotted_track.enterocyteness_values, linewidth=3,
                           color=lib_figures.CELL_TYPE_PALETTE["ENTEROCYTE"], label="Enterocyte")
        ax_cell_types.plot(plotted_track.times_h, plotted_track.stemness_values, linewidth=3,
                           color=lib_figures.CELL_TYPE_PALETTE["STEM"], label="Stem")
        ax_cell_types.set_ylim(0, 1)
        ax_cell_types.legend()
        ax_sphericity.plot(plotted_track.times_h, plotted_track.sphericity_values)
        ax_sphericity.set_xlabel("Time (h)")
        ax_sphericity.set_ylim(0.68, 0.92)

        # Add a dotted vertical line for each division
        for j in range(1, axes.shape[0]):
            ax = axes[j, i]
            for division_time_h in plotted_track.division_times_h:
                ax.axvline(division_time_h, color="black", linestyle="--")

    figure.tight_layout()
    plt.show()


def _get_first_track(final_track: LinkingTrack) -> LinkingTrack:
    """Recursive function to get the first track in a chain of tracks. May return the passed track if it doesn't have
    any parent tracks."""
    parent_tracks = final_track.get_previous_tracks()
    if len(parent_tracks) >= 1:
        return _get_first_track(parent_tracks.pop())
    return final_track


def _get_crop_image(experiment: Experiment, position: Position) -> ndarray:
    offset = experiment.images.offsets.of_time_point(position.time_point())
    image_2d = experiment.images.get_image_slice_2d(position.time_point(), _NUCLEUS_CHANNEL, round(position.z))
    crop_size = 50
    output_array = numpy.zeros((crop_size, crop_size), dtype=image_2d.dtype)
    crop_2d(image_2d, int(position.x - offset.x - crop_size / 2), int(position.y - offset.y - crop_size / 2), output_array)
    return output_array


def _analyze_track(experiment: Experiment, origin_position: Position) -> Optional[_PlottedTrack]:
    """Analyze a track and return the data to plot it, or None if no track was found for this position in the
     experiment."""
    enterocyte_index = experiment.global_data.get_data("ct_probabilities").index("ENTEROCYTE")
    stem_index = experiment.global_data.get_data("ct_probabilities").index("STEM")
    timings = experiment.images.timings()

    final_track = experiment.links.get_track(origin_position)

    fret_values = []
    times_h = []
    enterocyteness_values = []
    stemness_values = []
    sphericity_values = []
    position_names = []
    division_times_h = []
    if final_track is None:
        raise ValueError(f"Position {origin_position} not found in any track in the experiment.")
    for track in final_track.find_all_previous_tracks(include_self=True):
        if track.will_divide():
            division_times_h.append(timings.get_time_h_since_start(track.last_time_point_number()))
        for position in reversed(list(track.positions())):
            fret_signal = experiment.position_data.get_position_data(position, _FRET_SIGNAL_KEY)
            ct_probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
            data_array = lib_data.get_data_array(experiment.position_data, position, ["sphericity"])

            if fret_signal is not None and ct_probabilities is not None and data_array is not None \
                    and not numpy.any(numpy.isnan(data_array)):
                fret_values.append(fret_signal)
                enterocyteness_values.append(ct_probabilities[enterocyte_index])
                stemness_values.append(ct_probabilities[stem_index])
                times_h.append(timings.get_time_h_since_start(position.time_point_number()))
                sphericity_values.append(math.log(data_array[0]))
                position_names.append(str(position))

    if len(sphericity_values) < 2:
        return None  # Not enough data for adata object

    # Collect images
    image_last = _get_crop_image(experiment, final_track.find_last_position())
    image_first = _get_crop_image(experiment, _get_first_track(final_track).find_first_position())

    return _PlottedTrack(times_h=times_h, fret_values=fret_values,
                         enterocyteness_values=enterocyteness_values, stemness_values=stemness_values,
                         position_names=position_names, sphericity_values=sphericity_values,
                         division_times_h=division_times_h, first_image=image_first, last_image=image_last)


if __name__ == "__main__":
    main()
