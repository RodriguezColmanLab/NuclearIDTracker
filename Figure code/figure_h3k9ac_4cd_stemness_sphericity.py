import math

from sklearn.decomposition import PCA
from typing import List, NamedTuple, Optional, Tuple

import numpy
import scanpy
import scanpy.preprocessing
import sklearn
import sklearn.decomposition
from scipy.stats import linregress
from anndata import AnnData
from matplotlib import pyplot as plt
from numpy import ndarray

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import io, list_io
import lib_figures
import lib_data
from organoid_tracker.imaging.cropper import crop_2d

_TRACKING_INPUT_FILE = "../../Data/H3K9ac reporter/Tracking data.autlist"
_NUCLEUS_CHANNEL = ImageChannel(index_one=3)  # For the displayed crops

_FRET_SIGNAL_KEY = "intensity"


class _PlottedTrack(NamedTuple):
    # The following lists are all the same length, and share the same index
    times_h: List[float]
    z_values: List[float]
    fret_values: List[float]
    enterocyteness_values: List[float]
    stemness_values: List[float]
    panethness_values: List[float]
    position_names: List[str]
    sphericity_values: List[float]

    # The indices of the next list don't correspond to the above lists. This list is simply a list of all the
    # division time points
    division_times_h: List[float]

    # Just for show, to judge the quality of the nuclei
    first_image: ndarray
    last_image: ndarray

    def _filter_values(self, array: List[float]):
        array = numpy.array(array)
        times_h = numpy.array(self.times_h)
        z_values = numpy.array(self.z_values)
        #array = array[times_h > 15]
        return array

    def get_enterocyteness_mean(self) -> Optional[float]:
        return numpy.mean(self._filter_values(self.enterocyteness_values))

    def get_panethness_mean(self) -> Optional[float]:
        return numpy.mean(self._filter_values(self.panethness_values))

    def get_stemness_mean(self) -> Optional[float]:
        return numpy.mean(self._filter_values(self.stemness_values))

    def get_fret_mean(self) -> Optional[float]:
        return numpy.mean(self._filter_values(self.fret_values))

    def get_sphericity_mean(self) -> Optional[float]:
        return numpy.mean(self._filter_values(self.sphericity_values))

    def passes_filters(self) -> bool:
        return len(self._filter_values(self.enterocyteness_values)) >= 5


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
            if analyzed_track is not None and analyzed_track.passes_filters():
                plotted_tracks.append(analyzed_track)

    # Collect all the FRET and sphericity values
    fret_values = numpy.array([track.get_fret_mean() for track in plotted_tracks])
    stemness_values = numpy.array([track.get_stemness_mean() for track in plotted_tracks])
    sphericity_values = numpy.array([track.get_sphericity_mean() for track in plotted_tracks])

    # Filter outliers (otherwise the correlation is artificially high - p=0.02 thanks to two data points)
    to_keep = sphericity_values < 0.9
    fret_values = fret_values[to_keep]
    stemness_values = stemness_values[to_keep]
    sphericity_values = sphericity_values[to_keep]
    cell_names = numpy.array([track.position_names[-1] for track in plotted_tracks])[to_keep]
    print(f"Filtered {numpy.sum(~to_keep)} outliers of {len(to_keep)}")

    # Plot all three
    figure = lib_figures.new_figure(size=(2.5, 2))
    ax = figure.gca()
    mappable = ax.scatter(stemness_values, sphericity_values, c=fret_values, s=20, linewidth=0, cmap="Blues")
    ax.set_xlabel("Mean stemness")
    ax.set_ylabel("Mean sphericity")
    ax.set_xlim(0.8 * min(stemness_values), max(stemness_values) * 1.1)
    figure.colorbar(mappable).set_label("Mean H3K9ac FRET signal")
    figure.tight_layout()
    plt.show()

    # Sphericity vs FRET
    figure = lib_figures.new_figure(size=(2, 2))
    ax = figure.gca()
    ax.scatter(sphericity_values, fret_values, c="black", s=10, marker="s", linewidth=0)
    p_value = linregress(sphericity_values, fret_values).pvalue
    ax.set_xlabel("Mean sphericity")
    ax.set_ylabel("Mean H3K9ac FRET signal")
    ax.text(0.1, 0.9, f"p={p_value:.4f}", ha="center", va="center", transform=ax.transAxes)
    figure.tight_layout()
    plt.show()

    # Stemness vs FRET
    figure = lib_figures.new_figure(size=(2, 2))
    ax = figure.gca()
    ax.scatter(stemness_values, fret_values, c="black", s=10, marker="s", linewidth=0)
    regression = linregress(stemness_values, fret_values)
    p_value = regression.pvalue
    ax.set_xlabel("Mean stem cell likelihood")
    ax.set_ylabel("Mean H3K9ac FRET signal")
    ax.text(0.1, 0.9, f"p={p_value:.4f}", ha="center", va="center", transform=ax.transAxes)
    ax.plot([0, 1], [regression.intercept, regression.intercept + regression.slope], color="black")
    ax.set_xlim(min(stemness_values) * 0.8, max(stemness_values) * 1.1)
    ax.set_ylim(min(fret_values) * 0.8, max(fret_values) * 1.1)

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
    paneth_index = experiment.global_data.get_data("ct_probabilities").index("PANETH")
    timings = experiment.images.timings()

    final_track = experiment.links.get_track(origin_position)

    fret_values = []
    z_values = []
    times_h = []
    enterocyteness_values = []
    stemness_values = []
    panethness_values = []
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
                panethness_values.append(ct_probabilities[paneth_index])
                times_h.append(timings.get_time_h_since_start(position.time_point_number()))
                z_values.append(position.z)
                sphericity_values.append(math.log(data_array[0]))
                position_names.append(str(position))

    if len(sphericity_values) < 2:
        return None  # Not enough data for adata object

    # Collect images
    image_last = _get_crop_image(experiment, final_track.find_last_position())
    image_first = _get_crop_image(experiment, _get_first_track(final_track).find_first_position())

    return _PlottedTrack(times_h=times_h, fret_values=fret_values, z_values=z_values,
                         enterocyteness_values=enterocyteness_values, stemness_values=stemness_values,
                         panethness_values=panethness_values,
                         position_names=position_names, sphericity_values=sphericity_values,
                         division_times_h=division_times_h, first_image=image_first, last_image=image_last)


if __name__ == "__main__":
    main()
