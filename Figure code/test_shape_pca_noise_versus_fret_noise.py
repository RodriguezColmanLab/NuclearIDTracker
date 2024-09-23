from sklearn.decomposition import PCA
from typing import List, NamedTuple, Optional, Tuple

import numpy
import scanpy
import scanpy.preprocessing
import sklearn
import sklearn.decomposition
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
_TRAINING_DATA_INPUT_FILE = "../../Data/all_data.h5ad"

_SELECTED_POSITIONS = [
    Position(159.00, 350.00, 2.00, time_point_number=154),
    Position(389.48, 368.90, 21.00, time_point_number=154)

]
_FRET_SIGNAL_KEY = "intensity"
_FRET_NOISE_MEASUREMENT_PERIOD_H = 5


class _PlottedTrack(NamedTuple):
    # The following lists are all the same length, and share the same index
    times_h: List[float]
    fret_values: List[float]
    enterocyteness_values: List[float]
    stemness_values: List[float]
    position_names: List[str]
    pca_coords: ndarray

    # The indices of the next list don't correspond to the above lists. This list is simply a list of all the
    # division time points
    division_times_h: List[float]

    # Just for show, to judge the quality of the nuclei
    first_image: ndarray
    last_image: ndarray

    def get_min_max_fret_noise(self) -> Tuple[float, float]:
        """Get the minimum and maximum fret noise in the track, measured as the standard deviation over windows of 5h.
        """
        min_time_h = min(self.times_h)
        max_time_h = max(self.times_h)

        times_h = numpy.array(self.times_h)
        fret_values = numpy.array(self.fret_values)

        # Loop over windows of 5h
        fret_noise_values = []
        time_h = min_time_h
        while time_h < max_time_h:
            window_mask = (times_h >= time_h) & (times_h < time_h + _FRET_NOISE_MEASUREMENT_PERIOD_H)
            if numpy.any(window_mask):
                fret_noise_values.append(numpy.std(fret_values[window_mask], ddof=1))
            time_h += _FRET_NOISE_MEASUREMENT_PERIOD_H
        return min(fret_noise_values), max(fret_noise_values)


def _find_first_time_point_number(track: LinkingTrack) -> int:
    """Find the first time point number of a track, considering all its parent tracks."""
    first_time_point_number = track.first_time_point_number()
    for some_track in track.find_all_previous_tracks(include_self=False):
        if some_track.first_time_point_number() < first_time_point_number:
            first_time_point_number = some_track.first_time_point_number()
    return first_time_point_number


def main():
    adata = scanpy.read_h5ad(_TRAINING_DATA_INPUT_FILE)
    adata = lib_figures.standard_preprocess(adata)

    # Remove cells that we cannot train on, and then scale
    adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Do the PCA
    print(f"n_samples: {adata.X.shape[0]}, n_features: {adata.X.shape[1]}")
    pca = sklearn.decomposition.PCA(n_components=adata.X.shape[1])
    pca.fit(adata.X)

    # Load and analyze the tracking data
    plotted_tracks = list()
    for experiment in list_io.load_experiment_list_file(_TRACKING_INPUT_FILE):
        for track in experiment.links.find_ending_tracks():
            if track.last_time_point_number() != experiment.positions.last_time_point_number():
                continue  # Only analyze tracks that end at the last time point
            first_time_point_number = _find_first_time_point_number(track)
            if first_time_point_number != experiment.positions.first_time_point_number():
                continue  # Only analyze tracks that start at the first time point

            analyzed_track = _analyze_track(adata, experiment, track.find_last_position(), pca)
            if analyzed_track is not None:
                plotted_tracks.append(analyzed_track)

    # Find min and max fret noise
    min_fret_noise = list()
    max_fret_noise = list()
    for analyzed_track in plotted_tracks:
        min_noise, max_noise = analyzed_track.get_min_max_fret_noise()
        min_fret_noise.append(min_noise)
        max_fret_noise.append(max_noise)

    # Find the three lowest and highest fret noise tracks
    min_fret_noise = numpy.array(min_fret_noise)
    max_fret_noise = numpy.array(max_fret_noise)
    min_indices = numpy.argsort(min_fret_noise)
    max_indices = numpy.argsort(max_fret_noise)
    plotted_tracks = [plotted_tracks[i] for i in numpy.concatenate([min_indices[:3], max_indices[-3:]])]

    # Plot the selected tracks
    figure = lib_figures.new_figure(size=(12, 7))
    axes = figure.subplots(nrows=5, ncols=len(plotted_tracks), sharex=True, sharey="row", squeeze=False)
    for i, plotted_track in enumerate(plotted_tracks):
        ax_images = axes[0, i]
        ax_fret = axes[1, i]
        ax_cell_types = axes[2, i]
        ax_pca_123 = axes[3, i]
        ax_pca_456 = axes[4, i]

        max_x = max(plotted_track.times_h) * 1.1
        ax_images.imshow(plotted_track.first_image, cmap="gray", extent=[0, max_x * 0.48, 0, max_x * 0.48])
        ax_images.imshow(plotted_track.last_image, cmap="gray", extent=[max_x * 0.52, max_x, 0, max_x * 0.48])

        ax_fret.scatter(plotted_track.times_h, plotted_track.fret_values, color="black", s=5, linewidth=0)
        ax_fret.plot(plotted_track.times_h, plotted_track.fret_values, linewidth=0.5, color="black")
        ax_fret.set_xlabel("Time (h)")
        ax_fret.set_ylabel("FRET signal")
        ax_fret.set_ylim(0.5, 1)
        ax_cell_types.plot(plotted_track.times_h, plotted_track.enterocyteness_values, linewidth=3,
                           color=lib_figures.CELL_TYPE_PALETTE["ENTEROCYTE"], label="Enterocyte")
        ax_cell_types.plot(plotted_track.times_h, plotted_track.stemness_values, linewidth=3,
                           color=lib_figures.CELL_TYPE_PALETTE["STEM"], label="Stem")
        ax_cell_types.set_ylim(0, 1)
        ax_cell_types.set_ylabel("Predicted likelihood")
        ax_cell_types.legend()
        for pca_axis in range(3):
            ax_pca_123.plot(plotted_track.times_h, plotted_track.pca_coords[:, pca_axis], label=f"PCA {pca_axis + 1}")
        for pca_axis in range(3, 6):
            ax_pca_456.plot(plotted_track.times_h, plotted_track.pca_coords[:, pca_axis], label=f"PCA {pca_axis + 1}")
        ax_pca_123.set_ylabel("PCA value")
        ax_pca_456.set_ylabel("PCA value")
        ax_pca_456.set_xlabel("Time (h)")
        ax_pca_123.legend()
        ax_pca_456.legend()

        # Add a dotted vertical line for each division
        for j in range(1, axes.shape[0]):
            ax = axes[j, i]
            for division_time_h in plotted_track.division_times_h:
                ax.axvline(division_time_h, color="black", linestyle="--")

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


def _analyze_track(adata: AnnData, experiment: Experiment, origin_position: Position, pca: PCA
                   ) -> Optional[_PlottedTrack]:
    """Analyze a track and return the data to plot it, or None if no track was found for this position in the
     experiment."""
    enterocyte_index = experiment.global_data.get_data("ct_probabilities").index("ENTEROCYTE")
    stem_index = experiment.global_data.get_data("ct_probabilities").index("STEM")
    input_names = list(adata.var_names)
    timings = experiment.images.timings()

    final_track = experiment.links.get_track(origin_position)

    fret_values = []
    times_h = []
    enterocyteness_values = []
    stemness_values = []
    all_position_data = []
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
            data_array = lib_data.get_data_array(experiment.position_data, position, input_names)

            if fret_signal is not None and ct_probabilities is not None and data_array is not None \
                    and not numpy.any(numpy.isnan(data_array)):
                fret_values.append(fret_signal)
                enterocyteness_values.append(ct_probabilities[enterocyte_index])
                stemness_values.append(ct_probabilities[stem_index])
                times_h.append(timings.get_time_h_since_start(position.time_point_number()))
                all_position_data.append(data_array)
                position_names.append(str(position))

    if len(all_position_data) < 2:
        return None  # Not enough data for adata object

    # Calculate the PCA coords of the track
    adata_track = AnnData(numpy.array(all_position_data))
    adata_track.var_names = input_names
    adata_track.obs_names = position_names
    adata_track.obs["time_h"] = times_h

    # Preprocess, but scale using the same method as used for adata
    adata_track = lib_figures.standard_preprocess(adata_track, filter=False, scale=False)
    adata_track.X -= numpy.array(adata.var["mean"])
    adata_track.X /= numpy.array(adata.var["std"])
    pca_coords = pca.transform(adata_track.X)

    # Collect images
    image_last = _get_crop_image(experiment, final_track.find_last_position())
    image_first = _get_crop_image(experiment, _get_first_track(final_track).find_first_position())

    return _PlottedTrack(times_h=times_h, fret_values=fret_values,
                         enterocyteness_values=enterocyteness_values, stemness_values=stemness_values,
                         position_names=position_names, pca_coords=pca_coords,
                         division_times_h=division_times_h, first_image=image_first, last_image=image_last)


if __name__ == "__main__":
    main()
