from typing import Optional, List, Set

import numpy
import scipy
import skimage
import tifffile
from numpy import ndarray

import lib_figures
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io
from organoid_tracker.linking import nearby_position_finder

_LIST_FILE = "../../Data/Predicted data.autlist"
_OUTPUT_FILE = "../Figures/movie_tracked_{experiment}.tif"

_NUCLEUS_CHANNEL = ImageChannel(index_zero=0)
_SEGMENTATION_CHANNEL = ImageChannel(index_zero=2)

_Z = 6
_TRACKED_CELL = Position(192.02, 299.39, 5.00, time_point_number=1)  # Will track this cell and all future positions
_EXPERIMENT_NAME = "x20190926pos01"


class _PlottedPositions:
    _experiment: Experiment
    _tracks: Set[LinkingTrack]
    _min_probabilities: ndarray
    _max_probabilities: ndarray

    def __init__(self, experiment: Experiment, starting_position: Position):
        self._experiment = experiment

        # Find tracks to plot
        track = experiment.links.get_track(starting_position)
        if track is None:
            raise ValueError("Got position without links: " + str(starting_position))
        self._tracks = set(track.find_all_descending_tracks(include_self=True))

        # Find min and max probabilities, for coloring
        self._min_probabilities, self._max_probabilities = lib_figures.get_min_max_chance_per_cell_type(experiment)

    def is_plotted(self, position: Position) -> bool:
        track = self._experiment.links.get_track(position)
        return track in self._tracks

    def scale_probabilities(self, probabilities: List[float]) -> ndarray:
        probabilities = numpy.array(probabilities)
        probabilities = (probabilities - self._min_probabilities) / (self._max_probabilities - self._min_probabilities)
        numpy.clip(probabilities, 0, 1, out=probabilities)
        return probabilities


def _search_probabilities(experiment: Experiment, position: Position) -> Optional[List[float]]:
    """Gives the cell type probabilities of the position. If not found, then it checks whether they are there in the
    previous or next time point, and returns those."""
    probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
    if probabilities is not None:
        return probabilities

    past_position = experiment.links.find_single_past(position)
    future_position = experiment.links.find_single_future(position)
    for i in range(20):
        if past_position is not None:
            probabilities = experiment.position_data.get_position_data(past_position, "ct_probabilities")
            if probabilities is not None:
                return probabilities
            past_position = experiment.links.find_single_past(past_position)

        if future_position is not None:
            probabilities = experiment.position_data.get_position_data(future_position, "ct_probabilities")
            if probabilities is not None:
                return probabilities
            future_position = experiment.links.find_single_future(future_position)

    return None


def main():
    for experiment in list_io.load_experiment_list_file(_LIST_FILE):
        if experiment.name.get_name() != _EXPERIMENT_NAME:
            continue

        print(f"Working on {experiment.name}...")
        time_point_count = len(list(experiment.positions.time_points()))
        plotted_positions = _PlottedPositions(experiment, _TRACKED_CELL)

        movie = None
        for i, time_point in enumerate(experiment.positions.time_points()):
            print(time_point.time_point_number(), end="  ")
            image_nuclei = _get_nuclear_image_2d_gray(experiment, time_point)
            image_colored = _get_cell_types_image_rgb(experiment, time_point, plotted_positions)
            image = image_colored * image_nuclei
            image = (image * 255).astype(numpy.uint8)

            if movie is None:
                movie = numpy.zeros((time_point_count,) + image.shape, dtype=image.dtype)
            movie[i] = image

        tifffile.imwrite(_OUTPUT_FILE.format(experiment=experiment.name.get_save_name()), movie,
                         compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})
        break


def _get_nuclear_image_2d_gray(experiment: Experiment, time_point: TimePoint) -> ndarray:
    image_3d = experiment.images.get_image_stack(time_point)
    image_3d = image_3d[0:16]

    # Average each z layer with the one above and below it, for nicer-looking max intensity projection
    image_some_averaging = image_3d[1:-1].astype(numpy.float32) + image_3d[0:-2] + image_3d[2:]

    # Max intensity projection, scale from 0 to 1
    image = image_some_averaging.max(axis=0)
    image /= image.max()

    # Convert to RGB format (but keep grayscale)
    image_rgb = numpy.empty(image.shape + (3,), dtype=numpy.float32)
    for i in range(image_rgb.shape[-1]):
        image_rgb[..., i] = image

    return image_rgb


def _create_topdown_segmentation_image(segmentation_3d: ndarray) -> ndarray:
    segmentation_2d = segmentation_3d[0].copy()
    for i in range(1, segmentation_3d.shape[0]):
        segmentation_2d[segmentation_2d == 0] = segmentation_3d[i][segmentation_2d == 0]
    return segmentation_2d


def _get_cell_types_image_rgb(experiment: Experiment, time_point: TimePoint, plotted_positions: _PlottedPositions):
    resolution = experiment.images.resolution()
    segmentation_image = experiment.images.get_image(time_point, _SEGMENTATION_CHANNEL)

    offset = experiment.images.offsets.of_time_point(time_point)
    mask_image = _create_topdown_segmentation_image(segmentation_image.array)

    colored_image = numpy.full(fill_value=0.5, shape=mask_image.shape + (3,), dtype=numpy.float32)
    cell_types = experiment.global_data.get_data("ct_probabilities")

    positions = list(experiment.positions.of_time_point(time_point))
    for region in skimage.measure.regionprops(segmentation_image.array):
        # We match each region to the closest detected position
        centroid = Position(region.centroid[2], region.centroid[1], region.centroid[0], time_point=time_point) + offset
        position = nearby_position_finder.find_closest_position(positions,
                                                                around=centroid,
                                                                resolution=resolution,
                                                                max_distance_um=20)
        if not plotted_positions.is_plotted(position):
            continue
        probabilities = _search_probabilities(experiment, position)
        if probabilities is None:
            continue

        probabilities = plotted_positions.scale_probabilities(probabilities)

        # Calculate the desired color
        color = numpy.array([probabilities[cell_types.index("PANETH")],
                             probabilities[cell_types.index("STEM")],
                             probabilities[cell_types.index("ENTEROCYTE")]])

        # Color the target image by the calculated color
        for i in range(len(color)):
            colored_image[..., i][mask_image == region.label] = color[i]

    # Blur the image, for smoother results
    for i in range(colored_image.shape[-1]):
        colored_image[..., i] = scipy.ndimage.gaussian_filter(colored_image[..., i], 1)

    return colored_image


if __name__ == "__main__":
    main()
