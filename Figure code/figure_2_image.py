from typing import Optional, List

import numpy
import scipy
from matplotlib import pyplot as plt
from numpy import ndarray
import skimage.segmentation
import skimage.measure

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import io, list_io
import figure_lib
from organoid_tracker.linking import nearby_position_finder

_EXPERIMENT_NAME = "x20190926pos01"
_LIST_FILE = "../../Data/Predicted data.autlist"

_NUCLEUS_CHANNEL = ImageChannel(index_zero=0)
_SEGMENTATION_CHANNEL = ImageChannel(index_zero=2)
_Z = 6
_X_MIN = 15
_X_MAX = _X_MIN + 250
_Y_MIN = 250
_Y_MAX = _Y_MIN + 250
_TIME_POINT = TimePoint(330)


def main():
    for experiment in list_io.load_experiment_list_file(_LIST_FILE):
        if experiment.name.get_name() != _EXPERIMENT_NAME:
            continue

        image_nuclei = _get_nuclear_image_2d_gray(experiment, _TIME_POINT)
        image_colored = _get_cell_types_image_rgb(experiment, _TIME_POINT)
        image = image_colored * image_nuclei
        image = image[_Y_MIN:_Y_MAX, _X_MIN:_X_MAX]

        figure = figure_lib.new_figure()
        ax = figure.gca()
        ax.imshow(image, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        return


def _search_probabilities(experiment: Experiment, position: Position) -> Optional[List[float]]:
    """Gives the cell type probabilities of the position. If not found, then it checks whether they are there in the
    previous or next time point, and returns those."""
    probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
    if probabilities is not None:
        return probabilities

    past_postiion = experiment.links.find_single_past(position)
    future_position = experiment.links.find_single_future(position)
    past_past_position = None if past_postiion is None else experiment.links.find_single_past(past_postiion)
    future_future_position = None if future_position is None else experiment.links.find_single_future(future_position)

    for search_position in [past_postiion, future_position, past_past_position, future_future_position]:
        if search_position is None:
            continue
        probabilities = experiment.position_data.get_position_data(search_position, "ct_probabilities")
        if probabilities is not None:
            return probabilities
    return None


def _get_cell_types_image_rgb(experiment: Experiment, time_point: TimePoint):
    resolution = experiment.images.resolution()
    segmentation_image = experiment.images.get_image(time_point, _SEGMENTATION_CHANNEL)

    offset = experiment.images.offsets.of_time_point(time_point)
    image_z = int(_Z - offset.z)
    min_z = max(0, image_z - 2)
    max_z = image_z + 2
    mask_image = numpy.max(segmentation_image.array[min_z:max_z], axis=0)
    mask_image = skimage.segmentation.watershed(numpy.zeros_like(mask_image), markers=mask_image)

    colored_image = numpy.full(fill_value=0.4, shape=mask_image.shape + (3,), dtype=numpy.float32)
    cell_types = experiment.global_data.get_data("ct_probabilities")

    positions = list(experiment.positions.of_time_point(time_point))
    for region in skimage.measure.regionprops(segmentation_image.array):
        # We match each region to the closest detected position
        centroid = Position(region.centroid[2], region.centroid[1], region.centroid[0], time_point=time_point) + offset
        position = nearby_position_finder.find_closest_position(positions,
                                                                around=centroid,
                                                                resolution=resolution,
                                                                max_distance_um=20)
        probabilities = _search_probabilities(experiment, position)
        if probabilities is None:
            continue

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
        colored_image[..., i] -= colored_image[..., i].min()
        colored_image[..., i] /= colored_image[..., i].max()

    return colored_image


def _get_nuclear_image_2d_gray(experiment: Experiment, time_point: TimePoint) -> ndarray:
    image = experiment.images.get_image_slice_2d(time_point, _NUCLEUS_CHANNEL, _Z - 1).astype(numpy.float32) + \
            experiment.images.get_image_slice_2d(time_point, _NUCLEUS_CHANNEL, _Z) +\
            experiment.images.get_image_slice_2d(time_point, _NUCLEUS_CHANNEL, _Z + 1)
    image /= image.max()
    image **= (1/2)  # Makes the image brighter by taking the square root

    # Convert to RGB format (but keep grayscale)
    image_rgb = numpy.empty(image.shape + (3,), dtype=numpy.float32)
    for i in range(image_rgb.shape[-1]):
        image_rgb[..., i] = image

    return image_rgb


if __name__ == "__main__":
    main()
