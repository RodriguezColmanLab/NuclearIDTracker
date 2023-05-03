from typing import Optional, List

import numpy
import scipy
from matplotlib import pyplot as plt
from numpy import ndarray
import skimage.segmentation

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import io, list_io
import figure_lib

_EXPERIMENT_NAME = "x20200614pos002"
_LIST_FILE = "../../Data/Predicted data.autlist"

_NUCLEUS_CHANNEL = ImageChannel(index_zero=0)
_SEGMENTATION_CHANNEL = ImageChannel(index_zero=2)
_Z = 4
_X_MIN = 200
_X_MAX = _X_MIN + 250
_Y_MIN = 20
_Y_MAX = _Y_MIN + 250
_TIME_POINT = TimePoint(347)


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
    for search_position in [position,
                            experiment.links.find_single_future(position),
                            experiment.links.find_single_past(position)]:
        if search_position is None:
            continue
        probabilities = experiment.position_data.get_position_data(search_position, "ct_probabilities")
        if probabilities is not None:
            return probabilities
    return None


def _get_cell_types_image_rgb(experiment: Experiment, time_point: TimePoint):
    segmentation_image = experiment.images.get_image(time_point, _SEGMENTATION_CHANNEL)

    offset_z = experiment.images.offsets.of_time_point(time_point).z
    image_z = int(_Z - offset_z)
    min_z = max(0, image_z - 3)
    max_z = image_z + 3
    mask_image = numpy.max(segmentation_image.array[min_z:max_z], axis=0)
    mask_image = skimage.segmentation.watershed(numpy.zeros_like(mask_image), markers=mask_image)

    colored_image = numpy.full(fill_value=0.4, shape=mask_image.shape + (3,), dtype=numpy.float32)
    cell_types = experiment.global_data.get_data("ct_probabilities")

    for position in experiment.positions.of_time_point(time_point):
        segmentation_id = segmentation_image.value_at(position)
        if segmentation_id == 0:
            continue

        probabilities = _search_probabilities(experiment, position)
        if probabilities is None:
            continue
        else:
            color = numpy.array([probabilities[cell_types.index("PANETH")],
                probabilities[cell_types.index("STEM")],
                probabilities[cell_types.index("ENTEROCYTE")]])
            color -= color.min()
            color /= color.max()

        # Red, Paneth cells
        colored_image[..., 0][mask_image == segmentation_id] = color[0]
        # Green, stem cells
        colored_image[..., 1][mask_image == segmentation_id] = color[1]
        # Blue, enterocytes
        colored_image[..., 2][mask_image == segmentation_id] = color[2]

    # Blur the image, for smoother results
    colored_image = scipy.ndimage.gaussian_filter(colored_image, 1)

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
