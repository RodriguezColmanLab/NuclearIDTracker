import os
from typing import Optional, List, Tuple

import numpy
import scipy
import tifffile
from numpy import ndarray
import skimage.segmentation
import skimage.measure
from PIL import Image as PilImage

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io
from organoid_tracker.linking import nearby_position_finder

_LIST_FILE = "../../Data/Tracking data as controls/Dataset - full overweekend.autlist"
_OUTPUT_FILE = "E:/Scratch/Figures/movie_colored_{experiment}.tif"

_NUCLEUS_CHANNEL = ImageChannel(index_zero=0)
_SEGMENTATION_CHANNEL = ImageChannel(index_zero=2)


def main():
    os.makedirs(os.path.dirname(_OUTPUT_FILE), exist_ok=True)
    for experiment in list_io.load_experiment_list_file(_LIST_FILE):
        print(f"Working on {experiment.name}...")
        time_point_count = len(list(experiment.positions.time_points()))

        movie = None
        for i, time_point in enumerate(experiment.positions.time_points()):
            print(time_point.time_point_number(), end="  ")
            image = _get_image(experiment, time_point, _NUCLEUS_CHANNEL, _SEGMENTATION_CHANNEL)

            if movie is None:
                movie = numpy.zeros((time_point_count,) + image.shape, dtype=image.dtype)
            movie[i] = image

        tifffile.imwrite(_OUTPUT_FILE.format(experiment=experiment.name.get_save_name()), movie,
                         compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})


def _get_image(experiment: Experiment, time_point: TimePoint, nucleus_channel: ImageChannel,
               segmentation_channel: ImageChannel, background_rgba: Tuple[int, int, int, int] = (0, 0, 0, 255)
               ) -> ndarray:
    segmentation_image = experiment.images.get_image(time_point, segmentation_channel)
    nucleus_image = experiment.images.get_image(time_point, nucleus_channel)

    background_image = numpy.zeros((segmentation_image.shape_y, segmentation_image.shape_x, 4), dtype=numpy.uint8)
    for i in range(4):
        background_image[:, :, i] = background_rgba[i]
    color_image_pil = PilImage.fromarray(background_image)

    slice_buffer = numpy.zeros((segmentation_image.shape_y, segmentation_image.shape_x, 4),
                               dtype=numpy.float32)  # 2D RGBA
    slice_buffer_uint8 = numpy.zeros((segmentation_image.shape_y, segmentation_image.shape_x, 4),
                                     dtype=numpy.uint8)  # 2D RGBA
    for z in range(segmentation_image.limit_z - 1, segmentation_image.min_z - 1, -1):
        image_nuclei = _get_nuclear_image_2d_gray(z, nucleus_image)
        image_colored = _get_cell_types_image_rgb(experiment, time_point, z, segmentation_image)

        # Place the image in the temporary slice
        slice_buffer[:, :, 0] = image_colored[:, :, 0] * 255
        slice_buffer[:, :, 1] = image_colored[:, :, 1] * 255
        slice_buffer[:, :, 2] = image_colored[:, :, 2] * 255

        # Set the alpha layer to the image slice
        slice_buffer[:, :, 3] = image_nuclei[:, :, 0]
        slice_buffer[:, :, 3] **= 2  # To suppress noise
        slice_buffer[:, :, 3] *= 255

        # Add to existing image
        slice_buffer_uint8[...] = slice_buffer
        slice_buffer_pil = PilImage.fromarray(slice_buffer_uint8)
        if color_image_pil is None:
            color_image_pil = slice_buffer_pil
        else:
            result = PilImage.alpha_composite(color_image_pil, slice_buffer_pil)
            color_image_pil.close()
            slice_buffer_pil.close()
            color_image_pil = result

    color_image = numpy.asarray(color_image_pil, dtype=numpy.float32)
    color_image_pil.close()
    color_image /= 255  # Scale to 0-1
    color_image = color_image[:, :, 0:3]  # Remove alpha channel

    return color_image


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


def _clip(probability: float) -> float:
    if probability < 0:
        return 0
    if probability > 1:
        return 1
    return probability


def _scale_probabilities(probabilities: List[float]) -> List[float]:
    # Scales the probabilities so that the max is 1, and everything less than 50% of the max is 0
    # In this way, we mostly see the dominant cell type
    max_probability = max(probabilities)
    min_plotted_probability = max_probability * 0.5

    probabilities = [(probability - min_plotted_probability) / (max_probability - min_plotted_probability) for probability in probabilities]

    return [_clip(probability) for probability in probabilities]


def _get_cell_types_image_rgb(experiment: Experiment, time_point: TimePoint, z: int, segmentation_image: Image):
    resolution = experiment.images.resolution()

    offset = experiment.images.offsets.of_time_point(time_point)
    image_z = int(z - offset.z)
    min_z = max(0, image_z - 2)
    max_z = image_z + 2
    mask_image = numpy.max(segmentation_image.array[min_z:max_z], axis=0)

    colored_image = numpy.full(fill_value=0.25, shape=mask_image.shape + (3,), dtype=numpy.float32)
    cell_types = experiment.global_data.get_data("ct_probabilities")

    positions = list(experiment.positions.of_time_point(time_point))
    for region in skimage.measure.regionprops(segmentation_image.array):
        # We match each region to the closest detected position
        centroid = Position(region.centroid[2], region.centroid[1], region.centroid[0], time_point=time_point) + offset
        position = nearby_position_finder.find_closest_position(positions,
                                                                around=centroid,
                                                                resolution=resolution,
                                                                max_distance_um=20)
        if position is None:
            continue
        probabilities = _search_probabilities(experiment, position)
        if probabilities is None:
            continue
        probabilities = _scale_probabilities(probabilities)

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

    numpy.clip(colored_image, 0, 1, out=colored_image)
    return colored_image


def _get_nuclear_image_2d_gray(z: int, nucleus_image: Image) -> ndarray:
    image = nucleus_image.get_image_slice_2d(z)
    image = image.astype(numpy.float32)
    image = image / nucleus_image.max()
    image **= (1 / 2)  # Makes the image brighter by taking the square root

    # Convert to RGB format (but keep grayscale)
    image_rgb = numpy.empty(image.shape + (3,), dtype=numpy.float32)
    for i in range(image_rgb.shape[-1]):
        image_rgb[..., i] = image

    return image_rgb


if __name__ == "__main__":
    main()
