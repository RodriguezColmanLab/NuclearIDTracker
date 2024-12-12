import os
from typing import Optional, List, Tuple, Any

import numpy
import scipy
import skimage
import tifffile
from PIL import Image as PilImage
from matplotlib import pyplot as plt
from numpy import ndarray

from organoid_tracker.core import TimePoint, UserError, min_none
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window
from organoid_tracker.linking import nearby_position_finder


def get_menu_items(window: Window):
    return {
        "File//Export-Export image//Projection-Max-intensity colored by cell types...": lambda: _show_colored_image(window),
        "File//Export-Export movie//Projection-Max-intensity colored by cell types...": lambda: _show_colored_movie(window)
    }


def _show_colored_image(window: Window):
    nucleus_channel = window.display_settings.image_channel
    experiments = list(window.get_active_experiments())
    channel_count = min_none([len(experiment.images.get_channels()) for experiment in experiments])
    if channel_count is None or channel_count == 0:
        raise UserError("No images", "No images have been loaded.")

    segmentation_channel = dialog.prompt_int("Segmentation channel", "In which channel did you store the segmentation?",
                                             minimum=1, maximum=channel_count, default=channel_count)
    if segmentation_channel is None:
        return
    segmentation_channel = ImageChannel(index_zero=segmentation_channel - 1)

    if len(experiments) == 1:
        # Prompt to save to single file
        save_file = dialog.prompt_save_file("Save image", [("PNG files", "*.png")])
        if save_file is None:
            return
        image = _get_image(experiments[0], window.display_settings.time_point, nucleus_channel, segmentation_channel)
        plt.imsave(save_file, image)
    else:
        # Prompt to save to folder
        save_folder = dialog.prompt_save_file("Save folder for images", [("Folder", "*")])
        if save_folder is None:
            return
        if os.path.isfile(save_folder):
            os.unlink(save_folder)
        os.makedirs(save_folder, exist_ok=True)
        for i, experiment in enumerate(experiments):
            save_file = os.path.join(save_folder, f"{i + 1}. {experiment.name.get_save_name()}.png")
            image = _get_image(experiment, window.display_settings.time_point, nucleus_channel,
                               segmentation_channel)
            plt.imsave(save_file, image)


def _show_colored_movie(window: Window):
    nucleus_channel = window.display_settings.image_channel
    experiments = list(window.get_active_experiments())
    channel_count = min_none([len(experiment.images.get_channels()) for experiment in experiments])
    if channel_count is None or channel_count == 0:
        raise UserError("No images", "No images have been loaded.")

    segmentation_channel = dialog.prompt_int("Segmentation channel", "In which channel did you store the segmentation?",
                                             minimum=1, maximum=channel_count, default=channel_count)
    if segmentation_channel is None:
        return
    segmentation_channel = ImageChannel(index_zero=segmentation_channel - 1)

    if len(experiments) == 1:
        # Prompt to save to single file
        save_file = dialog.prompt_save_file("Save movie", [("TIF files", "*.tif")])
        if save_file is None:
            return
        window.get_scheduler().add_task(_CreateMovieTask(experiments, nucleus_channel, segmentation_channel, save_file))
    else:
        # Prompt to save to folder
        save_folder = dialog.prompt_save_file("Save folder for movies", [("Folder", "*")])
        if save_folder is None:
            return
        os.makedirs(save_folder, exist_ok=True)
        window.get_scheduler().add_task(_CreateMovieTask(experiments, nucleus_channel, segmentation_channel, save_folder))


class _CreateMovieTask(Task):
    """Creates a movie by writing an image for every time point."""

    _experiment_copies: List[Experiment]
    _nucleus_channel: ImageChannel
    _segmentation_channel: ImageChannel
    _save_file_or_folder: str

    def __init__(self, experiments: List[Experiment], nucleus_channel: ImageChannel, segmentation_channel: ImageChannel,
                 save_file_or_folder: str):
        """save_file_or_folder is a folder if you have multiple experiments, or a file if you only have one."""
        self._experiment_copies = [experiment.copy_selected(images=True, positions=True, position_data=True,
                                                            global_data=True)
                                   for experiment in experiments]
        self._nucleus_channel = nucleus_channel
        self._segmentation_channel = segmentation_channel
        self._save_file_or_folder = save_file_or_folder

    def compute(self) -> Any:
        array = None

        for j, experiment in enumerate(self._experiment_copies):
            print(f"\nWorking on {experiment.name}...")
            for i, time_point in enumerate(experiment.positions.time_points()):
                print(time_point.time_point_number(), end="  ")
                image = _get_image(experiment, time_point, self._nucleus_channel, self._segmentation_channel)
                if array is None:
                    time_point_count = experiment.positions.last_time_point_number() \
                                       - experiment.first_time_point_number() + 1
                    array = numpy.zeros((time_point_count, image.shape[0], image.shape[1], 3), dtype=numpy.uint8)
                array[i] = (image * 255).astype(numpy.uint8)[:, :, 0:3]  # Convert to 8bit, remove useless alpha channel

            if len(self._experiment_copies) == 1:
                save_file = self._save_file_or_folder
            else:
                save_file = os.path.join(self._save_file_or_folder, f"{j + 1}. {experiment.name.get_save_name()}.tif")
            tifffile.imwrite(save_file, array, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})

        return "Done"  # Output not used, we directly save

    def on_finished(self, result: Any):
        if len(self._experiment_copies) == 1:
            dialog.popup_message("Cell type movie done", f"Done creating the cell type movie. It can be found in"
                                                         f" \"{self._save_file_or_folder}\".")
        else:
            dialog.popup_message("Cell type movies done", f"Done creating the cell type movie. They can be found in"
                                                          f" \"{self._save_file_or_folder}\".")


def _get_image(experiment: Experiment, time_point: TimePoint, nucleus_channel: ImageChannel,
               segmentation_channel: ImageChannel, background_rgba: Tuple[int, int, int, int] = (0, 0, 0, 255)
               ) -> ndarray:
    segmentation_image = experiment.images.get_image(time_point, segmentation_channel)

    background_image = numpy.zeros((segmentation_image.shape_y, segmentation_image.shape_x, 4), dtype=numpy.uint8)
    for i in range(4):
        background_image[:, :, i] = background_rgba[i]
    color_image_pil = PilImage.fromarray(background_image)

    slice_buffer = numpy.zeros((segmentation_image.shape_y, segmentation_image.shape_x, 4),
                               dtype=numpy.float32)  # 2D RGBA
    slice_buffer_uint8 = numpy.zeros((segmentation_image.shape_y, segmentation_image.shape_x, 4),
                                     dtype=numpy.uint8)  # 2D RGBA
    for z in range(segmentation_image.limit_z - 1, segmentation_image.min_z - 1, -1):
        image_nuclei = _get_nuclear_image_2d_gray(experiment, time_point, z, nucleus_channel)
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

    return color_image


def _search_probabilities(experiment: Experiment, position: Position) -> Optional[List[float]]:
    """Gives the cell type probabilities of the position. If not found, then it checks whether they are there in the
    previous or next time point, and returns those."""
    probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
    if probabilities is not None:
        return probabilities

    past_position = experiment.links.find_single_past(position)
    future_position = experiment.links.find_single_future(position)
    past_past_position = None if past_position is None else experiment.links.find_single_past(past_position)
    future_future_position = None if future_position is None else experiment.links.find_single_future(future_position)

    for search_position in [past_position, future_position, past_past_position, future_future_position]:
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
    # Scales the probabilities so that the max is 1, and everything less than 75% of the max is 0
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


def _get_nuclear_image_2d_gray(experiment: Experiment, time_point: TimePoint, z: int,
                               nucleus_channel: ImageChannel) -> ndarray:
    image = experiment.images.get_image_slice_2d(time_point, nucleus_channel, z)
    image = image / image.max()
    image **= (1 / 2)  # Makes the image brighter by taking the square root

    # Convert to RGB format (but keep grayscale)
    image_rgb = numpy.empty(image.shape + (3,), dtype=numpy.float32)
    for i in range(image_rgb.shape[-1]):
        image_rgb[..., i] = image

    return image_rgb
