import os
from typing import Optional, List, Any

import numpy
import scipy
import skimage
import tifffile
from numpy import ndarray

from organoid_tracker.core import TimePoint, UserError, min_none
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog, worker_job
from organoid_tracker.gui.window import Window
from organoid_tracker.gui.worker_job import WorkerJob
from organoid_tracker.linking import nearby_position_finder


def get_menu_items(window: Window):
    return {
        "File//Export-Export image//Projection-Mean-intensity projection...": lambda: _show_mean_intensity_projection_image(window),
        "File//Export-Export movie//Projection-Mean-intensity projection...": lambda: _show_mean_intensity_projection_movie(window)
    }


def _show_mean_intensity_projection_image(window: Window):
    nucleus_channel = window.display_settings.image_channel
    experiments = list(window.get_active_experiments())
    channel_count = min_none([len(experiment.images.get_channels()) for experiment in experiments])
    if channel_count is None or channel_count == 0:
        raise UserError("No images", "No images have been loaded.")

    if len(experiments) == 1:
        # Prompt to save to single file
        save_file = dialog.prompt_save_file("Save image", [("TIF files", "*.tif")])
        if save_file is None:
            return
        image = _get_image(experiments[0], window.display_settings.time_point, nucleus_channel)
        tifffile.imwrite(save_file, image)
    else:
        # Prompt to save to folder
        save_folder = dialog.prompt_save_file("Save folder for images", [("Folder", "*")])
        if save_folder is None:
            return
        if os.path.isfile(save_folder):
            os.unlink(save_folder)
        os.makedirs(save_folder, exist_ok=True)
        for i, experiment in enumerate(experiments):
            save_file = os.path.join(save_folder, f"{i + 1}. {experiment.name.get_save_name()}.tif")
            image = _get_image(experiment, window.display_settings.time_point, nucleus_channel)
            tifffile.imwrite(save_file, image)


def _show_mean_intensity_projection_movie(window: Window):
    nucleus_channel = window.display_settings.image_channel
    experiments = list(window.get_active_experiments())
    channel_count = min_none([len(experiment.images.get_channels()) for experiment in experiments])
    if channel_count is None or channel_count == 0:
        raise UserError("No images", "No images have been loaded.")

    if len(experiments) == 1:
        # Prompt to save to single file
        save_file = dialog.prompt_save_file("Save movie", [("TIF files", "*.tif")])
        if save_file is None:
            return
        worker_job.submit_job(window, _CreateMovieJob(nucleus_channel, save_file, has_multiple_experiments=False))
    else:
        # Prompt to save to folder
        save_folder = dialog.prompt_save_file("Save folder for movies", [("Folder", "*")])
        if save_folder is None:
            return
        os.makedirs(save_folder, exist_ok=True)
        worker_job.submit_job(window, _CreateMovieJob(nucleus_channel, save_folder, has_multiple_experiments=True))


class _CreateMovieJob(WorkerJob):
    """Creates a movie by writing an image for every time point."""

    _nucleus_channel: ImageChannel
    _save_file_or_folder: str
    _has_multiple_experiments: bool

    def __init__(self, nucleus_channel: ImageChannel, save_file_or_folder: str, has_multiple_experiments: bool):
        """save_file_or_folder is a folder if you have multiple experiments, or a file if you only have one."""
        self._nucleus_channel = nucleus_channel
        self._save_file_or_folder = save_file_or_folder
        self._has_multiple_experiments = has_multiple_experiments

    def copy_experiment(self, experiment: Experiment) -> Experiment:
        return experiment.copy_selected(images=True, positions=True, global_data=True, name=True)

    def gather_data(self, experiment_copy: Experiment) -> Any:
        print(f"\nWorking on {experiment_copy.name}...")

        array = None
        i = 0
        for time_point in self.reporting_progress(experiment_copy.positions.time_points()):
            image = _get_image(experiment_copy, time_point, self._nucleus_channel)
            if array is None:
                time_point_count = experiment_copy.positions.last_time_point_number() \
                                   - experiment_copy.first_time_point_number() + 1
                array = numpy.zeros((time_point_count, image.shape[0], image.shape[1]), dtype=numpy.float32)
            array[i] = image
            i += 1

        if self._has_multiple_experiments:
            save_file = os.path.join(self._save_file_or_folder, f"{experiment_copy.name.get_save_name()}.tif")
        else:
            save_file = self._save_file_or_folder
        tifffile.imwrite(save_file, array, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})

        return "Done"  # Output not used, so anything that's not None works

    def on_finished(self, result: Any):
        if self._has_multiple_experiments == 1:
            dialog.popup_message("Cell type movie done", f"Done creating the cell type movie. It can be found in"
                                                         f" \"{self._save_file_or_folder}\".")
        else:
            dialog.popup_message("Cell type movies done", f"Done creating the cell type movie. They can be found in"
                                                          f" \"{self._save_file_or_folder}\".")


def _get_image(experiment: Experiment, time_point: TimePoint, nucleus_channel: ImageChannel) -> ndarray:
    nucleus_image = experiment.images.get_image_stack(time_point, nucleus_channel)
    if nucleus_image is None:
        raise UserError("Missing image", f"Image is missing for time point {time_point.time_point_number()}.")

    nucleus_image = nucleus_image.astype(numpy.float32)
    return numpy.mean(nucleus_image, axis=0)


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
