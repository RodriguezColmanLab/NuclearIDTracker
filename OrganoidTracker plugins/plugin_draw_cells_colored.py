import os
from typing import Optional, List, Tuple, Any, Iterable

import numpy
import scipy
import skimage
import tifffile
from PIL import Image as PilImage
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent
from numpy import ndarray
from skimage.measure._regionprops import RegionProperties

from organoid_tracker.core import TimePoint, UserError, min_none
from organoid_tracker.core.bounding_box import BoundingBox
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.gui import dialog, worker_job
from organoid_tracker.gui.window import Window
from organoid_tracker.gui.worker_job import WorkerJob
from organoid_tracker.linking import nearby_position_finder
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer
from organoid_tracker.visualizer.track_visualizer import TrackVisualizer


def get_menu_items(window: Window):
    return {
        "File//Export-Export image//Projection-Max-intensity colored by cell types...": lambda: _create_colored_image(window),
        "File//Export-Export movie//Projection-Max-intensity colored by cell types...": lambda: _create_colored_movie(window),
        "File//Export-Export movie//Projection-Max-intensity colored by cell types of selected cells...":
            lambda: _create_colored_movie_selectable(window)
    }


class _CellFilter:

    NO_FILTER = None  # Set after class creation

    def should_include(self, position: Position, *, is_inside: bool) -> bool:
        """Returns True if we should include the position for cell type prediction display.

        is_inside is whether the position is inside the segmentation region. Normally, we also color nearby
        segmentations, assuming those are oversegmentations. To prevent this, simply return False if is_inside is False.
        """
        return True
_CellFilter.NO_FILTER = _CellFilter()


class _SetFilter(_CellFilter):
    _included_positions: set[Position]

    def __init__(self, included_positions: set[Position]):
        self._included_positions = included_positions

    def should_include(self, position: Position, *, is_inside: bool) -> bool:
        return is_inside and position in self._included_positions


class _SelectionForColoredDrawingVisualizer(TrackVisualizer):
    """Select the track(s) that you want to have highlighted, and then use the Export menu to create the movie."""


    def __init__(self, window: Window):
        super().__init__(window)

    def get_extra_menu_options(self) -> dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Export//Export-Create max-intensity movie": self._create_movie,
            "Export//Export-Create max-intensity image of current time point": self._create_image
        }

    def _get_figure_title(self) -> str:
        return f"Tracks for movie export at time point {self._time_point.time_point_number()} (z={self._z})"

    def _create_movie(self):
        positions = self._collect_positions()
        channels = self._experiment.images.get_channels()
        segmentation_channel = dialog.prompt_int("Segmentation channel",
                                                 "In which channel did you store the segmentation?",
                                                 minimum=1, maximum=len(channels), default=len(channels))
        if segmentation_channel is None:
            return
        segmentation_channel = ImageChannel(index_one=segmentation_channel)
        save_file = dialog.prompt_save_file("Save movie", [("TIF files", "*.tif")])
        if save_file is None:
            return
        worker_job.submit_job(self._window, _CreateMovieJob(self._window.display_settings.image_channel,
                                                            segmentation_channel,
                                                            save_file_or_folder=save_file,
                                                            has_multiple_experiments=False,
                                                            cell_filter=_SetFilter(positions)))


    def _create_image(self):
        positions = self._collect_positions()
        channels = self._experiment.images.get_channels()
        segmentation_channel = dialog.prompt_int("Segmentation channel",
                                                 "In which channel did you store the segmentation?",
                                                 minimum=1, maximum=len(channels), default=len(channels))
        if segmentation_channel is None:
            return
        segmentation_channel = ImageChannel(index_one=segmentation_channel)
        save_file = dialog.prompt_save_file("Save image", [("PNG files", "*.png")])
        if save_file is None:
            return
        image = _get_image(self._experiment, self._time_point, self._window.display_settings.image_channel, segmentation_channel,
                           cell_filter=_SetFilter(positions))
        plt.imsave(save_file, image)

    def _collect_positions(self) -> set[Position]:
        """Gets all positions in the selected tracks. Raises UserError if no positions have been selected."""
        positions = set()
        for selected_position in self._selected_positions:
            selected_track = self._experiment.links.get_track(selected_position)
            for track in selected_track.find_all_previous_and_descending_tracks(include_self=True):
                positions.update(track.positions())
        if len(positions) == 0:
            raise UserError("No positions selected",
                            "No positions have been selected. Please select at least one position by clicking on it.")
        return positions


def _create_colored_image(window: Window):
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

def _create_colored_movie_selectable(window: Window):
    activate(_SelectionForColoredDrawingVisualizer(window))


def _create_colored_movie(window: Window):
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
    cell_filter = _CellFilter.NO_FILTER

    if len(experiments) == 1:
        # Prompt to save to single file
        save_file = dialog.prompt_save_file("Save movie", [("TIF files", "*.tif")])
        if save_file is None:
            return
        worker_job.submit_job(window, _CreateMovieJob(nucleus_channel, segmentation_channel, save_file,
                                                      has_multiple_experiments=False, cell_filter=cell_filter))
    else:
        # Prompt to save to folder
        save_folder = dialog.prompt_save_file("Save folder for movies", [("Folder", "*")])
        if save_folder is None:
            return
        os.makedirs(save_folder, exist_ok=True)
        worker_job.submit_job(window, _CreateMovieJob(nucleus_channel, segmentation_channel, save_folder,
                                                      has_multiple_experiments=True, cell_filter=cell_filter))


class _CreateMovieJob(WorkerJob):
    """Creates a movie by writing an image for every time point."""

    _nucleus_channel: ImageChannel
    _segmentation_channel: ImageChannel
    _save_file_or_folder: str
    _has_multiple_experiments: bool
    _cell_filter: _CellFilter

    def __init__(self, nucleus_channel: ImageChannel, segmentation_channel: ImageChannel,
                 save_file_or_folder: str, has_multiple_experiments: bool, cell_filter: _CellFilter):
        """save_file_or_folder is a folder if you have multiple experiments, or a file if you only have one."""
        self._nucleus_channel = nucleus_channel
        self._segmentation_channel = segmentation_channel
        self._save_file_or_folder = save_file_or_folder
        self._has_multiple_experiments = has_multiple_experiments
        self._cell_filter = cell_filter

    def copy_experiment(self, experiment: Experiment) -> Experiment:
        return experiment.copy_selected(images=True, positions=True, global_data=True, name=True)

    def gather_data(self, experiment_copy: Experiment) -> Any:
        print(f"\nWorking on {experiment_copy.name}...")

        array = None
        i = 0
        for time_point in self.reporting_progress(experiment_copy.positions.time_points()):
            image = _get_image(experiment_copy, time_point, self._nucleus_channel, self._segmentation_channel,
                               cell_filter=self._cell_filter)
            if array is None:
                time_point_count = experiment_copy.positions.last_time_point_number() \
                                   - experiment_copy.first_time_point_number() + 1
                array = numpy.zeros((time_point_count, image.shape[0], image.shape[1], 3), dtype=numpy.uint8)
            array[i] = (image * 255).astype(numpy.uint8)[:, :, 0:3]  # Convert to 8bit, remove useless alpha channel
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


def _get_image(experiment: Experiment, time_point: TimePoint, nucleus_channel: ImageChannel,
               segmentation_channel: ImageChannel, background_rgba: Tuple[int, int, int, int] = (0, 0, 0, 255),
               cell_filter: _CellFilter = _CellFilter.NO_FILTER) -> ndarray:
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
        image_colored = _get_cell_types_image_rgb(experiment, time_point, z, segmentation_image, cell_filter)

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
    probabilities = experiment.positions.get_position_data(position, "ct_probabilities")
    if probabilities is not None:
        return probabilities

    past_position = experiment.links.find_single_past(position)
    future_position = experiment.links.find_single_future(position)
    past_past_position = None if past_position is None else experiment.links.find_single_past(past_position)
    future_future_position = None if future_position is None else experiment.links.find_single_future(future_position)

    for search_position in [past_position, future_position, past_past_position, future_future_position]:
        if search_position is None:
            continue
        probabilities = experiment.positions.get_position_data(search_position, "ct_probabilities")
        if probabilities is not None:
            return probabilities
    return None


def _clip(probability: float) -> float:
    if probability < 0:
        return 0
    if probability > 1:
        return 1
    return probability


def _get_cell_types_image_rgb(experiment: Experiment, time_point: TimePoint, z: int, segmentation_image: Image,
                              cell_filter: _CellFilter) -> numpy.ndarray:
    resolution = experiment.images.resolution()

    offset = experiment.images.offsets.of_time_point(time_point)
    image_z = int(z - offset.z)
    min_z = max(0, image_z - 2)
    max_z = image_z + 2
    mask_image = numpy.max(segmentation_image.array[min_z:max_z], axis=0)

    colored_image = numpy.full(fill_value=0.25, shape=mask_image.shape + (3,), dtype=numpy.float32)
    cell_types = experiment.global_data.get_data("ct_probabilities")

    for region in skimage.measure.regionprops(segmentation_image.array):
        # We match each region to a position
        positions_in_region = list(_find_all_inside(region, time_point, experiment.positions, segmentation_image))
        if len(positions_in_region) == 0:
            # No positions inside, match to close by position (useful in case of oversegmentation)
            centroid = Position(region.centroid[2], region.centroid[1], region.centroid[0],
                                time_point=time_point) + offset
            position = nearby_position_finder.find_closest_position(experiment.positions.of_time_point(time_point),
                                                                    around=centroid,
                                                                    resolution=resolution,
                                                                    max_distance_um=20)
            if position is None or not cell_filter.should_include(position, is_inside=False):
                continue
        else:
            # Find any position inside that passes the filter
            position = None
            for position_in_region in positions_in_region:
                if cell_filter.should_include(position_in_region, is_inside=True):
                    position = position_in_region
            if position is None:
                continue  # None matched

        probabilities = _search_probabilities(experiment, position)
        if probabilities is None:
            continue
        color = _get_mixed_cell_type_color(cell_types, probabilities)

        # Color the target image by the calculated color
        for i in range(len(color)):
            colored_image[..., i][mask_image == region.label] = color[i]

    # Blur the image, for smoother results
    for i in range(colored_image.shape[-1]):
        colored_image[..., i] = scipy.ndimage.gaussian_filter(colored_image[..., i], 1)

    numpy.clip(colored_image, 0, 1, out=colored_image)
    return colored_image


def _find_all_inside(region: RegionProperties, time_point: TimePoint, positions: PositionCollection, segmentation_image: Image) -> Iterable[Position]:
    """Finds all positions inside the given segmentation region."""
    offset = segmentation_image.offset
    for position in positions.of_time_point(time_point):
        image_position = position - offset
        if image_position.x < region.slice[2].start or image_position.x >= region.slice[2].stop:
            continue
        if image_position.y < region.slice[1].start or image_position.y >= region.slice[1].stop:
            continue
        if image_position.z < region.slice[0].start or image_position.z >= region.slice[0].stop:
            continue
        if segmentation_image.value_at(position) == region.label:
            yield position


def _get_mixed_cell_type_color(cell_type_names: List[str], cell_type_probabilities: List[float]) -> Tuple[
    float, float, float]:
    max_probability = max(cell_type_probabilities)
    min_plotted_probability = max_probability * 0.5

    cell_type_probabilities = [(probability - min_plotted_probability) / (max_probability - min_plotted_probability)
                               for probability in cell_type_probabilities]

    cell_type_probabilities = [_clip(probability) for probability in cell_type_probabilities]
    return (cell_type_probabilities[cell_type_names.index("PANETH")],
            cell_type_probabilities[cell_type_names.index("STEM")],
            cell_type_probabilities[cell_type_names.index("ENTEROCYTE")])


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
