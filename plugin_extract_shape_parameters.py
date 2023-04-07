import math
from typing import Any, Dict, List

import numpy
import scipy.ndimage
import skimage.measure
import skimage.morphology
import skimage.segmentation

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog, action
from organoid_tracker.gui.gui_experiment import GuiExperiment, SingleGuiTab
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window
from organoid_tracker.util.moving_average import MovingAverage

_EXTRACTED_PARAMETERS = [
    "volume_um3",
    "solidity",
    "surface_um2",
    "neighbor_distance_mean_um",
    "neighbor_distance_variation",
    "feret_diameter_max_um",
    "intensity_factor",
    "intensity_factor_local",
    "ellipticity",
    "organoid_relative_z_um"
]
_AVERAGING_H = 2


def get_menu_items(window: Window):
    return {
        "Tools//Process-Segmentation//Extract shape parameters...": lambda: _extract_segmentation_parameters(window)
    }


def _extract_segmentation_parameters(window: Window):
    if not dialog.popup_message_cancellable("Incorporate segmentation", f"Please make sure that you have the channel"
                                                                        f" with the segmentation masks selected (so not the nuclear fluorescence)."):
        return

    # Start!
    open_tabs = window.get_gui_experiment().get_active_tabs()
    window.get_scheduler().add_task(_AnalyzeShapesTask(open_tabs, window.display_settings.image_channel))


class _AnalyzeShapesTask(Task):
    _open_tabs: List[SingleGuiTab]
    _experiment_copies: List[Experiment]
    _segmentation_channel: ImageChannel

    def __init__(self, open_tabs: List[SingleGuiTab], segmentation_channel: ImageChannel):
        self._open_tabs = open_tabs
        self._experiment_copies = [
            open_tab.experiment.copy_selected(positions=True, links=True, connections=True, images=True)
            for open_tab in open_tabs]
        self._segmentation_channel = segmentation_channel

    def compute(self) -> List[PositionData]:
        return [_analyze_shapes(experiment_copy, self._segmentation_channel) for experiment_copy in
                self._experiment_copies]

    def on_finished(self, results: List[PositionData]):
        for open_tab, result in zip(self._open_tabs, results):
            position_data = open_tab.experiment.position_data
            # Remove old data
            for data_name in _EXTRACTED_PARAMETERS:
                position_data.delete_data_with_name(data_name)

            open_tab.experiment.position_data.merge_data(result)
            open_tab.undo_redo.clear()
        dialog.popup_message("Extraction finished", "Stored all the metadata of the positions.")


def _analyze_shapes(experiment: Experiment, segmentation_channel: ImageChannel) -> PositionData:
    """Measures on the experiment."""
    resolution = experiment.images.resolution()
    results = PositionData()

    import skimage.measure
    for time_point in experiment.images.time_points():
        print(f"Working on time point {time_point.time_point_number()}...")
        segmented_image = experiment.images.get_image(time_point, segmentation_channel)
        nuclei_image = experiment.images.get_image(time_point)  # TODO allow selection of nucleus channel
        if segmented_image is None or nuclei_image is None:
            continue

        # Index the properties
        regionprops_by_label = dict()
        image_size_z, image_size_y, image_size_x = segmented_image.array.shape
        for properties in skimage.measure.regionprops(segmented_image.array, intensity_image=nuclei_image.array):
            regionprops_by_label[properties.label] = properties

        # Index positions to label
        positions_by_label = _get_positions_by_label(experiment, time_point, segmented_image)

        # Remove positions bordering the image
        for label, properties in regionprops_by_label.items():
            if label not in positions_by_label:
                continue  # Already didn't have a position
            slice_z, slice_y, slice_x = properties.slice
            if slice_z.start == 0 or slice_z.stop >= image_size_z \
                    or slice_y.start == 0 or slice_y.stop >= image_size_y \
                    or slice_x.start == 0 or slice_x.stop >= image_size_x:
                del positions_by_label[label]

        # First pass: characterize the shape of the nuclei
        _measure_shape(positions_by_label, regionprops_by_label, resolution, results)

        # Second pass: characterize the environment
        _measure_neighborhood(positions_by_label, regionprops_by_label, resolution, results)

    # Average all values
    print("Averaging values over time...")
    for track in experiment.links.find_all_tracks():
        _average_track(experiment, track, results)

    return results


def _average_track(experiment: Experiment, track: LinkingTrack, results: PositionData):
    resolution = experiment.images.resolution()

    for data_name in _EXTRACTED_PARAMETERS:
        # Record all values
        times_h = list()
        values = list()
        for position in track.positions():
            value = results.get_position_data(position, data_name)
            if value is None:
                continue
            times_h.append(position.time_point_number() * resolution.time_point_interval_h)
            values.append(value)
        if len(values) < 3:
            continue

        # Store the averaged values
        average = MovingAverage(times_h, values, window_width=_AVERAGING_H,
                                x_step_size=resolution.time_point_interval_h)
        for position in track.positions():
            value = average.get_mean_at(position.time_point_number() * resolution.time_point_interval_h)
            results.set_position_data(position, data_name, value)


def _measure_neighborhood(positions_by_label: Dict[int, Position],
                          properties_by_label: Dict[int, "skimage.measure._regionprops.RegionProperties"],
                          resolution: ImageResolution, results: PositionData):
    """Looks at the distances to nearby cells."""
    resolution_zyx_um = numpy.array(resolution.pixel_size_zyx_um, dtype=numpy.float32)

    # Construct table of *all* positions (including of regions that were segmented, but that we did not track)
    other_positions_table = numpy.empty((len(properties_by_label), 4), dtype=numpy.float32)
    for i, properties in enumerate(properties_by_label.values()):
        other_positions_table[i, 0:3] = properties.centroid * resolution_zyx_um
        other_positions_table[i, 3] = properties.intensity_mean

    for position in positions_by_label.values():
        if position.is_zero():
            continue

        position_zyx_um = numpy.array([position.z, position.y, position.x], dtype=numpy.float32) * resolution_zyx_um
        distance_matrix = other_positions_table[:, 0:3] - position_zyx_um
        distance_matrix = numpy.sum(distance_matrix ** 2, axis=1)

        # Find the nearby positions
        closest_positions_indices = numpy.argpartition(distance_matrix, 7)[0:7]
        closest_distances_intensities = other_positions_table[closest_positions_indices, 3]
        closest_distances_um = distance_matrix[closest_positions_indices] ** 0.5

        # We ignore the closest distance, since that is the position itself
        closest_index = numpy.argmin(closest_distances_um)
        own_intensity = float(closest_distances_intensities[closest_index])
        closest_distances_um = numpy.delete(closest_distances_um, closest_index)
        nearby_intensities = float(numpy.mean(numpy.delete(closest_distances_intensities, closest_index)))

        neighbor_distance_median_um = float(numpy.median(closest_distances_um))
        neighbor_distance_mad_um = float(numpy.median(numpy.abs(closest_distances_um - neighbor_distance_median_um)))
        results.set_position_data(position, "neighbor_distance_median_um", neighbor_distance_median_um)
        results.set_position_data(position, "neighbor_distance_variation",
                                  neighbor_distance_mad_um / neighbor_distance_median_um)
        results.set_position_data(position, "intensity_factor_local", own_intensity / nearby_intensities)


def _measure_shape(positions_by_label: Dict[int, Position],
                   properties_by_label: Dict[int, "skimage.measure._regionprops.RegionProperties"],
                   resolution: ImageResolution, results: PositionData):
    """Looks at the shape of a cell."""
    if len(positions_by_label) == 0:
        return
    pixel_volume_um3 = resolution.pixel_size_x_um * resolution.pixel_size_y_um * resolution.pixel_size_z_um
    median_intensity = numpy.median([properties.intensity_mean for properties in properties_by_label.values()])
    lowest_z = min(position.z for position in positions_by_label.values())

    for label, properties in properties_by_label.items():
        position = positions_by_label.get(label)
        if position is None or position.is_zero():
            # Segmentation failed for this label
            continue

        padded = numpy.pad(properties.image_filled, 2, mode='constant', constant_values=0)
        padded = scipy.ndimage.binary_opening(padded, structure=numpy.ones((3, 5, 5)))
        volume_um3 = float(numpy.sum(padded) * pixel_volume_um3)
        if volume_um3 == 0:
            continue  # Nothing was left after opening

        convex_hull_image = skimage.morphology.convex_hull_image(padded)
        solidity = float(numpy.sum(padded) / numpy.sum(convex_hull_image))

        vertices, faces, _, _ = skimage.measure.marching_cubes(padded, level=.5,
                                                               spacing=(
                                                                   resolution.pixel_size_z_um,
                                                                   resolution.pixel_size_y_um,
                                                                   resolution.pixel_size_x_um))
        surface_um2 = float(skimage.measure.mesh_surface_area(vertices, faces))
        distances = scipy.spatial.distance.pdist(vertices, 'sqeuclidean')
        feret_diameter_max_um = math.sqrt(numpy.max(distances))
        intensity_factor = float(properties.intensity_mean / median_intensity)
        ellipticity = float((properties.axis_major_length - properties.axis_minor_length) / properties.axis_major_length)
        organoid_relative_z_um = (position.z - lowest_z) * resolution.pixel_size_z_um

        results.set_position_data(position, "volume_um3", volume_um3)
        results.set_position_data(position, "solidity", solidity)
        results.set_position_data(position, "surface_um2", surface_um2)
        results.set_position_data(position, "feret_diameter_max_um", feret_diameter_max_um)
        results.set_position_data(position, "intensity_factor", intensity_factor)
        results.set_position_data(position, "ellipticity", ellipticity)
        results.set_position_data(position, "organoid_relative_z_um", organoid_relative_z_um)


def _get_positions_by_label(experiment: Experiment, time_point: TimePoint, segmented_image: Image):
    positions_by_label = dict()
    for position in experiment.positions.of_time_point(time_point):
        label = segmented_image.value_at(position)
        if label is None:
            print("No label for position")
            continue  # Outside image
        if label in positions_by_label:
            # Two positions in one label, segmentation failed
            # Mark this label as failed. (Note: setting it to None wouldn't work, as then if we add a
            positions_by_label[label] = Position(0, 0, 0, time_point=time_point)
            continue
        positions_by_label[label] = position
    return positions_by_label
