import math
from enum import Enum, auto
from typing import Dict, List, Tuple

import numpy
import scipy.ndimage
import skimage.measure
import skimage.morphology
import skimage.segmentation
from numpy import ndarray

from organoid_tracker.core import TimePoint, max_none
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.gui_experiment import SingleGuiTab
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window
from organoid_tracker.util.moving_average import MovingAverage


class _SingleCellParam(Enum):
    volume_um3 = auto()
    solidity = auto()
    surface_um2 = auto()
    feret_diameter_max_um = auto()
    intensity_factor = auto()
    ellipticity = auto()
    organoid_relative_z_um = auto()
    extent = auto()
    minor_axis_length_um = auto()
    intermediate_axis_length_um = auto()
    major_axis_length_um = auto()


_AVERAGING_H = 2


def _get_all_keys() -> List[str]:
    """Gets all keys that will get values during the metadata extraction process."""
    keys = list()
    for param in _SingleCellParam:
        keys.append(param.name)
        keys.append(param.name + "_local")
    keys.append("neighbor_distance_mean_um")
    keys.append("neighbor_distance_median_um")
    keys.append("neighbor_distance_variation")
    return keys


class _OurProperties:
    """Stores all properties that we use in one table."""

    _values: ndarray
    _positions_zyx_um: ndarray
    _neighbor_distances_and_variation: ndarray
    _relative_values: ndarray

    _SENTINEL_VALUE = -999999  # Used if no position has been set

    def __init__(self, label_count: int):
        self._values = numpy.zeros((label_count + 1, len(_SingleCellParam)), dtype=numpy.float32)
        self._positions_zyx_um = numpy.full((label_count + 1, 3), fill_value=self._SENTINEL_VALUE, dtype=numpy.float32)
        self._neighbor_distances_and_variation = numpy.zeros((label_count + 1, 3), dtype=numpy.float32)
        self._relative_values = numpy.zeros((label_count + 1, len(_SingleCellParam)), dtype=numpy.float32)

    def add_position(self, label: int, x_um: float, y_um: float, z_um: float):
        self._positions_zyx_um[label, 0] = x_um
        self._positions_zyx_um[label, 1] = y_um
        self._positions_zyx_um[label, 2] = z_um

    def add(self, label: int, param: _SingleCellParam, value: float):
        self._values[label, param.value - 1] = value

    def get(self, label: int, param: _SingleCellParam) -> float:
        return self._values[label, param.value - 1]

    def store(self, label_to_position: Dict[int, Position], position_data: PositionData):
        self._calculate_neighborhood()

        for label, position in label_to_position.items():
            position_data.set_position_data(position, "neighbor_distance_median_um",
                                            float(self._neighbor_distances_and_variation[label, 0]))
            position_data.set_position_data(position, "neighbor_distance_mean_um",
                                            float(self._neighbor_distances_and_variation[label, 1]))
            position_data.set_position_data(position, "neighbor_distance_variation",
                                            float(self._neighbor_distances_and_variation[label, 2]))
            for key in _SingleCellParam:
                position_data.set_position_data(position, key.name, float(self._values[label, key.value - 1]))
                position_data.set_position_data(position, key.name + "_local",
                                                float(self._relative_values[label, key.value - 1]))

    def _calculate_neighborhood(self):
        for label, position_zyx_um in enumerate(self._positions_zyx_um):
            if position_zyx_um[0] == self._SENTINEL_VALUE:
                continue

            distance_matrix = self._positions_zyx_um - position_zyx_um
            distance_matrix = numpy.sum(distance_matrix ** 2, axis=1)

            # Find the nearby positions
            closest_positions_labels = numpy.argpartition(distance_matrix, 7)[0:7]
            closest_distances_um = distance_matrix[closest_positions_labels] ** 0.5

            # We ignore the closest distance, since that is the position itself
            closest_i = int(numpy.argmin(closest_distances_um))
            closest_label = closest_positions_labels[closest_i]
            own_values = self._values[closest_label, :]
            closest_distances_um = numpy.delete(closest_distances_um, closest_i)
            neighbor_labels = numpy.delete(closest_positions_labels, closest_i)
            neighbor_values = self._values[neighbor_labels, :]
            relative_values = own_values / numpy.mean(neighbor_values, axis=0)

            neighbor_distance_median_um = float(numpy.median(closest_distances_um))
            neighbor_distance_mean_um = float(numpy.mean(closest_distances_um))
            neighbor_distance_mad_um = float(
                numpy.median(numpy.abs(closest_distances_um - neighbor_distance_median_um)))

            self._relative_values[label] = relative_values
            self._neighbor_distances_and_variation[label] = neighbor_distance_median_um, neighbor_distance_mean_um, neighbor_distance_mad_um


def get_menu_items(window: Window):
    return {
        "Tools//Process-Cell types//2. Extract shape parameters...": lambda: _extract_segmentation_parameters(window)
    }


def _extract_segmentation_parameters(window: Window):
    result = dialog.prompt_options("Extracting segmentation parameters", "Do you want to extract parameters of nuclei"
                                   " bordering the image? (Not recommended for training data.)",
                                   option_1="Segment", option_2="Segment (keep bordering)")
    if result is None:
        return
    remove_bordering = result == 1

    # Start!
    open_tabs = window.get_gui_experiment().get_active_tabs()
    max_channel = max_none([len(tab.experiment.images.get_channels()) for tab in open_tabs])
    if max_channel is None:
        return
    nucleus_channel_index = dialog.prompt_int("Nucleus channel", "Which channel is the nucleus channel? (The"
                                              " fluorescence, not the segmentation.)",
                                              minimum=1, maximum=max_channel, default=1)
    if nucleus_channel_index is None:
        return
    segmentation_channel_index = dialog.prompt_int("Segmentation channel", "Which channel contains the segmented nuclei?",
                                              minimum=1, maximum=max_channel, default=max_channel)
    if segmentation_channel_index is None:
        return

    window.get_scheduler().add_task(_AnalyzeShapesTask(open_tabs, ImageChannel(index_zero=segmentation_channel_index - 1),
                                                       ImageChannel(index_zero=nucleus_channel_index - 1), remove_bordering))


class _AnalyzeShapesTask(Task):
    _open_tabs: List[SingleGuiTab]
    _experiment_copies: List[Experiment]
    _segmentation_channel: ImageChannel
    _nucleus_channel: ImageChannel
    _remove_bordering: bool

    def __init__(self, open_tabs: List[SingleGuiTab], segmentation_channel: ImageChannel, nucleus_channel: ImageChannel,
                 remove_bordering: bool):
        self._open_tabs = open_tabs
        self._experiment_copies = [
            open_tab.experiment.copy_selected(positions=True, links=True, connections=True, images=True)
            for open_tab in open_tabs]
        self._segmentation_channel = segmentation_channel
        self._nucleus_channel = nucleus_channel
        self._remove_bordering = remove_bordering

    def compute(self) -> List[PositionData]:
        return [_analyze_shapes(experiment_copy, self._segmentation_channel, self._nucleus_channel,
                                self._remove_bordering) for experiment_copy in
                self._experiment_copies]

    def on_finished(self, results: List[PositionData]):
        for open_tab, result in zip(self._open_tabs, results):
            position_data = open_tab.experiment.position_data
            # Remove old data
            for data_name in _get_all_keys():
                position_data.delete_data_with_name(data_name)

            open_tab.experiment.position_data.merge_data(result)
            open_tab.undo_redo.clear()
        dialog.popup_message("Extraction finished", "Stored all the metadata of the positions.")


def _analyze_shapes(experiment: Experiment, segmentation_channel: ImageChannel, nucleus_channel: ImageChannel,
                    remove_bordering: bool = True) -> PositionData:
    """Measures on the experiment."""
    resolution = experiment.images.resolution()
    results = PositionData()

    import skimage.measure
    for time_point in experiment.images.time_points():
        if len(experiment.positions.of_time_point(time_point)) == 0:
            continue  # Skip time points without positions

        print(f"Working on time point {time_point.time_point_number()}...")
        segmented_image = experiment.images.get_image(time_point, segmentation_channel)
        nuclei_image = experiment.images.get_image(time_point, nucleus_channel)
        if segmented_image is None or nuclei_image is None:
            continue

        # Relabel to have continuous labels
        segmented_image.array, _, _ = skimage.segmentation.relabel_sequential(segmented_image.array)

        # Index the properties
        regionprops_by_label = dict()
        image_size_z, image_size_y, image_size_x = segmented_image.array.shape
        for properties in skimage.measure.regionprops(segmented_image.array, intensity_image=nuclei_image.array):
            regionprops_by_label[properties.label] = properties

        # Index positions to label
        positions_by_label = _get_positions_by_label(experiment, time_point, segmented_image)

        # Remove positions bordering the image
        if remove_bordering:
            for label, properties in regionprops_by_label.items():
                if label not in positions_by_label:
                    continue  # Already didn't have a position
                slice_z, slice_y, slice_x = properties.slice
                if slice_z.start == 0 or slice_z.stop >= image_size_z \
                        or slice_y.start == 0 or slice_y.stop >= image_size_y \
                        or slice_x.start == 0 or slice_x.stop >= image_size_x:
                    del positions_by_label[label]

        # First pass: characterize the shape of the nuclei
        time_point_results = _measure_shape(regionprops_by_label, resolution)
        time_point_results.store(positions_by_label, results)

    # Average all values
    # print("Averaging values over time...")
    # for track in experiment.links.find_all_tracks():
    #     _average_track(experiment, track, results)

    return results


def _average_track(experiment: Experiment, track: LinkingTrack, results: PositionData):
    resolution = experiment.images.resolution()

    for data_name in _get_all_keys():
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
            if value is not None:
                results.set_position_data(position, data_name, float(value))


def _ellipsoid_axis_lengths(central_moments: ndarray) -> Tuple[float, ...]:
    """Gets the maior, intermediate and minor axis length of the (3D) ellipsoid."""
    m0 = central_moments[0, 0, 0]
    sxx = central_moments[2, 0, 0] / m0
    syy = central_moments[0, 2, 0] / m0
    szz = central_moments[0, 0, 2] / m0
    sxy = central_moments[1, 1, 0] / m0
    sxz = central_moments[1, 0, 1] / m0
    syz = central_moments[0, 1, 1] / m0
    S = numpy.asarray([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    # determine eigenvalues in descending order
    eigvals = numpy.sort(numpy.linalg.eigvalsh(S))[::-1]
    return tuple([math.sqrt(20.0 * e) for e in eigvals])


def _measure_shape(properties_by_label: Dict[int, "skimage.measure._regionprops.RegionProperties"],
                   resolution: ImageResolution) -> _OurProperties:
    """Looks at the shape of a cell."""
    if len(properties_by_label) == 0:
        return _OurProperties(0)
    pixel_volume_um3 = resolution.pixel_size_x_um * resolution.pixel_size_y_um * resolution.pixel_size_z_um
    median_intensity = numpy.median([properties.intensity_mean for properties in properties_by_label.values()])
    lowest_z = min(properties.centroid[0] for properties in properties_by_label.values())

    storage = _OurProperties(len(properties_by_label))
    for label, properties in properties_by_label.items():
        padded = numpy.pad(properties.image_filled, 2, mode='constant', constant_values=0)
        padded = scipy.ndimage.binary_opening(padded, structure=numpy.ones((3, 5, 5)))
        volume_um3 = numpy.sum(padded) * pixel_volume_um3
        if volume_um3 == 0:
            continue  # Nothing was left after opening

        convex_hull_image = skimage.morphology.convex_hull_image(padded)
        solidity = numpy.sum(padded) / numpy.sum(convex_hull_image)

        vertices, faces, _, _ = skimage.measure.marching_cubes(padded, level=.5,
                                                               spacing=(
                                                                   resolution.pixel_size_z_um,
                                                                   resolution.pixel_size_y_um,
                                                                   resolution.pixel_size_x_um))
        surface_um2 = skimage.measure.mesh_surface_area(vertices, faces)
        distances = scipy.spatial.distance.pdist(vertices, 'sqeuclidean')
        feret_diameter_max_um = math.sqrt(numpy.max(distances))
        intensity_factor = properties.intensity_mean / median_intensity

        axis_major_length, axis_intermediate_length, axis_minor_length = _ellipsoid_axis_lengths(
            properties.moments_central)

        ellipticity = (axis_major_length - axis_minor_length) / axis_major_length
        organoid_relative_z_um = (properties.centroid[0] - lowest_z) * resolution.pixel_size_z_um
        z, y, x = properties.centroid

        storage.add_position(label, x * resolution.pixel_size_x_um, y * resolution.pixel_size_y_um,
                             z * resolution.pixel_size_z_um)
        storage.add(label, _SingleCellParam.solidity, numpy.sum(padded) / numpy.sum(convex_hull_image))
        storage.add(label, _SingleCellParam.volume_um3, volume_um3)
        storage.add(label, _SingleCellParam.solidity, solidity)
        storage.add(label, _SingleCellParam.surface_um2, surface_um2)
        storage.add(label, _SingleCellParam.feret_diameter_max_um, feret_diameter_max_um)
        storage.add(label, _SingleCellParam.intensity_factor, intensity_factor)
        storage.add(label, _SingleCellParam.ellipticity, ellipticity)
        storage.add(label, _SingleCellParam.organoid_relative_z_um, organoid_relative_z_um)
        storage.add(label, _SingleCellParam.extent, properties.extent)
        storage.add(label, _SingleCellParam.minor_axis_length_um, axis_minor_length * resolution.pixel_size_x_um)
        storage.add(label, _SingleCellParam.intermediate_axis_length_um,
                    axis_intermediate_length * resolution.pixel_size_x_um)
        storage.add(label, _SingleCellParam.major_axis_length_um, axis_major_length * resolution.pixel_size_x_um)

    return storage


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
