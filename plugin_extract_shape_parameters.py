import math
from typing import Any, Dict

import numpy
import skimage.measure

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import Image
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog, action
from organoid_tracker.gui.gui_experiment import GuiExperiment
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window):
    return {
        "Segmentation//Segment-Extract shape parameters...": lambda: _extract_segmentation_parameters(window)
    }


def _extract_segmentation_parameters(window: Window):
    experiment_gui = window.get_gui_experiment()
    resolution = experiment_gui.get_experiment().images.resolution()
    temporary_experiment = experiment_gui.get_experiment().copy_selected(positions=True, links=True, connections=True)

    if not dialog.popup_message_cancellable("Incorporate segmentation", f"Please select the segmentation for this organoid."):
        return

    # Load images
    from organoid_tracker.gui import image_series_loader_dialog
    if not image_series_loader_dialog.prompt_image_series(temporary_experiment):
        return
    # Copy over resolution and offsets
    temporary_experiment.images.set_resolution(resolution)
    temporary_experiment.images.offsets = experiment_gui.get_experiment().images.offsets

    # Start!
    window.get_scheduler().add_task(_AnalyzeShapesTask(experiment_gui, temporary_experiment))


class _AnalyzeShapesTask(Task):

    _experiment_original: GuiExperiment
    _experiment_copy: Experiment

    def __init__(self, experiment_original: GuiExperiment, experiment_copy: Experiment):
        self._experiment_original = experiment_original
        self._experiment_copy = experiment_copy

    def compute(self) -> PositionData:
        return _analyze_shapes(self._experiment_copy)

    def on_finished(self, result: PositionData):
        position_data = self._experiment_original.get_experiment().position_data
        for data_name in ["volume_um3", "solidity", "surface_um2", "surface_to_volume_ratio", "label",
                          "neighbor_distance_avg_um", "neighbor_separation_avg_um"]:
            position_data.delete_data_with_name(data_name)

        self._experiment_original.get_experiment().position_data.merge_data(result)
        self._experiment_original.undo_redo.clear()
        dialog.popup_message("Extraction finished", "Stored all the metadata of the positions.")


def _analyze_shapes(experiment: Experiment) -> PositionData:
    """Measures on the experiment. The experiment must have only one channel, which contains the segmentation."""
    resolution = experiment.images.resolution()
    results = PositionData()

    import skimage.measure
    for time_point in experiment.images.time_points():
        print(f"Working on time point {time_point.time_point_number()}...")
        segmented_image = experiment.images.get_image(time_point)
        if segmented_image is None:
            continue

        # Index the properties
        regionprops_by_label = dict()
        for properties in skimage.measure.regionprops(segmented_image.array):
            regionprops_by_label[properties.label] = properties

        # Index positions to label
        positions_by_label = _get_positions_by_label(experiment, time_point, segmented_image)

        # First pass: characterize the shape of the nuclei
        _measure_shape(positions_by_label, regionprops_by_label, resolution, results)

        # Second pass: characterize the environment
        _measure_neighborhood(experiment, positions_by_label, resolution, results)

    return results


def _measure_neighborhood(experiment: Experiment, positions_by_label: Dict[int, Position], resolution: ImageResolution,
                          results: PositionData):
    """Looks at the distances to nearby cells."""
    for position in positions_by_label.values():
        if position.is_zero():
            continue

        volume_um3 = results.get_position_data(position, "volume_um3")
        if volume_um3 is None:
            continue
        spherical_radius_um = (volume_um3 / (math.pi * 4 / 3)) ** (1 / 3)

        neighbor_count = 0
        neighbor_distance_sum = 0
        neighbor_separation_sum = 0
        for neighbor in experiment.connections.find_connections(position):
            neighbor_volume_um3 = results.get_position_data(neighbor, "volume_um3")
            if neighbor_volume_um3 is None:
                continue
            neighbor_spherical_radius_um = (volume_um3 / (math.pi * 4 / 3)) ** (1 / 3)

            center_to_center_distance = neighbor.distance_um(position, resolution)
            neighbor_count += 1
            neighbor_distance_sum += center_to_center_distance
            neighbor_separation_sum += center_to_center_distance - spherical_radius_um - neighbor_spherical_radius_um
        if neighbor_count < 3:
            continue  # Too few neighbors, likely at edge

        results.set_position_data(position, "neighbor_distance_avg_um",
                                  neighbor_distance_sum / neighbor_count)
        results.set_position_data(position, "neighbor_separation_avg_um",
                                  neighbor_separation_sum / neighbor_count)


def _measure_shape(positions_by_label: Dict[int, Position], properties_by_label: Dict[int, Any],
                   resolution: ImageResolution, results: PositionData):
    """Looks at the shape of a cell."""
    pixel_volume_um3 = resolution.pixel_size_x_um * resolution.pixel_size_y_um * resolution.pixel_size_z_um

    for label, properties in properties_by_label.items():
        position = positions_by_label.get(label)
        if position is None or position.is_zero():
            # Segmentation failed for this label
            continue

        volume_um3 = float(properties.area * pixel_volume_um3)
        solidity = float(properties.solidity)

        padded = numpy.pad(properties.image, 2, mode='constant', constant_values=0)
        vertices, faces, _, _ = skimage.measure.marching_cubes(padded, level=.5,
                                                               spacing=(
                                                               resolution.pixel_size_z_um, resolution.pixel_size_y_um,
                                                               resolution.pixel_size_x_um))
        surface_um2 = float(skimage.measure.mesh_surface_area(vertices, faces))

        surface_to_volume_ratio = surface_um2 / volume_um3
        results.set_position_data(position, "volume_um3", volume_um3)
        results.set_position_data(position, "solidity", solidity)
        results.set_position_data(position, "surface_um2", surface_um2)
        results.set_position_data(position, "surface_to_volume_ratio", surface_to_volume_ratio)
        results.set_position_data(position, "label", int(label))


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
            print("Multiple positions for label " + str(label))
            positions_by_label[label] = Position(0, 0, 0, time_point=time_point)
            continue
        positions_by_label[label] = position
    return positions_by_label


