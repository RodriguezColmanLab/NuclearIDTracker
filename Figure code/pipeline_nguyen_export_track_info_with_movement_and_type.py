# For all the organoids (control and DT remove): per cells: cell ID, cell type, total time recorded,
# cell speed (total move/time), travel distance abs(late - begin position), distance to the axis (as you calculated)
import os
from typing import Optional, List

import numpy

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution, ImageTimings
from organoid_tracker.core.spline import SplineCollection
from organoid_tracker.imaging import list_io
from organoid_tracker.linking import cell_division_finder

_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_DATA_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_OUTPUT_FOLDER = "../../Data/CSV exports Nguyen's format/By track, with movement and cell type"


def main():
    os.makedirs(_OUTPUT_FOLDER, exist_ok=True)
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_CONTROL, load_images=False):
        _export_organoid_track_info(experiment, os.path.join(_OUTPUT_FOLDER, experiment.name.get_save_name() + ".csv"))
        _export_organoid_divisions(experiment,
                                   os.path.join(_OUTPUT_FOLDER, experiment.name.get_save_name() + "_divisions.csv"))
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_REGENERATION, load_images=False):
        _export_organoid_track_info(experiment, os.path.join(_OUTPUT_FOLDER, experiment.name.get_save_name() + ".csv"))
        _export_organoid_divisions(experiment,
                                   os.path.join(_OUTPUT_FOLDER, experiment.name.get_save_name() + "_divisions.csv"))


def filter_lineages(experiment: Experiment, starting_track: LinkingTrack):
    required_time_point_number = experiment.positions.last_time_point_number() // 2

    return starting_track.first_time_point_number() == experiment.positions.first_time_point_number() and \
        _last_time_point_number(starting_track) >= required_time_point_number


def _last_time_point_number(starting_track: LinkingTrack) -> int:
    last_time_point_number = starting_track.last_time_point_number()
    for track in starting_track.find_all_descending_tracks(include_self=False):
        last_time_point_number = max(last_time_point_number, track.last_time_point_number())
    return last_time_point_number


def _export_organoid_track_info(experiment: Experiment, data_file: str):
    cell_types = experiment.global_data.get_data("ct_probabilities")
    if cell_types is None:
        raise ValueError("Cell type probabilities are not available")
    position_data = experiment.position_data
    timings = experiment.images.timings()
    resolution = experiment.images.resolution()
    splines = experiment.splines

    with open(data_file, "w") as handle:
        handle.write(
            "Cell ID, Cell type, Total hours recorded, Cell speed (um/min), Distance traveled (um), Difference on crypt-villus axis\n")

        for track in experiment.links.find_all_tracks():
            if not filter_lineages(experiment, track):
                continue

            track_id = experiment.links.get_track_id(track)
            cell_type = _get_cell_type(cell_types, position_data, track)
            time_start_h = timings.get_time_h_since_start(track.first_time_point())
            time_end_h = timings.get_time_h_since_start(track.last_time_point() + 1)
            total_time_recorded = time_end_h - time_start_h
            cell_speed_um_m = _get_cell_speed_um_m(resolution, timings, track)
            travel_distance_um = track.find_last_position().distance_um(track.find_first_position(), resolution)
            crypt_axis_um = _get_crypt_axis_change_um(splines, resolution, track)

            handle.write(
                f"{track_id}, {cell_type}, {total_time_recorded}, {cell_speed_um_m}, {travel_distance_um}, {crypt_axis_um}\n")


def _get_cell_type(cell_types: List[str], position_data: PositionData, track: LinkingTrack) -> Optional[str]:
    overall_probabilities = numpy.zeros(len(cell_types), dtype=float)
    overall_probabilities_count = 0
    for position in track.positions():
        probabilities = position_data.get_position_data(position, "ct_probabilities")
        if probabilities is None:
            continue
        overall_probabilities += probabilities
        overall_probabilities_count += 1

    if overall_probabilities_count == 0:
        return None
    return cell_types[numpy.argmax(overall_probabilities)]


def _get_cell_speed_um_m(resolution: ImageResolution, timings: ImageTimings, track: LinkingTrack) -> float:
    total_distance = 0
    total_time_m = 0

    previous_position = None
    for position in track.positions():
        if previous_position is not None:
            total_distance += position.distance_um(previous_position, resolution)
            time_difference_m = timings.get_time_m_since_start(position.time_point_number()) - \
                                timings.get_time_m_since_start(previous_position.time_point_number())
            total_time_m += time_difference_m
        previous_position = position

    if total_time_m == 0:
        return 0
    return total_distance / total_time_m


def _get_crypt_axis_change_um(splines: SplineCollection, resolution: ImageResolution, track: LinkingTrack) -> Optional[
    float]:
    axis_position_first = splines.to_position_on_spline(track.find_first_position(), only_axis=True)
    axis_position_last = splines.to_position_on_spline(track.find_last_position(), only_axis=True)

    if axis_position_first is None or axis_position_last is None:
        return None

    axis_position_difference = axis_position_last.pos - axis_position_first.pos
    return axis_position_difference * resolution.pixel_size_x_um


def _export_organoid_divisions(experiment: Experiment, output_file: str):
    timings = experiment.images.timings()
    links = experiment.links

    with open(output_file, "w") as handle:
        handle.write(
            "Mother cell ID, Mother track TZYX, Division time (hours), Daughter 1 cell ID, Daughter 2 cell ID, Daughter 3 cell ID, Daughter 4 cell ID,"
            "Daughter 1 track TZYX, Daughter 2 track TZYX, Daughter 3 track TZYX, Daughter 4 track TZYX\n")
        for track in links.find_all_tracks():
            if not filter_lineages(experiment, track):
                continue

            if not track.will_divide():
                continue

            division_time = timings.get_time_h_since_start(track.last_time_point() + 1)
            daughter_ids = [str(links.get_track_id(t)) for t in track.get_next_tracks()]
            while len(daughter_ids) < 4:
                daughter_ids.append("")
            if len(daughter_ids) > 4:
                raise ValueError("A cell divided into more than 4 daughters")
            daughter_txyz = [t.find_first_position().to_dict_key() for t in track.get_next_tracks()]
            while len(daughter_txyz) < 4:
                daughter_txyz.append("")

            handle.write(f"{links.get_track_id(track)}, {track.find_first_position().to_dict_key()}, {division_time},"
                         f" {', '.join(daughter_ids)}, {', '.join(daughter_txyz)}\n")


if __name__ == "__main__":
    main()
