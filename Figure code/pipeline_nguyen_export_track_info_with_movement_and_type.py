# For all the organoids (control and DT remove): per cells: cell ID, cell type, total time recorded,
# cell speed (total move/time), travel distance abs(late - begin position), distance to the axis (as you calculated)
import os
from typing import Optional, List, Tuple

import numpy

from organoid_tracker.core.beacon_collection import BeaconCollection
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution, ImageTimings
from organoid_tracker.core.spline import SplineCollection, SplinePosition
from organoid_tracker.imaging import list_io
from organoid_tracker.linking import cell_division_finder
import lib_data

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
    beacons = experiment.beacons

    with open(data_file, "w") as handle:
        handle.write(
            "Cell ID, Cell type, Total hours recorded, Cell speed (um/min), Distance traveled (um), Crypt axis id,"
            "Crypt-villus axis start (um), Crypt-villus axis end (um), Difference on crypt-villus axis (um), "
            "Crypt-villus axis start (relative to neck), Crypt-villus axis end (relative to neck), Difference on crypt-villus axis (relative to neck), "
            "Crypt-villus axis start (relative to lumen center), Crypt-villus axis end (relative to lumen center), Difference on crypt-villus axis (relative to lumen center), "
            "Stem to EC location\n")

        for track in experiment.links.find_all_tracks():
            if not filter_lineages(experiment, track):
                continue

            track_id = experiment.links.get_track_id(track)
            cell_type_probabilities = _get_cell_type_scores(cell_types, position_data, track)
            cell_type = None if cell_type_probabilities is None else cell_types[numpy.argmax(cell_type_probabilities)]
            stem_to_ec_location = lib_data.find_stem_to_ec_location(cell_types, cell_type_probabilities)
            time_start_h = timings.get_time_h_since_start(track.first_time_point())
            time_end_h = timings.get_time_h_since_start(track.last_time_point() + 1)
            total_time_recorded = time_end_h - time_start_h
            cell_speed_um_m = _get_cell_speed_um_m(resolution, timings, track)
            travel_distance_um = track.find_last_position().distance_um(track.find_first_position(), resolution)
            spline_id, crypt_axis_first_um, crypt_axis_last_um = _get_crypt_axis_change_um(splines, resolution, track)
            crypt_axis_relative_to_neck_first, crypt_axis_relative_to_neck_last = _get_crypt_axis_change_relative_to_neck(splines, beacons, track)
            crypt_axis_relative_to_lumen_first, crypt_axis_relative_to_lumen_last = _get_crypt_axis_change_relative_to_lumen(splines, beacons, track)

            handle.write(
                f"{track_id}, {cell_type}, {total_time_recorded}, {cell_speed_um_m}, {travel_distance_um}, {spline_id}, "
                f"{crypt_axis_first_um}, {crypt_axis_last_um}, {_subtract(crypt_axis_last_um, crypt_axis_first_um)}, "
                f"{crypt_axis_relative_to_neck_first}, {crypt_axis_relative_to_neck_last},  {_subtract(crypt_axis_relative_to_neck_last, crypt_axis_relative_to_neck_first)}, "
                f"{crypt_axis_relative_to_lumen_first}, {crypt_axis_relative_to_lumen_last},  {_subtract(crypt_axis_relative_to_lumen_last, crypt_axis_relative_to_lumen_first)}, "
                f"{stem_to_ec_location}\n")


def _subtract(value1: Optional[float], value2: Optional[float]) -> Optional[float]:
    if value1 is None or value2 is None:
        return None
    return value1 - value2


def _get_cell_type_scores(cell_types: List[str], position_data: PositionData, track: LinkingTrack) -> Optional[str]:
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
    return overall_probabilities / overall_probabilities_count


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


def _get_crypt_axis_change_um(splines: SplineCollection, resolution: ImageResolution, track: LinkingTrack
                              ) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    axis_position_first = splines.to_position_on_spline(track.find_first_position(), only_axis=True)
    axis_position_last = splines.to_position_on_spline(track.find_last_position(), only_axis=True)

    if (axis_position_first is None or axis_position_last is None
            or axis_position_first.spline_id != axis_position_last.spline_id):
        return None, None, None

    return (axis_position_first.spline_id, axis_position_first.pos * resolution.pixel_size_x_um,
            axis_position_last.pos * resolution.pixel_size_x_um)


def _make_axis_position_relative_to_neck(beacons: BeaconCollection, spline_position: SplinePosition) -> float:
    """Makes the axis position relative, such that the position of the closest beacon to the spline without a defined
     type  is defined as 1.0."""
    closest_beacon_spline_position = None
    for beacon in beacons.of_time_point_with_type(spline_position.time_point):
        if beacon.beacon_type is not None:
            continue  # Some specialized beacon, ignore
        beacon_spline_position = spline_position.spline.to_position_on_axis(beacon.position)
        if closest_beacon_spline_position is None or beacon_spline_position.distance < closest_beacon_spline_position.distance:
            closest_beacon_spline_position = beacon_spline_position
    return spline_position.pos / closest_beacon_spline_position.pos


def _make_axis_position_relative_to_lumen(beacons: BeaconCollection, spline_position: SplinePosition) -> float:
    """Makes the axis position relative, such that the position of the closest beacon defined as "LUMEN" to the spline
     is defined as 1.0.
    """
    closest_beacon_spline_position = None
    for beacon in beacons.of_time_point_with_type(spline_position.time_point):
        if beacon.beacon_type != "LUMEN":
            continue  # Not the annotation of the lumen
        beacon_spline_position = spline_position.spline.to_position_on_axis(beacon.position)
        if closest_beacon_spline_position is None or beacon_spline_position.distance < closest_beacon_spline_position.distance:
            closest_beacon_spline_position = beacon_spline_position
    return spline_position.pos / closest_beacon_spline_position.pos


def _get_crypt_axis_change_relative_to_neck(splines: SplineCollection, beacons: BeaconCollection, track: LinkingTrack
                                            ) -> Tuple[Optional[float], Optional[float]]:
    axis_position_first = splines.to_position_on_spline(track.find_first_position(), only_axis=True)
    axis_position_last = splines.to_position_on_spline(track.find_last_position(), only_axis=True)

    if (axis_position_first is None or axis_position_last is None
            or axis_position_first.spline_id != axis_position_last.spline_id):
        return None, None

    last_pos = _make_axis_position_relative_to_neck(beacons, axis_position_last)
    first_pos = _make_axis_position_relative_to_neck(beacons, axis_position_first)
    return first_pos, last_pos


def _get_crypt_axis_change_relative_to_lumen(splines: SplineCollection, beacons: BeaconCollection, track: LinkingTrack
                                            ) -> Tuple[Optional[float], Optional[float]]:
    axis_position_first = splines.to_position_on_spline(track.find_first_position(), only_axis=True)
    axis_position_last = splines.to_position_on_spline(track.find_last_position(), only_axis=True)

    if (axis_position_first is None or axis_position_last is None
            or axis_position_first.spline_id != axis_position_last.spline_id):
        return None, None

    last_pos = _make_axis_position_relative_to_lumen(beacons, axis_position_last)
    first_pos = _make_axis_position_relative_to_lumen(beacons, axis_position_first)
    return first_pos, last_pos


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
