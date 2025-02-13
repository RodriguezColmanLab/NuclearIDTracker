import os
from typing import List, Iterable, Optional

import numpy

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.imaging import list_io

# Using the format of "P:\Rodriguez_Colman\stemcell_cancer_metab\tnguyen\1 - paper work\1 - PAPER1-MET CRC cell ident\Figure 2-final\Organoid-Tracker analysis\raw csv files\02022020pos1 tpo3 control.csv"
# Difference with version 1 is that the final cell type is now included in the CSV file, and that it no longer cuts
# off tracking data by itself. (All tracking data is assumed to be cut to 25h already.)

_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_DATA_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_OUTPUT_FOLDER = "../../Data/CSV exports Nguyen's format"

_CELL_TYPE_TIME_POINT_COUNT = 10
_MAX_TRACKS_IN_SEQUENCE = 5


def filter_lineages(experiment: Experiment, starting_track: LinkingTrack):
    required_time_point_number = experiment.positions.last_time_point_number() // 2

    return starting_track.first_time_point_number() == experiment.positions.first_time_point_number() and \
        _last_time_point_number(starting_track) >= required_time_point_number


def _last_time_point_number(starting_track: LinkingTrack) -> int:
    last_time_point_number = starting_track.last_time_point_number()
    for track in starting_track.find_all_descending_tracks(include_self=False):
        last_time_point_number = max(last_time_point_number, track.last_time_point_number())
    return last_time_point_number


class _ExportTable:

    _rows: List[List[str]]

    def __init__(self):
        self._rows = []

    def set_cell(self, row_number: int, column_number: int, value: str):
        while len(self._rows) <= row_number:
            self._rows.append([])
        row = self._rows[row_number]
        while len(row) <= column_number:
            row.append("")
        row[column_number] = value

    def save(self, output_file: str):
        with open(output_file, "w") as file:
            file.write("sep=,\n")
            for row in self._rows:
                file.write(",".join(row) + "\n")


def _find_track_sequences(starting_track: LinkingTrack) -> Iterable[List[LinkingTrack]]:
    """Finds all sequences of tracks that start with the given track and end with a final track."""
    for track in starting_track.find_all_descending_tracks(include_self=True):
        if len(track.get_next_tracks()) == 0:
            # Found a final track, now go back to the start to find the full sequence
            track_sequence = list()
            current_track = track
            while current_track is not None:
                track_sequence.append(current_track)
                previous_tracks = current_track.get_previous_tracks()
                if len(previous_tracks) != 1:
                    break
                current_track = previous_tracks.pop()
            if track_sequence[-1] != starting_track:
                continue  # Couldn't find the full sequence (cell merge?)
            track_sequence.reverse()
            yield track_sequence


def _get_cell_type_start(position_data: PositionData, all_cell_types: List[str], track_sequence: List[LinkingTrack]
                         ) -> Optional[str]:
    """Gets the most likely cell type at the start of the track sequence."""
    probabilities = numpy.full((len(all_cell_types), _CELL_TYPE_TIME_POINT_COUNT), fill_value=numpy.NAN, dtype=numpy.float32)
    i = 0
    for track in track_sequence:
        if i >= _CELL_TYPE_TIME_POINT_COUNT:
            break
        for position in track.positions():
            probabilities_position = position_data.get_position_data(position, "ct_probabilities")
            if probabilities_position is not None:
                probabilities[:, i] = probabilities_position
            i += 1
            if i >= _CELL_TYPE_TIME_POINT_COUNT:
                break
    probabilities_mean = numpy.nanmean(probabilities, axis=1)

    # Return None if there are any NaN values
    if numpy.any(numpy.isnan(probabilities_mean)):
        return None

    # Otherwise, return the most likely cell type
    return all_cell_types[numpy.argmax(probabilities_mean)]


def _get_cell_type_end(position_data: PositionData, all_cell_types: List[str], track_sequence: List[LinkingTrack]
                         ) -> Optional[str]:
    """Gets the most likely cell type at the end of the track sequence."""
    probabilities = numpy.full((len(all_cell_types), _CELL_TYPE_TIME_POINT_COUNT), fill_value=numpy.NAN, dtype=numpy.float32)
    i = 0
    for track in reversed(track_sequence):
        if i >= _CELL_TYPE_TIME_POINT_COUNT:
            break
        for position in reversed(list(track.positions())):
            probabilities_position = position_data.get_position_data(position, "ct_probabilities")
            if probabilities_position is not None:
                probabilities[:, i] = probabilities_position
            i += 1
            if i >= _CELL_TYPE_TIME_POINT_COUNT:
                break
    probabilities_mean = numpy.nanmean(probabilities, axis=1)

    # Return None if there are any NaN values
    if numpy.any(numpy.isnan(probabilities_mean)):
        return None

    # Otherwise, return the most likely cell type
    return all_cell_types[numpy.argmax(probabilities_mean)]


def _get_distance_change_um(track_sequence: List[LinkingTrack], resolution: ImageResolution) -> float:
    first_position_vector = track_sequence[0].find_first_position().to_vector_um(resolution)
    last_position_vector = track_sequence[-1].find_last_position().to_vector_um(resolution)
    return first_position_vector.distance(last_position_vector)


def _get_traveled_distance_um(track_sequence: List[LinkingTrack], resolution: ImageResolution) -> float:
    total_traveled_distance_um = 0
    previous_position = None
    for track in track_sequence:
        for position in track.positions():
            if previous_position is not None:
                total_traveled_distance_um += position.distance_um(previous_position, resolution)

            previous_position = position
    return total_traveled_distance_um


def _get_crypt_axis_change_um(track_sequence: List[LinkingTrack], experiment: Experiment) -> Optional[float]:
    spline_position_start = experiment.splines.to_position_on_spline(track_sequence[0].find_first_position(), only_axis=True)
    spline_position_end = experiment.splines.to_position_on_spline(track_sequence[-1].find_last_position(), only_axis=True)
    if spline_position_start is None or spline_position_end is None:
        return None
    if spline_position_start.spline_id != spline_position_end.spline_id:
        return None  # Cell moved to another crypt, cannot compare the distance
    return (spline_position_end.distance - spline_position_start.distance) * experiment.images.resolution().pixel_size_x_um


def _export_tracks(experiment: Experiment, output_file: str, included_starting_tracks: Iterable[LinkingTrack]):
    table = _ExportTable()
    position_data = experiment.position_data
    links = experiment.links
    timings = experiment.images.timings()
    resolution = experiment.images.resolution()
    cell_types = experiment.global_data.get_data("ct_probabilities")
    if cell_types is None:
        raise UserError("No cell type probabilities found", "No cell type probabilities found in the experiment. Did you run the predict cell types step?")

    # Write the header
    table.set_cell(0, 0, "Cell id")
    table.set_cell(0, 1, "Time observed (h)")
    table.set_cell(0, 2, "Starting state")
    table.set_cell(0, 3, "Ending state")
    table.set_cell(0, 4, "Movement speed (um/h)")
    table.set_cell(0, 5, "Traveled distance (um)")
    table.set_cell(0, 6, "Traveled distance along crypt-villus axis (um)")
    for i in range(_MAX_TRACKS_IN_SEQUENCE):
        counting_word = str(i + 1) + "th"
        if i == 0:
            counting_word = "First"
        elif i == 1:
            counting_word = "Second"
        elif i == 2:
            counting_word = "Third"
        table.set_cell(0, 7 + i, counting_word + " track TZYX")

    # Write the rows
    row_number = 1
    for starting_track in included_starting_tracks:
        for track_sequence in _find_track_sequences(starting_track):
            # Collect track ids
            track_ids = [links.get_track_id(track) for track in track_sequence]

            # Get the other variables
            time_start = timings.get_time_h_since_start(starting_track.first_time_point())
            time_end = timings.get_time_h_since_start(track_sequence[-1].last_time_point() + 1)

            # Set the table cells
            table.set_cell(row_number, 0, " ".join(str(track_id) for track_id in track_ids))
            table.set_cell(row_number, 1, str(time_end - time_start))

            # Get the cell types at the start
            cell_type_start = _get_cell_type_start(position_data, cell_types, track_sequence)
            cell_type_end = _get_cell_type_end(position_data, cell_types, track_sequence)
            table.set_cell(row_number, 2, str(cell_type_start))
            table.set_cell(row_number, 3, str(cell_type_end))

            # Get the movement speed
            table.set_cell(row_number, 4, str(_get_traveled_distance_um(track_sequence, resolution) / (time_end - time_start)))
            table.set_cell(row_number, 5, str(_get_distance_change_um(track_sequence, resolution)))
            table.set_cell(row_number, 6, str(_get_crypt_axis_change_um(track_sequence, experiment)))

            # Get the tracsk TZYX
            dict_keys = [track.find_first_position().to_dict_key() for track in track_sequence]
            while len(dict_keys) < _MAX_TRACKS_IN_SEQUENCE:
                dict_keys.append("")
            if len(dict_keys) > _MAX_TRACKS_IN_SEQUENCE:
                raise ValueError("Too many tracks in sequence")
            for i, dict_key in enumerate(dict_keys):
                table.set_cell(row_number, 7 + i, dict_key)

            row_number += 1

    table.save(output_file)


def main():
    # First do the unfiltered files of both
    export_filtered(_DATA_FILE_REGENERATION, "Stem cell regeneration multi-generation")
    export_filtered(_DATA_FILE_CONTROL, "Controls for stem cell regeneration multi-generation")


def export_filtered(data_file: str, output_folder_name: str):
    """Exports the (part of) the tracks that passes the filter to the CSV files."""
    output_folder = os.path.join(_OUTPUT_FOLDER, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    for experiment in list_io.load_experiment_list_file(data_file, load_images=False):
        output_file = os.path.join(output_folder, experiment.name.get_save_name() + ".csv")

        starting_tracks = {track for track in experiment.links.find_starting_tracks() if filter_lineages(experiment, track)}

        _export_tracks(experiment, output_file, starting_tracks)


if __name__ == "__main__":
    main()
