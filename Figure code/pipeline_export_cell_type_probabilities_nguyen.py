import os
from typing import List, Iterable

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.imaging import list_io

# Using the format of "P:\Rodriguez_Colman\stemcell_cancer_metab\tnguyen\1 - paper work\1 - PAPER1-MET CRC cell ident\Figure 2-final\Organoid-Tracker analysis\raw csv files\02022020pos1 tpo3 control.csv"

_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_DATA_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_OUTPUT_FOLDER = "../../Data/CSV exports Nguyen's format"
_MAX_TIME_POINT_FOR_FILTERED = 118  # Makes all experiments have the same length


def filter_lineages(experiment: Experiment, starting_track: LinkingTrack):
    required_time_point_number = _MAX_TIME_POINT_FOR_FILTERED // 2

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


def _export_tracks(experiment: Experiment, output_file: str, included_tracks: Iterable[LinkingTrack]):
    table = _ExportTable()
    position_data = experiment.position_data
    cell_types = experiment.global_data.get_data("ct_probabilities")
    if cell_types is None:
        raise UserError("No cell type probabilities found", "No cell type probabilities found in the experiment. Did you run the predict cell types step?")

    # Write the time row
    timings = experiment.images.timings()
    first_time_point_number = experiment.first_time_point_number()
    table.set_cell(1, 0, "Time (h)")
    for time_point in experiment.time_points():
        table.set_cell(time_point.time_point_number() - first_time_point_number + 2, 0, str(timings.get_time_h_since_start(time_point)))

    # Write the cells
    working_column = 1
    for track in included_tracks:
        track_id = experiment.links.get_track_id(track)
        table.set_cell(0, working_column, f"Cell {track_id}")

        previous_tracks = track.get_previous_tracks()
        if len(previous_tracks) == 1:
            previous_track = previous_tracks.pop()
            table.set_cell(0, working_column + 1, f"(daughter of cell {experiment.links.get_track_id(previous_track)})")

        table.set_cell(1, working_column, "X")
        table.set_cell(1, working_column + 1, "Y")
        table.set_cell(1, working_column + 2, "Z")
        for i, cell_type in enumerate(cell_types):
            table.set_cell(1, working_column + 3 + i, cell_type)
        for position in track.positions():
            table.set_cell(position.time_point_number() - first_time_point_number + 2, working_column, str(position.x))
            table.set_cell(position.time_point_number() - first_time_point_number + 2, working_column + 1, str(position.y))
            table.set_cell(position.time_point_number() - first_time_point_number + 2, working_column + 2, str(position.z))

            ct_probabilities = position_data.get_position_data(position, "ct_probabilities")
            if ct_probabilities is None:
                ct_probabilities = [float("nan")] * len(cell_types)
            for i, probability in enumerate(ct_probabilities):
                table.set_cell(position.time_point_number() - first_time_point_number + 2, working_column + 3 + i, str(probability))

        working_column += 3 + len(cell_types)

    table.save(output_file)


def main():
    # First do the unfiltered files of both
    export_unfiltered(_DATA_FILE_REGENERATION, "Stem cell regeneration")
    export_unfiltered(_DATA_FILE_CONTROL, "Controls for stem cell regeneration")

    # Now do the filtered files
    export_filtered(_DATA_FILE_REGENERATION, "Stem cell regeneration (filtered)")
    export_filtered(_DATA_FILE_CONTROL, "Controls for stem cell regeneration (filtered)")


def export_unfiltered(data_file: str, output_folder_name: str):
    """Exports the tracks of all experiments in the data file to CSV files."""
    output_folder_regen = os.path.join(_OUTPUT_FOLDER, output_folder_name)
    os.makedirs(output_folder_regen, exist_ok=True)
    for experiment in list_io.load_experiment_list_file(data_file, load_images=False):
        output_file = os.path.join(output_folder_regen, experiment.name.get_save_name() + ".csv")
        _export_tracks(experiment, output_file, experiment.links.find_all_tracks())


def export_filtered(data_file: str, output_folder_name: str):
    """Exports the (part of) the tracks that passes the filter to the CSV files."""
    output_folder_regen = os.path.join(_OUTPUT_FOLDER, output_folder_name)
    os.makedirs(output_folder_regen, exist_ok=True)
    for experiment in list_io.load_experiment_list_file(data_file, load_images=False, max_time_point=_MAX_TIME_POINT_FOR_FILTERED):
        output_file = os.path.join(output_folder_regen, experiment.name.get_save_name() + ".csv")

        starting_tracks = {track for track in experiment.links.find_starting_tracks() if filter_lineages(experiment, track)}
        tracks_including_offspring = set()
        for track in starting_tracks:
            for some_track in track.find_all_descending_tracks(include_self=True):
                tracks_including_offspring.add(some_track)

        _export_tracks(experiment, output_file, tracks_including_offspring)


if __name__ == "__main__":
    main()
