import os
from typing import Optional

from organoid_tracker.core import min_none
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io

_TRACKING_INPUT_FILE = "../../Data/H3K9ac reporter/Tracking data.autlist"
_OUTPUT_FOLDER = "../../Data/H3K9ac reporter/Paraview export"


def _hours_to_division(experiment: Experiment, position: Position) -> Optional[float]:
    track = experiment.links.get_track(position)
    timings = experiment.images.timings()
    if track is None:
        return None

    time_to_division = None
    if track.will_divide():
        # Found a division in the future
        time_of_division = timings.get_time_h_since_start(track.last_time_point() + 1)
        time_to_division = time_of_division - timings.get_time_h_since_start(position.time_point())
    if len(track.get_previous_tracks()) > 0:
        # Found a division in the past
        time_of_division = timings.get_time_h_since_start(track.first_time_point())
        time_to_division = min_none(time_to_division, timings.get_time_h_since_start(position.time_point()) - time_of_division)
    return time_to_division


def main():
    for experiment in list_io.load_experiment_list_file(_TRACKING_INPUT_FILE, load_images=False):
        experiment_output_folder = os.path.join(_OUTPUT_FOLDER, experiment.name.get_save_name())
        os.makedirs(experiment_output_folder, exist_ok=True)

        stem_cell_index = experiment.global_data.get_data("ct_probabilities").index("STEM")
        enterocyte_index = experiment.global_data.get_data("ct_probabilities").index("ENTEROCYTE")
        position_data = experiment.position_data
        links = experiment.links
        resolution = experiment.images.resolution()

        for time_point in experiment.positions.time_points():
            with open(os.path.join(experiment_output_folder, f"positions.{time_point.time_point_number()}.csv"), "w") as file:
                file.write("X,Y,Z,H3K9ac intensity,Stemness,Enterocyteness,Time to division(h)\n")
                for position in experiment.positions.of_time_point(time_point):
                    h3k9ac_value = position_data.get_position_data(position, "intensity")
                    cell_type_probabilities = position_data.get_position_data(position, "ct_probabilities")
                    stemness = cell_type_probabilities[stem_cell_index] if cell_type_probabilities is not None else None
                    enterocyteness = cell_type_probabilities[enterocyte_index] if cell_type_probabilities is not None else None
                    time_h_to_division = _hours_to_division(experiment, position)
                    vector = position.to_vector_um(resolution)
                    file.write(f"{vector.x},{vector.y},{vector.z},{_none_to_nan(h3k9ac_value)},{_none_to_nan(stemness)},{_none_to_nan(enterocyteness)},{_none_to_nan(time_h_to_division)}\n")


def _none_to_nan(value):
    return value if value is not None else "nan"

if __name__ == "__main__":
    main()