import csv
import os

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.vector import Vector3
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window


def export_probabilities_to_csv(window: Window):
    """Writes the probabilities of a cell having a certain type to a folder of CSV files, for import in Paraview."""
    experiment = window.get_experiment()

    resolution = experiment.images.resolution()

    cell_type_names = experiment.global_data.get_data("ct_probabilities")
    if cell_type_names is None:
        raise UserError("No predicted cell types", f"The experiment {experiment.name} doesn't contain automatically"
                                                   f" predicted cell types")

    folder_name = dialog.prompt_save_file("Folder for CSV files", [("Folder", "*")])
    if folder_name is None:
        return

    os.makedirs(folder_name, exist_ok=True)
    for time_point in experiment.positions.time_points():
        file_name = os.path.join(folder_name, experiment.name.get_save_name() + ".csv." + str(time_point.time_point_number()))
        with open(file_name, "w", newline='') as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(["x", "y", "z"] + cell_type_names)
            for position in experiment.positions.of_time_point(time_point):
                probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
                if probabilities is None:
                    probabilities = [float("nan")] * len(cell_type_names)
                vector = position.to_vector_um(resolution)
                csv_writer.writerow([vector.x, vector.y, vector.z] + probabilities)
    window.set_status("Exported all cell type probabilities to " + folder_name + ".")
