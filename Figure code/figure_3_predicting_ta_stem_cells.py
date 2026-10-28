"""How far in advance can we predict the cell type of a cell?

Easiest case is to predict the type of the cell itself.
Next challenge is to predict the cell type of the daughter cells, and then the grand-daughter cells.

"""
import numpy
import pandas
import sklearn
from numba.cuda.cudadrv import ndarray
from sklearn.model_selection import KFold

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers
import lib_data
import lib_models

_DATA_FILE_PREDICTED = "../../Data/Predicted data.autlist"
_DATA_FILE_GROUND_TRUTH = "../../Data/Training data.autlist"

_TRACK_SAMPLING_PERCENTAGES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
_PLOT_GENERATIONS_BACK = 4
_NUM_FOLDS = 5  # For K-fold cross-validation

class _SampledNucleusFeatures:
    """Represents the features of a nucleus, sampled at 10% of the cell cycle."""

    _features: numpy.ndarray | None  # Shape: (len(_TRACK_SAMPLING_PERCENTAGES), len(lib_data.STANDARD_METADATA_NAMES))

    def __init__(self, linking_track: LinkingTrack, positions: PositionCollection):
        linking_track_length = linking_track.get_duration_in_time_points()

        if linking_track_length < 10:
            # Comically short track, likely a tracking error. Don't calculate features
            self._features = None
            return

        features_over_time = []
        for percentage in _TRACK_SAMPLING_PERCENTAGES:
            time_point_number = linking_track.first_time_point_number() + int(linking_track_length * percentage / 100)
            position = linking_track.find_position_at_time_point_number(time_point_number)
            data_values = lib_data.get_data_array(positions, position, lib_data.STANDARD_METADATA_NAMES)
            if data_values is None:
                # Check one time point earlier
                position = linking_track.find_position_at_time_point_number(time_point_number - 1)
                data_values = lib_data.get_data_array(positions, position, lib_data.STANDARD_METADATA_NAMES)
                if data_values is None:
                    # Check one time point later
                    position = linking_track.find_position_at_time_point_number(time_point_number + 1)
                    data_values = lib_data.get_data_array(positions, position, lib_data.STANDARD_METADATA_NAMES)
                    if data_values is None:
                        # If we still can't find data, use NaN
                        data_values = [float("nan")] * len(lib_data.STANDARD_METADATA_NAMES)

            features_over_time.append(data_values)
        self._features = numpy.array(features_over_time)

    def get_features(self, percentage_back: int) -> numpy.ndarray:
        """Returns the features of the nucleus, sampled at the given percentage back in time."""
        if self._features is None:
            return numpy.full(len(lib_data.STANDARD_METADATA_NAMES), float("nan"))
        try:
            index = _TRACK_SAMPLING_PERCENTAGES.index(percentage_back)
        except ValueError:
            raise ValueError(f"Invalid percentage_back: {percentage_back}. Must be one of {_TRACK_SAMPLING_PERCENTAGES}")
        return self._features[index]

class _PastOfCell:
    """Represents the past of a cell, including its parent tracks."""
    _last_track: LinkingTrack  # Track reaching the end of the movie, so most recent

    _full_parent_tracks: list[LinkingTrack]  # Lineage of self._last_track. [0] is the direct parent, [1] is further back, etc.
    _cut_parent_track: LinkingTrack | None = None  # Oldest parent track, which is not a full track (doesn't start at a cell birth)

    _full_parent_features: list[_SampledNucleusFeatures]  # Features of the full parent tracks, in the same order as _full_parent_tracks

    def __init__(self, last_track: LinkingTrack, positions: PositionCollection):
        self._last_track = last_track

        self._full_parent_tracks = []
        parent_tracks = last_track.get_previous_tracks()
        while len(parent_tracks) == 1:
            parent_track = parent_tracks.pop()
            self._full_parent_tracks.append(parent_track)
            parent_tracks = parent_track.get_previous_tracks()
        if len(parent_tracks) > 0:
            # Remove the last parent track, as it's not a full track
            self._cut_parent_track = parent_tracks.pop()

        self._full_parent_features = []
        for track in self._full_parent_tracks:
            self._full_parent_features.append(_SampledNucleusFeatures(track, positions))

    def has_full_parent_track(self, generations_back: int = 1) -> bool:
        """Returns True if the cell has a full parent track (so from one mitosis to the next).
        With generations_back = 1, we check if the cell has a full parent track,
        with generations_back = 2, we check if the cell has a full grandparent track, etc."""
        if generations_back <= 0:
            raise ValueError("generations_back must be at least 1")
        return len(self._full_parent_tracks) > (generations_back - 1)

    def get_features_of_parent_track(self, generations_back: int, percentage_cell_cycle: int) -> numpy.ndarray:
        """Returns the features of the parent track, sampled at 10% of the cell cycle.
        With generations_back = 1, we get the features of the parent track,
        with generations_back = 2, we get the features of the grandparent track, etc."""
        if generations_back <= 0:
            raise ValueError("generations_back must be at least 1")
        if not self.has_full_parent_track(generations_back):
            raise ValueError(f"Cell does not have a full parent track {generations_back} generations back")
        return self._full_parent_features[generations_back - 1].get_features(percentage_cell_cycle)


class _CelltypeTracks:

    _tracks_by_cell_type: dict[str, list[_PastOfCell]]

    def __init__(self):
        self._tracks_by_cell_type = dict()

    def add_track(self, cell_type: str, track: LinkingTrack, positions: PositionCollection):
        """Adds a track to the collection, under the given cell type. The positions parameter is used to extract the
        features of the track."""
        if cell_type not in self._tracks_by_cell_type:
            self._tracks_by_cell_type[cell_type] = [_PastOfCell(track, positions)]
        else:
            self._tracks_by_cell_type[cell_type].append(_PastOfCell(track, positions))

    def count_parents(self, cell_type: str, generations_back: int = 1) -> int:
        """Returns the number of tracks for the given cell type, and how many of them have a full parent track."""
        if cell_type not in self._tracks_by_cell_type:
            return 0

        tracks = self._tracks_by_cell_type[cell_type]
        tracks_with_full_parent = sum(1 for track in tracks if track.has_full_parent_track(generations_back))
        return tracks_with_full_parent

    def get_training_data(self, generations_back: int, percentage_cell_cycle: int) -> tuple[numpy.ndarray, pandas.Series]:
        """Gets the training data for the given number of generations back, and the percentage of the cell cycle to
        sample the features at, to check whether we can predict cell types."""
        all_features = []
        all_labels = []

        for cell_type, tracks in self._tracks_by_cell_type.items():
            for track in tracks:
                if not track.has_full_parent_track(generations_back):
                    continue
                features = track.get_features_of_parent_track(generations_back, percentage_cell_cycle)
                if numpy.isnan(features).any():
                    continue
                all_features.append(features)
                all_labels.append(cell_type)
        return numpy.array(all_features), pandas.Series(all_labels, dtype="category")


class _LikelyhoodTable:

    _table: numpy.ndarray  # Columns are cell types, rows are: final cell
    _cell_types: list[str]

    def __init__(self, cell_types: list[str]):
        self._cell_types = cell_types
        self._table = numpy.zeros((1 + len(_TRACK_SAMPLING_PERCENTAGES) * _PLOT_GENERATIONS_BACK,
                                   len(cell_types)), dtype=numpy.float32)

    def time_points(self) -> list[tuple[int, int]]:
        """Returns a list of tuples (percentage_cell_cycle, generations_back) for each row in the table."""
        time_points = [(0, 0)]  # Final cell is a special case

        # Then we run backwards in time.
        # First one generation back at 90% of the cell cycle, then 80%, etc. Then two generations back, etc.
        for generations_back in range(1, _PLOT_GENERATIONS_BACK + 1):
            for percentage_cell_cycle in reversed(_TRACK_SAMPLING_PERCENTAGES):
                time_points.append((percentage_cell_cycle, generations_back))
        return time_points

    def add_results(self, cell_types: list[str], percentage_cell_cycle: int, generations_back: int, accuracies: list[float]):
        """Adds the accuracy for the given cell type, percentage of the cell cycle, and number of generations back."""
        if set(cell_types) != set(self._cell_types):
            raise ValueError(f"Cell types {cell_types} do not match table cell types {self._cell_types}")
        if percentage_cell_cycle not in _TRACK_SAMPLING_PERCENTAGES:
            raise ValueError(f"Percentage {percentage_cell_cycle} not in table")
        if generations_back < 1 or generations_back > _PLOT_GENERATIONS_BACK:
            raise ValueError(f"Generations back {generations_back} not in table")

        # row_index mirrors the time_points() method
        row_index = 1 + (generations_back - 1) * len(_TRACK_SAMPLING_PERCENTAGES) + _TRACK_SAMPLING_PERCENTAGES.index(percentage_cell_cycle)

        # Insert at the right spot in the table
        for i, cell_type in enumerate(cell_types):
            column_index = self._cell_types.index(cell_type)
            self._table[row_index, column_index] = accuracies[i]


def main():
    # First load the predicted experiments, which have the nucleus shape parameters
    cell_types = set()
    experiments_predicted = {}
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_PREDICTED, load_images=False):
        if experiment.name.get_name().startswith("Interleukin_"):
            continue  # Skip the interleukin experiments
        experiments_predicted[experiment.name.get_name()] = experiment

        # Record which cell types are present
        cell_types.update(experiment.global_data.get_data("ct_probabilities"))

    # Then, load the ground truth experiments, which have immunostaining
    cell_type_tracks = _CelltypeTracks()
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_GROUND_TRUTH, load_images=False):
        if experiment.name.get_name().startswith("Interleukin_"):
            continue  # Skip the interleukin experiments
        if experiment.name.get_name() not in experiments_predicted:
            raise ValueError(f"Experiment {experiment.name.get_name()} not found in predicted data. Predicted experiments: {list(experiments_predicted.keys())}")

        _collect_tracking_data(experiment, experiments_predicted[experiment.name.get_name()], into=cell_type_tracks)

    # Train a model for each time point
    cell_types = list(cell_types)
    results = _LikelyhoodTable(cell_types)
    for percentage_cell_cycle, generations_back in results.time_points():
        if generations_back == 0:
            continue  # Todo, implement
        training_features, training_cell_types = cell_type_tracks.get_training_data(generations_back, percentage_cell_cycle)

        cell_type_counts = training_cell_types.value_counts().to_dict()

        print(f"Training data for {percentage_cell_cycle}% of cell cycle, {generations_back} generations back: {len(training_features)} samples, counts: {cell_type_counts}")
        confusion_matrix, cell_types = _train_model(training_features, training_cell_types)
        print("    Mean likelihood by cell type:\n ", confusion_matrix, "legend:", cell_types)


def _train_model(training_features: numpy.ndarray, training_labels: pandas.Series) -> tuple[numpy.ndarray, list[str]]:
    cell_types = list(training_labels.cat.categories)
    input_output = lib_models.ModelInputOutput(cell_types, lib_data.STANDARD_METADATA_NAMES)
    class_weights = lib_models.calculate_class_weights(training_labels)

    kfold = KFold(n_splits=_NUM_FOLDS, shuffle=True)

    confusion_matrix = numpy.zeros((len(cell_types), len(cell_types)))

    for train_index, test_index in kfold.split(training_features):

        x_train, x_test = training_features[train_index], training_features[test_index]
        y_train, y_test = numpy.array(training_labels.iloc[train_index].cat.codes), numpy.array(training_labels.iloc[test_index].cat.codes)
        model = lib_models.build_shallow_model(input_output, x_train, hidden_neurons=0)
        model.fit(x_train, y_train, class_weights=class_weights)

        predicted_cell_types = model.predict(x_test).argmax(axis=1)  # Get the predicted cell type index for each sample
        confusion_matrix += sklearn.metrics.confusion_matrix(y_test, predicted_cell_types, labels=numpy.arange(len(cell_types)))


    return confusion_matrix, cell_types


def _collect_tracking_data(experiment_ground_gruth: Experiment, experiment_predicted: Experiment, *, into: _CelltypeTracks, ):
    for position in experiment_ground_gruth.positions.of_time_point(experiment_ground_gruth.positions.last_time_point()):
        cell_type = position_markers.get_position_type(experiment_ground_gruth.positions, position)
        cell_type = lib_data.convert_cell_type(cell_type)
        if cell_type == "NONE":
            continue

        track = experiment_ground_gruth.links.get_track(position)
        if track is None:
            continue

        into.add_track(cell_type, track, experiment_predicted.positions)


if __name__ == "__main__":
    main()
