import json
import math
from typing import List, Dict, NamedTuple, Optional, Tuple, Iterable, Set

from ._cell_type_converter import ClassifiedCellType, UNKNOWN_CELL_TYPE, convert_cell_type
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.position_analysis import position_markers

TRANSITION_FILE = "cell_type_transition_counts.json"


class _TransitionMatrix:
    _transition_dict: Dict[ClassifiedCellType, Dict[ClassifiedCellType, int]]

    def __init__(self):
        self._transition_dict = dict()

    def print(self):
        for from_type, from_dict in self._transition_dict.items():
            sum_of_type = sum(from_dict.values())
            for to_type, count in from_dict.items():
                print(from_type, to_type, count, count / sum_of_type)

    def load_from_serialized(self, serialized_dict: Dict[str, Dict[str, int]]):
        """Loads the data from a dictionary created by self.serialize(). Clears any existing data in this object."""
        self._transition_dict.clear()
        for from_type_str, from_dict in serialized_dict.items():
            from_type = ClassifiedCellType(from_type_str)
            self._transition_dict[from_type] = dict()
            for to_type_str, count in from_dict.items():
                to_type = ClassifiedCellType(to_type_str)
                self._transition_dict[from_type][to_type] = count

    def get_full_transition_probabilities(self) -> Dict[ClassifiedCellType, Dict[ClassifiedCellType, float]]:
        """Gets the full dictionary (all combinations of cell types are present) of all cell type transition
        probabilities."""
        cell_types = list(self._transition_dict.keys())

        # Loop over all possible of cell types
        return_dict = dict()
        for from_cell_type in cell_types:
            cell_dict = dict()
            for to_cell_type in cell_types:
                cell_dict[to_cell_type] = self._get_transition_probability(from_cell_type, to_cell_type)
            return_dict[from_cell_type] = cell_dict
        return return_dict

    def _get_transition_probability(self, from_type: ClassifiedCellType, to_type: ClassifiedCellType) -> float:
        """Gets the probability that a given cell type switches to another cell type in a single time step."""
        from_dict = self._transition_dict[from_type]
        if to_type in from_dict:
            total_count = sum(from_dict.values())
            return from_dict[to_type] / total_count
        return 0


class _EmittingProbabilitiesOfTrack(NamedTuple):
    # List of positions, in chronological order.
    # Warning: certain time points may be skipped, if the neural network didn't predict a chance for the position
    # at the time point. So the next time point for which we have data isn't always simply t + 1.
    # You can use self.get_next_time_point(index) to get the next time point.
    positions: List[Position]

    log_probabilities: List[Dict[ClassifiedCellType, float]]

    def indices_and_positions_backwards(self) -> Iterable[Tuple[int, Position]]:
        """Gets the positions 0 to one-to-last in reverse order. Useful for the Viterbi iteration. Also returns
        the indices, which you can use in the other methods of this class."""
        indices = list(range(len(self.positions)))

        indices_reversed = indices[::-1]
        positions_reversed = self.positions[::-1]

        return zip(indices_reversed[1:], positions_reversed[1:])

    def get(self, index: int, state: ClassifiedCellType) -> float:
        """Gets the log probability for the given time index and cell type."""
        return self.log_probabilities[index][state]

    def get_next_time_point(self, index: int) -> TimePoint:
        """Gets the time point for the next position after self.positions[index]. This is equal to
        `self.positions[index + 1].time_point()`. Not that this is NOT equal to self.positions[index].time_point() + 1,
        as time points may be skipped in the data."""
        return self.positions[index + 1].time_point()


class _StateEntry(NamedTuple):
    """Stores the probability for a certain state, given the previous state."""
    log_prob: float
    next: Optional[ClassifiedCellType]

    def __repr__(self):
        return f"_StateEntry({self.log_prob:.4f}, {self.next})"

    @property
    def prob(self) -> float:
        return math.exp(self.log_prob)


class _TrellisAndPointersOfTrack:
    """Holds the path of each cell type at each time point for a single track (chain)."""
    _trellis_and_pointers: Dict[TimePoint, Dict[ClassifiedCellType, _StateEntry]]

    def __init__(self):
        self._trellis_and_pointers = dict()

    def set(self, time_point: TimePoint, cell_type: ClassifiedCellType, entry: _StateEntry):
        if time_point not in self._trellis_and_pointers:
            self._trellis_and_pointers[time_point] = dict()

        self._trellis_and_pointers[time_point][cell_type] = entry

    def get(self, time_point: TimePoint, cell_type: ClassifiedCellType) -> _StateEntry:
        return self._trellis_and_pointers[time_point][cell_type]

    def optimum(self, *, start_cell_type: Optional[ClassifiedCellType]
                ) -> Iterable[Tuple[TimePoint, ClassifiedCellType]]:
        """Based on all the entries set using self.set(...), this method calculates the optimum path.

        If this track is at the start of a lineage tree, you can pass None to start_cell_type. Otherwise, you need to
        pass the cell type of the parent.

        Returned iterable follows chronological order.
        """
        time_points = list(self._trellis_and_pointers.keys())
        time_points.sort()

        optimum = []

        if start_cell_type is None:
            # Find the starting cell type

            log_max_prob = float("-inf")
            # Get most probable state at the start and its forwards track
            for state, data in self._trellis_and_pointers[time_points[0]].items():
                if data.log_prob > log_max_prob:
                    log_max_prob = data.log_prob
                    start_cell_type = state

        optimum.append(start_cell_type)
        next_cell_type = start_cell_type

        # Go forward, following the path laid out by .next
        for time_point in time_points[:-1]:
            optimum.append(self._trellis_and_pointers[time_point][next_cell_type].next)
            next_cell_type = self._trellis_and_pointers[time_point][next_cell_type].next

        return zip(time_points, optimum)

    def first_time_point(self) -> TimePoint:
        min_time_point = None
        for time_point in self._trellis_and_pointers.keys():
            if min_time_point is None or time_point.time_point_number() < min_time_point.time_point_number():
                min_time_point = time_point
        if min_time_point is None:
            raise ValueError("Trellis is empty")
        return min_time_point

    def is_empty(self) -> bool:
        return len(self._trellis_and_pointers) == 0


class _TrellisTree:
    """Makes a tree structure out of all the individual trellis chains"""

    _pending_tracks: Set[LinkingTrack]
    _done_tracks: Dict[LinkingTrack, _TrellisAndPointersOfTrack]

    def __init__(self, end_tracks: Iterable[LinkingTrack]):
        self._pending_tracks = set(end_tracks)
        self._done_tracks = dict()

    def pop_pending_track(self) -> Tuple[LinkingTrack, List[_TrellisAndPointersOfTrack]]:
        """Gets a new track that needs to be handled, as well as the already-completed daughter tracks.
        (For tracks at the end of a lineage tree, the cell won't have any daughters.)"""
        pending_track = self._pending_tracks.pop()
        next_tracks = pending_track.get_next_tracks()
        next_trellis = [self._done_tracks[next_track] for next_track in next_tracks if
                        not self._done_tracks[next_track].is_empty()]
        return pending_track, next_trellis

    def submit_trellis(self, track: LinkingTrack, trellis: _TrellisAndPointersOfTrack):
        """Call this when you have calculated a new trellis.

        Once the trellis chain of all siblings has been calculated, the parent will automatically be added to the set of
        panding tracks.
        """
        self._done_tracks[track] = trellis

        parent_tracks = track.get_previous_tracks()
        if len(parent_tracks) == 1:
            # Found a parent

            # Check whether we can already process the parent track
            missing_sibling = False
            parent_track = next(iter(parent_tracks))
            for sibling_track in parent_track.get_next_tracks():
                if sibling_track not in self._done_tracks:
                    # Nope, still waiting for a sibling to complete
                    missing_sibling = True

            if not missing_sibling:
                # Yes, we have all information! We can continue with the parent
                self._pending_tracks.add(parent_track)

    def has_pending_tracks(self) -> bool:
        return len(self._pending_tracks) > 0

    def set_cell_types(self, position_data: PositionData):
        """Fills out the cell types over all lineage trees."""
        for track, trellis in self._done_tracks.items():
            if len(track.get_previous_tracks()) != 0:
                continue  # Skip daughter tracks here, they will be handled starting from the parent

            self._set_cell_types_of_sublineage(position_data, track, start_cell_type=None)

    def _set_cell_types_of_sublineage(self, position_data: PositionData, track: LinkingTrack, *,
                                      start_cell_type: Optional[ClassifiedCellType]):
        """Fills out of the cell types over the given sublineage. Works using recursion. Only use None for
        start_cell_type if this is the start of the lineage tree, otherwise pass the last cell type of the parent."""
        trellis = self._done_tracks[track]

        last_cell_type = None
        for time_point, cell_type in trellis.optimum(start_cell_type=start_cell_type):
            position = track.find_position_at_time_point_number(time_point.time_point_number())
            position_markers.set_position_type(position_data, position, cell_type.identifier)

            last_cell_type = cell_type

        for daughter_track in track.get_next_tracks():
            # Go to the daughters recursively
            self._set_cell_types_of_sublineage(position_data, daughter_track, start_cell_type=last_cell_type)


def _log(value: float):
    """Natural logarithm, except that log(0) is defined as float("-inf")."""
    if value == 0:
        return float("-inf")
    return math.log(value)


def _fetch_emitting_probabilities_in_log(experiment: Experiment,
                                         track: LinkingTrack) -> _EmittingProbabilitiesOfTrack:
    cell_type_order = experiment.global_data.get_data("ct_probabilities")
    cell_types = [ClassifiedCellType(cell_type.lower()) for cell_type in cell_type_order]

    positions = list()
    log_emitting_probability = list()

    for position in track.positions():
        probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
        if probabilities is None:
            continue

        # Sanity check - probabilities must sum to 1
        if abs(sum(probabilities) - 1) > 0.0001:
            raise ValueError(f"Emitting probabilities from {position} must sum to 1, but were {sum(probabilities)}")

        # Convert to log scale
        log_probablities = dict()
        for state_out, probability in zip(cell_types, probabilities):
            log_probablities[state_out] = _log(probability)

        positions.append(position)
        log_emitting_probability.append(log_probablities)
    return _EmittingProbabilitiesOfTrack(positions=positions, log_probabilities=log_emitting_probability)


class Viterbi:
    log_start_probability: Dict[ClassifiedCellType, float]
    log_transition_probability: Dict[ClassifiedCellType, Dict[ClassifiedCellType, float]]

    def __init__(self, *, start_probability: Dict[ClassifiedCellType, float],
                 transition_probability: Dict[ClassifiedCellType, Dict[ClassifiedCellType, float]]):
        # Sanity checks
        if abs(sum(start_probability.values()) - 1) > 0.0001:
            raise ValueError(f"Starting probabilities must sum to 1, but is {sum(start_probability.values())}")
        for state_from, probabilities in transition_probability.items():
            if abs(sum(probabilities.values()) - 1) > 0.0001:
                raise ValueError(
                    f"Transition probabilities from {state_from} must sum to 1, but is {sum(probabilities.values())}")

        # Convert to log scale
        self.log_start_probability = dict()
        self.log_transition_probability = dict()
        for state, probability in start_probability.items():
            self.log_start_probability[state] = _log(probability)
        for state_from, probabilities in transition_probability.items():
            self.log_transition_probability[state_from] = dict()
            for state_to, probability in probabilities.items():
                self.log_transition_probability[state_from][state_to] = _log(probability)

    def run_viterbi(self, experiment: Experiment):
        """Runs Viterbi in the entire experiment. Make sure that all cell types are deleted beforehand."""
        trellis_tree = _TrellisTree(experiment.links.find_ending_tracks())

        while trellis_tree.has_pending_tracks():
            track, next_trelles = trellis_tree.pop_pending_track()
            trellis_and_pointers = self._run_on_track(experiment, track, next_trelles)
            trellis_tree.submit_trellis(track, trellis_and_pointers)

        trellis_tree.set_cell_types(experiment.position_data)

    def _run_on_track(self, experiment: Experiment, track: LinkingTrack,
                      next_trelles: List[_TrellisAndPointersOfTrack]) -> _TrellisAndPointersOfTrack:
        internal_states = list(self.log_start_probability.keys())
        log_emitting_probability = _fetch_emitting_probabilities_in_log(experiment, track)
        trellis_and_pointers = _TrellisAndPointersOfTrack()

        if len(log_emitting_probability.positions) == 0:
            return trellis_and_pointers  # Nothing to calculate

        # For every time point: chance of getting there * emitting probability
        # For t = last: chance of getting there == ending probability
        # For t < last: chance of getting there = previous chance * transition chance

        # Last time point is a special case
        if len(next_trelles) == 0:
            # End of tree
            position = log_emitting_probability.positions[-1]
            for state in internal_states:
                trellis_and_pointers.set(position.time_point(), state, _StateEntry(
                    log_prob=self.log_start_probability[state] + log_emitting_probability.get(-1, state),
                    next=None))
        else:
            # Connect to next branches of tree
            position = log_emitting_probability.positions[-1]

            for state in internal_states:
                log_max_tr_prob = float("-inf")
                next_state_selected = None
                for next_state in internal_states:
                    log_tr_prob = 0
                    for next_trellis in next_trelles:
                        log_tr_prob += next_trellis.get(next_trellis.first_time_point(), next_state).log_prob + \
                                       self.log_transition_probability[state][next_state]
                    if log_tr_prob > log_max_tr_prob:
                        log_max_tr_prob = log_tr_prob
                        next_state_selected = next_state

                log_max_prob = log_max_tr_prob + log_emitting_probability.get(-1, state)
                trellis_and_pointers.set(position.time_point(), state,
                                         _StateEntry(log_prob=log_max_prob, next=next_state_selected))

        # Run Viterbi backwards in time
        for i, position in log_emitting_probability.indices_and_positions_backwards():
            next_time_point = log_emitting_probability.get_next_time_point(i)

            # Run a single time step
            for state in internal_states:
                log_max_tr_prob = float("-inf")
                next_state_selected = None
                for next_state in internal_states:
                    log_tr_prob = trellis_and_pointers.get(next_time_point, next_state).log_prob + \
                                  self.log_transition_probability[state][next_state]
                    if log_tr_prob > log_max_tr_prob:
                        log_max_tr_prob = log_tr_prob
                        next_state_selected = next_state

                log_max_prob = log_max_tr_prob + log_emitting_probability.get(i, state)
                trellis_and_pointers.set(position.time_point(), state,
                                         _StateEntry(log_prob=log_max_prob, next=next_state_selected))

        return trellis_and_pointers


def load_transition_probabilities(file: str) -> Viterbi:
    """Loads a Viterbi model from a folder where it was saved using self.save_transition_probabilities(...)."""
    transition_matrix = _TransitionMatrix()
    with open(file, "r") as handle:
        transition_matrix.load_from_serialized(json.load(handle))

    transition_probabilities = transition_matrix.get_full_transition_probabilities()

    # We give each cell an equal start probability
    # (calculating equilibrium probabilities from the transition matrix wouldn't work, since that matrix doesn't
    # take cell divisions and cell death into account)
    cell_types = list(transition_probabilities.keys())
    start_probabilities = dict()
    for cell_type in cell_types:
        start_probabilities[cell_type] = 1 / len(cell_types)

    return Viterbi(start_probability=start_probabilities, transition_probability=transition_probabilities)


def _get_links_with_cell_types(experiment: Experiment, *, allow_cell_type_guessing: bool = False
                               ) -> Iterable[Tuple[Position, ClassifiedCellType, Position, ClassifiedCellType]]:
    """Gets all links, along with their cell types."""
    position_data = experiment.position_data
    for position_from, position_to in experiment.links.find_all_links():
        cell_type_from = convert_cell_type(position_markers.get_position_type(position_data, position_from),
                                           allow_guessing=allow_cell_type_guessing)
        cell_type_to = convert_cell_type(position_markers.get_position_type(position_data, position_to),
                                         allow_guessing=allow_cell_type_guessing)
        yield position_from, cell_type_from, position_to, cell_type_to
