from . import _viterbi as viterbi
from organoid_tracker.core.experiment import Experiment


def apply_viterbi(experiment: Experiment, file: str):
    """Replaced all predicted cell types by those from the Viterbi algorithm, based on the output of
    the network (which must already be stored in the experiment) and the cell type transition probabilities from
    the model folder."""
    viterbi_er = viterbi.load_transition_probabilities(file)
    experiment.position_data.delete_data_with_name("type")

    viterbi_er.run_viterbi(experiment)
