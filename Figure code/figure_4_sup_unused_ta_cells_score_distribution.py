from collections import defaultdict

import math
import matplotlib.colors
import numpy
import pandas
import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
from anndata import AnnData
from enum import Enum, auto
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Dict, NamedTuple, Iterable, List

import lib_data
import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

LDA_FILE = "../../Data/all_data.h5ad"

_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_DATA_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"


_TA_MIN_SCORE = 0.4
_TA_MAX_SCORE = 0.55

def _get_stem_to_ec_scores(experiments: Iterable[Experiment]) -> List[float]:
    """Gets the raw features, so that we can create a heatmap. Cells are indexed by _get_cell_key."""

    stem_to_ec_score_list = list()

    # Collect position data for last 10 time points of each experiment
    for experiment in experiments:
        print("Loading", experiment.name)
        cell_types = experiment.global_data.get_data("ct_probabilities")
        position_data = experiment.positions

        for position in experiment.positions.of_time_point(experiment.positions.last_time_point()):
            probabilities = position_data.get_position_data(position, "ct_probabilities")
            stem_to_enterocyte_score = lib_data.find_stem_to_ec_location(cell_types, probabilities)

            if  stem_to_enterocyte_score is not None:
                stem_to_ec_score_list.append(stem_to_enterocyte_score)

    return stem_to_ec_score_list


def main():
    # Loading and preprocessing
    adata = scanpy.read_h5ad(LDA_FILE)
    adata = lib_figures.standard_preprocess(adata, log1p=False, scale=False)

    # Remove cells that we cannot train on
    adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Do the LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(adata.X, adata.obs["cell_type_training"])

    # Load the features and immunostaining results
    experiments_regeneration = list_io.load_experiment_list_file(_DATA_FILE_REGENERATION, load_images=False)
    experiments_control = list_io.load_experiment_list_file(_DATA_FILE_CONTROL, load_images=False)

    data_regeneration = _get_stem_to_ec_scores(experiments_regeneration)
    data_control = _get_stem_to_ec_scores(experiments_control)

    # Plotting
    figure = plt.figure(figsize=(4, 3))
    ax = figure.gca()

    bins = numpy.linspace(0, 1, 50)
    ax.hist(data_control, bins=bins, alpha=0.3, label="Control", color="#0000ff")

    ax_twin = ax.twinx()
    ax_twin.hist(data_regeneration, bins=bins, alpha=0.3, label="Regeneration", color="#ff0000")

    ax.set_xlabel("Stem to Enterocyte score")
    ax.set_ylabel("Number of cells (Control)", color="#0000ff")
    ax_twin.set_ylabel("Number of cells (Regeneration)", color="#ff0000")
    ax.set_xlim(1, 0)

    # Highlight TA cell score range
    ax.axvline(x=_TA_MIN_SCORE, color="black", linestyle="--")
    ax.axvline(x=_TA_MAX_SCORE, color="black", linestyle="--")

    figure.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
