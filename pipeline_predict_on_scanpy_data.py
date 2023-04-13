import numpy
import pandas
import scanpy
from anndata import AnnData

import lib_models

_MODEL_FOLDER = r"Output models\epochs-1-neurons-0"
_INPUT_FILE = "all_data.h5ad"
_OUTPUT_FILE = "all_data_with_predictions.h5ad"
_PREDICTIONS_FILE = "predictions_only.h5ad"


def main():
    model = lib_models.load_model(_MODEL_FOLDER)
    adata = scanpy.read_h5ad(_INPUT_FILE)

    result = model.predict(adata.X)
    cell_types = numpy.array(model.get_input_output().cell_type_mapping)
    resulting_cell_types = cell_types[numpy.argmax(result, axis=1)]
    resulting_cell_types = pandas.Categorical(resulting_cell_types, cell_types)
    adata.obs["predicted_cell_type"] = resulting_cell_types
    for i, cell_type in enumerate(cell_types):
        adata.obs["predicted_" + cell_type + "_score"] = result[:, i]
    adata.write_h5ad(_OUTPUT_FILE, compression="gzip")

    predictions_only = AnnData(numpy.array(result, dtype=numpy.float32))
    predictions_only.var_names = numpy.array(["predicted_" + cell_type.lower() + "_score" for cell_type in cell_types])
    predictions_only.obs_names = adata.obs_names
    predictions_only.obs["predicted_cell_type"] = pandas.Categorical(resulting_cell_types)
    predictions_only.obs["cell_type"] = adata.obs["cell_type"]
    predictions_only.obs["organoid"] = adata.obs["organoid"]
    predictions_only.write_h5ad(_PREDICTIONS_FILE, compression="gzip")


if __name__ == "__main__":
    main()
