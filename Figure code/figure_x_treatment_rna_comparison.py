import matplotlib.pyplot as plt
import pandas
from anndata import AnnData
import lib_figures

_INPUT_FILE = "../../Data/GSE114113_geneExpression_readcounts_normalized.txt.gz"


def _get_treatment(column_name: str) -> str:
    if "CV" in column_name:
        return "CV"
    if "ENR" in column_name:
        return "ENR"
    if "EN" in column_name:
        return "EN"
    raise ValueError(f"Cannot extract treatment from column name \"{column_name}\".")

def _get_sample_name(column_name: str) -> str:
    return column_name.split("_")[1]


def main():
    data_frame = pandas.read_csv(_INPUT_FILE, compression='gzip', header=0, sep='\t', quotechar='"')
    gene_names = data_frame["geneSymbol"].values
    counts_columns = [column_name for column_name in data_frame.columns if column_name.startswith("counts_")]
    sample_names = [_get_sample_name(column_name) for column_name in counts_columns]

    adata = AnnData(X=data_frame[counts_columns].values.T)
    adata.var_names = gene_names
    adata.obs_names = sample_names

    adata.obs["treatment"] = [_get_treatment(column_name) for column_name in counts_columns]

    # Make bar plot of Krt20 expression in each sample
    for gene_name in ["Krt20", "Lgr5", "Lyz1"]:
        gene_expression = adata[:, gene_name].X.flatten()
        figure = lib_figures.new_figure(size=(4, 4))
        ax = figure.gca()
        ax.bar(range(len(adata.obs_names)), gene_expression)
        ax.set_xticks(range(len(adata.obs_names)), adata.obs_names, rotation=90)
        ax.set_title(gene_name)
        plt.show()




if __name__ == "__main__":
    main()
