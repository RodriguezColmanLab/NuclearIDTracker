import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.linear_model import LogisticRegression

import lib_figures
import lib_models

_MODEL_PATH = "../../Data/Models/epochs-1-neurons-0/"
_NUMBER_OF_GENES = 6


def main():
    model = lib_models.load_model(_MODEL_PATH)
    regressor: LogisticRegression = model.logistic_regression
    input_output = model.get_input_output()

    figure = lib_figures.new_figure(size=(3, 6))
    axes = figure.subplots(nrows=len(input_output.cell_type_mapping), ncols=1, sharex=True, sharey=True)

    for ax, cell_type, coefficients in zip(axes, model.get_input_output().cell_type_mapping, regressor.coef_):
        ax: Axes
        ax.set_title(lib_figures.style_cell_type_name(cell_type))

        sorting = numpy.argsort(coefficients)
        sorting = numpy.concatenate([sorting[:_NUMBER_OF_GENES // 2], sorting[-_NUMBER_OF_GENES // 2:]])

        features = [lib_figures.style_variable_name(input_output.input_mapping[i]) for i in sorting]
        coefficients = coefficients[sorting]

        for i in range(len(coefficients)):
            alignment = "right" if coefficients[i] > 0 else "left"
            x_position = 0.1 if coefficients[i] < 0 else -0.1
            ax.text(x_position, i, features[i], ha=alignment, va="center", in_layout=False, color="#666666")
        ax.barh(y=range(len(coefficients)), width=coefficients, color=lib_figures.CELL_TYPE_PALETTE[cell_type], height=0.5)

        # Hide vertical axis line
        ax.set_yticks([])
        ax.spines[['right', 'top', 'left']].set_visible(False)

    axes[-1].set_ylim(-1, _NUMBER_OF_GENES - 0.5)
    axes[-1].set_xlim(-3, 3)
    axes[-1].set_xlabel("Coefficient")
    figure.tight_layout(w_pad=0)
    plt.show()


if __name__ == "__main__":
    main()
