import os

from typing import List

import math
import matplotlib.colors
import numpy
from matplotlib import pyplot as plt
import lib_figures

_IMAGE_SCALE = 1000

_SQRT_3_OVER_2 = math.sqrt(3) / 2


def _get_plotted_probabilities(x, y):
    # Given the Cartesian x and y coords, give the a, b and c values of the Ternary plot
    b = y / _SQRT_3_OVER_2
    a = x - b / 2.
    c = 1 - a - b

    if a < 0 or a > 1 or b < 0 or b > 1 or c < 0 or c > 1:
        return numpy.nan, numpy.nan, numpy.nan
    return a, b, c


def main():
    array = numpy.full((_IMAGE_SCALE, _IMAGE_SCALE, 3), dtype=numpy.float32, fill_value=numpy.nan)
    for x in range(_IMAGE_SCALE):
        for y in range(_IMAGE_SCALE):
            a, b, c = _get_plotted_probabilities(x / _IMAGE_SCALE, y / _IMAGE_SCALE)
            if numpy.isnan(a) or numpy.isnan(b) or numpy.isnan(c):
                continue
            a, b, c, _ = _scale_probabilities([a, b, c, 0])
            array[y, x] = a, b, c

    ax = plt.gca()
    ax.imshow(array)

    output_filename = os.path.join(os.path.dirname(__file__), "scale_triangle.png")
    plt.imsave(output_filename, array)
    ax.invert_yaxis()
    plt.show()


def _scale_probabilities(probabilities: List[float]) -> List[float]:
    max_probability = max(probabilities)
    min_plotted_probability = 0

    probabilities = [(probability - min_plotted_probability) / (max_probability - min_plotted_probability) for
                     probability in probabilities]

    return [_clip(probability) for probability in probabilities]


def _clip(probability: float) -> float:
    if probability < 0:
        return 0
    if probability > 1:
        return 1
    return probability


if __name__ == "__main__":
    main()
