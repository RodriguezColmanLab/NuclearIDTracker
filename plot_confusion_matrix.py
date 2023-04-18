import numpy
from matplotlib import pyplot as plt

cell_types = ['stem', 'enterocyte', 'Paneth', 'other secretory']
confusion_matrix = numpy.array([[ 699, 17, 20, 7],
                                [ 84 , 485, 37, 14],
                                [ 71 , 25, 244, 6],
                                [ 8  , 54, 41, 12]
                                ]
                               )

fraction_correct = numpy.diagonal(confusion_matrix).sum() / confusion_matrix.sum()

scaled_matrix = confusion_matrix.astype(dtype=numpy.float64)
for i in range(scaled_matrix.shape[1]):
    scaled_matrix[i] /= scaled_matrix[i].sum()

ax = plt.gca()
ax.imshow(scaled_matrix, cmap="Oranges")
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        color = "white" if scaled_matrix[i, j] > 0.7 else "black"
        ax.text(j, i, f"{scaled_matrix[i, j] * 100:.0f}%", horizontalalignment="center", verticalalignment="center",
                color=color)

ax.set_xticks(list(range(len(cell_types))))
ax.set_xticklabels(cell_types, rotation=-45, ha='left')
ax.set_yticks(list(range(len(cell_types))))
ax.set_yticklabels(cell_types)
ax.set_title(f"Accuracy: {fraction_correct * 100:.1f}%")
plt.show()
