*Work in progress*

# NuclearIDTracker

In time lapse imaging, there is a severe limitation in the number of channels one can image simultanously. This prevents the simultanous imaging of multiple cell type reporters, especially when combined with signalling or metabolic reporters. For mouse small intestinal organoids, we noticed that nuclei of different types have a slightly different appearance: enterocyte nuclei are for example more round than stem cell nuclei. Here, we exploit those differences to predict cell type. Therefore, using only the nucleus reporter H2B-mCherry, we are able to predict for each cell at every time point the cell type, giving new insights into organoid development and disease.

The scripts here are based on [OrganoidTracker](https://jvzonlab.github.io/OrganoidTracker/). While we have only tested the scripts on mouse intestinal organoids, in principle they should work on any cell tracking experiment, provided you have segmentation data and cell types available, which indeed morphological differences between cell types.

![Example image](Documentation/example_image.png)
<p align="center"><i>Nuclei colored by cell types that were predicted using the nucleus shape</i></p>

## Intended workflow for training:
1. Obtain a set of organoids with fluorescent nuclei where you figure out the cell types, for example using antibody staining.
2. Load those images in OrganoidTracker, and annotate the nuclei with their cell type.
3. Run a segmentation algorithm (CellPose, ActiveUnetSegmentation, etc.) on the nucleus images.
4. Measure the features of the nuclei, done using a provided plugin.
5. Run the provided script to train a logistic regression network.

## Intended workflow for inference of cell type:
1. Take time lapses of fluorescent nuclei.
2. Do tracking of the cells.
3. Run the same segmentation algorithm (CellPose, ActiveUnetSegmentation, etc.) as used for training on the nucleus images.
4. Measure the features of the nuclei, also done using a provided plugin.
5. Use the logistic regression model to predict the cell types over time.

# Installation
1. Install OrganoidTracker as usual.
2. Copy the files and folders in the `OrganoidTracker plugins` folder into the plugin folder of OrganoidTracker. You can find this folder through the `File` menu of OrganoidTracker.

# Notices for code maintainers

- The `OrganoidTracker plugins` folder contains the plugins that are used in OrganoidTracker.
- The `Figure code` folder contains the code that is used to generate the figures in the paper. There's some code duplication with the plugins folder. Both contain a training script, make sure those have the same logic. Also make sure that the respective lib_data and lib_models files stay identical.
- The `Jupyter Notebooks` folder contains some Jupyter notebooks that were used for data inspection.
