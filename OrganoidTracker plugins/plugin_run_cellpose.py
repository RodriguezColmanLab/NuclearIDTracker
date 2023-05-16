import json
import os
from typing import NamedTuple, Optional, List

import numpy
import tifffile

from organoid_tracker.config import ConfigFile, config_type_int, config_type_float
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.gui import dialog, action
from organoid_tracker.gui.dialog import DefaultOption
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging import list_io
from organoid_tracker.util.run_script_creator import create_run_script


def get_menu_items(window: Window):
    return {
        "Tools//Process-Cell types//1. Segment nuclei using CellPose...": lambda: _create_segmentation_script(window)
    }


def get_commands():
    return {
        "cellpose_segment": _run_segmentation,
    }


class _ParsedConfig(NamedTuple):
    model_name: str
    main_channel: ImageChannel
    optional_channel: Optional[ImageChannel]
    diameter_px: float
    dataset_path: str
    output_folder: str


def _parse_config(config: ConfigFile, default_channel: ImageChannel) -> _ParsedConfig:
    """Parses the config, and adds any missing comments or default settings."""
    model_name = config.get_or_default("cellpose_model", "nuclei",
        comment="Name of the CellPose model, like \"nuclei\", \"cyto\", \"TN1\", etc.")
    main_channel = config.get_or_default("main_channel", str(default_channel.index_one), type=config_type_int,
                                         comment="Main channel for segmentation. The first channel is channel 1.")
    main_channel = ImageChannel(index_zero=main_channel - 1)
    optional_channel = config.get_or_default("optional_channel", "None",
                                             comment="Extra channel for segmentation, used in some CellPose models.")
    if optional_channel == "None":
        optional_channel = None
    else:
        optional_channel = ImageChannel(index_zero=int(optional_channel) - 1)
    diameter_px = config.get_or_default("diameter_px", "30", type=config_type_float,
                                        comment="Typical diameter of the object to segment.")
    dataset_path = config.get_or_default("dataset_to_segment", "Dataset" + list_io.FILES_LIST_EXTENSION,
                                         comment="Path to the dataset that you are using.")
    output_folder = config.get_or_default("output_folder", "masks_{experiment_number}_{experiment_name}/", comment="Path to the folder that will"
                                          "contain the masks.")
    return _ParsedConfig(
        model_name=model_name,
        main_channel=main_channel,
        optional_channel=optional_channel,
        diameter_px=diameter_px,
        dataset_path=dataset_path,
        output_folder=output_folder
    )


def _create_segmentation_script(window: Window):
    """Creates a segmentation script in the GUI. Called from get_menu_items()."""
    for experiment in window.get_active_experiments():
        experiment.images.resolution()  # Check whether a resolution is set

    if not dialog.prompt_confirmation("CellPose nucleus segmentation", "This will segment the nuclei using the"
        " CellPose software. Please make sure that you have selected the nucleus channel."
        "\n\nPlease select an output folder."):
        return

    # Creates an output folder
    output_folder = dialog.prompt_save_file("Output folder", [("*", "Folder")])
    if output_folder is None:
        return
    os.makedirs(output_folder)

    # Save dataset information
    data_structure = action.to_experiment_list_file_structure(window.get_gui_experiment().get_active_tabs())
    with open(os.path.join(output_folder, "Dataset" + list_io.FILES_LIST_EXTENSION), "w") as handle:
        json.dump(data_structure, handle)

    # Save run script
    create_run_script(output_folder, "cellpose_segment")

    # Save config file
    config = ConfigFile("cellpose_segment", folder_name=output_folder)
    _parse_config(config, window.display_settings.image_channel)
    config.save()

    # Done!
    if dialog.prompt_options("Run folder created", f"The configuration files were created successfully. Please"
                             f" run the cellpose_segment script from that directory:\n\n{output_folder}",
                             option_default=DefaultOption.OK, option_1="Open that directory") == 1:
        dialog.open_file(output_folder)


def _run_segmentation(args: List[str]):
    config = ConfigFile("cellpose_segment")
    parsed_config = _parse_config(config, ImageChannel(index_zero=0))
    config.save_and_exit_if_changed()

    import torch
    import cellpose.models
    model = cellpose.models.Cellpose(gpu=torch.cuda.is_available(), model_type=parsed_config.model_name)

    for i, experiment in enumerate(list_io.load_experiment_list_file(parsed_config.dataset_path)):
        do_3d = True
        image_size = experiment.images.image_loader().get_image_size_zyx()
        if image_size is not None and image_size[0] == 1:
            do_3d = False
        resolution = experiment.images.resolution()
        anisotropy = resolution.pixel_size_z_um / resolution.pixel_size_x_um
        output_folder = parsed_config.output_folder.format(experiment_number = i + 1, experiment_name = experiment.name.get_save_name())
        os.makedirs(output_folder, exist_ok=True)

        print(f"\nWorking on experiment \"{experiment.name}\"...")
        print("  ", end="")
        for time_point in experiment.images.time_points():
            print(time_point.time_point_number(), end=" ", flush=True)

            image = experiment.images.get_image_stack(time_point, parsed_config.main_channel)
            cellpose_channels = [0, 0]
            if image is None:
                print(f"\n  No image in time point {time_point.time_point_number()}\n  ", end="")
                continue
            if parsed_config.optional_channel is not None:
                # Convert to RGB color image, so that we can pass that to CellPose
                secondary_image = experiment.images.get_image_stack(time_point, parsed_config.optional_channel)
                color_image = numpy.zeros(image.shape + (3,), dtype=image.dtype)
                color_image[0] = image
                color_image[1] = secondary_image
                image = color_image
                cellpose_channels = [1, 2]

            masks, flows, styles, estimated_diameter = model.eval(image, diameter=parsed_config.diameter_px,
                channels=cellpose_channels, anisotropy=anisotropy, do_3D=do_3d)
            output_file = os.path.join(output_folder, f"masks_{time_point.time_point_number()}.tif")
            tifffile.imwrite(output_file, masks.astype(numpy.uint16), compression=tifffile.COMPRESSION.ADOBE_DEFLATE,
                             compressionargs={"level": 9})

