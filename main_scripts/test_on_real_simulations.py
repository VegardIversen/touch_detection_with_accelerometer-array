from os import path
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from main_scripts.generate_ideal_signal import add_noise

from main_scripts.generate_signals_for_matlab import generate_signals_for_matlab
from utils.data_processing.detect_echoes import get_envelopes
from utils.data_processing.preprocessing import crop_data, filter_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals
from utils.global_constants import SAMPLE_RATE


def prepare_simulation_data(
    array_type: str,
    noise: bool = False,
    crop: bool = False,
    number_of_sensors: int = 8,
    critical_frequency_Hz: int = 0,
    filter_order: int = 1,
    crop_start: float = 0.0,
    crop_end: float = 0.002,
):
    # If the simulation data is not already generated, generate it
    if not path.exists(
        f"Measurements/Plate_10mm/COMSOL/simulation_data_formatted_{array_type}.csv"
    ):
        import_simulation_data(array_type=array_type)

    # Import the simulation_data_formatted.csv to a Pandas DataFrame
    simulation_data = pd.read_csv(
        f"Measurements/Plate_10mm/COMSOL/simulation_data_formatted_{array_type}.csv",
        header=0,
    )

    if array_type == "ULA":
        simulation_data = simulation_data.iloc[:, 4 : (4 + number_of_sensors)]

    simulation_data = interpolate_signal(simulation_data)

    # Rename each column to name to start with "Sensor [column index + 1]"
    simulation_data.rename(
        columns={
            column_name: f"Sensor {column_index + 1}"
            for column_index, column_name in enumerate(simulation_data.columns)
        },
        inplace=True,
    )

    if noise:
        # Add noise to the signal
        simulation_data = add_noise(
            simulation_data,
            snr_dB=50,
        )

    if crop:
        simulation_data = crop_data(
            simulation_data,
            # time_start=0.0008,
            time_start=crop_start,
            time_end=crop_end,
            # time_end=5,
            apply_window_function=False,
        )

    fig, axs = plt.subplots(number_of_sensors, 1, squeeze=False)
    envelopes = get_envelopes(simulation_data)
    compare_signals(
        fig,
        axs,
        measurements=[simulation_data[sensor] for sensor in simulation_data.columns],
        plots_to_plot=["time"],
        sharey=True,
    )

    return simulation_data


def import_simulation_data(array_type: str):
    """Call to convert the simulation data file to a more convenient format."""
    if array_type == "ULA":
        data_file = "Measurements/Plate_10mm/COMSOL/az_on_plate_top_Teflon_25kHz_pulse_5cmfromedge.txt"
        simulation_data = pd.read_csv(
            data_file,
            delim_whitespace=True,
            comment="%",
        )
    elif array_type == "UCA":
        data_file = "Measurements/Plate_10mm/COMSOL/az_on_plate_bottom_Teflon_25kHz_pulse_circular_receivers.txt"
        simulation_data = pd.read_csv(
            data_file,
            delim_whitespace=True,
            comment="%",
            header=None,
        )
    else:
        raise ValueError("Invalid array type")
    # Read the file into a Pandas DataFrame, but a bit different for ULA and UCA, don't know why
    # Drop x and z columns
    simulation_data.drop(
        columns=[simulation_data.columns[0], simulation_data.columns[2]],
        inplace=True,
    )
    simulation_data = simulation_data.T
    simulation_data.to_csv(
        "/home/sniklad/GitHub/touch_detection_with_accelerometer-array/"
        f"Measurements/Plate_10mm/COMSOL/simulation_data_formatted_{array_type}.csv",
        header=False,
        index=False,
    )
