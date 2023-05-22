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


def test_on_real_simulations_ULA(
    noise: bool = False,
    crop: bool = False,
    number_of_sensors: int = 8,
    critical_frequency_Hz: int = 25000,
    filter_order: int = 8,
):
    # Import the simulation_data_formatted.csv to a Pandas DataFrame
    simulation_data = pd.read_csv(
        "simulation_data_formatted.csv",
        header=0,
    )

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
            critical_frequency=critical_frequency_Hz,
            snr_dB=50,
        )

    if crop:
        simulation_data = crop_data(
            simulation_data,
            time_start=0.0009,
            time_end=0.0009 + 0.00085,
            apply_window_function=True,
        )

    # Make a dataframe with 15 columns of 2 * simulation_data.size[0] rows of zeros
    # simulation_data = simulation_data.append(
    #     pd.DataFrame(
    #         np.zeros((2 * simulation_data.shape[0], simulation_data.shape[1])),
    #         columns=simulation_data.columns,
    #     ),
    #     ignore_index=True,
    # )

    if critical_frequency_Hz:
        simulation_data = filter_signal(
            simulation_data,
            filtertype="bandpass",
            critical_frequency=critical_frequency_Hz,
            plot_response=False,
            order=filter_order,
            sample_rate=SAMPLE_RATE,
            q=0.04,
        )

    # if crop:

    #     simulation_data = crop_data(
    #         simulation_data,
    #         time_start=0.0006,
    #         time_end=0.002,
    #         apply_window_function=False,
    #     )
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


def import_simulation_data_ULA():
    """Call to convert the simulation data file to a more convenient format."""
    # Set the path to your data file
    data_file = "az_on_plate_top_Teflon_25kHz_pulse_5cmfromedge.txt"

    # Read the file into a Pandas DataFrame
    simulation_data = pd.read_csv(
        data_file,
        delim_whitespace=True,
        comment="%",
    )

    # Drop x and z columns
    simulation_data.drop(
        columns=[simulation_data.columns[0], simulation_data.columns[2]],
        inplace=True,
    )

    # Transpose dataframe
    simulation_data = simulation_data.T

    # Save as a csv file named "simulation_data_formatted.csv"
    simulation_data.to_csv(
        "simulation_data_formatted.csv",
        header=False,
        index=False,
    )


def test_on_real_simulations_UCA(
    noise: bool = False,
    crop: bool = False,
    number_of_sensors: int = 8,
    critical_frequency_Hz: int = 23000,
    filter_order: int = 1,
):
    import_simulation_data_UCA()

    # Import the simulation_data_formatted.csv to a Pandas DataFrame
    simulation_data = pd.read_csv(
        "simulation_data_formatted.csv",
        header=0,
    )

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
            critical_frequency=critical_frequency_Hz,
            snr_dB=50,
        )

    if crop:
        simulation_data = crop_data(
            simulation_data,
            time_start=0.0008,
            time_end=0.00155,
            apply_window_function=False,
        )

    # Make a dataframe with 15 columns of 2 * simulation_data.size[0] rows of zeros
    # simulation_data = simulation_data.append(
    #     pd.DataFrame(
    #         np.zeros((2 * simulation_data.shape[0], simulation_data.shape[1])),
    #         columns=simulation_data.columns,
    #     ),
    #     ignore_index=True,
    # )

    if critical_frequency_Hz:
        simulation_data = filter_signal(
            simulation_data,
            filtertype="bandpass",
            critical_frequency=critical_frequency_Hz,
            plot_response=True,
            order=filter_order,
            sample_rate=SAMPLE_RATE,
            q=0.1,
        )

    # if crop:

    #     simulation_data = crop_data(
    #         simulation_data,
    #         time_start=0.0006,
    #         time_end=0.002,
    #         apply_window_function=False,
    #     )
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


def import_simulation_data_UCA():
    """Call to convert the simulation data file to a more convenient format."""
    # Set the path to your data file
    data_file = "az_on_plate_bottom_Teflon_25kHz_pulse_circular_receivers.txt"

    # Read the file into a Pandas DataFrame
    simulation_data = pd.read_csv(
        data_file,
        delim_whitespace=True,
        comment="%",
        header=None,
    )

    # Drop x and z columns
    simulation_data.drop(
        columns=[simulation_data.columns[0], simulation_data.columns[2]],
        inplace=True,
    )

    # Transpose dataframe
    simulation_data = simulation_data.T

    # Save as a csv file named "simulation_data_formatted.csv"
    simulation_data.to_csv(
        "simulation_data_formatted.csv",
        header=False,
        index=False,
    )
