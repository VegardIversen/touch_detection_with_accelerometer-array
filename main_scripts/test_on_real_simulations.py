from os import path
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from main_scripts.generate_ideal_signal import add_noise

from main_scripts.generate_signals_for_matlab import generate_signals_for_matlab
from utils.data_processing.detect_echoes import get_envelopes
from utils.data_processing.preprocessing import filter_signal
from utils.data_visualization.visualize_data import compare_signals
from utils.global_constants import SAMPLE_RATE


def test_on_real_simulations(
    noise: bool = True,
):
    # Import the simulation_data_formatted.csv to a Pandas DataFrame
    simulation_data = pd.read_csv(
        "simulation_data_formatted.csv",
        header=0,
    )

    # Rename each column to name to start with "Sensor [column index + 1]"
    simulation_data.rename(
        columns={
            column_name: f"Sensor {column_index + 1}"
            for column_index, column_name in enumerate(simulation_data.columns)
        },
        inplace=True,
    )

    CRITICAL_FREQUENCY = 25000

    if noise:
        # Add noise to the signal
        simulation_data = add_noise(
            simulation_data,
            critical_frequency=CRITICAL_FREQUENCY,
            snr_dB=50,
        )

    # Bandpass filter at 25 kHz
    simulation_data = filter_signal(
        simulation_data,
        filtertype="bandpass",
        critical_frequency=CRITICAL_FREQUENCY,
        plot_response=False,
        order=2,
        sample_rate=SAMPLE_RATE,
    )

    time_axis = np.linspace(
        start=0,
        stop=1e6 * simulation_data.shape[0] / SAMPLE_RATE,
        num=simulation_data.shape[0],
    )
    simulation_data["Time [s]"] = time_axis

    fig, axs = plt.subplots(3, 2, squeeze=False)

    envelopes = get_envelopes(simulation_data)
    compare_signals(
        fig,
        axs,
        measurements=[
            simulation_data["y=0.01"],
            simulation_data["y=0.02"],
            simulation_data["y=0.03"],
        ],
        plots_to_plot=["time", "fft"],
    )


def import_simulation_data():
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
