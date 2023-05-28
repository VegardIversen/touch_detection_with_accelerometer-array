# A funtion to combine any number of csv files as arguments, and combine them into a DataFrame with columns "Sensor i" for each csv column i.
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from main_scripts.generate_ideal_signal import compare_to_ideal_signal

from utils.data_processing.preprocessing import crop_data, filter_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals
from utils.global_constants import SAMPLE_RATE
from utils.plate_setups import Setup


def combine_measurements_into_dataframe(
    file_folder: str,
    file_names: list[str],
    setup: Setup,
    group_velocity_mps: float,
    center_frequency: float,
    filter_order: int,
    filter_q_value: float,
):
    # Create a list of the arguments
    args_list = file_names
    # Create an empty list to store the dataframes
    dataframes = []
    # Loop through the arguments
    for arg in args_list:
        # Read the csv file into a dataframe
        file_path = os.path.join("Measurements", file_folder, f"{arg}.csv")
        measurements = pd.read_csv(
            filepath_or_buffer=file_path,
            header=None,
        )
        # Append the dataframe to the list of dataframes
        dataframes.append(measurements)
    # Concatenate the dataframes into a single dataframe
    measurements = pd.concat(
        dataframes,
        axis=1,
    )
    measurements.columns = (
        ["Actuator 123"]
        + [f"Sensor {i}" for i in range(1, 4)]
        + ["Sync Signal 123"]
        + ["Actuator 456"]
        + [f"Sensor {i}" for i in range(4, 7)]
        + ["Sync Signal 456"]
        + ["Actuator 78"]
        + [f"Sensor {i}" for i in range(7, 10)]
        + ["Sync Signal 78"]
    )

    measurements = crop_data(
        signals=measurements,
        time_start=0,
        time_end=0.001,
    )
    measurements = interpolate_signal(measurements)

    # Get rid of 50 Hz and potential DC offset
    measurements = filter_signal(
        signals=measurements,
        critical_frequency=250,
        filtertype="highpass",
        order=2,
        plot_response=False,
        sample_rate=SAMPLE_RATE,
    )
    # Get rid of frequencies above 50 kHz due to sensor responses
    measurements = filter_signal(
        signals=measurements,
        critical_frequency=50000,
        filtertype="lowpass",
        order=2,
        plot_response=False,
        sample_rate=SAMPLE_RATE,
    )

    align_transmitted_signal(measurements)

    measurements = measurements.drop(
        columns=[
            "Sync Signal 123",
            "Sync Signal 456",
            "Sync Signal 78",
            "Sensor 9",
        ]
    )

    plot_time_corrected_signals(measurements, plot=True)

    measurements["Actuator"] = measurements["Actuator 123"]

    measurements = measurements.drop(
        columns=[
            "Actuator 123",
            "Actuator 456",
            "Actuator 78",
            # "Sensor 7",
            "Sensor 8",
        ]
    )

    # compare_to_ideal_signal(
    #     setup=setup,
    #     measurements=measurements,
    #     attenuation_dBpm=17,
    #     group_velocity_mps=group_velocity_mps,
    #     signal_model="gaussian",
    #     critical_frequency=center_frequency,
    #     filter_order=filter_order,
    #     filter_q_value=filter_q_value,
    # )

    return measurements


def plot_time_corrected_signals(
    measurements,
    plot: bool,
):
    if plot:
        fig, axs = plt.subplots(
            nrows=measurements.shape[1],
            ncols=2,
            figsize=(10, 12),
        )
        compare_signals(
            fig,
            axs,
            [
                measurements["Actuator 123"],
                measurements["Actuator 456"],
                measurements["Actuator 78"],
            ]
            + [measurements[f"Sensor {i}"] for i in range(1, 9)],
            plots_to_plot=[
                "time",
                # "spectrogram",
                "fft",
            ],
            nfft=2**6,
        )
        plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)


def align_transmitted_signal(measurements: pd.DataFrame) -> None:
    # Find the delay of the sync signals using the maximum of the "Actuator" signal
    delay_456 = np.argmax(measurements["Actuator 456"]) - np.argmax(
        measurements["Actuator 123"]
    )
    delay_78 = np.argmax(measurements["Actuator 78"]) - np.argmax(
        measurements["Actuator 123"]
    )
    # Shift sensors 123, 456, and 78 by the delay of their sync signals
    measurements["Sensor 4"] = np.roll(measurements["Sensor 4"], -delay_456)
    measurements["Sensor 5"] = np.roll(measurements["Sensor 5"], -delay_456)
    measurements["Sensor 6"] = np.roll(measurements["Sensor 6"], -delay_456)
    measurements["Sensor 7"] = np.roll(measurements["Sensor 7"], -delay_78)
    measurements["Sensor 8"] = np.roll(measurements["Sensor 8"], -delay_78)
    # Shift the sync signals by the delay of their sync signals
    measurements["Sync Signal 456"] = np.roll(
        measurements["Sync Signal 456"], -delay_456
    )
    measurements["Sync Signal 78"] = np.roll(measurements["Sync Signal 78"], -delay_78)
    # Shift the actuators by the delay of their sync signals
    measurements["Actuator 456"] = np.roll(measurements["Actuator 456"], -delay_456)
    measurements["Actuator 78"] = np.roll(measurements["Actuator 78"], -delay_78)


def measure_phase_velocity(
    measurements: pd.DataFrame,
    crop_length=None,
):
    SENSOR_SPACING = 0.01

    # Crop the signal to the specified length (if specified)
    if crop_length is not None:
        measurements = measurements.iloc[:crop_length]

    # Calculate the time delay between adjacent sensors
    delays = []
    for i in range(1, len(measurements.columns) - 1):
        corr = np.correlate(
            measurements[f"Sensor {i}"],
            measurements[f"Sensor {i + 1}"],
            mode="same",
        )
        delays.append(
            (np.argmax(corr) - len(measurements[measurements.columns[i - 1]]) + 1)
            / SAMPLE_RATE
        )

    # Calculate the phase difference between adjacent sensors
    phase_diff = []
    for i in range(len(delays)):
        phase_diff.append(delays[i] * 2 * np.pi * SAMPLE_RATE / SENSOR_SPACING)

    # Calculate the phase velocity
    phase_velocity = np.mean(phase_diff)
    print(f"Phase velocity: {phase_velocity} m/s")

    return phase_velocity
