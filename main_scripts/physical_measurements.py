# A funtion to combine any number of csv files as arguments, and combine them into a DataFrame with columns "Sensor i" for each csv column i.
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.data_processing.preprocessing import crop_data
from utils.data_processing.processing import interpolate_signal
from utils.global_constants import SAMPLE_RATE


def combine_measurements_into_dataframe(
    file_folder: str,
    *args,
):
    # Create a list of the arguments
    args_list = list(args)
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
    # Rename the columns
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

    align_with_trigger(measurements)

    plot_time_corrected_signals(measurements, plot=False)

    measurements = measurements.drop(
        columns=[
            "Actuator 123",
            "Actuator 456",
            "Actuator 78",
            "Sync Signal 123",
            "Sync Signal 456",
            "Sync Signal 78",
            "Sensor 8",
            "Sensor 9",
        ]
    )
    return measurements


def plot_time_corrected_signals(
    measurements,
    plot: bool,
):
    if plot:
        fig, axs = plt.subplots(
            nrows=measurements.shape[1] - 1,
            ncols=1,
            sharex=True,
        )
        time_axis = np.linspace(
            start=0,
            stop=1000 * measurements.shape[0] / SAMPLE_RATE,
            num=measurements.shape[0],
        )
        axs[0].plot(time_axis, measurements["Sync Signal 123"], label="Sync Signal 123")
        axs[1].plot(time_axis, measurements["Sync Signal 456"], label="Sync Signal 456")
        axs[2].plot(time_axis, measurements["Sync Signal 78"], label="Sync Signal 78")
        axs[3].plot(time_axis, measurements["Actuator 123"], label="Actuator 123")
        axs[4].plot(time_axis, measurements["Actuator 456"], label="Actuator 456")
        axs[5].plot(time_axis, measurements["Actuator 78"], label="Actuator 78")
        axs[6].plot(time_axis, measurements["Sensor 8"], label="Sensor 8")
        axs[7].plot(time_axis, measurements["Sensor 7"], label="Sensor 7")
        axs[8].plot(time_axis, measurements["Sensor 6"], label="Sensor 6")
        axs[9].plot(time_axis, measurements["Sensor 5"], label="Sensor 5")
        axs[10].plot(time_axis, measurements["Sensor 4"], label="Sensor 4")
        axs[11].plot(time_axis, measurements["Sensor 3"], label="Sensor 3")
        axs[12].plot(time_axis, measurements["Sensor 2"], label="Sensor 2")
        axs[13].plot(time_axis, measurements["Sensor 1"], label="Sensor 1")
        axs[13].set_xlabel("Time (ms)")
        plt.show()


def align_with_trigger(measurements):
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
