# A funtion to combine any number of csv files as arguments, and combine them into a DataFrame with columns "Sensor i" for each csv column i.
import os
import numpy as np
import pandas as pd

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
        ["Actuator"]
        + [f"Sensor {i}" for i in range(1, 4)]
        + ["Sync Signal"]
        + ["Actuator"]
        + [f"Sensor {i}" for i in range(4, 7)]
        + ["Sync Signal"]
        + ["Actuator"]
        + [f"Sensor {i}" for i in range(7, 10)]
        + ["Sync Signal"]
    )
    measurements = measurements.drop(columns=["Actuator", "Sync Signal", "Sensor 9"])
    return measurements


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
