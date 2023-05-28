import os
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from main_scripts.generate_ideal_signal import compare_to_ideal_signal
from utils.csv_to_df import import_measurements
from utils.data_processing.detect_echoes import get_envelopes
from utils.data_processing.preprocessing import crop_data, crop_to_signal, filter_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals
from utils.global_constants import ORIGINAL_SAMPLE_RATE, SAMPLE_RATE
from utils.plate_setups import Setup, Setup3, Setup_ULA


def inspect_touch():
    SETUP = Setup3()
    SETUP.draw()

    FILE_FOLDER = "Plate_10mm/Setup3/touch"
    FILE_NAME = "nik_touch_v3"
    measurements = import_measurements(file_folder=FILE_FOLDER, file_name=FILE_NAME)
    correct_sensitivities(SETUP, measurements)
    measurements = crop_to_signal(measurements)
    measurements = crop_data(
        measurements,
        time_start=0.061,
        time_end=0.068,
        sample_rate=ORIGINAL_SAMPLE_RATE,
    )
    CRITICAL_FREQUENCY = 30e3
    measurements = filter_signal(
        measurements,
        filtertype="highpass",
        critical_frequency=CRITICAL_FREQUENCY,
        order=1,
    )
    measurements = interpolate_signal(measurements)

    measurements["Sensor 3"] = np.roll(
        measurements["Sensor 3"], int(-0.00009 * SAMPLE_RATE)
    )

    measurements["A0 Wave"] = measurements["Sensor 2"] - measurements["Sensor 3"]
    measurements["S0 Wave"] = measurements["Sensor 2"] + measurements["Sensor 3"]

    PLOTS_TO_PLOT = [
        "time",
        # "spectrogram",
        "fft",
    ]
    fig, axs = plt.subplots(
        nrows=5,
        ncols=len(PLOTS_TO_PLOT),
        squeeze=False,
        figsize=(6, 10),
    )
    compare_signals(
        fig,
        axs,
        [measurements[sensor.name] for sensor in SETUP.sensors]
        + [measurements["A0 Wave"]]
        + [measurements["S0 Wave"]],
        plots_to_plot=PLOTS_TO_PLOT,
        freq_max=40000,
        dynamic_range_db=14,
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [get_envelopes(measurements[sensor.name]) for sensor in SETUP.sensors]
        + [get_envelopes(measurements["A0 Wave"])]
        + [get_envelopes(measurements["S0 Wave"])],
        plots_to_plot=PLOTS_TO_PLOT,
        freq_max=40000,
        dynamic_range_db=14,
        sharey=True,
    )

    fig.tight_layout(pad=0.1, h_pad=0.5)

    # compare_to_ideal_signal(
    #     SETUP,
    #     measurements,
    #     attenuation_dBpm=15,
    #     critical_frequency=CRITICAL_FREQUENCY,
    # )


def inspect_swipe():
    SETUP = Setup_ULA(
        array_start_coordinates=np.array([0.05, 0.10]),
        array_spacing_m=0.01,
        actuator_coordinates=np.array([0.47, 0.40]),
        number_of_sensors=3,
    )
    SETUP.draw()

    FILE_FOLDER = "Plate_10mm/Setup5/swipes/"
    FILE_NAME = "x47y40y30_sensors678_v1"
    measurements = import_measurements_for_swipes(FILE_FOLDER, FILE_NAME, SETUP)

    CRITICAL_FREQUENCY = 250
    # measurements = filter_signal(
    #     measurements,
    #     filtertype="highpass",
    #     critical_frequency=CRITICAL_FREQUENCY,
    #     order=1,
    # )
    measurements = interpolate_signal(measurements)

    PLOTS_TO_PLOT = [
        "time",
        # "spectrogram",
        "fft",
    ]
    fig, axs = plt.subplots(
        nrows=3,
        ncols=len(PLOTS_TO_PLOT),
        squeeze=False,
        figsize=(8, 10),
    )
    compare_signals(
        fig,
        axs,
        [measurements[sensor.name] for sensor in SETUP.sensors],
        plots_to_plot=PLOTS_TO_PLOT,
        freq_max=40000,
        dynamic_range_db=14,
        sharey=True,
    )
    fig.tight_layout(pad=0.5, h_pad=0, w_pad=0.5)


def import_measurements_for_swipes(
    file_folder: str,
    file_name: str,
    setup: Setup,
):
    # Read the csv file into a dataframe
    file_path = os.path.join("Measurements", file_folder, f"{file_name}.csv")
    measurements = pd.read_csv(
        filepath_or_buffer=file_path,
        header=None,
    )
    measurements.columns = (
        ["Actuator"] + [f"Sensor {i}" for i in range(1, 4)] + ["Sync Signal"]
    )
    measurements = measurements.drop(columns=["Actuator", "Sync Signal"])

    # measurements = crop_to_signal(measurements, padding_percent=0.3)

    # Correct for the wrong sensitivity in the amplifier.

    correct_sensitivities(setup, measurements)

    # Get rid of 50 Hz and potential DC offset
    measurements = filter_signal(
        signals=measurements,
        critical_frequency=50,
        filtertype="highpass",
        order=2,
        plot_response=True,
        sample_rate=ORIGINAL_SAMPLE_RATE,
    )
    # Get rid of frequencies above 50 kHz due to sensor responses
    measurements = filter_signal(
        signals=measurements,
        critical_frequency=50000,
        filtertype="lowpass",
        order=2,
        plot_response=False,
        sample_rate=ORIGINAL_SAMPLE_RATE,
    )

    measurements = interpolate_signal(measurements)

    return measurements


def correct_sensitivities(setup, measurements):
    for sensor in setup.sensors:
        # If sensor name is "Sensor 1", "Sensor 4", "Sensor 7", etc.
        if int(sensor.name[-1]) % 3 == 1:
            # Measurements at 100 mV/ms^-2, should be 0.293 mV/ms^-2
            measurements[sensor.name] = measurements[sensor.name] * 0.293 / 100
        # If sensor name is "Sensor 2", "Sensor 5", "Sensor 8", etc.
        elif int(sensor.name[-1]) % 3 == 2:
            # Measurements at 100 mV/ms^-2, should be 0.489 mV/ms^-2
            measurements[sensor.name] = measurements[sensor.name] * 0.489 / 100
        # If sensor name is "Sensor 3", "Sensor 6", "Sensor 9", etc.
        elif int(sensor.name[-1]) % 3 == 0:
            # Measurements at 100 mV/ms^-2, should be 0.293 mV/ms^-2
            measurements[sensor.name] = measurements[sensor.name] * 0.293 / 100


if __name__ == "__main__":
    raise RuntimeError("This script is not meant to run standalone.")
