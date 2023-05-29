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
from utils.global_constants import FIGURES_SAVE_PATH, ORIGINAL_SAMPLE_RATE, SAMPLE_RATE
from utils.little_helpers import to_linear
from utils.plate_setups import Setup, Setup3, Setup_ULA


def inspect_touch():
    SETUP = Setup3()
    SETUP.draw()
    CRITICAL_FREQUENCY = 3e3

    FILE_FOLDER = "Plate_10mm/Setup3/touch"
    FILE_NAME = "nik_touch_v3"
    measurements = import_measurements(file_folder=FILE_FOLDER, file_name=FILE_NAME)
    correct_sensitivities(SETUP, measurements, CRITICAL_FREQUENCY)
    measurements = crop_to_signal(measurements)
    measurements = crop_data(
        measurements,
        time_start=0.0622,
        time_end=0.067,
        sample_rate=ORIGINAL_SAMPLE_RATE,
    )
    measurements = filter_signal(
        measurements,
        filtertype="highpass",
        critical_frequency=CRITICAL_FREQUENCY,
        order=2,
    )
    measurements = filter_signal(
        measurements,
        filtertype="lowpass",
        critical_frequency=50e3,
        order=2,
    )
    measurements = interpolate_signal(measurements)

    # plot_shift_testing(measurements)
    plot_raw_shifted_signals(measurements)
    # plot_A0_and_S0_large_shift(measurements)
    # plot_A0_and_S0_small_shift(measurements)
    # plot_raw_signals()

    # compare_to_ideal_signal(
    #     SETUP,
    #     measurements,
    #     attenuation_dBpm=15,
    #     critical_frequency=CRITICAL_FREQUENCY,
    # )


def plot_shift_testing(measurements):
    SHIFT_AMOUNT_S = 0.0
    measurements["Sensor 3 Shifts"] = np.roll(
        measurements["Sensor 3"], int(-SHIFT_AMOUNT_S * SAMPLE_RATE)
    )
    measurements["A0 Wave"] = measurements["Sensor 3 Shifts"] - measurements["Sensor 2"]
    measurements["S0 Wave"] = measurements["Sensor 3 Shifts"] + measurements["Sensor 2"]

    PLOTS_TO_PLOT = [
        "time",
        # "spectrogram",
        # "fft",
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
        [
            measurements["Sensor 1"],
            measurements["Sensor 2"],
            measurements["Sensor 3 Shifts"],
            measurements["A0 Wave"],
            measurements["S0 Wave"],
        ],
        plots_to_plot=PLOTS_TO_PLOT,
        freq_max=40000,
        dynamic_range_db=14,
        sharey=True,
    )
    plt.tight_layout(pad=0.5, h_pad=0)


def plot_raw_shifted_signals(measurements):
    SHIFT_AMOUNT_S = 0.0000944
    measurements["Sensor 3"] = np.roll(
        measurements["Sensor 3"], int(-SHIFT_AMOUNT_S * SAMPLE_RATE)
    )

    PLOTS_TO_PLOT = [
        "time",
        # "spectrogram",
        "fft",
    ]
    fig, axs = plt.subplots(
        nrows=3,
        ncols=len(PLOTS_TO_PLOT),
        squeeze=False,
        figsize=(10, 6),
    )
    compare_signals(
        fig,
        axs,
        [
            measurements["Sensor 1"],
            measurements["Sensor 2"],
            measurements["Sensor 3"],
        ],
        plots_to_plot=PLOTS_TO_PLOT,
        freq_max=50000,
        dynamic_range_db=14,
        sharey=True,
    )
    # Set the ylims on the second column
    for axs in axs[:, 1]:
        axs.set_ylim(-90, -19)
    plt.tight_layout(pad=0.5, h_pad=0)
    plt.subplots_adjust(wspace=0.25)
    plt.savefig(
        f"{FIGURES_SAVE_PATH}/touch_over_and_under.pdf",
        bbox_inches="tight",
    )

    measurements["A0 Wave"] = measurements["Sensor 3"] - measurements["Sensor 2"]
    measurements["S0 Wave"] = measurements["Sensor 3"] + measurements["Sensor 2"]

    PLOTS_TO_PLOT = [
        "time",
        # "spectrogram",
        # "fft",
    ]
    fig, axs = plt.subplots(
        nrows=2,
        ncols=len(PLOTS_TO_PLOT),
        squeeze=False,
        figsize=(6, 4),
    )
    compare_signals(
        fig,
        axs,
        [measurements["A0 Wave"], measurements["S0 Wave"]],
        plots_to_plot=PLOTS_TO_PLOT,
        freq_max=40000,
        dynamic_range_db=14,
        sharey=True,
    )
    fig.tight_layout(pad=0.5, h_pad=0)
    plt.savefig(
        f"{FIGURES_SAVE_PATH}/touch_A0_and_S0_{SHIFT_AMOUNT_S * 1e6:.0f}us.pdf",
        bbox_inches="tight",
    )

    fig, axs = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(6, 6),
    )
    for ax_i in range(0, axs.shape[0]):
        frequencies, time, Sxx = signal.spectrogram(
            measurements[f"Sensor {ax_i + 1}"],
            SAMPLE_RATE,
            nperseg=2**7,
            nfft=2**7,
        )
        im = axs[ax_i].pcolormesh(
            time,
            frequencies / 1000,  # Modified line
            10 * np.log10(Sxx),
            vmin=-160,
            cmap="viridis",
        )
        axs[ax_i].set_ylabel("Frequency (kHz)")  # Modified line
        axs[ax_i].set_xlabel("Time (s)")
        axs[ax_i].set_ylim([0, 50])
        fig.colorbar(
            im,
            ax=axs[ax_i],
            label="Power Spectral \n Density (dB)",
        )
    fig.tight_layout(pad=0.5, h_pad=0)
    plt.savefig(
        f"{FIGURES_SAVE_PATH}/touch_spectrogram_{SHIFT_AMOUNT_S * 1e6:.0f}us.pdf",
        bbox_inches="tight",
    )


def plot_A0_and_S0_large_shift(measurements):
    # SHIFT_AMOUNT_S = 0.00129 # v1
    SHIFT_AMOUNT_S = 0.00147  # v2
    measurements["Sensor 3 Large Shifts"] = np.roll(
        measurements["Sensor 3"], int(-SHIFT_AMOUNT_S * SAMPLE_RATE)
    )
    measurements["A0 Wave"] = (
        measurements["Sensor 3 Large Shifts"] - measurements["Sensor 2"]
    )
    measurements["S0 Wave"] = (
        measurements["Sensor 3 Large Shifts"] + measurements["Sensor 2"]
    )

    PLOTS_TO_PLOT = [
        "time",
        # "spectrogram",
        # "fft",
    ]
    fig, axs = plt.subplots(
        nrows=2,
        ncols=len(PLOTS_TO_PLOT),
        squeeze=False,
        figsize=(6, 4),
    )
    compare_signals(
        fig,
        axs,
        [measurements["A0 Wave"], measurements["S0 Wave"]],
        plots_to_plot=PLOTS_TO_PLOT,
        freq_max=40000,
        dynamic_range_db=14,
        sharey=True,
    )
    plt.tight_layout(pad=0.5, h_pad=0)
    plt.savefig(
        f"{FIGURES_SAVE_PATH}/touch_A0_and_S0_{SHIFT_AMOUNT_S * 10e3:.0f}ms.pdf",
        bbox_inches="tight",
    )


def plot_A0_and_S0_small_shift(measurements):
    SHIFT_AMOUNT_S = 0.0000944
    measurements["Sensor 3 Small Shift"] = np.roll(
        measurements["Sensor 3"], int(-SHIFT_AMOUNT_S * SAMPLE_RATE)
    )
    measurements["A0 Wave"] = (
        measurements["Sensor 3 Small Shift"] - measurements["Sensor 2"]
    )
    measurements["S0 Wave"] = (
        measurements["Sensor 3 Small Shift"] + measurements["Sensor 2"]
    )

    PLOTS_TO_PLOT = [
        "time",
        # "spectrogram",
        # "fft",
    ]
    fig, axs = plt.subplots(
        nrows=2,
        ncols=len(PLOTS_TO_PLOT),
        squeeze=False,
        figsize=(6, 4),
    )
    compare_signals(
        fig,
        axs,
        [measurements["A0 Wave"], measurements["S0 Wave"]],
        plots_to_plot=PLOTS_TO_PLOT,
        freq_max=40000,
        dynamic_range_db=14,
        sharey=True,
    )
    plt.tight_layout(pad=0.5, h_pad=0)
    plt.savefig(
        f"{FIGURES_SAVE_PATH}/touch_A0_and_S0_{SHIFT_AMOUNT_S * 1e6:.0f}us.pdf",
        bbox_inches="tight",
    )


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

    # CRITICAL_FREQUENCY = 250
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


def correct_sensitivities(
    setup,
    measurements,
    critical_frequency=None,
):
    for sensor in setup.sensors:
        # If sensor name is "Sensor 1", "Sensor 4", "Sensor 7", etc.
        if int(sensor.name[-1]) % 3 == 1:
            # Measurements at 100 mV/ms^-2, should be 0.293 mV/ms^-2
            correction_factor = -0.293 / 100
            measurements[sensor.name] = measurements[sensor.name] * correction_factor
        # If sensor name is "Sensor 2", "Sensor 5", "Sensor 8", etc.
        elif int(sensor.name[-1]) % 3 == 2:
            # Measurements at 100 mV/ms^-2, should be 0.489 mV/ms^-2
            # Handle som cases of bandpass/highpass above 30 kHz
            if critical_frequency == 30e3:
                # Handle the case of 30 kHz, where there is a 3 dB boost
                correction_factor = 0.489 / 100 * to_linear(-3)
                measurements[sensor.name] = (
                    measurements[sensor.name] * correction_factor
                )
            elif critical_frequency == 35e3:
                # Handle the case of 35 kHz, where there is a 4.5 dB boost
                correction_factor = 0.489 / 100 * to_linear(-4.5)
                measurements[sensor.name] = (
                    measurements[sensor.name] * correction_factor
                )
            elif critical_frequency == 40e3:
                # Handle the case of 40 kHz, where there is a 6.3 dB boost
                correction_factor = 0.489 / 100 * to_linear(-6.3)
                measurements[sensor.name] = (
                    measurements[sensor.name] * correction_factor
                )
            else:
                correction_factor = 0.300 / 100
                measurements[sensor.name] = (
                    measurements[sensor.name] * correction_factor
                )
        # If sensor name is "Sensor 3", "Sensor 6", "Sensor 9", etc.
        elif int(sensor.name[-1]) % 3 == 0:
            # Measurements at 100 mV/ms^-2, should be 0.293 mV/ms^-2
            correction_factor = 0.305 / 100  # Actual sensitivity
            measurements[sensor.name] = measurements[sensor.name] * correction_factor


if __name__ == "__main__":
    raise RuntimeError("This script is not meant to run standalone.")
