"""Generate results for the paper.
This will be a collection of functions that will be called
and print and plot different results with a common configuration.
"""
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from global_constants import (CHIRP_CHANNEL_NAMES,
                              SAMPLE_RATE,
                              FIGSIZE_ONE_SIGNAL,
                              FIGSIZE_ONE_COLUMN)
from csv_to_df import csv_to_df
from data_processing.detect_echoes import (get_hilbert,
                                           get_travel_times)
from data_processing.preprocessing import (compress_chirp,
                                           crop_data,
                                           filter_general,
                                           window_signals)
from data_processing.processing import (avg_waveform,
                                        interpolate_waveform,
                                        normalize,
                                        var_waveform,
                                        correct_drift,
                                        to_dB)
from data_visualization.visualize_data import (compare_signals,
                                               wave_statistics,
                                               envelopes_with_lines,
                                               subplots_adjust)
from setups import (Setup,
                    Setup3_2_without_sensor2,
                    Setup7)


"""Setup 3_2"""


def results_setup3_2():
    """Run some general commands for all functions:
        - Choose file and open it
        - Channel selection
        - Choose setup and draw it
        - Interpolation
        - Generate results from functions
    """
    """Choose file"""
    FILE_FOLDER = 'prop_speed_files/setup3_2'
    FILE_NAME = 'prop_speed_chirp3_setup3_2_v1'
    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Delete sensor 2 as it doesn't have the required bandwidth"""
    measurements = measurements.drop(['Sensor 2'], axis='columns')
    CHIRP_CHANNEL_NAMES.remove('Sensor 2')

    """Choose setup"""
    SETUP = Setup3_2_without_sensor2()
    """Draw setup"""
    SETUP.draw()

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Set everything but the signal to zero"""
    signal_length_seconds = 2 + 0.05  # Length of chirp + time for sensor 3 to die down
    threshold = 0.001  # Determine empirically
    measurements, signal_start_seconds = window_signals(measurements,
                                                        signal_length_seconds,
                                                        threshold)

    """Run functions to generate results"""
    plot_time_signals_setup3_2(measurements,
                               signal_start_seconds,
                               signal_length_seconds)
    plot_spectrogram_signals_setup3_2(measurements,
                                      signal_start_seconds,
                                      signal_length_seconds)
    plot_fft_signals_setup_3_2(measurements,
                               signal_start_seconds,
                               signal_length_seconds)


def plot_time_signals_setup3_2(measurements: pd.DataFrame,
                               signal_start_seconds: float,
                               signal_length_seconds: float) -> None:
    """SETTINGS FOR PLOTTING"""
    plots_to_plot = ['time']

    """Plot the raw measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    subplots_adjust('time', rows=3, columns=1)

    """Compress chirps"""
    measurements = compress_chirp(measurements)

    """Plot the measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=True)

    """Use scientific notation if values are greater than 1000"""
    for ax in axs.flatten():
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    """Adjust for correct spacing in plot"""
    subplots_adjust('time', rows=3, columns=1)


def plot_spectrogram_signals_setup3_2(measurements: pd.DataFrame,
                                      signal_start_seconds: float,
                                      signal_length_seconds: float) -> None:
    """SETTINGS FOR PLOTTING"""
    NFFT = 1024
    plots_to_plot = ['spectrogram']

    """Plot the raw measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    nfft=NFFT,
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    subplots_adjust('spectrogram', rows=3, columns=1)

    """Compress chirps"""
    measurements = compress_chirp(measurements)

    """Plot the measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    nfft=NFFT,
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=True)

    """Adjust for correct spacing in plot"""
    subplots_adjust('spectrogram', rows=3, columns=1)


def plot_fft_signals_setup_3_2(measurements: pd.DataFrame,
                               signal_start_seconds: float,
                               signal_length_seconds: float) -> None:
    """SETTINGS FOR PLOTTING"""
    plots_to_plot = ['fft']

    """Plot the raw measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    subplots_adjust('fft', rows=3, columns=1)

    """Limit y axis"""
    for ax in axs.flatten():
        ax.set_ylim(bottom=-70, top=120)

    """Compress chirps"""
    measurements = compress_chirp(measurements)

    """Plot the measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=True)

    """Adjust for correct spacing in plot"""
    subplots_adjust('fft', rows=3, columns=1)

    """Limit y axis"""
    for ax in axs.flatten():
        ax.set_ylim(bottom=-50, top=230)


"""Setup 7"""


def results_setup7():
    """Run some general commands for all functions:
        - Choose file and open it
        - Channel selection
        - Choose setup and draw it
        - Interpolation
        - Generate results from functions
    """
    """CONFIG"""
    FILE_FOLDER = 'setup7'
    FILE_NAME = 'notouchThenHoldB2_20to40khz_125ms_10vpp_v1'
    SETUP = Setup7()
    TIME_START = 0.114  # s
    TIME_END = 0.246  # s


    SETUP.draw()

    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME,
                             channel_names=CHIRP_CHANNEL_NAMES)

    """Crop data"""
    # measurements = crop_data(measurements,
    #                          time_start=TIME_START,
    #                          time_end=TIME_END)

    plot_raw_time_signal_setup7(measurements)
    plt.show()


def plot_raw_time_signal_setup7(measurements: pd.DataFrame):
    """Plot raw signal"""
    time_axis = np.linspace(start=0,
                            stop=1000 * len(measurements) / SAMPLE_RATE,
                            num=len(measurements))
    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=FIGSIZE_ONE_SIGNAL)
    ax.set_title('Chirp from 20 khz to 40 kHz in 125 ms')
    ax.plot(time_axis, 1000 * measurements['Sensor 1'], label='Sensor 1')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (mV)')
    ax.legend()
    ax.grid()

    """Adjust for correct spacing in plot"""
    subplots_adjust('time', rows=1, columns=1)


if __name__ == '__main__':
    raise RuntimeError('This file is not meant to be run directly.')
