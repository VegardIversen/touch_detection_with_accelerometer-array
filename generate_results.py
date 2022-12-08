"""Generate results for the paper.
This will be a collection of functions that will be called
and print and plot different results with a common configuration.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal

from constants import CHIRP_CHANNEL_NAMES, SAMPLE_RATE, INTERPOLATION_FACTOR
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
                                        correct_drift)
from data_viz_files.visualise_data import (compare_signals,
                                           wave_statistics,
                                           set_fontsizes,
                                           envelopes_with_lines)
from setups import Setup


def plot_time_signals(measurements: pd.DataFrame,
                      signal_start_seconds: float,
                      signal_length_seconds: float,
                      FIGSIZE: tuple) -> None:
    """SETTINGS FOR PLOTTING"""
    plots_to_plot = ['time']
    subplots_adjust = {'left': 0.125, 'right': 0.965,
                       'top': 0.955, 'bottom': 0.07,
                       'hspace': 0.28, 'wspace': 0.2}

    """Plot the raw measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    plt.subplots_adjust(left=subplots_adjust['left'],
                        right=subplots_adjust['right'],
                        top=subplots_adjust['top'],
                        bottom=subplots_adjust['bottom'],
                        hspace=subplots_adjust['hspace'],
                        wspace=subplots_adjust['wspace'])

    """Compress chirps"""
    measurements = compress_chirp(measurements)

    """Plot the measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE,
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
    plt.subplots_adjust(left=subplots_adjust['left'],
                        right=subplots_adjust['right'],
                        top=subplots_adjust['top'],
                        bottom=subplots_adjust['bottom'],
                        hspace=subplots_adjust['hspace'],
                        wspace=subplots_adjust['wspace'])


def plot_spectrogram_signals(measurements: pd.DataFrame,
                             signal_start_seconds: float,
                             signal_length_seconds: float,
                             FIGSIZE: tuple) -> None:
    """SETTINGS FOR PLOTTING"""
    NFFT = 1024
    plots_to_plot = ['spectrogram']
    subplots_adjust = {'left': 0.125, 'right': 1.05,
                       'top': 0.955, 'bottom': 0.07,
                       'hspace': 0.28, 'wspace': 0.2}

    """Plot the raw measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    nfft=NFFT,
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    plt.subplots_adjust(left=subplots_adjust['left'],
                        right=subplots_adjust['right'],
                        top=subplots_adjust['top'],
                        bottom=subplots_adjust['bottom'],
                        hspace=subplots_adjust['hspace'],
                        wspace=subplots_adjust['wspace'])

    """Compress chirps"""
    measurements = compress_chirp(measurements)

    """Plot the measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    nfft=NFFT,
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=True)

    """Adjust for correct spacing in plot"""
    plt.subplots_adjust(left=subplots_adjust['left'],
                        right=subplots_adjust['right'],
                        top=subplots_adjust['top'],
                        bottom=subplots_adjust['bottom'],
                        hspace=subplots_adjust['hspace'],
                        wspace=subplots_adjust['wspace'])


def plot_fft_signals(measurements: pd.DataFrame,
                     signal_start_seconds: float,
                     signal_length_seconds: float,
                     FIGSIZE: tuple) -> None:
    """SETTINGS FOR PLOTTING"""
    plots_to_plot = ['fft']
    subplots_adjust = {'left': 0.125, 'right': 0.95,
                       'top': 0.955, 'bottom': 0.07,
                       'hspace': 0.28, 'wspace': 0.2}

    """Plot the raw measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    plt.subplots_adjust(left=subplots_adjust['left'],
                        right=subplots_adjust['right'],
                        top=subplots_adjust['top'],
                        bottom=subplots_adjust['bottom'],
                        hspace=subplots_adjust['hspace'],
                        wspace=subplots_adjust['wspace'])

    """Limit y axis"""
    for ax in axs.flatten():
        ax.set_ylim(bottom=-70, top=120)

    """Compress chirps"""
    measurements = compress_chirp(measurements)

    """Plot the measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=True)

    """Adjust for correct spacing in plot"""
    plt.subplots_adjust(left=subplots_adjust['left'],
                        right=subplots_adjust['right'],
                        top=subplots_adjust['top'],
                        bottom=subplots_adjust['bottom'],
                        hspace=subplots_adjust['hspace'],
                        wspace=subplots_adjust['wspace'])

    """Limit y axis"""
    for ax in axs.flatten():
        ax.set_ylim(bottom=-50, top=230)


if __name__ == '__main__':
    raise RuntimeError('This file is not meant to be run directly.')
