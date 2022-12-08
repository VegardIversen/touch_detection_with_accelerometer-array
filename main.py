import timeit

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
                                           zero_all_but_signal)
from data_processing.processing import (avg_waveform,
                                        interpolate_waveform,
                                        normalize,
                                        var_waveform,
                                        correct_drift)
from data_viz_files.visualise_data import (compare_signals,
                                           wave_statistics,
                                           set_fontsizes,
                                           envelopes_with_lines)
from objects import Table, Actuator, Sensor
from setups import Setup3_2


def main():
    """CONFIG"""
    FILE_FOLDER = 'prop_speed_files/setup3_2'
    FILE_NAME = 'prop_speed_chirp3_setup3_2_v1'
    SETUP = Setup3_2()

    """Pyplot adjustments"""
    set_fontsizes()

    """Draw setup"""
    SETUP.draw()

    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Delete sensor 2 as it doesn't have the required bandwidth"""
    measurements = measurements.drop(['Sensor 2'], axis='columns')
    CHIRP_CHANNEL_NAMES.remove('Sensor 2')

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Set everything but the signal to zero"""
    signal_length_seconds = 2 + 0.05  # Length of chirp + time for sensor 3 to die down
    threshold = 0.001  # Determine empirically
    measurements, signal_start_seconds = window_signals(measurements,
                                                        signal_length_seconds,
                                                        threshold)

    """Compress chirps"""
    measurements = compress_chirp(measurements)
    COMPRESSED = True

    """Plot the measurements"""
    plots_to_plot = ['time']
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=(8, 9),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    nfft=1024,
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=COMPRESSED)
    plt.show()


if __name__ == '__main__':
    main()
