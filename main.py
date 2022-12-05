import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal

from constants import CHIRP_CHANNEL_NAMES, SAMPLE_RATE, INTERP_FACTOR
from csv_to_df import csv_to_df
from data_processing.detect_echoes import (get_hilbert,
                                           get_travel_times)
from data_processing.preprocessing import (compress_chirp,
                                           crop_data,
                                           filter_general)
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
from data_viz_files.drawing import draw_a_setup

def main():
    """CONFIG
    NOTE:   Set SETUP in setups.py"""
    FILE_FOLDER = 'div_files/setup3'
    FILE_NAME = 'sign_integ_test_chirp1_150k_5s_setup3_2_v1'
    SETUP = Setup3_2()
    CROP = False
    TIME_START = 1.5   # s
    TIME_END = 2.1  # s
    FILTER = False
    BANDWIDTH = (0, 60000)

    """Pyplot adjustments"""
    set_fontsizes()

    """Draw setup"""
    SETUP.draw()

    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements,
                                        INTERP_FACTOR * measurements['Actuator'].size)

    """Calculate wave propagation speed"""
    prop_speed = SETUP.get_propagation_speed(measurements)
    print(f'Propagation speed: {np.round(prop_speed, 2)} m/s.')

    """Calculate wave arrival times"""
    arrival_times = np.array([])
    for sensor in SETUP.sensors:
        time, _ = get_travel_times(SETUP.actuators[0],
                                   sensor,
                                   prop_speed,
                                   ms=False,
                                   print_info=True,
                                   relative_first_reflection=True)
        arrival_times = np.append(arrival_times, time)

    """Plot the signal in fullscreen"""
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(16, 9))
    compare_signals(fig, axs,
                    [measurements['Sensor 1'],
                     measurements['Sensor 2'],
                     measurements['Sensor 3']],
                     nfft = 1024)
    # plt.show()

    """Plot the correlation between sensor 1 and sensor 2"""
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(16, 9))
    compare_signals(fig, axs,
                    [get_hilbert(measurements['Sensor 1']),
                     get_hilbert(measurements['Sensor 2']),
                     get_hilbert(signal.correlate(measurements['Sensor 2'],
                                                  measurements['Sensor 1'],
                                                  mode='same'))],
                    nfft=1024)
    plt.show()


    """Compress chirp signals"""
    measurements = compress_chirp(measurements)


if __name__ == '__main__':
    main()
