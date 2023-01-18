"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd

from utils.global_constants import SAMPLE_RATE

from utils.csv_to_df import csv_to_df
from utils.data_processing.preprocessing import filter_general
from utils.data_processing.detect_echoes import find_first_peak, find_indices_of_peaks, get_envelope


def find_propagation_speed_with_delay(df, ch1, ch2, height, distance=0.1, hilbert=True):
    if hilbert:
        peak_indices_ch1 = find_indices_of_peaks(df[ch1].to_numpy(), height, hilbert=hilbert, plot=False)
        peak_indices_ch2 = find_indices_of_peaks(df[ch2].to_numpy(), height, hilbert=hilbert, plot=False)
    else:
        peak_indices_ch1 = find_indices_of_peaks(df[ch1].to_numpy(), height, hilbert=hilbert, plot=False)
        peak_indices_ch2 = find_indices_of_peaks(df[ch2].to_numpy(), height, hilbert=hilbert, plot=False)
    diff = np.abs(peak_indices_ch1[0] - peak_indices_ch2[0])
    time = diff / SAMPLE_RATE
    speed = distance / time
    return speed


def find_propagation_speed_plot(chirp_df,
                                start_freq,
                                end_freq,
                                steps=1000):
    """Return an array of frequencies and an array of propagation speeds"""
    frequencies = np.array([])
    freq_speeds = np.array([])
    chirp_bps = np.array([])

    for freq in range(start_freq, end_freq, steps):
        chirp_bp = filter_general(signals=chirp_df,
                                  filtertype='bandpass',
                                  cutoff_lowpass=freq * 0.9,
                                  cutoff_highpass=freq * 1.1,
                                  order=4)
        freq_prop_speed = find_propagation_speed(df=chirp_bp,
                                                 ch1='Sensor 1',
                                                 ch2='Sensor 3')
        frequencies = np.append(frequencies, freq)
        freq_speeds = np.append(freq_speeds, freq_prop_speed)
        chirp_bps = np.append(chirp_bps, chirp_bp)

    return frequencies, freq_speeds, chirp_bps



if __name__ == '__main__':
    pass