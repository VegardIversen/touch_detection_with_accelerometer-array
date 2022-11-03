import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd

from constants import *

from csv_to_df import csv_to_df
from data_processing.preprocessing import filter_general
from data_processing.detect_echoes import find_first_peak, find_indices_of_peaks, get_hilbert_envelope


def find_propagation_speed(df1, df2, distance):
    """Use the cross correlation between the two channels
    to find the propagation speed. Based on:
    https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
    """
    n = len(df1)
    # Convert to df if np.array
    if type(df1) == np.ndarray:
        df1 = pd.DataFrame(df1)
    if type(df2) == np.ndarray:
        df2 = pd.DataFrame(df2)

    corr = signal.correlate(df1, df2, mode='same') \
           / np.sqrt(signal.correlate(df2, df2, mode='same')[int(n / 2)]
           * signal.correlate(df1, df1, mode='same')[int(n / 2)])

    delay_arr = np.linspace(-0.5 * n / SAMPLE_RATE, 0.5 * n / SAMPLE_RATE, n)
    delay = delay_arr[np.argmax(corr)]
    # print('\n' + f'The delay between {df1} and {ch2} is {str(np.round(1000 * np.abs(delay), decimals=4))} ms.')

    propagation_speed = np.round(np.abs(distance / delay), decimals=2)
    # print("\n" + f"Propagation speed is {propagation_speed} m/SAMPLE_RATE \n")

    return propagation_speed


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
        chirp_bp = filter_general(sig=chirp_df,
                                  filtertype='bandpass',
                                  cutoff_lowpass=freq * 0.9,
                                  cutoff_highpass=freq * 1.1,
                                  order=4)
        freq_prop_speed = find_propagation_speed(df=chirp_bp,
                                                 ch1='channel 1',
                                                 ch2='channel 3')
        frequencies = np.append(frequencies, freq)
        freq_speeds = np.append(freq_speeds, freq_prop_speed)
        chirp_bps = np.append(chirp_bps, chirp_bp)

    return frequencies, freq_speeds, chirp_bps



if __name__ == '__main__':
    pass