import scipy.signal as signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_viz_files.visualise_data import compare_signals, plot_data_vs_noiseavg, plot_data, plot_fft, plot_2fft
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
from data_processing.find_propagation_speed import find_propagation_speed, find_propagation_speed_func
from data_processing.detect_echoes import find_first_peak, find_indices_of_peaks, get_hilbert_envelope
from data_processing.preprocessing import crop_data, crop_data_threshold, hp_or_lp_filter
from data_processing.transfer_function import transfer_function
from csv_to_df import csv_to_df


print('\n' + __file__ + '\n')

# Sample rate in Hz
SAMPLE_RATE = 150000

# Crop limits in seconds
TIME_START = 0
TIME_END = 5

FREQ_START = 20000
FREQ_END = 60000

CHIRP_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'chirp']


def main():


if __name__ == '__main__':
    main()
