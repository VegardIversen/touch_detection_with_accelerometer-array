from calendar import c
from turtle import color
import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_viz_files.visualise_data import compare_signals, plot_data_vs_noiseavg, plot_data, plot_fft, plot_2fft
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
from data_processing.find_propagation_speed import find_propagation_speed, find_propagation_speed_plot
from data_processing.detect_echoes import find_first_peak, find_indices_of_peaks, get_hilbert_envelope, get_expected_reflections_pos
from data_processing.preprocessing import crop_data, crop_data_threshold, filter_general, filter_notches
from data_processing.transfer_function import transfer_function
import matplotlib.pyplot as plt
from data_processing.signal_separation import signal_sep
from csv_to_df import csv_to_df


def main():
    # Crop limits, in seconds
    TIME_START = 0
    TIME_END = 0.1

    FREQ_START = 20000
    FREQ_END = 40000

    CHIRP_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'chirp']

    CROP = True
    FILTER = True

    chirp_meas_df = csv_to_df(file_folder='div_files',
                              file_name='chirp_30000_30005_finger_hold_B2_setup4_5_v1',
                              channel_names=CHIRP_CHANNEL_NAMES,)

    if CROP:
        chirp_meas_df = crop_data(chirp_meas_df,
                                  time_start=TIME_START,
                                  time_end=TIME_END)

    if FILTER:
        chirp_meas_df = filter_general(chirp_meas_df,
                                       filtertype='highpass',
                                       cutoff_highpass=5000,
                                       order=32)

    chirp_ideal_df = csv_to_df(file_folder='div_files',
                               file_name='chirp_custom_fs_150000_tmax_1_20000-40000_method_linear',
                               channel_names=['chirp'])

    compare_signals(chirp_meas_df['channel 1'],
                    chirp_meas_df['chirp'],
                    sync_time=True)


if __name__ == '__main__':
    main()
