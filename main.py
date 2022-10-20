from calendar import c
from sys import orig_argv
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
    TIME_START = 1.01734
    TIME_END = 5

    CHIRP_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'chirp']

    CROP = True
    FILTER = True

    chirp_meas_df = csv_to_df(file_folder='div_files',
                              file_name='chirp_test_fs_150000_t_max_2s_20000-40000hz_1vpp_1cyc_setup3_v1',
                              channel_names=CHIRP_CHANNEL_NAMES,)

    if FILTER:
        chirp_meas_df = filter_general(chirp_meas_df,
                                       filtertype='highpass',
                                       cutoff_lowpass=1000,
                                       order=4)

    compressed_chirps = signal.correlate(chirp_meas_df['channel 1'],
                                         chirp_meas_df['chirp'],
                                         mode='same')

    compressed_hilbert = get_hilbert_envelope(compressed_chirps)
    original_len = len(chirp_meas_df['channel 1'])

    if CROP:
        chirp_meas_df = crop_data(chirp_meas_df,
                                  time_start=TIME_START,
                                  time_end=TIME_END)

    compressed_hilbert = get_hilbert_envelope(compressed_chirps)

    chirp_meas_hilbert_df = get_hilbert_envelope(chirp_meas_df)


    # peak_indices = find_indices_of_peaks(chirp_meas_hilbert_df)

    # compare_signals(chirp_meas_df['channel 1'],
    #                 chirp_meas_df['chirp'],
    #                 plot_1_name='Measured signal',
    #                 plot_2_name='Chirp signal',
    #                 sync_time=True)

    """Plot the chirp signal and the measured signal along with the hilbert envelopes"""
    time_axis = np.linspace(0, len(chirp_meas_df['channel 1']) / 150000, len(chirp_meas_df['channel 1']))
    ax1 = plt.subplot(311)
    plt.title('Chirp signal')
    plt.plot(time_axis, chirp_meas_df['chirp'] / 150, label='Chirp signal')
    plt.plot(time_axis, chirp_meas_hilbert_df['chirp'] / 150, label='Chirp Hilbert envelope')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.grid()

    plt.subplot(312, sharex=ax1)
    plt.title('Measured signal')
    plt.plot(time_axis, chirp_meas_df['channel 1'], label='Measured signal')
    plt.plot(time_axis, chirp_meas_hilbert_df['channel 1'], label='Hilbert envelope')
    # plt.plot(time_axis[peak_indices], chirp_meas_hilbert_df['channel 1'][peak_indices], 'x', label='Peaks')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.grid()

    """Plot the correlation between the chirp signal and the measured signal"""
    time_axis_corr = np.linspace(-original_len / 150000,
                                 (original_len) / 150000,
                                 (original_len))
    plt.subplot(313, sharex=ax1)
    plt.title('Correlation between chirp and measured signal')
    plt.plot(time_axis_corr, compressed_chirps, label='Correlation')
    plt.plot(time_axis_corr, compressed_hilbert, label='Hilbert envelope')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.grid()

    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == '__main__':
    main()
