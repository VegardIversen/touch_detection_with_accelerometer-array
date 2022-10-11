import scipy.signal as signal
from pandas import DataFrame as df
import numpy as np
import matplotlib.pyplot as plt
from data_viz_files.visualise_data import compare_signals, plot_data_vs_noiseavg, plot_data
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
from data_processing.find_propagation_speed import find_propagation_speed, find_propagation_speed_func
from data_processing.detect_echoes import find_indices_of_peaks, get_hilbert_envelope
from data_processing.preprocessing import crop_data, crop_data_threshold, hp_or_lp_filter
from data_processing.transfer_function import transfer_function
from csv_to_df import csv_to_df


print('\n' + __file__ + '\n')

# Sample rate in Hz
SAMPLE_RATE = 150000

# Crop limits in seconds
TIME_START = 0
TIME_END = 5

FREQ_START = 2000
FREQ_END = 60000

CHIRP_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'chirp']


def main():
    chirp_df = csv_to_df(file_folder='div_files',
                         file_name='chirp_test_fs_150000_t_max_2s_20000-60000hz_1vpp_1cyc_setup3_v1',
                         channel_names=CHIRP_CHANNEL_NAMES)

    # compare_signals(chirp_df['channel 1'],
    #                 chirp_df['channel 3'],
    #                 sample_rate=SAMPLE_RATE,
    #                 time_start=TIME_START,
    #                 time_end=TIME_END)

    frequencies, freq_speeds = find_propagation_speed_func(chirp_df,
                                                           start_freq=FREQ_START,
                                                           end_freq=FREQ_END,
                                                           steps=10)

    # Plot freq_speeds as a function of frequencies
    plt.subplot(2, 1, 1)
    plt.plot(frequencies, freq_speeds)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Propagation speed (m/s)')
    plt.grid()
    # time_axis = np.linspace(TIME_START, TIME_END, len(chirp_df['channel 1']))
    # plt.subplot(2, 1, 2)
    # plt.plot(time_axis, chirp_df['channel 1'].values, time_axis, chirp_df['channel 3'].values)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude (V)')
    # plt.grid()
    plt.show()

    # find_indices_of_peaks(signal_df_filtered, plot=True)

    # compare_signals(signal_df['channel 1'], signal_df['channel 1'], sample_rate=SAMPLE_RATE, time_start=TIME_START, time_end=TIME_END)
    # find_propagation_speed(chirp_df, sr=SAMPLE_RATE)


if __name__ == '__main__':
    main()
