import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_viz_files.visualise_data import compare_signals, plot_data_vs_noiseavg
from data_viz_files.drawing import draw_a_setup, draw_setup_2
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
from data_processing.find_propagation_speed import find_propagation_speed, find_propagation_speed_plot
from data_processing.detect_echoes import get_hilbert_envelope, get_travel_distances, get_travel_distances_firsts
from data_processing.preprocessing import crop_data, crop_data_threshold, filter_general, filter_notches
from data_processing.transfer_function import transfer_function
import matplotlib.pyplot as plt
from data_processing.signal_separation import signal_sep
from csv_to_df import csv_to_df
from objects import Table, Actuator, Sensor


SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def main():
    actuator = Actuator(np.array([1 / 2 * Table.LENGTH, 1 / 3 * Table.WIDTH]))
    sensor = Sensor(np.array([2 / 3 * Table.LENGTH, 2 / 3 * Table.WIDTH]))
    distances_first = get_travel_distances_firsts(actuator.coordinates, sensor.coordinates)
    distances_second = get_travel_distances(actuator, sensor)

    # draw_a_setup(np.array([actuator]), np.array([sensor]))

    # Crop limits, in seconds
    TIME_START = 0
    TIME_END = 5

    CHIRP_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'chirp']

    CROP = True
    FILTER = True

    chirp_meas_df = csv_to_df(file_folder='div_files',
                              file_name='chirp_test_fs_150000_t_max_2s_1000-40000hz_1vpp_1cyc_setup3_v2',
                              channel_names=CHIRP_CHANNEL_NAMES)

    chirp = chirp_meas_df['chirp']
    chirp = crop_data(chirp)

    if CROP:
        chirp_meas_df = crop_data(chirp_meas_df,
                                  time_start=TIME_START,
                                  time_end=TIME_END)

    if FILTER:
        chirp_meas_filt_df = filter_general(chirp_meas_df,
                                            filtertype='highpass',
                                            cutoff_highpass=1000,
                                            order=4)
    else:
        chirp_meas_filt_df = chirp_meas_df

    compressed_chirps = signal.correlate(chirp_meas_filt_df['channel 1'],
                                         chirp,
                                         mode='same')

    compressed_hilbert = get_hilbert_envelope(compressed_chirps)
    original_len = len(chirp_meas_df['channel 1'])

    # compressed_hilbert = crop_data(compressed_hilbert,
    #                                time_start=TIME_START,
    #                                time_end=TIME_END)

    chirp_meas_hilbert_df = get_hilbert_envelope(chirp_meas_df)

    # peak_indices = find_indices_of_peaks(chirp_meas_hilbert_df)

    compare_signals(chirp_meas_df['channel 1'],
                    chirp_meas_filt_df['channel 1'],
                    plot_1_name='Chirp signal',
                    plot_2_name='Measured signal',
                    sync_time=True)

    """Plot the chirp signal and the measured signal along with the hilbert envelopes"""
    time_axis = np.linspace(0, len(chirp_meas_df['channel 1']) / 150000, len(chirp_meas_df['channel 1']))
    time_axis_chirp = np.linspace(0, len(chirp) / 150000, len(chirp))
    ax1 = plt.subplot(311)
    plt.title('Chirp signal')
    plt.plot(time_axis_chirp, chirp / 150, label='Chirp signal')
    # plt.plot(time_axis, chirp_meas_hilbert_df['chirp' / 150, label='Chirp Hilbert envelope')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.grid()

    plt.subplot(312, sharex=ax1)
    plt.title('Measured signal')
    plt.plot(time_axis, chirp_meas_filt_df['channel 1'], label='Measured signal')
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
