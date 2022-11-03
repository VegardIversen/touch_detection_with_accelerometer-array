from cgi import print_arguments
from turtle import color
import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_viz_files.visualise_data import compare_signals, plot_data_vs_noiseavg
from data_viz_files.drawing import draw_a_setup, draw_setup_2, draw_setup_3_2, plot_legend_without_duplicates, draw_setup_ideal
from data_processing.noise import adaptive_filter_RLS, adaptive_filter_NLMS, noise_reduce_signal
from data_processing.find_propagation_speed import find_propagation_speed, find_propagation_speed_plot, find_propagation_speed_with_delay
from data_processing.detect_echoes import find_first_peak, get_hilbert_envelope, get_travel_distances, get_travel_distances_firsts
from data_processing.preprocessing import crop_data, crop_data_threshold, filter_general, filter_notches
from data_processing.transfer_function import transfer_function
from data_processing.signal_separation import signal_sep
from csv_to_df import csv_to_df
from objects import Table, Actuator, Sensor


def main():
    SAMPLE_RATE = 150000
    CHIRP_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'chirp']

    """Crop limits, in seconds"""
    CROP = False
    TIME_START = 0
    TIME_END = 5

    FILTER = True

    chirp_meas_df = csv_to_df(file_folder='div_files\\setup3',
                              file_name='sign_integ_test_chirp2_150k_5s_setup3_4_v1',
                              channel_names=CHIRP_CHANNEL_NAMES)

    if CROP:
        chirp_meas_df = crop_data(chirp_meas_df,
                                  time_start=TIME_START,
                                  time_end=TIME_END)

    if FILTER:
        chirp_meas_filt_df = filter_general(chirp_meas_df,
                                            filtertype='bandpass',
                                            cutoff_highpass=20000,
                                            cutoff_lowpass=40000,
                                            order=4)
    else:
        chirp_meas_filt_df = chirp_meas_df

    compressed_chirp_ch1 = signal.correlate(chirp_meas_filt_df['channel 1'],
                                            chirp_meas_filt_df['chirp'],
                                            mode='same')

    compressed_chirp_ch2 = signal.correlate(chirp_meas_filt_df['channel 2'],
                                            chirp_meas_filt_df['chirp'],
                                            mode='same')

    """Generate Hilbert transforms"""
    chirp_meas_hilbert_df = get_hilbert_envelope(chirp_meas_filt_df)
    compressed_hilbert_ch1 = get_hilbert_envelope(compressed_chirp_ch1)
    compressed_hilbert_ch2 = get_hilbert_envelope(compressed_chirp_ch2)

    original_len = len(chirp_meas_df['channel 1'])


    """Place sensors, actuator and find their wave arrival times"""
    actuator, sensor_1, sensor_2, sensor_3 = draw_setup_3_2()
    distances_ch1 = get_travel_distances(actuator, sensor_1, print_distances=True)
    distances_ch2 = get_travel_distances(actuator, sensor_2)
    distances_ch3 = get_travel_distances(actuator, sensor_3)

    # peak_indices_ch1 = find_first_peak(chirp_meas_filt_df['channel 1'], 0.0001)
    # peak_indices_ch2 = find_first_peak(chirp_meas_filt_df['channel 2'], 0.0001)
    # diff_time = np.abs(peak_indices_ch1 - peak_indices_ch2)


    # prop_speed = np.abs(sensor_2.x - sensor_1.x) / diff_time * SAMPLE_RATE
    prop_speed = find_propagation_speed(chirp_meas_filt_df,
                                        'channel 1',
                                        'channel 2',
                                        distance=np.abs(sensor_2.x-sensor_1.x))
    prop_speed_peak = find_propagation_speed_with_delay(chirp_meas_filt_df,
                                                        'channel 1',
                                                        'channel 2',
                                                        height=0.0005,
                                                        distance=np.abs(sensor_2.x-sensor_1.x))
    print(f'Prop speed peak: {prop_speed_peak}')

    arrival_times_ch1 = np.array([])
    arrival_times_ch2 = np.array([])
    arrival_times_ch3 = np.array([])
    for d_1, d_2, d_3 in zip(distances_ch1, distances_ch2, distances_ch3):
        arrival_times_ch1 = np.append(arrival_times_ch1, d_1 / prop_speed)
        arrival_times_ch2 = np.append(arrival_times_ch2, d_2 / prop_speed)
        arrival_times_ch3 = np.append(arrival_times_ch3, d_3 / prop_speed)

    # Print the first peak of the chirp
    print(f'First peak of the chirp: {chirp_meas_filt_df["chirp"].idxmax() * 1 / SAMPLE_RATE}')

    """Plot the measured signals along with their hilbert envelopes"""
    time_axis = np.linspace(0, len(chirp_meas_df['channel 1']) / SAMPLE_RATE, len(chirp_meas_df['channel 1']))
    ax1 = plt.subplot(221)
    plt.title('Channel 1')
    plt.plot(time_axis, chirp_meas_filt_df['channel 1'], label='Measured signal')
    plt.plot(time_axis, chirp_meas_hilbert_df['channel 1'], label='Hilbert envelope')
    plt.axvline(arrival_times_ch1[0], linestyle='--', color='r')
    [plt.axvline(line, linestyle='--', color='g') for line in (arrival_times_ch1[1:5])]
    [plt.axvline(line, linestyle='--', color='purple') for line in (arrival_times_ch1[5:])]
    # plt.plot(time_axis[peak_indices], chirp_meas_hilbert_df['channel 1'][peak_indices], 'x', label='Peaks')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.grid()

    plt.subplot(223, sharex=ax1)
    plt.title('Channel 2')
    plt.plot(time_axis, chirp_meas_filt_df['channel 2'], label='Measured signal')
    plt.plot(time_axis, chirp_meas_hilbert_df['channel 2'], label='Hilbert envelope')
    plt.axvline(arrival_times_ch2[0], linestyle='--', color='r')
    [plt.axvline(line, linestyle='--', color='g') for line in (arrival_times_ch2[1:5])]
    [plt.axvline(line, linestyle='--', color='purple') for line in (arrival_times_ch2[5:])]
    # plt.plot(time_axis[peak_indices], chirp_meas_hilbert_df['channel 1'][peak_indices], 'x', label='Peaks')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.grid()

    """Plot the correlation between the chirp signal and the measured signal"""
    time_axis_corr = np.linspace(-original_len / SAMPLE_RATE,
                                 (original_len) / SAMPLE_RATE,
                                 (original_len))
    ax2 = plt.subplot(222)
    plt.title('Correlation between chirp and channel 1')
    plt.plot(time_axis_corr, compressed_chirp_ch1, label='Correlation')
    plt.plot(time_axis_corr, compressed_hilbert_ch1, label='Hilbert envelope')
    plt.axvline(arrival_times_ch2[0] + 0.001, linestyle='--', color='r', label='Direct wave')
    [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times_ch2[1:5] + 0.001)]
    [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times_ch2[5:] + 0.001)]
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plot_legend_without_duplicates()
    plt.grid()

    """Plot the correlation between the chirp signal and the measured signal"""
    time_axis_corr = np.linspace(-original_len / SAMPLE_RATE,
                                 (original_len) / SAMPLE_RATE,
                                 (original_len))
    plt.subplot(224, sharex=ax2)
    plt.title('Correlation between chirp and channel 2')
    plt.plot(time_axis_corr, compressed_chirp_ch2, label='Correlation')
    plt.plot(time_axis_corr, compressed_hilbert_ch2, label='Hilbert envelope')
    plt.axvline(arrival_times_ch2[0] + 0.0006, linestyle='--', color='r', label='Direct wave')
    [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times_ch2[1:5] + 0.0006)]
    [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times_ch2[5:] + 0.0006)]
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plot_legend_without_duplicates()
    plt.grid()

    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == '__main__':
    main()
