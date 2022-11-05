from cgi import print_arguments
from turtle import color
import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from objects import Table, Actuator, Sensor
from setups import Setup2, Setup3_2, Setup3_4
from constants import SAMPLE_RATE, CHIRP_CHANNEL_NAMES

from csv_to_df import csv_to_df
from data_viz_files.visualise_data import compare_signals
from data_processing.preprocessing import crop_data, filter_general, compress_chirp
from data_processing.detect_echoes import find_first_peak, get_hilbert_envelope, get_travel_times
from data_processing.find_propagation_speed import find_propagation_speed_with_delay
from data_viz_files.drawing import plot_legend_without_duplicates


def main():
    """CONFIG"""
    CROP = True
    TIME_START = 0.75724  # s
    TIME_END = 1  # s
    FILTER = True
    BANDWIDTH = np.array([200, 2000 * 1.1])
    SETUP = Setup2()

    """Open file"""
    measurements = csv_to_df(file_folder='holdfinger_test_active_setup2_5',
                             file_name='active_plate_no_activity_sinus_2khz_10vpp_burstP_1s_cyccount_1_startphase0_150kzsample_5s')

    """Preprocessing"""
    if CROP:
        measurements = crop_data(measurements, TIME_START, TIME_END)
    if FILTER:
        measurements_filt = filter_general(measurements,
                                           filtertype='highpass',
                                           cutoff_highpass=BANDWIDTH[0],
                                           order=4)
    else:
        measurements_filt = measurements

    """Compress chirp signals"""
    t = np.linspace(0, 0.0005, int(0.0005 * SAMPLE_RATE))
    burst = np.sin(2 * np.pi * 2000 * t)
    measurements_comp = compress_chirp(measurements_filt, burst)

    """Generate Hilbert transforms"""
    measurements_hilb = get_hilbert_envelope(measurements_filt)
    measurements_comp_hilb = get_hilbert_envelope(measurements_comp)

    """Place setup objects"""
    SETUP.draw()
    actuator, sensors = SETUP.get_objects()

    """Calculate wave propagation speed"""
    prop_speed = SETUP.get_propagation_speed(measurements_filt['channel 1'],
                                             measurements_filt['channel 2'])
    print(f'Prop speed: {prop_speed}')

    """Calculate wave arrival times"""
    arrival_times_ch1, _ = get_travel_times(actuator[0],
                                            sensors[0],
                                            prop_speed,
                                            ms=True,
                                            print_info=False,
                                            relative_first_reflection=True)

    arrival_times_ch2, _ = get_travel_times(actuator[0],
                                            sensors[1],
                                            prop_speed,
                                            ms=True,
                                            print_info=False,
                                            relative_first_reflection=True)

    arrival_times_ch3, _ = get_travel_times(actuator[0],
                                            sensors[2],
                                            prop_speed,
                                            ms=True,
                                            print_info=False,
                                            relative_first_reflection=True)

    """Plot the measurements"""
    compare_signals(measurements_filt['channel 1'],
                    measurements_filt['channel 2'],
                    measurements_filt['channel 3'],
                    freq_max=10000,
                    nfft=128,
                    sync_time=True)

    """Plot the correlation between the chirp signal and the measured signal"""
    time_axis_corr = np.linspace(0,
                                 1000 * len(measurements_comp) / SAMPLE_RATE,
                                 (len(measurements_comp)))
    ax2 = plt.subplot(311)
    plt.title('Correlation between chirp and channel 1')
    plt.plot(time_axis_corr, measurements_comp['channel 1'], label='Correlation')
    plt.plot(time_axis_corr, measurements_comp_hilb['channel 1'], label='Hilbert envelope')
    plt.axvline(arrival_times_ch1[0], linestyle='--', color='r', label='Direct wave')
    [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times_ch1[1:5])]
    [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times_ch1[5:])]
    plt.xlabel('Time [ms]')
    plt.ylabel('Amplitude [V]')
    plot_legend_without_duplicates()
    plt.grid()

    """Plot the correlation between the chirp signal and the measured signal"""
    plt.subplot(312, sharex=ax2)
    plt.title('Correlation between chirp and channel 2')
    plt.plot(time_axis_corr, measurements_comp['channel 2'], label='Correlation')
    plt.plot(time_axis_corr, measurements_comp_hilb['channel 2'], label='Hilbert envelope')
    plt.axvline(arrival_times_ch2[0], linestyle='--', color='r', label='Direct wave')
    [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times_ch2[1:5])]
    [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times_ch2[5:])]
    plt.xlabel('Time [ms]')
    plt.ylabel('Amplitude [V]')
    plot_legend_without_duplicates()
    plt.grid()

    """Plot the correlation between the chirp signal and the measured signal"""
    plt.subplot(313, sharex=ax2)
    plt.title('Correlation between chirp and channel 3')
    plt.plot(time_axis_corr, measurements_comp['channel 3'], label='Correlation')
    plt.plot(time_axis_corr, measurements_comp_hilb['channel 3'], label='Hilbert envelope')
    plt.axvline(arrival_times_ch3[0], linestyle='--', color='r', label='Direct wave')
    [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times_ch3[1:5])]
    [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times_ch3[5:])]
    plt.xlabel('Time [ms]')
    plt.ylabel('Amplitude [V]')
    plot_legend_without_duplicates()
    plt.grid()

    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == '__main__':
    main()
