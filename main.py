
import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from objects import Table, Actuator, Sensor
from setups import Setup2, Setup3_2, Setup3_4, Setup6
from constants import SAMPLE_RATE, CHANNEL_NAMES, CHIRP_CHANNEL_NAMES

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
    TIME_END = TIME_START + 0.010  # s
    FILTER = True
    BANDWIDTH = np.array([200, 40000])
    SETUP = Setup2()

    """Open file"""
    measurements = csv_to_df(file_folder='holdfinger_test_active_setup2_5',
                             file_name='active_plate_no_activity_sinus_2khz_10vpp_burstP_1s_cyccount_1_startphase0_150kzsample_5s',
                             channel_names=CHANNEL_NAMES)

    """Preprocessing"""
    if CROP:
        measurements = crop_data(measurements, TIME_START, TIME_END)
    if FILTER:
        measurements_filt = filter_general(measurements,
                                           filtertype='highpass',
                                           cutoff_highpass=BANDWIDTH[0],
                                           # cutoff_lowpass=BANDWIDTH[1],
                                           order=4)
    else:
        measurements_filt = measurements

    """Compress chirp signals"""
    t = np.linspace(0, 0.0005, int(0.0005 * SAMPLE_RATE))
    burst = np.sin(2 * np.pi * 2000 * t)
    measurements_comp = compress_chirp(measurements_filt, custom_chirp=burst)

    """Generate Hilbert transforms"""
    measurements_hilb = get_hilbert_envelope(measurements_filt)
    measurements_comp_hilb = get_hilbert_envelope(measurements_comp)

    """Place setup objects"""
    SETUP.draw()
    actuator, sensors = SETUP.get_objects()

    """Calculate wave propagation speed"""
    prop_speed = SETUP.get_propagation_speed(measurements_comp['channel 1'],
                                             measurements_comp['channel 2'])
    prop_speed *= 1.3
    print(f'Prop speed: {prop_speed}')

    """Calculate wave arrival times"""
    arrival_times = np.array([])
    for sensor in sensors:
        time, _ = get_travel_times(actuator[0],
                                   sensor,
                                   prop_speed,
                                   ms=False,
                                   print_info=False,
                                   relative_first_reflection=False)
        arrival_times = np.append(arrival_times, time)
    """Reshape arrival_times to a 2D array with len(sensor) rows"""
    arrival_times = np.reshape(arrival_times, (len(sensors), len(arrival_times) // len(sensors)))

    """Plot the measurements"""
    compare_signals(measurements_comp['channel 1'],
                    measurements_comp['channel 2'],
                    measurements_comp['channel 3'],
                    freq_max=BANDWIDTH[1],
                    nfft=16,
                    sync_time=True)

    """Plot the spectrograms along with lines for expected reflections"""
    dynamic_range_db = 60
    vmin = 10 * np.log10(np.max(measurements_comp['channel 1'])) - dynamic_range_db
    for i, sensor in enumerate(sensors):
        plt.subplot(311 + i, sharex=plt.gca())
        plt.title('Correlation between chirp and channel ' + str(i + 1))
        plt.specgram(measurements_comp['channel ' + str(i + 1)], Fs=SAMPLE_RATE, NFFT=16, noverlap=(16 // 2), vmin=vmin)
        plt.axis(ymax=BANDWIDTH[1])
        plt.title('Spectrogram')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.colorbar()
        plt.axvline(arrival_times[i][0], linestyle='--', color='r', label='Direct wave')
        [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])]
        [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times[i][5:])]
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [V]')
        plot_legend_without_duplicates()
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    """Plot the correlation between the chirp signal and the measured signal"""
    # time_axis_corr = np.linspace(-1000 * len(measurements_comp) / SAMPLE_RATE,
    time_axis_corr = np.linspace(0,
                                 1000 * len(measurements_comp) / SAMPLE_RATE,
                                 (len(measurements_comp)))

    arrival_times *= 1000   # Convert to ms
    for i, sensor in enumerate(sensors):
        plt.subplot(311 + i, sharex=plt.gca())
        plt.title('Correlation between chirp and channel ' + str(i + 1))
        plt.plot(time_axis_corr, measurements_comp['channel ' + str(i + 1)], label='Correlation')
        plt.plot(time_axis_corr, measurements_comp_hilb['channel ' + str(i + 1)], label='Hilbert envelope')
        plt.axvline(arrival_times[i][0], linestyle='--', color='r', label='Direct wave')
        [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])]
        [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times[i][5:])]
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [V]')
        plot_legend_without_duplicates()
        plt.grid()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    arrival_times /= 1000   # Convert back to s


if __name__ == '__main__':
    main()
