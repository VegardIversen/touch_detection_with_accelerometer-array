import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

from objects import Table, Actuator, Sensor
from setups import Setup2, Setup3_2, Setup3_4, Setup4_5, Setup6, Setup7

from constants import SAMPLE_RATE, CHANNEL_NAMES, CHIRP_CHANNEL_NAMES

from csv_to_df import csv_to_df
from data_viz_files.visualise_data import compare_signals
from data_viz_files.drawing import plot_legend_without_duplicates
from data_processing.preprocessing import crop_data, filter_general, compress_chirp
from data_processing.detect_echoes import find_first_peak, get_hilbert_envelope, get_travel_times


def main():
    """CONFIG"""
    CROP = False
    TIME_START = 0.6  # s
    TIME_END = 5  # s
    FILTER = False
    BANDWIDTH = np.array([200, 40000])  # Should be between ~200 Hz and 40 kHz
    SETUP = Setup7()

    """Open file"""
    measurements = csv_to_df(file_folder='setup7',
                             file_name='notouchThenHoldB2_20to40khz_125ms_10vpp_v1',
                             channel_names=CHIRP_CHANNEL_NAMES)

    """Preprocessing"""
    if CROP:
        measurements = crop_data(measurements, TIME_START, TIME_END)
    if FILTER:
        measurements_filt = filter_general(measurements,
                                           filtertype='bandpass',
                                           cutoff_highpass=BANDWIDTH[0],
                                           cutoff_lowpass=BANDWIDTH[1],
                                           order=4)
    else:
        measurements_filt = measurements

    # SHIFT_BY = int(TIME_START * SAMPLE_RATE)
    # for channel in measurements_filt:
    #     measurements_filt[channel] = np.roll(measurements_filt[channel],
    #                                          -SHIFT_BY)
    #     measurements_filt[channel][-SHIFT_BY:] = 0
    """Compress chirp signals"""
    measurements_comp = compress_chirp(measurements_filt, custom_chirp=None)
    "Separate the channels into arrays of length 18750 samples"
    measurements_comp_split = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES)
    for channel in measurements:
        measurements_comp_split[channel] = np.split(measurements_comp[channel], 40)

    """Generate Hilbert transforms"""
    measurements_hilb = get_hilbert_envelope(measurements_filt)
    measurements_comp_hilb = get_hilbert_envelope(measurements_comp)

    """Place setup objects"""
    SETUP.draw()

    """Calculate wave propagation speed"""
    prop_speed = SETUP.get_propagation_speed(measurements_comp)
    print(f'Propagation speed: {prop_speed}')

    """Calculate wave arrival times"""
    arrival_times = np.array([])
    for sensor in SETUP.sensors:
        time, _ = get_travel_times(SETUP.actuators[0],
                                   sensor,
                                   prop_speed,
                                   ms=False,
                                   print_info=False,
                                   relative_first_reflection=False)
        arrival_times = np.append(arrival_times, time)
    """Reshape arrival_times to a 2D array with len(sensor) rows"""
    arrival_times = np.reshape(arrival_times, (len(SETUP.sensors), len(arrival_times) // len(SETUP.sensors)))

    """Plot the measurements"""
    compare_signals([measurements_comp['Sensor 1'],
                     measurements_comp['Sensor 2'],
                     measurements_comp['Actuator']],
                    freq_max=BANDWIDTH[1] + 20000,
                    nfft=256)

    """Plot the spectrograms along with lines for expected reflections"""
    arrival_times += 2.5
    for i, sensor in enumerate(SETUP.sensors):
        plt.subplot(311 + i, sharex=plt.gca())
        plt.title(f'Correlation between chirp and {sensor}')
        spec = plt.specgram(measurements_comp[sensor.name], Fs=SAMPLE_RATE, NFFT=16, noverlap=(16 // 2))
        plt.clim(10 * np.log10(np.max(spec[0])) - 60, 10 * np.log10(np.max(spec[0])))
        plt.axis(ymax=BANDWIDTH[1] + 20000)
        plt.title('Spectrogram')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.colorbar()
        plt.axvline(arrival_times[i][0], linestyle='--', color='r', label='Direct wave')
        [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])]
        [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times[i][5:])]
        plt.xlabel('Time [ms]')
        plt.ylabel('Frequency [Hz]')
        plot_legend_without_duplicates()
    arrival_times -= 2.5
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    """Plot the correlation between the chirp signal and the measured signal"""
    # time_axis_corr = np.linspace(-1000 * len(measurements_comp) / SAMPLE_RATE,
    time_axis_corr = np.linspace(-1000 * len(measurements_comp) / SAMPLE_RATE,
                                 1000 * len(measurements_comp) / SAMPLE_RATE,
                                 (len(measurements_comp)))

    arrival_times *= 1000   # Convert to ms
    for i, sensor in enumerate(SETUP.sensors):
        plt.subplot(311 + i, sharex=plt.gca())
        plt.title(f'Correlation between {SETUP.actuators[0]} and {sensor}')
        plt.plot(time_axis_corr, measurements_comp[sensor.name], label='Correlation')
        plt.plot(time_axis_corr, measurements_comp_hilb[sensor.name], label='Hilbert envelope')
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
