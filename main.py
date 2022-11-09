import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

from objects import Table, Actuator, Sensor
from setups import Setup2, Setup3_2, Setup3_4, Setup4_5, Setup6, Setup7

from constants import SAMPLE_RATE, CHANNEL_NAMES, CHIRP_CHANNEL_NAMES

from csv_to_df import csv_to_df
from data_viz_files.drawing import plot_legend_without_duplicates
from data_viz_files.visualise_data import compare_signals, specgram_with_lines
from data_processing.preprocessing import crop_data, filter_general, compress_chirp
from data_processing.detect_echoes import find_first_peak, get_hilbert_envelope, get_travel_times


def main():
    """CONFIG"""
    CROP = False
    TIME_START = 0.2  # s
    TIME_END = 5  # s
    FILTER = False
    BANDWIDTH = (200, 40000)  # Should be between ~200 Hz and 40 kHz
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

    """Compress chirp signals"""
    measurements_comp = compress_chirp(measurements_filt, custom_chirp=None)

    SHIFT_BY = int(TIME_START * SAMPLE_RATE)
    # Could consider using a windows also/instead
    for channel in measurements_filt:
        measurements_comp[channel] = np.roll(measurements_comp[channel],
                                             -SHIFT_BY)
        measurements_comp[channel][-SHIFT_BY:] = 0
    "Separate the channels into arrays of length 125 ms"
    measurements_comp_split = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES)
    for channel in measurements:
        measurements_comp_split[channel] = np.split(measurements_comp[channel], 40)

    """Generate Hilbert transforms"""
    measurements_hilb = get_hilbert_envelope(measurements_filt)
    measurements_comp_hilb = get_hilbert_envelope(measurements_comp)

    """Draw setup"""
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
    fig, axs = plt.subplots(nrows=3, ncols=3)
    compare_signals(fig, axs,
                    [measurements_comp_split['Sensor 1'][0],
                     measurements_comp_split['Sensor 2'][0],
                     measurements_comp_split['Sensor 3'][0]],
                    freq_max=BANDWIDTH[1] + 20000,
                    nfft=256)
    plt.show()

    """TODO: Make up for drift in signal generator"""
    # object_1 = SETUP.sensors[0]
    # object_2 = SETUP.sensors[0]
    # delays = np.array([])
    # for chirp in range(len(measurements_comp_split['Sensor 1'])):
    #     n = len(measurements_comp_split[object_1.name][0])
    #     corr = signal.correlate(measurements_comp_split[object_1.name][0], measurements_comp_split[object_2.name][chirp], mode='same') \
    #          / np.sqrt(signal.correlate(measurements_comp_split[object_2.name][chirp], measurements_comp_split[object_2.name][chirp], mode='same')[int(n / 2)]
    #          * signal.correlate(measurements_comp_split[object_1.name][0], measurements_comp_split[object_1.name][0], mode='same')[int(n / 2)])
    #     delay_arr = np.linspace(-0.5 * n / SAMPLE_RATE, 0.5 * n / SAMPLE_RATE, n)
    #     delay = delay_arr[np.argmax(corr)]
    #     delays = np.append(delays, delay)


    # measurements_comp_split['Sensor 1'] = np.roll(measurements_comp_split['Sensor 1'], (np.rint(delays * SAMPLE_RATE)).astype(int))



    fig, axs = plt.subplots(nrows=3, ncols=3)
    for chirp in range(len(measurements_comp_split['Sensor 1'])):
        compare_signals(fig, axs,
                        [measurements_comp_split['Sensor 1'][chirp],
                         measurements_comp_split['Sensor 2'][chirp],
                         measurements_comp_split['Sensor 3'][chirp]],
                        freq_max=BANDWIDTH[1] + 20000,
                        nfft=256)
    plt.show()


if __name__ == '__main__':
    main()
