import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

from setups import Setup7

from constants import SAMPLE_RATE, CHIRP_CHANNEL_NAMES

from csv_to_df import csv_to_df
from data_viz_files.visualise_data import compare_signals
from data_processing.preprocessing import (crop_data,
                                           filter_general,
                                           compress_chirp)
from data_processing.detect_echoes import (get_travel_times)


def main():
    """CONFIG"""
    FILE_FOLDER = 'setup7'
    FILE_NAME = 'notouchThenHoldB2_20to40khz_125ms_10vpp_v1'
    SETUP = Setup7()
    CROP = False
    TIME_START = 0.05  # s
    TIME_END = 5  # s
    FILTER = False
    BANDWIDTH = (15000, 40000)

    def normalize(arr, t_min, t_max):
        """Normalize array to be between t_min and t_max"""
        arr = arr - np.min(arr)
        if np.max(arr) == 0:
            raise ValueError('Array is all zeros')
        arr = arr / np.max(arr)
        arr = arr * (t_max - t_min)
        arr = arr + t_min
        return arr

    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME,
                             channel_names=CHIRP_CHANNEL_NAMES)

    """Preprocessing"""
    if CROP:
        measurements = crop_data(measurements, TIME_START, TIME_END)
    if FILTER:
        measurements_filt = filter_general(measurements,
                                           filtertype='bandpass',
                                           cutoff_highpass=BANDWIDTH[0],
                                           order=4)
    else:
        measurements_filt = measurements

    """Compress chirp signals"""
    measurements_comp = compress_chirp(measurements_filt, custom_chirp=None)

    """Shift to make up for the drift in the signal generator"""
    SHIFT_BY = int(TIME_START * SAMPLE_RATE)
    for chan in measurements_filt:
        measurements_comp[chan] = np.roll(measurements_comp[chan],
                                          -SHIFT_BY)
        measurements_comp[chan][-SHIFT_BY:] = 0
    "Separate the channels into arrays of length 125 ms"
    measurements_split = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES)
    for chan in measurements:
        measurements_split[chan] = np.split(measurements_comp[chan],
                                            indices_or_sections=40)


    """Generate Hilbert transforms"""
    # measurements_hilb = get_hilbert_envelope(measurements_filt)
    # measurements_comp_hilb = get_hilbert_envelope(measurements_comp)

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
    arrival_times = np.reshape(arrival_times,
                               (len(SETUP.sensors),
                                len(arrival_times) // len(SETUP.sensors)))

    """Plot the shifted measurements"""
    # fig, axs = plt.subplots(nrows=3, ncols=3)
    # compare_signals(fig, axs,
    #                 [measurements_comp['Sensor 1'],
    #                  measurements_comp['Sensor 2'],
    #                  measurements_comp['Sensor 3']],
    #                 freq_max=BANDWIDTH[1] + 20000,
    #                 nfft=256)
    # plt.show()

    """Make up for drift in singal generator"""
    for i, chan in enumerate(measurements_comp):
        for chirp in range(len(measurements_split['Sensor 1'])):
            n = len(measurements_split[chan][0])
            corr = signal.correlate(measurements_split[chan][0],
                                    measurements_split[chan][chirp],
                                    mode='same')
            delay_arr = np.linspace(start=-0.5 * n,
                                    stop=0.5 * n,
                                    num=n)
            delay = delay_arr[np.argmax(corr)]
            SHIFT_BY = (np.rint(delay)).astype(int)
            measurements_split.at[chirp, chan] = np.roll(measurements_split.at[chirp, chan],
                                                         SHIFT_BY)
            measurements_split.at[chirp, chan] = normalize(measurements_split.at[chirp, chan], 0, 1)
            measurements_split.at[chirp, chan] = signal.detrend(measurements_split.at[chirp, chan])

    """Find the average waveforms"""
    avg_waveforms_1 = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES,
                                   data=np.empty((1, 4), np.ndarray))
    for chan in avg_waveforms_1:
        avg_waveforms_1.at[0, chan] = np.empty(18750)
    avg_waveforms_2 = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES,
                                   data=np.empty((1, 4), np.ndarray))
    for chan in avg_waveforms_2:
        avg_waveforms_2.at[0, chan] = np.empty(18750)
    for chan in measurements_comp:
        for n in range(len(measurements_split[chan][chirp])):
            avg_buf = np.array([])
            for chirp in range(len(measurements_split[chan]) // 3):
                # Trippel for-loop, baby
                avg_buf = np.append(avg_buf, [measurements_split.at[chirp, chan][n]])
            avg_waveforms_1.at[0, chan][n] = np.mean(avg_buf)
            avg_buf = np.array([])
            for chirp in range(2 * len(measurements_split[chan]) // 3,
                               len(measurements_split[chan]) - 1):
                avg_buf = np.append(avg_buf, [measurements_split.at[chirp, chan][n]])
            avg_waveforms_2.at[0, chan][n] = np.mean(avg_buf)

    """Plot the average waveforms"""
    fig, axs = plt.subplots(nrows=3, ncols=3)
    for chirp in range(len(measurements_split['Sensor 1']) - 1):
        compare_signals(fig, axs,
                        [avg_waveforms_2.at[0, 'Sensor 1'] - avg_waveforms_1.at[0, 'Sensor 1'],
                         avg_waveforms_2.at[0, 'Sensor 2'] - avg_waveforms_1.at[0, 'Sensor 2'],
                         avg_waveforms_2.at[0, 'Sensor 3'] - avg_waveforms_1.at[0, 'Sensor 3']],
                        freq_max=BANDWIDTH[1] + 20000,
                        nfft=16)
    # time_axis = np.linspace(start=0,
    #                         stop=0.125,
    #                         num=len(avg_waveforms_1.at[0, 'Sensor 1']))
    # for i in range(3):
    #     axs[i, 0].plot(time_axis,
    #                    avg_waveforms_1.at[0, avg_waveforms_1.columns[i]],
    #                    color='black',
    #                    linestyle='--')
    plt.show()

    """Plot individual waveforms, their average and the diff"""
    for chirp in range(len(avg_waveforms_2['Sensor 1'])):
        fig, axs = plt.subplots(nrows=3, ncols=3)
        compare_signals(fig, axs,
                        [avg_waveforms_2['Sensor 1'][chirp] - avg_waveforms_1['Sensor 1'][0],
                         avg_waveforms_2['Sensor 2'][chirp] - avg_waveforms_1['Sensor 2'][0],
                         avg_waveforms_2['Sensor 3'][chirp] - avg_waveforms_1['Sensor 3'][0]],
                        freq_max=BANDWIDTH[1] + 20000,
                        nfft=16)
        axs[0, 0].axvline(np.argmax(measurements_split['Sensor 1'][chirp]) / SAMPLE_RATE + (0.411909 / prop_speed),
                          color='red',
                          linestyle='--')
        axs[1, 0].axvline(np.argmax(measurements_split['Sensor 2'][chirp]) / SAMPLE_RATE + (0.421637 / prop_speed),
                          color='red',
                          linestyle='--')
        axs[2, 0].axvline(np.argmax(measurements_split['Sensor 3'][chirp]) / SAMPLE_RATE + (0.433977 / prop_speed),
                          color='red',
                          linestyle='--')
        axs[0, 1].axvline(np.argmax(measurements_split['Sensor 1'][chirp]) / SAMPLE_RATE + (0.411909 / prop_speed),
                          color='red',
                          linestyle='--')
        axs[1, 1].axvline(np.argmax(measurements_split['Sensor 2'][chirp]) / SAMPLE_RATE + (0.421637 / prop_speed),
                          color='red',
                          linestyle='--')
        axs[2, 1].axvline(np.argmax(measurements_split['Sensor 3'][chirp]) / SAMPLE_RATE + (0.433977 / prop_speed),
                          color='red',
                          linestyle='--')
        # compare_signals(fig, axs,
        #                 [measurements_split['Sensor 1'][chirp],
        #                  measurements_split['Sensor 2'][chirp],
        #                  measurements_split['Sensor 3'][chirp]],
        #                 freq_max=BANDWIDTH[1] + 20000,
        #                 nfft=256)
        # compare_signals(fig, axs,
        #                 [avg_waveforms['Sensor 1'][0],
        #                  avg_waveforms['Sensor 2'][0],
        #                  avg_waveforms['Sensor 3'][0]],
        #                 freq_max=BANDWIDTH[1] + 20000,
        #                 nfft=256)
    plt.show()

    """Plot individual waveforms, their average and the diff"""
    # for chirp in range(1, len(measurements_split['Sensor 1'])):
    #     fig, axs = plt.subplots(nrows=3, ncols=3)
    #     compare_signals(fig, axs,
    #                     [measurements_split['Sensor 1'][chirp] - measurements_split['Sensor 1'][chirp - 1],
    #                      measurements_split['Sensor 2'][chirp] - measurements_split['Sensor 2'][chirp - 1],
    #                      measurements_split['Sensor 3'][chirp] - measurements_split['Sensor 3'][chirp - 1]],
    #                     freq_max=BANDWIDTH[1] + 20000,
    #                     nfft=256)
    # plt.show()


if __name__ == '__main__':
    main()
