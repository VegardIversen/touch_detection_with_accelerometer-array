import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal

from constants import CHIRP_CHANNEL_NAMES, SAMPLE_RATE
from csv_to_df import csv_to_df
from data_processing.detect_echoes import (get_hilbert_envelope,
                                           get_travel_times)
from data_processing.preprocessing import (compress_chirp,
                                           crop_data,
                                           filter_general)
from data_processing.processing import (avg_waveform,
                                        interpolate_waveform,
                                        normalize,
                                        var_waveform,
                                        correct_drift)
from data_viz_files.visualise_data import compare_signals, wave_statistics
from objects import Table
from setups import Setup7


def main():
    """CONFIG"""
    FILE_FOLDER = 'setup7'
    FILE_NAME_1 = 'notouch_20to40khz_100ms_10vpp_v1'
    FILE_NAME_2 = 'holdB1_20to40khz_125ms_10vpp_v1'
    SETUP = Setup7()
    CROP = False
    TIME_START = 0.03  # s
    TIME_END = 0.8  # s
    FILTER = False
    BANDWIDTH = (200, 40000)

    """Open file"""
    measurements_1 = csv_to_df(file_folder=FILE_FOLDER,
                               file_name=FILE_NAME_1,
                               channel_names=CHIRP_CHANNEL_NAMES)
    measurements_2 = csv_to_df(file_folder=FILE_FOLDER,
                               file_name=FILE_NAME_2,
                               channel_names=CHIRP_CHANNEL_NAMES)

    """Preprocessing"""
    if CROP:
        measurements_1 = crop_data(measurements_1, TIME_START, TIME_END)
        measurements_2 = crop_data(measurements_2, TIME_START, TIME_END)
    if FILTER:
        measurements_filt_1 = filter_general(measurements_1,
                                             filtertype='bandpass',
                                             cutoff_highpass=BANDWIDTH[0],
                                             order=4)
        measurements_filt_2 = filter_general(measurements_2,
                                             filtertype='bandpass',
                                             cutoff_highpass=BANDWIDTH[0],
                                             order=4)
    else:
        measurements_filt_1 = measurements_1
        measurements_filt_2 = measurements_2

    """Compress chirp signals"""
    measurements_comp_1 = compress_chirp(measurements_filt_1)
    measurements_comp_2 = compress_chirp(measurements_filt_2)

    """Shift to align chirps better with their time intervals"""
    SHIFT_BY = int(TIME_START * SAMPLE_RATE)
    for chan in measurements_comp_1:
        measurements_comp_1[chan] = np.roll(measurements_comp_1[chan],
                                            -SHIFT_BY)
        measurements_comp_1[chan][-SHIFT_BY:] = 0
    for chan in measurements_comp_2:
        measurements_comp_2[chan] = np.roll(measurements_comp_2[chan],
                                            -SHIFT_BY)
        measurements_comp_2[chan][-SHIFT_BY:] = 0
    "Separate the channels into arrays of length 125 ms"
    measurements_split_1 = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES)
    for chan in measurements_split_1:
        measurements_split_1[chan] = np.split(measurements_comp_1[chan],
                                              indices_or_sections=50)
    measurements_split_2 = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES)
    for chan in measurements_split_2:
        measurements_split_2[chan] = np.split(measurements_comp_2[chan],
                                              indices_or_sections=40)

    """Draw setup"""
    SETUP.draw()

    """Start a runtime timer after setup is drawn"""
    start = timeit.default_timer()

    """Calculate wave propagation speed"""
    prop_speed_1 = SETUP.get_propagation_speed(measurements_comp_1)
    # prop_speed_1 *= 1.3
    ANGLE_B1_SENSOR_2 = np.sin((Table.C2[0] - Table.B1[0]) /
                        np.linalg.norm(Table.B1 - Table.C2))
    ANNGLE_B1_SENSOR_3 = np.sin((Table.C2[0] + 0.013 - Table.B1[0]) /
                         np.linalg.norm(Table.B1 - (Table.C2 + 0.013)))
    print(f'Propagation speed 1: {np.round(prop_speed_1, 2)} m/s.')
    print(f'Angle B1 sensor 2: {np.round(np.arcsin(ANGLE_B1_SENSOR_2), 4)} rad.')
    print(f'Angle B1 sensor 3: {np.round(np.arcsin(ANNGLE_B1_SENSOR_3), 4)} rad.')
    print(f'Propagation speed B1 sensor 2 v_x: {np.round(prop_speed_1 * ANGLE_B1_SENSOR_2, 2)} m/s.')
    print(f'Propagation speed B1 sensor 3 v_x: {np.round(prop_speed_1 * ANNGLE_B1_SENSOR_3, 2)} m/s.')
    print(f'Peaks should appear with {0.013 / (prop_speed_1 * ANGLE_B1_SENSOR_2) * 1000} ms spacing.')
    print(f'Peaks should appear with {0.013 / (prop_speed_1 * ANNGLE_B1_SENSOR_3) * 1000} ms spacing.')
    prop_speed_2 = SETUP.get_propagation_speed(measurements_comp_2)
    print(f'Propagation speed 2: {np.round(prop_speed_2, 2)} m/s')

    """Calculate wave arrival times"""
    # arrival_times = np.array([])
    # for sensor in SETUP.sensors:
    #     time, _ = get_travel_times(SETUP.actuators[0],
    #                                sensor,
    #                                prop_speed,
    #                                ms=False,
    #                                print_info=False,
    #                                relative_first_reflection=False)
    #     arrival_times = np.append(arrival_times, time)
    """Reshape arrival_times to a 2D array with len(sensor) rows"""
    # arrival_times = np.reshape(arrival_times,
    #                            (len(SETUP.sensors),
    #                             len(arrival_times) // len(SETUP.sensors)))

    """Plot the shifted measurements"""
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    time_axs = np.linspace(0, 5, measurements_comp_1.shape[0])
    for chan in measurements_comp_1.columns[0:3]:
        axs[0].plot(time_axs, measurements_comp_1[chan], label=chan)
    axs[0].set_title('Shifted measurements 1')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Amplitude [V]')
    axs[0].legend()
    axs[0].grid()
    for chan in measurements_comp_2.columns[0:3]:
        axs[1].plot(time_axs, measurements_comp_2[chan], label=chan)
    axs[1].set_title('Shifted measurements 2')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Amplitude [V]')
    axs[1].legend()
    axs[1].grid()
    plt.show()

    """Make up for drift in signal generator"""
    measurements_split_1 = correct_drift(measurements_split_1,
                                         data_to_sync_with=measurements_split_1,
                                         n_interp=len(measurements_split_1['Sensor 1'][0]))
    measurements_split_2 = correct_drift(measurements_split_2,
                                         data_to_sync_with=measurements_split_1,
                                         n_interp=len(measurements_split_1['Sensor 1'][0]))

    """Find the average and variance of the waveforms"""
    chirp_range = [1,
                   len(measurements_split_1['Sensor 1']) - 1]
    avg_waveforms_1 = avg_waveform(measurements_split_1,
                                   chirp_range)
    var_waveforms_1 = var_waveform(measurements_split_1,
                                   chirp_range)
    chirp_range = [1,
                   len(measurements_split_2['Sensor 1']) - 1]
    avg_waveforms_2 = avg_waveform(measurements_split_2,
                                   chirp_range)
    var_waveforms_2 = var_waveform(measurements_split_2,
                                   chirp_range)


    """Generate Hilbert transforms"""
    avg_waveforms_1_hilb = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES,
                                        data=np.empty((1, 4), np.ndarray))
    avg_waveforms_2_hilb = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES,
                                        data=np.empty((1, 4), np.ndarray))
    for chan in avg_waveforms_1:
        avg_waveforms_1_hilb.at[0, chan] = get_hilbert_envelope(avg_waveforms_1.at[0, chan])
        avg_waveforms_2_hilb.at[0, chan] = get_hilbert_envelope(avg_waveforms_2.at[0, chan])


    """Normalize the average waveforms"""
    avg_waveforms_1 = normalize(avg_waveforms_1)
    avg_waveforms_2 = normalize(avg_waveforms_2)
    avg_waveforms_1_hilb = normalize(avg_waveforms_1_hilb)
    avg_waveforms_2_hilb = normalize(avg_waveforms_2_hilb)

    """Sum the waveforms to detect periodic peaks"""
    avg_waveforms_1_sum = avg_waveforms_1_hilb[avg_waveforms_1.columns[:3]].sum(axis=1)
    avg_waveforms_2_sum = avg_waveforms_2_hilb[avg_waveforms_2.columns[:3]].sum(axis=1)
    time_axs = np.linspace(start=0,
                           stop=len(avg_waveforms_1_sum.values[0]) / SAMPLE_RATE,
                           num=len(avg_waveforms_1_sum.values[0]))
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0, 0].plot(time_axs, 10 * np.log10(avg_waveforms_1_sum.values[0]))
    axs[0, 0].set_title('Sum of all channels')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid()
    axs[1, 0].plot(time_axs, 10 * np.log10(avg_waveforms_2_sum.values[0]))
    axs[1, 0].set_title('Sum of all channels')
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].sharex(axs[0, 0])
    axs[1, 0].sharey(axs[0, 0])
    axs[1, 0].grid()
    """Plot FFTs of the sum of the waveforms"""
    freq_axs = np.linspace(start=0,
                           stop=(SAMPLE_RATE / 2),
                           num=len(avg_waveforms_1_sum.values[0]) // 2)
    fft_1 = np.fft.fft(avg_waveforms_1_sum.values[0])
    fft_2 = np.fft.fft(avg_waveforms_2_sum.values[0])
    axs[0, 1].plot(freq_axs, 10 * np.log10(np.abs(fft_1[:len(fft_1) // 2])))
    axs[0, 1].set_title('FFT of sum of all channels')
    axs[0, 1].set_xlabel('Frequency [Hz]')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].grid()
    axs[1, 1].plot(freq_axs, 10 * np.log10(np.abs(fft_2[:len(fft_2) // 2])))
    axs[1, 1].set_title('FFT of sum of all channels')
    axs[1, 1].set_xlabel('Frequency [Hz]')
    axs[1, 1].set_ylabel('Amplitude')
    axs[1, 1].sharex(axs[0, 1])
    axs[1, 1].sharey(axs[0, 1])
    axs[1, 1].grid()
    plt.show()

    """Stop runtime timer before plotting"""
    stop = timeit.default_timer()
    print(f'\nRuntime: {np.round(stop - start, 2)} s')

    """Plot statistics for waveforms"""
    # fig, axs = plt.subplots(3, 1, sharex=True)
    # wave_statistics(fig, axs, measurements_split_1)
    # plt.show()


    """Plot the average waveforms"""
    # fig, axs = plt.subplots(nrows=3, ncols=3)
    # for chirp in range(len(measurements_split_1['Sensor 1']) - 1):
    #     compare_signals(fig, axs,
    #                     [measurements_split_1['Sensor 1'][chirp],
    #                      measurements_split_1['Sensor 2'][chirp],
    #                      measurements_split_1['Sensor 3'][chirp]],
    #                     freq_max=BANDWIDTH[1] + 20000,
    #                     nfft=16)
        # compare_signals(fig, axs,
        #                 [avg_waveforms_2['Sensor 1'][0],
        #                  avg_waveforms_2['Sensor 2'][0],
        #                  avg_waveforms_2['Sensor 3'][0]],
        #                 freq_max=BANDWIDTH[1] + 20000,
        #                 nfft=16)
    # time_axis = np.linspace(start=0,
    #                         stop=0.125,
    #                         num=len(avg_waveforms_1.at[0, 'Sensor 1']))
    # for i in range(3):
    #     axs[i, 0].plot(time_axis,
    #                    avg_waveforms_1.at[0, avg_waveforms_1.columns[i]],
    #                    color='black',
    #                    linestyle='--')
    #     axs[i, 0].plot(time_axis,
    #                    avg_waveforms_2.at[0, avg_waveforms_1.columns[i]],
    #                    color='gray',
    #                    linestyle='--')
    # plt.show()

    """Plot individual waveforms, their average and the diff"""
    for chirp in range(1):
        fig, axs = plt.subplots(nrows=3, ncols=3)
        compare_signals(fig, axs,
                        [normalize(avg_waveforms_1_hilb['Sensor 1'][chirp]),
                         normalize(avg_waveforms_1_hilb['Sensor 2'][chirp]),
                         normalize(avg_waveforms_1_hilb['Sensor 3'][chirp])],
                         freq_max=BANDWIDTH[1] + 20000,
                         nfft=16,
                         dynamic_range_db=20)
        compare_signals(fig, axs,
                        [normalize(avg_waveforms_2_hilb['Sensor 1'][chirp]),
                         normalize(avg_waveforms_2_hilb['Sensor 2'][chirp]),
                         normalize(avg_waveforms_2_hilb['Sensor 3'][chirp])],
                         freq_max=BANDWIDTH[1] + 20000,
                         nfft=16,
                         dynamic_range_db=20)
        compare_signals(fig, axs,
                        [normalize(np.abs(signal.hilbert((avg_waveforms_2['Sensor 1'][chirp] -
                                   avg_waveforms_1['Sensor 1'][0])))),
                         normalize(np.abs(signal.hilbert((avg_waveforms_2['Sensor 2'][chirp] -
                                   avg_waveforms_1['Sensor 2'][0])))),
                         normalize(np.abs(signal.hilbert((avg_waveforms_2['Sensor 3'][chirp] -
                                   avg_waveforms_1['Sensor 3'][0]))))],
                        freq_max=BANDWIDTH[1] + 20000,
                        nfft=16,
                        dynamic_range_db=25)
        # compare_signals(fig, axs,
        #                 [(np.abs(signal.hilbert(avg_waveforms_2['Sensor 1'][chirp] -
        #                                        avg_waveforms_1['Sensor 1'][0]))),
        #                  (np.abs(signal.hilbert(avg_waveforms_2['Sensor 2'][chirp] -
        #                                        avg_waveforms_1['Sensor 2'][0]))),
        #                  (np.abs(signal.hilbert(avg_waveforms_2['Sensor 3'][chirp] -
        #                                        avg_waveforms_1['Sensor 3'][0])))],
        #                 freq_max=BANDWIDTH[1] + 20000,
        #                 nfft=16,
        #                 dynamic_range_db=10)
        # compare_signals(fig, axs,
        #                 [(var_waveforms_1['Sensor 1'][0]),
        #                  (var_waveforms_1['Sensor 2'][0]),
        #                  (var_waveforms_1['Sensor 3'][0])],
        #                 freq_max=BANDWIDTH[1] + 20000,
        #                 nfft=16,
        #                 dynamic_range_db=32)
        # compare_signals(fig, axs,
        #                 [(var_waveforms_2['Sensor 1'][0]),
        #                  (var_waveforms_2['Sensor 2'][0]),
        #                  (var_waveforms_2['Sensor 3'][0])],
        #                 freq_max=BANDWIDTH[1] + 20000,
        #                 nfft=16,
        #                 dynamic_range_db=32)
        axs[0, 0].axvline(np.argmax(avg_waveforms_1['Sensor 1'][chirp]) /
                          SAMPLE_RATE + np.abs((0.1203333 - 0.563398) / prop_speed_1),
                          color='red',
                          linestyle='--')
        axs[1, 0].axvline(np.argmax(avg_waveforms_1['Sensor 2'][chirp]) /
                          SAMPLE_RATE + np.abs((0.133333 - 0.573703) / prop_speed_1),
                          color='red',
                          linestyle='--')
        axs[2, 0].axvline(np.argmax(avg_waveforms_1['Sensor 3'][chirp]) /
                          SAMPLE_RATE + np.abs((0.146333 - 0.584192) / prop_speed_1),
                          color='red',
                          linestyle='--')
        axs[0, 1].axvline(np.argmax(avg_waveforms_1['Sensor 1'][chirp]) /
                          SAMPLE_RATE + np.abs((0.1203333 - 0.563398) / prop_speed_1),
                          color='red',
                          linestyle='--')
        axs[1, 1].axvline(np.argmax(avg_waveforms_1['Sensor 2'][chirp]) /
                          SAMPLE_RATE + np.abs((0.133333 - 0.573703) / prop_speed_1),
                          color='red',
                          linestyle='--')
        axs[2, 1].axvline(np.argmax(avg_waveforms_1['Sensor 3'][chirp]) /
                          SAMPLE_RATE + np.abs((0.146333 - 0.584192) / prop_speed_1),
                          color='red',
                          linestyle='--')
    plt.show()


if __name__ == '__main__':
    main()
