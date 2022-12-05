import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal

from constants import CHIRP_CHANNEL_NAMES, SAMPLE_RATE
from csv_to_df import csv_to_df
from data_processing.detect_echoes import (get_hilbert,
                                           get_travel_times)
from data_processing.preprocessing import (compress_chirp,
                                           crop_data,
                                           filter_general)
from data_processing.processing import (avg_waveform,
                                        interpolate_waveform,
                                        normalize,
                                        var_waveform,
                                        correct_drift)
from data_viz_files.visualise_data import compare_signals, wave_statistics, set_fontsizes
from objects import Table
from setups import Setup7


def main():
    """CONFIG"""
    FILE_FOLDER = 'setup7'
    FILE_NAME = 'notouchThenHoldB2_20to40khz_125ms_10vpp_v1'
    SETUP = Setup7()
    CROP = False
    TIME_START = 0.05  # s
    TIME_END = 0.8  # s
    FILTER = False
    BANDWIDTH = (200, 40000)

    set_fontsizes()

    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME,
                             channel_names=CHIRP_CHANNEL_NAMES)

    """Preprocessing"""
    if CROP:
        measurements = crop_data(measurements, TIME_START, TIME_END)

    """Compress chirp signals"""
    measurements = compress_chirp(measurements)

    """Shift to align chirps better with their time intervals"""
    SHIFT_BY = int(TIME_START * SAMPLE_RATE)
    for chan in measurements:
        measurements[chan] = np.roll(measurements[chan],
                                     -SHIFT_BY)
        measurements[chan][-SHIFT_BY:] = 0

    "Separate the channels into arrays of length 125 ms"
    measurements_split = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES)
    for chan in measurements_split:
        measurements_split[chan] = np.split(measurements[chan],
                                            indices_or_sections=40)

    """Draw setup"""
    # SETUP.draw()

    """Start a runtime timer after setup is drawn"""
    start = timeit.default_timer()

    """Calculate wave propagation speed"""
    prop_speed = SETUP.get_propagation_speed(measurements)
    prop_speed = 1700
    ANGLE_B1_SENSOR_2 = np.sin((Table.C2[0] - Table.B1[0]) /
                               np.linalg.norm(Table.B1 - Table.C2))
    ANGLE_B1_SENSOR_3 = np.sin((Table.C2[0] + 0.013 - Table.B1[0]) /
                                np.linalg.norm(Table.B1 - (Table.C2 + 0.013)))
    print(f'Propagation speed: {np.round(prop_speed, 2)} m/s.')
    # print(f'Angle B1 sensor 2: {np.round(np.arcsin(ANGLE_B1_SENSOR_2), 4)} rad.')
    # print(f'Angle B1 sensor 3: {np.round(np.arcsin(ANGLE_B1_SENSOR_3), 4)} rad.')
    # print(f'Propagation speed B1 sensor 2 v_x: {np.round(prop_speed_1 * ANGLE_B1_SENSOR_2, 2)} m/s.')
    # print(f'Propagation speed B1 sensor 3 v_x: {np.round(prop_speed_1 * ANGLE_B1_SENSOR_3, 2)} m/s.')
    # print(f'Peaks should appear with {0.013 / (prop_speed_1 * ANGLE_B1_SENSOR_2) * 1000} ms spacing.')
    # print(f'Peaks should appear with {0.013 / (prop_speed_1 * ANGLE_B1_SENSOR_3) * 1000} ms spacing.')

    """Calculate wave arrival times"""
    arrival_times = np.array([])
    for sensor in SETUP.sensors:
        time, _ = get_travel_times(SETUP.actuators[0],
                                   sensor,
                                   prop_speed,
                                   ms=False,
                                   print_info=True,
                                   relative_first_reflection=False)
        arrival_times = np.append(arrival_times, time)
    """Reshape arrival_times to a 2D array with len(sensor) rows"""
    arrival_times = np.reshape(arrival_times,
                               (len(SETUP.sensors),
                                len(arrival_times) // len(SETUP.sensors)))

    """Plot the shifted measurements"""
    # fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    # time_axs = np.linspace(0, 5, measurements.shape[0])
    # for chan in measurements.columns[0:3]:
    #     axs[0].plot(time_axs, measurements[chan], label=chan)
    # axs[0].set_title('Shifted measurements 1')
    # axs[0].set_xlabel('Time [s]')
    # axs[0].set_ylabel('Amplitude [V]')
    # axs[0].legend()
    # axs[0].grid()
    # plt.show()

    """Make up for drift in signal generator"""
    # measurements_split = correct_drift(measurements_split,
    #                                    data_to_sync_with=measurements_split,
    #                                    n_interp=20 * len(measurements_split['Sensor 1'][0]))

    """Find the average and variance of the waveforms"""
    chirp_range = [1,
                   len(measurements_split['Sensor 1']) - 1]
    avg_waveforms = avg_waveform(measurements_split,
                                 chirp_range)
    # var_waveforms = var_waveform(measurements_split,
    #                              chirp_range)

    """Generate Hilbert transforms"""
    # avg_waveforms_hilb = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES,
    #                                     data=np.empty((1, 4), np.ndarray))
    # for chan in avg_waveforms:
    #     avg_waveforms_hilb.at[0, chan] = get_hilbert_envelope(avg_waveforms.at[0, chan])

    """Normalize the average waveforms"""
    # avg_waveforms = normalize(avg_waveforms)
    # avg_waveforms_hilb = normalize(avg_waveforms_hilb)

    """Sum the waveforms to detect periodic peaks"""
    # avg_waveforms_1_sum = avg_waveforms_1_hilb[avg_waveforms.columns[:3]].sum(axis=1)
    # avg_waveforms_2_sum = avg_waveforms_2_hilb[avg_waveforms_2.columns[:3]].sum(axis=1)
    # time_axs = np.linspace(start=0,
    #                        stop=len(avg_waveforms_1_sum.values[0]) / SAMPLE_RATE,
    #                        num=len(avg_waveforms_1_sum.values[0]))
    # fig, axs = plt.subplots(nrows=2, ncols=2)
    # axs[0, 0].plot(time_axs, 10 * np.log10(avg_waveforms_1_sum.values[0]))
    # axs[0, 0].set_title('Sum of all channels')
    # axs[0, 0].set_xlabel('Time [s]')
    # axs[0, 0].set_ylabel('Amplitude')
    # axs[0, 0].grid()
    # axs[1, 0].plot(time_axs, 10 * np.log10(avg_waveforms_2_sum.values[0]))
    # axs[1, 0].set_title('Sum of all channels')
    # axs[1, 0].set_xlabel('Time [s]')
    # axs[1, 0].set_ylabel('Amplitude')
    # axs[1, 0].sharex(axs[0, 0])
    # axs[1, 0].sharey(axs[0, 0])
    # axs[1, 0].grid()
    """Plot FFTs of the sum of the waveforms"""
    # freq_axs = np.linspace(start=0,
    #                        stop=(SAMPLE_RATE / 2),
    #                        num=len(avg_waveforms_1_sum.values[0]) // 2)
    # fft_1 = np.fft.fft(avg_waveforms_1_sum.values[0])
    # fft_2 = np.fft.fft(avg_waveforms_2_sum.values[0])
    # axs[0, 1].plot(freq_axs, 10 * np.log10(np.abs(fft_1[:len(fft_1) // 2])))
    # axs[0, 1].set_title('FFT of sum of all channels')
    # axs[0, 1].set_xlabel('Frequency [Hz]')
    # axs[0, 1].set_ylabel('Amplitude')
    # axs[0, 1].grid()
    # axs[1, 1].plot(freq_axs, 10 * np.log10(np.abs(fft_2[:len(fft_2) // 2])))
    # axs[1, 1].set_title('FFT of sum of all channels')
    # axs[1, 1].set_xlabel('Frequency [Hz]')
    # axs[1, 1].set_ylabel('Amplitude')
    # axs[1, 1].sharex(axs[0, 1])
    # axs[1, 1].sharey(axs[0, 1])
    # axs[1, 1].grid()
    # plt.show()

    """Stop runtime timer before plotting"""
    stop = timeit.default_timer()
    print(f'\nRuntime: {np.round(stop - start, 2)} s')

    """Plot statistics for waveforms"""
    # fig, axs = plt.subplots(3, 1, sharex=True)
    # wave_statistics(fig, axs, measurements_split_1)
    # plt.show()

    """Plot all chirps on top of each other"""
    # time_axis = np.linspace(start=0,
    #                         stop=len(measurements_split['Sensor 1'][0]) / (20 * SAMPLE_RATE),
    #                         num=len(measurements_split['Sensor 1'][0]))
    # fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    # for i, _ in enumerate(measurements_split['Sensor 1'][0:39]):
    #     axs[0].plot(time_axis, measurements_split['Sensor 1'][i])
    #     axs[1].plot(time_axis, measurements_split['Sensor 2'][i])
    #     axs[2].plot(time_axis, measurements_split['Sensor 3'][i])
    # axs[0].set_title('39 chirps, sensor 1')
    # axs[1].set_title('39 chirps, sensor 2')
    # axs[2].set_title('39 chirps, sensor 3')
    # axs[2].set_xlabel('Time [s]')
    # axs[0].grid()
    # axs[1].grid()
    # axs[2].grid()
    # plt.show()

    """Plot a waveforms"""
    fig, axs = plt.subplots(nrows=3, ncols=3)
    compare_signals(fig, axs,
                    [measurements_split['Sensor 1'][1][0:int(0.0715 * SAMPLE_RATE)],
                     measurements_split['Sensor 2'][1][0:int(0.0715 * SAMPLE_RATE)],
                     measurements_split['Sensor 3'][1][0:int(0.0715 * SAMPLE_RATE)]],
                    nfft=16,
                    dynamic_range_db=20)
    """Plot the Hilert transforms"""
    # time_axis = np.linspace(start=0,
    #                         stop=len(measurements_split['Sensor 1'][1]) / SAMPLE_RATE,
    #                         num=len(measurements_split['Sensor 1'][1]))
    # axs[0, 0].plot(time_axis, np.abs(signal.hilbert(measurements_split['Sensor 1'][1])))
    # axs[1, 0].plot(time_axis, np.abs(signal.hilbert(measurements_split['Sensor 2'][1])))
    # axs[2, 0].plot(time_axis, np.abs(signal.hilbert(measurements_split['Sensor 3'][1])))
    """Plot the expected arrival times"""
    arrival_times += len(measurements_split['Sensor 1'][1]) / (2 * SAMPLE_RATE) + 0.00895
    for i in range(3):
        axs[i, 0].axvline(arrival_times[i][0],
                          linestyle='--',
                          color='r',
                          label='Direct wave')
        [axs[i, 0].axvline(line,
                           linestyle='--',
                           color='g',
                           label='1st reflections')
         for line in (arrival_times[i][1:5])]
        axs[i, 1].axvline(arrival_times[i][0],
                          linestyle='--',
                          color='r',
                          label='Direct wave')
        [axs[i, 1].axvline(line,
                           linestyle='--',
                           color='g',
                           label='1st reflections')
         for line in (arrival_times[i][1:5])]
        # [axs[i, 0].axvline(line,
        #              linestyle='--',
        #              color='purple',
        #              label='2nd reflections')
        #  for line in (arrival_times[i][5:])]
    plt.show()

    """Plot each chirp subtracted the previous chirp"""
    # time_axis = np.linspace(start=0,
    #                         stop=len(measurements_split['Sensor 1'][0]) / (10 * SAMPLE_RATE),
    #                         num=len(measurements_split['Sensor 1'][0]))

    # for i, chirp in enumerate(measurements_split['Sensor 1']):
    #     if i == 0:
    #         continue
    #     fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    #     spec = axs[0].specgram(measurements_split['Sensor 1'][i] - measurements_split['Sensor 1'][i - 1],
    #                            Fs=SAMPLE_RATE,
    #                            NFFT=16,
    #                            noverlap=(8))
    #     # fig.colorbar(spec[3], ax=axs[i, 1])
    #     spec[3].set_clim(10 * np.log10(np.max(spec[0])) - 40,
    #                      10 * np.log10(np.max(spec[0])))
    #     axs[0].set_xlabel('Time [s]')
    #     axs[0].set_ylabel('Frequency [Hz]')
    #     axs[0].axis(ymax=50000)
    #     axs[0].set_xlim(0.09126, 0.09207)
    #     axs[0].axvline(np.argmax(measurements_split['Sensor 1'][i]) /
    #                    SAMPLE_RATE + np.abs((0.1203333 - 0.563398) / prop_speed),
    #                    color='red',
    #                    linestyle='--')
    #     spec = axs[1].specgram(measurements_split['Sensor 2'][i] - measurements_split['Sensor 2'][i - 1],
    #                            Fs=SAMPLE_RATE,
    #                            NFFT=16,
    #                            noverlap=(8))
    #     spec[3].set_clim(10 * np.log10(np.max(spec[1])) - 40,
    #                   10 * np.log10(np.max(spec[1])))
    #     axs[1].set_xlabel('Time [s]')
    #     axs[1].set_ylabel('Frequency [Hz]')
    #     axs[1].axis(ymax=50000)
    #     axs[1].set_xlim(0.09126, 0.09207)
    #     axs[1].axvline(np.argmax(measurements_split['Sensor 2'][i]) /
    #                    SAMPLE_RATE + np.abs((0.133333 - 0.573703) / prop_speed),
    #                    color='red',
    #                    linestyle='--')
    #     spec = axs[2].specgram(measurements_split['Sensor 3'][i] - measurements_split['Sensor 3'][i - 1],
    #                            Fs=SAMPLE_RATE,
    #                            NFFT=16,
    #                            noverlap=(8))
    #     spec[3].set_clim(10 * np.log10(np.max(spec[2])) - 40,
    #                      10 * np.log10(np.max(spec[2])))
    #     axs[2].set_xlabel('Time [s]')
    #     axs[2].set_ylabel('Frequency [Hz]')
    #     axs[2].axis(ymax=50000)
    #     axs[2].set_xlim(0.09126, 0.09207)
    #     axs[2].axvline(np.argmax(measurements_split['Sensor 3'][i]) /
    #                    SAMPLE_RATE + np.abs((0.146333 - 0.584192) / prop_speed),
    #                    color='red',
    #                    linestyle='--')
    #     manager = plt.get_current_fig_manager()
    #     manager.full_screen_toggle()

    # plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.show()




    #     compare_signals(fig, axs,
    #                     [(np.abs(signal.hilbert(avg_waveforms_2['Sensor 1'][chirp] -
    #                                            avg_waveforms_1['Sensor 1'][0]))),
    #                      (np.abs(signal.hilbert(avg_waveforms_2['Sensor 2'][chirp] -
    #                                            avg_waveforms_1['Sensor 2'][0]))),
    #                      (np.abs(signal.hilbert(avg_waveforms_2['Sensor 3'][chirp] -
    #                                            avg_waveforms_1['Sensor 3'][0])))],
    #                     freq_max=BANDWIDTH[1] + 20000,
    #                     nfft=16,
    #                     dynamic_range_db=10)
    #     compare_signals(fig, axs,
    #                     [(var_waveforms_1['Sensor 1'][0]),
    #                      (var_waveforms_1['Sensor 2'][0]),
    #                      (var_waveforms_1['Sensor 3'][0])],
    #                     freq_max=BANDWIDTH[1] + 20000,
    #                     nfft=16,
    #                     dynamic_range_db=32)
    #     compare_signals(fig, axs,
    #                     [(var_waveforms_2['Sensor 1'][0]),
    #                      (var_waveforms_2['Sensor 2'][0]),
    #                      (var_waveforms_2['Sensor 3'][0])],
    #                     freq_max=BANDWIDTH[1] + 20000,
    #                     nfft=16,
    #                     dynamic_range_db=32)
    #     axs[0, 1].axvline(np.argmax(avg_waveforms_1['Sensor 1'][chirp]) /
    #                       SAMPLE_RATE + np.abs((0.1203333 - 0.563398) / prop_speed_1),
    #                       color='red',
    #                       linestyle='--')
    #     axs[1, 1].axvline(np.argmax(avg_waveforms_1['Sensor 2'][chirp]) /
    #                       SAMPLE_RATE + np.abs((0.133333 - 0.573703) / prop_speed_1),

    #                       color='red',
    #                       linestyle='--')
    #     axs[2, 1].axvline(np.argmax(avg_waveforms_1['Sensor 3'][chirp]) /
    #                       SAMPLE_RATE + np.abs((0.146333 - 0.584192) / prop_speed_1),
    #                       color='red',
    #                       linestyle='--')
    # plt.show()


if __name__ == '__main__':
    main()
