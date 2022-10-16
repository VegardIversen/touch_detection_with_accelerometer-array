import numpy as np
import scipy.signal as signal
from csv_to_df import csv_to_df
from data_processing.preprocessing import filter_general
from data_processing.detect_echoes import find_first_peak
import matplotlib.pyplot as plt


def find_propagation_speed(df, ch1, ch2, sr, distance_between_sensors=0.1):
    """Use the cross correlation between the two channels
    to find the propagation speed. Based on:
    https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
    """
    n = len(df['channel 1'])

    corr = signal.correlate(df[ch1], df[ch2], mode='same') \
        / np.sqrt(signal.correlate(df[ch2], df[ch2], mode='same')[int(n / 2)]
        * signal.correlate(df[ch1], df[ch1], mode='same')[int(n / 2)])

    delay_arr = np.linspace(-0.5 * n / sr, 0.5 * n / sr, n)
    delay = delay_arr[np.argmax(corr)]
    # print('\n' + 'The delay between ' + ch1 + ' and ' + ch2 + ' is ' + str(np.round(1000 * np.abs(delay), decimals=4)) + 'ms.')

    propagation_speed = np.round(np.abs(distance_between_sensors / delay), decimals=2)
    # print("\n" + "Propagation speed is", propagation_speed, "m/s \n")

    return propagation_speed


def find_propagation_speed_plot(chirp_df,
                                start_freq,
                                end_freq,
                                steps=1000,
                                sample_rate=150000):
    """Return an array of frequencies and an array of propagation speeds"""
    frequencies = np.array([])
    freq_speeds = np.array([])

    for freq in range(start_freq, end_freq, steps):
        chirp_bp = filter_general(sig=chirp_df,
                                  filtertype='bandpass',
                                  cutoff_low=freq * 0.9,
                                  cutoff_high=freq * 1.1,
                                  order=4)
        freq_prop_speed = find_propagation_speed(df=chirp_bp,
                                                 ch1='channel 1',
                                                 ch2='channel 3',
                                                 sr=sample_rate)
        frequencies = np.append(frequencies, freq)
        freq_speeds = np.append(freq_speeds, freq_prop_speed)

    return frequencies, freq_speeds


def find_propagation_speed_first_peak(chirp_df,
                                      start_freq,
                                      end_freq,
                                      time_start=0,
                                      time_end=5,
                                      distance=0.1,
                                      steps=100,
                                      sample_rate=150000,
                                      plot=False):
    """Find the propagation speed by looking at the first peaks"""
    frequencies = np.array([])
    freq_speeds = np.array([])

    for freq in range(start_freq,
                      end_freq + (end_freq - start_freq) // steps,
                      (end_freq - start_freq) // steps):
        time_axis = np.linspace(time_start, time_end, len(chirp_df['chirp']))

        chirp_bp = filter_general(sig=chirp_df,
                                   filtertype='lowpass',
                                   cutoff=freq,
                                   order=8)
        chirp_bp = filter_general(sig=chirp_bp,
                                   filtertype='highpass',
                                   cutoff=freq,
                                   order=8)

        # Apply a window function to the signal
        chirp_bp['channel 1'] = chirp_bp['channel 1'] * signal.windows.tukey(len(chirp_bp['channel 1']), alpha=0.1)
        chirp_bp['channel 3'] = chirp_bp['channel 3'] * signal.windows.tukey(len(chirp_bp['channel 3']), alpha=0.1)

        height = np.abs(np.max(chirp_bp['channel 1'].truncate(before=0, after=0.5 * sample_rate) * 1.5))

        peak_index_ch1 = find_first_peak(chirp_bp['channel 1'], height)
        peak_index_ch3 = find_first_peak(chirp_bp['channel 3'], height)
        # peak_index_chirp = find_first_peak(chirp_bp['chirp'], height)

        """Use the first peaks in ch1 and ch3 to find the propagation speed"""
        sample_delay_ch1_ch3 = np.abs(peak_index_ch1 - peak_index_ch3)
        time_delay_ch1_ch3 = sample_delay_ch1_ch3 / sample_rate
        if time_delay_ch1_ch3 != 0:
            freq_prop_speed_ch1_ch3 = np.abs(distance / time_delay_ch1_ch3)
        else:
            freq_prop_speed_ch1_ch3 = -1
        # print('\nPropagation speed (between channel 1 and channel 3) for',
        #       freq / 1000, 'kHz is', freq_prop_speed_ch1_ch3, 'm/s')

        """Could also use the chirp signal to find the propagation speed"""
        # sample_delay_chirp_ch1 = np.abs(peak_index_ch1 - peak_index_chirp)
        # time_delay_chirp_ch1 = sample_delay_chirp_ch1 / sample_rate
        # freq_prop_speed_chirp_ch1 = np.abs(distance / time_delay_chirp_ch1)
        # print('\nPropagation speed (between channel 1 and the chrip) for', freq / 1000, 'kHz is', freq_prop_speed_chirp_ch1, 'm/s')

        if plot:
            # Plot the propagation speed vs frequency
            time_axis = np.linspace(time_start, time_end, len(chirp_bp['channel 1']))
            plt.subplot(1, 1, 1)

            plt.plot(time_axis,
                     chirp_bp['channel 1'],
                     label='channel 1')

            plt.plot(time_axis[peak_index_ch1],
                     chirp_bp['channel 1'][peak_index_ch1],
                     'rx',
                     label='peak ch1')

            plt.plot(time_axis,
                     chirp_bp['channel 3'],
                     label='chirp')

            plt.plot(time_axis[peak_index_ch3],
                     chirp_bp['channel 3'][peak_index_ch3],
                     'rx',
                     label='peak ch3')

            plt.legend()
            plt.title('Chirp signal of frequency ' + str(freq) + ' Hz')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (V)')
            plt.grid()
            plt.show()

        frequencies = np.append(frequencies, freq)
        freq_speeds = np.append(freq_speeds, freq_prop_speed_ch1_ch3)

    return frequencies, freq_speeds


if __name__ == '__main__':
    chirp_df = csv_to_df(file_folder='div_files',
                         file_name='chirp_test_fs_150000_t_max_0_1s_20000-60000hz_1vpp_1cyc_setup3_v2')

    find_propagation_speed(chirp_df, sr=150000)
