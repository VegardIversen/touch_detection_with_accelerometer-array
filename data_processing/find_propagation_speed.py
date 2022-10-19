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
    chirp_bps = np.array([])

    for freq in range(start_freq, end_freq, steps):
        chirp_bp = filter_general(sig=chirp_df,
                                  filtertype='bandpass',
                                  cutoff_lowpass=freq * 0.9,
                                  cutoff_highpass=freq * 1.1,
                                  order=4)
        freq_prop_speed = find_propagation_speed(df=chirp_bp,
                                                 ch1='channel 1',
                                                 ch2='channel 3',
                                                 sr=sample_rate)
        frequencies = np.append(frequencies, freq)
        freq_speeds = np.append(freq_speeds, freq_prop_speed)
        chirp_bps = np.append(chirp_bps, chirp_bp)

    return frequencies, freq_speeds, chirp_bps


if __name__ == '__main__':
    """Chirp freq speed stuff"""
    chirp_df = csv_to_df(file_folder='div_files',
                                  file_name='chirp_test_fs_150000_t_max_1s_20000-40000hz_1vpp_1cyc_setup3_v1',
                                  channel_names=CHIRP_CHANNEL_NAMES)

    frequencies, freq_speeds, chirp_bps = find_propagation_speed_plot(chirp_df,
                                                                      start_freq=20000,
                                                                      end_freq=40000)


    plt.subplot(2, 1, 1)
    plt.plot(frequencies, freq_speeds)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Propagation speed [m/s]')
    plt.title('Propagation speed vs frequency')
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    time_axis = np.linspace(0, len(chirp_bps[0]) / SAMPLE_RATE, len(chirp_bps[0]))
    plt.plot(time_axis, chirp_bps)
    plt.show()

    """Burst stuff"""

    burst_measured_df = csv_to_df(file_folder='div_files',
                                  file_name='separate_signal_test_9vpp_chirp_30000_30000hz_0_00024s_150khzsr_setup3_trigon_v1',
                                  channel_names=CHIRP_CHANNEL_NAMES)

    burst_meas_crop_df = crop_data(burst_measured_df, TIME_START, TIME_END)

    burst_meas_crop_filt_df = filter_general(burst_meas_crop_df,
                                             filtertype='bandpass',
                                             cutoff_lowpass=29000,
                                             cutoff_highpass=31000,
                                             order=2)

    time_axis = np.linspace(start=0,
                            stop=len(burst_meas_crop_filt_df) / SAMPLE_RATE,
                            num=len(burst_meas_crop_filt_df))

    burst_meas_hilbert_df = get_hilbert_envelope(burst_meas_crop_df)

    corr = np.correlate(burst_meas_crop_filt_df['channel 1'],
                        burst_meas_crop_filt_df['chirp'],
                        mode='full')
    corr_hilbert = get_hilbert_envelope(corr)
    time_corr_ax = np.linspace(-TIME_END, TIME_END, len(corr))

    prop_speed = find_propagation_speed(burst_meas_crop_filt_df,
                                        ch1='channel 1',
                                        ch2='channel 3',
                                        sr=SAMPLE_RATE)
    print(prop_speed)
    ch2_hilbert_index_1st_peak = find_first_peak(sig_np=burst_meas_hilbert_df['channel 2'],
                                                 height=0.0002)

    plt.subplot(211)
    for channel in ['channel 2']:
        plt.plot(time_axis,
                 burst_meas_hilbert_df[channel],
                 label=channel + ' Hilbert')
        plt.plot(time_axis,
                 burst_meas_crop_df[channel],
                 label=channel)

    for d in [0.25, 0.337, 0.386, 0.41]:    # Distances to the edges
        plt.axvline(ch2_hilbert_index_1st_peak / SAMPLE_RATE + 2 * d / prop_speed,
                    color='k', linestyle='--')

    plt.plot(time_axis[ch2_hilbert_index_1st_peak],
             burst_meas_hilbert_df['channel 2'][ch2_hilbert_index_1st_peak],
             'x',
             color='red',
             label='first peak')


    plt.legend()
    plt.title('Chirp signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid()

    plt.subplot(212)
    plt.plot(time_axis, burst_meas_crop_df['channel 1'], label='channel 1')
    plt.plot(time_axis, burst_meas_crop_df['channel 2'], label='channel 2')
    plt.plot(time_axis, burst_meas_crop_filt_df['channel 1'], label='channel 1 filtered')
    plt.plot(time_axis, burst_meas_crop_filt_df['channel 2'], label='channel 2 filtered')
    plt.title('Chirp signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plt.show()