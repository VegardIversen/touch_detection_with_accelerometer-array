import numpy as np
import scipy.signal as signal
from csv_to_df import csv_to_df
from data_processing.preprocessing import hp_or_lp_filter


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




def find_propagation_speed_func(chirp_df, start_freq, end_freq, steps=1000, sample_rate=150000):
      frequencies = np.array([])
      freq_speeds = np.array([])

      for freq in range(start_freq, end_freq, steps):
            chirp_bp = hp_or_lp_filter(sig=chirp_df,
                                       filtertype='lowpass',
                                       cutoff=freq,
                                       order=5)
            chirp_bp = hp_or_lp_filter(sig=chirp_df,
                                       filtertype='highpass',
                                       cutoff=freq,
                                       order=5)
            freq_prop_speed = find_propagation_speed(df=chirp_bp,
                                                     ch1='channel 1',
                                                     ch2='chirp',
                                                     sr=sample_rate)
            frequencies = np.append(frequencies, freq)
            freq_speeds = np.append(freq_speeds, freq_prop_speed)

      return frequencies, freq_speeds


if __name__ == '__main__':
      chirp_df = csv_to_df(file_folder='div_files',
                           file_name='chirp_test_fs_150000_t_max_0_1s_20000-60000hz_1vpp_1cyc_setup3_v2')



      find_propagation_speed(chirp_df, sr=150000)
