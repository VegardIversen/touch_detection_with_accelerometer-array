import numpy as np
import scipy.signal as signal
from csv_to_df import csv_to_df


def find_propagation_speed(df, sr, distance_between_sensors=0.1):
    """Use the cross correlation between the two channels
    to find the propagation speed. Based on:
    https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
    """
    n = len(df['channel 1'])

    corr = signal.correlate(df['channel 1'], df['channel 2'], mode='same') \
         / np.sqrt(signal.correlate(df['channel 2'], df['channel 2'], mode='same')[int(n / 2)]
         * signal.correlate(df['channel 1'], df['channel 1'], mode='same')[int(n / 2)])

    delay_arr = np.linspace(-0.5 * n / sr, 0.5 * n / sr, n)
    delay = delay_arr[np.argmax(corr)]
    print('\n' + 'Channel 1 is ' + str(np.round(1000 * np.abs(delay), decimals=4))
          + ' ms behind channel 2\n')

    propagation_speed = distance_between_sensors / delay
    print("\n" + "Propagation speed is",
          np.round(np.abs(propagation_speed), decimals=2), "m/s \n")


if __name__ == '__main__':
    chirp_df = csv_to_df(file_folder='div_files',
                         file_name='chirp_test_fs_150000_t_max_0_1s_20000-40000hz_1vpp_1cyc_setup3_v1')

    find_propagation_speed(chirp_df, sr=150000)
