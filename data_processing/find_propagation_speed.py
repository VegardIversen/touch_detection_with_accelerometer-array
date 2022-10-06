import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pandas as pd
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
    print('Channel 1 is ' + str(1000 * delay) + 'ms behind channel 2')

    propagation_speed = distance_between_sensors / delay
    print("Propagation speed is", propagation_speed, "m/s")


if __name__ == '__main__':

    # Sine sample with some noise and copy to y1 and y2 with a 1-second lag
    sr = 150000

    chirp_df = csv_to_df(file_folder='div_files',
                         file_name='chirp_test_fs_96000_t_max_2s_20000-60000hz_1vpp_1cyc_setup3_method_linear_v3')

    find_propagation_speed(chirp_df, sr)
