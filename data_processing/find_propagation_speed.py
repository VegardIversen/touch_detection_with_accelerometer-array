import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pandas as pd
from pathlib import Path

# Set the path to the data
DATA_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
FILE_FOLDER = '\\div_files'
FILE_NAME = '\\chirp_test_fs_96000_t_max_2s_20000-60000hz_1vpp_1cyc_setup3_method_linear_v3'
FILE_EXTENSION = '.csv'
FILE_PATH = DATA_FOLDER + FILE_FOLDER + FILE_NAME + FILE_EXTENSION
print("\nUsing data file path:", FILE_PATH, "\n")

# Headers for the dataframe df
CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3']


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

    df = pd.read_csv(FILE_PATH, names=CHANNEL_NAMES)

    find_propagation_speed(df, sr)
