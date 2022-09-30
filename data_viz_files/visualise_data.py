from distutils.util import change_root
from lzma import FILTER_DELTA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
from scipy import signal


def crop_data(data, crop_mode):
    """CROP_MODE:
    Manual,
    Auto
    """
    if crop_mode == "Auto":
        # Removes zero sections of the data
        data_cropped = data.loc[(df != 0).any(1)]
    elif crop_mode == "Manual":
        data_cropped = data.truncate(before=CROP_BEFORE, after=CROP_AFTER)

    return data_cropped


def plot_fft(df, Fs=150000, window=False):
    
    if window:
        hamming_window = scipy.signal.hamming(len(df))
        data_fft = scipy.fft.fft(df.values * hamming_window)
    else:
        data_fft = scipy.fft.fft(df.values, axis=0)

    # print(df.values)

    fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / Fs)
    plt.grid()
    plt.title('fft of signal')
    plt.xlabel("Frequency [hz]")
    plt.ylabel("Amplitude")
    plt.plot(fftfreq, 20 * np.log10(np.abs(data_fft)))
    # Only plot positive frequencies
    ax = plt.subplot(1, 1, 1)
    ax.set_xlim(0)
    plt.show()

def filter_signal(sig, FS=150000, Q=10):
    b_notch, a_notch = signal.iirnotch(50/(0.5*FS), Q)
    signal_filt = signal.filtfilt(b_notch, a_notch, sig)
    return signal_filt

def plot_fft_with_hamming(df, Fs=150000):
    plot_fft(df, window=True)


def filter_signal(sig, FS=150000, Q=3):
    b_notch, a_notch = signal.iirnotch(50 / (0.5 * FS), Q)
    # signal_filt = signal.filtfilt(b_notch, a_notch, sig)

    for channel in sig:
        sig[channel] = signal.filtfilt(b_notch, a_notch, sig[channel].values)
    return sig


def plot_data(df, crop=True):
    if crop:
        df = crop_data(df, CROP_MODE)

    df.plot()
    plt.legend(df.columns)
    plt.grid()
    plt.show()


def plot_spectogram(df, include_signal=True, sample_rate=150000, channel='channel 1', freq_max=None):

    if include_signal:
        time_axis = np.linspace(0, len(df) // sample_rate, num=len(df))
        ax1 = plt.subplot(211)
        plt.grid()
        plt.plot(time_axis, filter_signal(df[channel]))
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        ax2 = plt.subplot(212, sharex=ax1)
        plt.specgram(df[channel], Fs=sample_rate)
        plt.axis(ymax=freq_max)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
    else:
        plt.specgram(df[channel], Fs=sample_rate)
        plt.axis(ymax=freq_max)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
    plt.show()


if __name__ == '__main__':

    # Config 
    SAMPLE_RATE = 150000     # Hz

    CROP_MODE = "Auto"      # Auto or Manual
    CROP_BEFORE = 80000     # samples
    CROP_AFTER = 120000     # samples

    DATA_DELIMITER = ","

    data_folder = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'

    test_file = data_folder + '\\first_test_touch_active_setup2_5\\touch_test_active_setup2_5_B2_v1.csv'
    print(test_file)
    df = pd.read_csv(test_file, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'] )
    # df = filter_signal(df, FS=SAMPLE_RATE, Q=3)
    # print(df.head())
    print(len(df['channel 1'].values))
    print(int(50e-3 * SAMPLE_RATE))
    df_crop = crop_data(df, CROP_MODE)
    plot_spectogram(df)
