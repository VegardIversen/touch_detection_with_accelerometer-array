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


def plot_fft_with_hamming(df, Fs=150000):
    plot_fft(df, window=True)


def filter_signal(sig, freq, FS=150000, Q=3):
    b_notch, a_notch = signal.iirnotch(freq / (0.5 * FS), Q)
    # signal_filt = signal.filtfilt(b_notch, a_notch, sig)
    sig_filtered = sig

    for channel in sig_filtered:
        sig_filtered[channel] = signal.filtfilt(b_notch, a_notch, sig[channel].values)
    return sig_filtered


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
        plt.plot(time_axis, df[channel])
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        ax2 = plt.subplot(212, sharex=ax1)
        plt.specgram(df[channel], Fs=sample_rate)
        plt.axis(ymax=freq_max)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency')
    else:
        plt.specgram(df[channel], Fs=sample_rate)
        plt.axis(ymax=freq_max)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency')
    plt.show()


def compare_signals(df1, df2, sample_rate=150000, channel='channel 1', freq_max=60000, time_start=0, time_end=None):
    time_axis = np.linspace(0, len(df1) // sample_rate, num=len(df1))
    ax1 = plt.subplot(231)
    ax1.set_xlim(time_start, time_end)
    # time_axis = np.clip(a=time_axis, a_min=0, a_max=1)
    plt.grid()
    plt.plot(time_axis, df1[channel])
    # ax1.set_xlim(left=0, right=1)   
    plt.title('Time signal 1')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    ax2 = plt.subplot(234, sharex=ax1)
    plt.grid()
    plt.plot(time_axis, df2[channel])
    plt.title('Time signal 2')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    ax3 = plt.subplot(232, sharex=ax1)
    plt.specgram(df1[channel], Fs=sample_rate)
    plt.axis(ymax=freq_max, xmin=time_start, xmax=time_end)
    plt.title('Spectrogram of signal 1')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    ax4 = plt.subplot(235, sharex=ax1)
    plt.specgram(df2[channel], Fs=sample_rate)
    plt.axis(ymin=0, ymax=freq_max, xmin=time_start, xmax=time_end)
    plt.title('Spectrogram of signal 2')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    ax5 = plt.subplot(233)
    ax5.set_xlim(left=0, right=freq_max)
    data_fft = scipy.fft.fft(df1[channel].values, axis=0)
    fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / sample_rate)
    plt.grid()
    plt.title('FFT of signal 1')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.plot(fftfreq, 20 * np.log10(np.abs(data_fft)))
    # Only plot positive frequencies

    ax6 = plt.subplot(236, sharex=ax5)
    data_fft = scipy.fft.fft(df2[channel].values, axis=0)
    fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / sample_rate)
    plt.grid()
    plt.title('FFT of signal 2')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.plot(fftfreq, 20 * np.log10(np.abs(data_fft)))
    # Only plot positive frequencies

    plt.tight_layout()
    plt.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.06, hspace=0.3, wspace=0.2)
    plt.show()


if __name__ == '__main__':

    # Config
    SAMPLE_RATE = 150000     # Hz

    CROP_MODE = "Manual"      # Auto or Manual
    CROP_BEFORE = 0     # samples
    CROP_AFTER = 1     # samples

    DATA_DELIMITER = ","

    data_folder = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
    test_file = data_folder + '\\first_test_touch_active_setup2_5\\touch_test_active_setup2_5_B2_v1.csv'
    df = pd.read_csv(test_file, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'] )
    df_filt_50hz = filter_signal(df.copy(), freq=50, FS=SAMPLE_RATE, Q=3)

    # print(len(df['channel 1'].values))
    # print(int(50e-3 * SAMPLE_RATE))
    
    df_crop = crop_data(df, CROP_MODE)

    # plot_spectogram(df)
    # plot_fft(df['channel 1'])
    compare_signals(df, df_filt_50hz, sample_rate=SAMPLE_RATE, channel='channel 1', time_start=1.37, time_end=1.6)
