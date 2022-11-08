import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
from scipy import signal

from constants import *
from csv_to_df import csv_to_df
from data_processing.preprocessing import crop_data


def plot_fft(df, window=False):
    if isinstance(df, pd.DataFrame):
        if window:
            hamming_window = scipy.signal.hamming(len(df))
            data_fft = scipy.fft.fft(df.values * hamming_window)
        else:
            data_fft = scipy.fft.fft(df.values, axis=0)
    else:
        if window:
            hamming_window = scipy.signal.hamming(len(df))
            data_fft = scipy.fft.fft(df * hamming_window)
        else:
            data_fft = scipy.fft.fft(df, axis=0)
    fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / SAMPLE_RATE)
    plt.grid()
    plt.title('fft of signal')
    plt.xlabel("Frequency [hz]")
    plt.ylabel("Amplitude")
    plt.plot(np.fft.fftshift(fftfreq), 20 * np.log10(np.abs(np.fft.fftshift(data_fft))))
    ax = plt.subplot(1, 1, 1)
    # Only plot positive frequencies
    ax.set_xlim(0)
    plt.show()


def plot_2fft(df1, df2, window=False):
    if window:
        hamming_window1 = scipy.signal.hamming(len(df1))
        data_fft1 = scipy.fft.fft(df1.values * hamming_window1, axis=0)
        hamming_window2 = scipy.signal.hamming(len(df1))
        data_fft2 = scipy.fft.fft(df2.values * hamming_window2, axis=0)
    else:
        data_fft1 = scipy.fft.fft(df1.values, axis=0)
        data_fft2 = scipy.fft.fft(df2.values, axis=0)
    fftfreq1 = scipy.fft.fftfreq(len(data_fft1),  1 / SAMPLE_RATE)
    fftfreq2 = scipy.fft.fftfreq(len(data_fft2),  1 / SAMPLE_RATE)
    plt.grid()
    ax1 = plt.subplot(211)
    plt.grid()
    plt.title(f'fft of {df1.name}')
    plt.xlabel("Frequency [hz]")
    plt.ylabel("Amplitude")
    plt.plot(np.fft.fftshift(fftfreq1), 20 * np.log10(np.abs(np.fft.fftshift(data_fft1))))
    plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.title(f'fft of {df2.name}')
    plt.xlabel("Frequency [hz]")
    plt.ylabel("Amplitude")
    plt.plot(np.fft.fftshift(fftfreq2), 20 * np.log10(np.abs(np.fft.fftshift(data_fft2))))
    plt.tight_layout()
    plt.show()


def plot_data(df, crop=True):
    if crop:
        df = crop_data(df)
    df.plot()
    plt.legend(df.columns)
    plt.grid()
    plt.show()


def plot_spectogram(df,
                    include_signal=True,
                    channel='Sensor 1',
                    freq_max=None):
    vmin = 10 * np.log10(np.max(df)) - 60
    if include_signal:
        time_axis = np.linspace(0, len(df) // SAMPLE_RATE, num=len(df))
        ax1 = plt.subplot(211)
        plt.grid()
        plt.plot(time_axis, df[channel])
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        ax2 = plt.subplot(212, sharex=ax1)
        plt.specgram(df[channel], vmin=vmin)
        plt.axis(ymax=freq_max)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency')
    else:
        plt.specgram(df[channel], vmin=vmin)
        plt.axis(ymax=freq_max)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency')
    plt.show()


def compare_signals(df1: pd.DataFrame or np.ndarray,
                    df2: pd.DataFrame or np.ndarray,
                    df3: pd.DataFrame or np.ndarray,
                    freq_max: int = 40000,
                    nfft: int = 256,
                    plot_diff: bool = False,
                    save: bool = False,
                    filename: str = 'compared_signal.png',
                    plot_1_name: str = 'Sensor 1',
                    plot_2_name: str = 'Sensor 2',
                    plot_3_name: str = 'Sensor 3',
                    sync_time: bool = False):
    """Visually compare two signals, by plotting:
    time signal, spectogram, fft and (optionally) difference signal
    """

    """Change numpy array to dataframe if needed"""
    if isinstance(df1, np.ndarray):
        df1 = pd.DataFrame(df1, columns=['Sensor 1'])
    if isinstance(df2, np.ndarray):
        df2 = pd.DataFrame(df2, columns=['Sensor 2'])
    if isinstance(df3, np.ndarray):
        df3_df = pd.DataFrame(df3, columns=['Sensor 3'])
        df3 = df3_df['Sensor 3']

    """Time signal 1"""
    time_axis_1 = np.linspace(0, len(df1) / SAMPLE_RATE, num=len(df1))
    ax1 = plt.subplot(331)
    plt.grid()
    plt.plot(time_axis_1, df1)
    plt.title(f'{plot_1_name}, time signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')

    """Time signal 2"""
    time_axis_2 = np.linspace(0, len(df2) / SAMPLE_RATE, num=len(df2))
    if sync_time:
        ax2 = plt.subplot(334, sharex=ax1)
    else:
        ax2 = plt.subplot(334)
    plt.grid()
    plt.plot(time_axis_2, df2)
    plt.title(f'{plot_2_name}, time signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')

    """Time signal 3"""
    time_axis_3 = np.linspace(0, len(df3) / SAMPLE_RATE, num=len(df3))
    if sync_time:
        ax3 = plt.subplot(337, sharex=ax1)
    else:
        ax3 = plt.subplot(337)
    plt.grid()
    plt.plot(time_axis_3, df3)
    plt.title(f'{plot_3_name}, time signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')

    """Spectrogram of signal 1"""
    dynamic_range_db = 60
    vmin = 10 * np.log10(np.max(df1)) - dynamic_range_db
    ax3 = plt.subplot(332, sharex=ax1)
    plt.specgram(df1, Fs=SAMPLE_RATE, NFFT=nfft, noverlap=(nfft // 2), vmin=vmin)
    plt.axis(ymax=freq_max)
    plt.title(f'{plot_1_name}, spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()

    """Spectrogram of signal 2"""
    plt.subplot(335, sharex=ax2, sharey=ax3)
    plt.specgram(df2, Fs=SAMPLE_RATE, NFFT=nfft, noverlap=(nfft // 2), vmin=vmin, )
    plt.axis(ymax=freq_max)
    plt.title(f'{plot_2_name}, spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()

    """Spectrogram of signal 3"""
    plt.subplot(338, sharex=ax3, sharey=ax3)
    plt.specgram(df3, Fs=SAMPLE_RATE, NFFT=nfft, noverlap=(nfft // 2), vmin=vmin)
    plt.axis(ymax=freq_max)
    plt.title(f'{plot_3_name}, spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()

    """FFT of signal 1"""
    ax5 = plt.subplot(333)
    ax5.set_xlim(left=0, right=freq_max)
    data_fft = scipy.fft.fft(df1.values, axis=0)
    fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / SAMPLE_RATE)
    data_fft = np.fft.fftshift(data_fft)
    fftfreq = np.fft.fftshift(fftfreq)
    plt.grid()
    plt.title(f'{plot_1_name}, FFT')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.plot(fftfreq, 20 * np.log10(np.abs(data_fft)))
    # data_fft_phase = data_fft
    # data_fft_phase[data_fft_phase < 0.1] = 0
    # plt.plot(fftfreq, (np.angle( data_fft_phase, deg=True)))

    """FFT of signal 2"""
    plt.subplot(336, sharex=ax5, sharey=ax5)
    data_fft = scipy.fft.fft(df2.values, axis=0)
    fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / SAMPLE_RATE)
    data_fft = np.fft.fftshift(data_fft)
    fftfreq = np.fft.fftshift(fftfreq)
    plt.grid()
    plt.title(f'{plot_2_name}, FFT')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.plot(fftfreq, 20 * np.log10(np.abs(data_fft)))

    """FFT of signal 3"""
    plt.subplot(339, sharex=ax5, sharey=ax5)
    data_fft = scipy.fft.fft(df3.values, axis=0)
    fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / SAMPLE_RATE)
    data_fft = np.fft.fftshift(data_fft)
    fftfreq = np.fft.fftshift(fftfreq)
    plt.grid()
    plt.title(f'{plot_3_name}, FFT')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.plot(fftfreq, 20 * np.log10(np.abs(data_fft)))

    """Adjust to look nice in fullscreen view"""
    plt.subplots_adjust(left=0.06, right=0.985,
                        top=0.97, bottom=0.06,
                        hspace=0.3, wspace=0.2)

    if save:
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
    plt.show()

    """Plot difference between signals
    NOTE:   Be careful if using on two different measurements,
            as the time axis might be different
    """
    if plot_diff:
        """Time signal difference"""
        signal_diff = np.abs(df1 - df2)
        ax1 = plt.subplot(311)
        plt.grid()
        plt.plot(time_axis_1, signal_diff)
        plt.title('Difference between signals 1 and 2')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [V]')

        """Spectogram of signal difference"""
        ax2 = plt.subplot(312, sharex=ax1)
        plt.specgram(signal_diff, Fs=SAMPLE_RATE)
        plt.axis(ymax=freq_max)
        plt.title('Spectrogram of the difference between signals 1 and 2')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')

        """FFT of signal difference"""
        ax5 = plt.subplot(313)
        ax5.set_xlim(left=0, right=freq_max)
        data_fft = scipy.fft.fft(signal_diff.values, axis=0)
        fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / SAMPLE_RATE)
        data_fft = np.fft.fftshift(data_fft)
        fftfreq = np.fft.fftshift(fftfreq)
        plt.grid()
        plt.title('FFT of the difference between signals 1 and 2')
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude [dB]")
        plt.plot(fftfreq, 20 * np.log10(np.abs(data_fft)))

        # plt.subplots_adjust(left=0.06, right=0.985,
        #                     top=0.97, bottom=0.06,
        #                     hspace=0.3, wspace=0.2)
        plt.show()


def plot_data_vs_noiseavg(df, channel='Sensor 1'):
    """Plot data vs noise average
    Input:  df with all channels or channel specified by argument
    Output: Plot of data vs noise average
    """
    noise_df = csv_to_df(file_folder='base_data',
                         file_name='df_average_noise')
    compare_signals(df[channel], noise_df[channel])


def plot_data_subtracted_noise(df, channel='Sensor 1'):
    """Plot data subtracted by noise average
    Input:  df with all channels or channel specified by argument
    Output: Plot of data subtracted by noise average
    """
    noise_df = csv_to_df(file_folder='base_data',
                         file_name='df_average_noise')
    df_sub_noise = noise_df - df
    compare_signals(df[channel], df_sub_noise[channel])


def plot_data_sub_ffts(df, channel='Sensor 1'):
    """Plot data subtracted by noise average FFT
    Input:  df with all channels or channel specified by argument
    Output: Plot of data subtracted by noise average
    """
    noise_df = csv_to_df(file_folder='base_data',
                         file_name='df_average_noise')
    noise_df_fft = scipy.fft.fft(noise_df.values, axis=0)
    df_fft = scipy.fft.fft(df.values, axis=0)
    df_fft_sub_noise_fft = df_fft - noise_df_fft
    df_sub_noise = pd.DataFrame(scipy.fft.ifft(df_fft_sub_noise_fft),
                                columns=['Sensor 1', 'Sensor 2', 'Sensor 3'])
    ax1 = plt.subplot(311)
    fftfreq_data = scipy.fft.fftfreq(len(df_fft),  1 / SAMPLE_RATE)
    data_fft = np.fft.fftshift(df_fft)
    fftfreq_data = np.fft.fftshift(fftfreq_data)
    plt.grid()
    plt.title('data')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.plot(fftfreq_data, 20 * np.log10(np.abs(data_fft)))

    plt.subplot(312, sharey=ax1, sharex=ax1)
    fftfreq_data_noise = scipy.fft.fftfreq(len(noise_df_fft),  1 / SAMPLE_RATE)
    data_noise_fft = np.fft.fftshift(noise_df_fft)
    fftfreq_data_noise = np.fft.fftshift(fftfreq_data_noise)
    plt.grid()
    plt.title('noise')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.plot(fftfreq_data_noise, 20 * np.log10(np.abs(data_noise_fft)))

    plt.subplot(313, sharey=ax1, sharex=ax1)
    fftfreq_data_sub_noise = scipy.fft.fftfreq(len(df_fft_sub_noise_fft),  1 / SAMPLE_RATE)
    data_sub_noise_fft = np.fft.fftshift(df_fft_sub_noise_fft)
    fftfreq_data_sub_noise = np.fft.fftshift(fftfreq_data_sub_noise)
    plt.grid()
    plt.title('data-noise')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.plot(fftfreq_data_sub_noise, 20 * np.log10(np.abs(data_sub_noise_fft)))

    plt.tight_layout()
    plt.show()
    # print(df_sub_noise.head())
    # compare_signals(df[channel], noise_df[channel])


def set_fontsizes():
    SMALL_SIZE = 15
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 25

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == '__main__':
    pass
