import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from setups import Setup
from constants import SAMPLE_RATE

from csv_to_df import csv_to_df
from data_processing.preprocessing import crop_data
from data_processing.detect_echoes import get_hilbert_envelope
from data_viz_files.drawing import plot_legend_without_duplicates


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
    plt.plot(np.fft.fftshift(fftfreq),
             20 * np.log10(np.abs(np.fft.fftshift(data_fft))))
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
    plt.plot(np.fft.fftshift(fftfreq1),
             20 * np.log10(np.abs(np.fft.fftshift(data_fft1))))
    plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.title(f'fft of {df2.name}')
    plt.xlabel("Frequency [hz]")
    plt.ylabel("Amplitude")
    plt.plot(np.fft.fftshift(fftfreq2),
             20 * np.log10(np.abs(np.fft.fftshift(data_fft2))))
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
        plt.subplot(212, sharex=ax1)
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


def compare_signals(fig, axs,
                    data: list,
                    freq_max: int = 40000,
                    nfft: int = 256):
    """Visually compare two signals, by plotting:
    time signal, spectogram, fft and (optionally) difference signal
    """
    for i, channel in enumerate(data):
        if isinstance(data[i], np.ndarray):
            data[i] = pd.Series(data[i], name='Sensor ' + str(i + 1))
        """Time signal"""
        time_axis = np.linspace(start=0,
                                stop=len(data[i]) / SAMPLE_RATE,
                                num=len(data[i]))
        axs[i, 0].sharex(axs[0, 0])
        axs[i, 0].grid()
        axs[i, 0].plot(time_axis, data[i])
        axs[i, 0].set_title(f'{data[i].name}, time signal')
        axs[i, 0].set_xlabel('Time [s]')
        axs[i, 0].set_ylabel('Amplitude [V]')
        axs[i, 0].plot()

        """Spectrogram"""
        spec = axs[i, 1].specgram(data[i],
                                  Fs=SAMPLE_RATE,
                                  NFFT=nfft,
                                  noverlap=(nfft // 2))
        # fig.colorbar(spec[3], ax=axs[i, 1])
        spec[3].set_clim(10 * np.log10(np.max(spec[0])) - 60,
                         10 * np.log10(np.max(spec[0])))
        axs[i, 1].sharex(axs[0, 0])
        axs[i, 1].axis(ymax=freq_max)
        axs[i, 1].set_title(f'{data[i].name}, spectrogram')
        axs[i, 1].set_xlabel('Time [s]')
        axs[i, 1].set_ylabel('Frequency [Hz]')
        axs[i, 1].plot(sharex=axs[0, 0])

        """FFT"""
        axs[i, 2].set_xlim(left=0, right=freq_max)
        data_fft = scipy.fft.fft(data[i].values, axis=0)
        fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / SAMPLE_RATE)
        data_fft = np.fft.fftshift(data_fft)[len(data[i]) // 2:]
        fftfreq = np.fft.fftshift(fftfreq)[len(data[i]) // 2:]
        axs[i, 2].sharex(axs[0, 2])
        axs[i, 2].grid()
        axs[i, 2].set_title(f'{data[i].name}, FFT')
        axs[i, 2].set_xlabel("Frequency [Hz]")
        axs[i, 2].set_ylabel("Amplitude [dB]")
        axs[i, 2].plot(fftfreq, 20 * np.log10(np.abs(data_fft)))
        # data_fft_phase = data_fft
        # data_fft_phase[data_fft_phase < 0.1] = 0
        # plt.plot(fftfreq, (np.angle( data_fft_phase, deg=True)))

    """Adjust to look nice in fullscreen view"""
    plt.subplots_adjust(left=0.06, right=0.985,
                        top=0.97, bottom=0.06,
                        hspace=0.3, wspace=0.2)
    # plt.show()


def specgram_with_lines(setup, measurements_comp, arrival_times, bandwidth):
    """Plot the spectrograms along with lines for expected reflections"""
    for i, sensor in enumerate(setup.sensors):
        plt.subplot(311 + i, sharex=plt.gca())
        plt.title(f'Correlation between chirp and {sensor}')
        spec = plt.specgram(measurements_comp[sensor.name],
                            Fs=SAMPLE_RATE,
                            NFFT=16,
                            noverlap=(16 // 2))
        plt.clim(10 * np.log10(np.max(spec[0])) - 60,
                 10 * np.log10(np.max(spec[0])))
        plt.axis(ymax=bandwidth[1] + 20000)
        plt.title('Spectrogram')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        # plt.colorbar()
        plt.axvline(arrival_times[i][0],
                    linestyle='--',
                    color='r',
                    label='Direct wave')
        [plt.axvline(line,
                     linestyle='--',
                     color='g',
                     label='1st reflections')
         for line in (arrival_times[i][1:5])]
        [plt.axvline(line,
                     linestyle='--',
                     color='purple',
                     label='2nd reflections')
         for line in (arrival_times[i][5:])]
        plt.xlabel('Time [ms]')
        plt.ylabel('Frequency [Hz]')
        plot_legend_without_duplicates()
    plt.subplots_adjust(hspace=0.5)
    # plt.show()


def envelopes_with_lines(setup: Setup,
                         measurements_comp: pd.DataFrame,
                         arrival_times: np.ndarray,
                         bandwidth: tuple):
    """Plot the correlation between the chirp signal and the measured signal"""
    time_axis_corr = np.linspace(-1000 * len(measurements_comp) / SAMPLE_RATE,
                                 1000 * len(measurements_comp) / SAMPLE_RATE,
                                 (len(measurements_comp)))
    measurements_comp_hilb = get_hilbert_envelope(measurements_comp)

    for i, sensor in enumerate(setup.sensors):
        plt.subplot(311 + i, sharex=plt.gca())
        plt.title(f'Correlation between {setup.actuators[0]} and {sensor}')
        plt.plot(time_axis_corr,
                 measurements_comp[sensor.name],
                 label='Correlation')
        plt.plot(time_axis_corr,
                 measurements_comp_hilb[sensor.name],
                 label='Hilbert envelope')
        plt.axvline(arrival_times[i][0],
                    linestyle='--',
                    color='r',
                    label='Direct wave')
        [plt.axvline(line,
                     linestyle='--',
                     color='g',
                     label='1st reflections')
         for line in (arrival_times[i][1:5])]
        [plt.axvline(line,
                     linestyle='--',
                     color='purple',
                     label='2nd reflections')
         for line in (arrival_times[i][5:])]
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [V]')
        plot_legend_without_duplicates()
        plt.grid()
    plt.subplots_adjust(hspace=0.5)
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
    fftfreq_data_sub_noise = scipy.fft.fftfreq(len(df_fft_sub_noise_fft),
                                               1 / SAMPLE_RATE)
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
