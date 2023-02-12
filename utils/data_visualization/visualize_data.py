"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.objects import Sensor
from utils.global_constants import SAMPLE_RATE

from utils.data_processing.detect_echoes import get_envelopes
from utils.data_visualization.drawing import plot_legend_without_duplicates
from utils.data_processing.processing import average_of_signals, variance_of_signals, to_dB


def compare_signals(fig, axs,
                    data: list,
                    nfft: int = 256,
                    sharey: bool = False,
                    freq_max: int = 45000,
                    set_index: int = None,
                    dynamic_range_db: int = 60,
                    log_time_signal: bool = False,
                    compressed_chirps: bool = False,
                    plots_to_plot: list = ['time', 'spectrogram', 'fft']):
    """Intended to be used for plotting time signal, spectogram, and fft
    of all channels, but can be used for individual plots as well.
    NOTE:   ['time', 'spectrogram', 'fft'] has to be in this order,
            but can be in any combination or subset of it.
    """
    for i, channel in enumerate(data):
        """Convert to pd.Series if necessary"""
        if isinstance(channel, np.ndarray):
            channel = pd.Series(channel, name='Sensor ' + str(i + 1))
        if set_index is not None:
            """Plot for instance a spectrogram under a time signal"""
            i = set_index
        if 'time' in plots_to_plot:
            if compressed_chirps:
                time_axis = np.linspace(start=-len(channel) / SAMPLE_RATE,
                                        stop=len(channel) / SAMPLE_RATE,
                                        num=len(channel))
                # axs[i, 0].set_xlim(left=-0.005,
                #                    right=(-0.005 + 0.035))
                axs[i, 0].set_ylabel('Correlation coefficient [-]')
            else:
                time_axis = np.linspace(start=0,
                                        stop=len(channel) / SAMPLE_RATE,
                                        num=len(channel))
                # axs[i, 0].set_xlim(left=signal_start_seconds,
                #                    right=(signal_start_seconds +
                #                           signal_length_seconds))
                axs[i, 0].set_ylabel('Amplitude [V]')
            axs[i, 0].sharex(axs[0, 0])
            if sharey:
                axs[i, 0].sharey(axs[0, 0])
            axs[i, 0].grid()
            if log_time_signal:
                axs[i, 0].plot(time_axis, to_dB(channel))
                axs[i, 0].set_ylim(bottom=np.max(to_dB(channel)) - 60)
            else:
                axs[i, 0].plot(time_axis, channel)
            # axs[i, 0].set_title(f'{channel.name}, time signal')
            axs[len(data) - 1, 0].set_xlabel('Time [s]')
            axs[i, 0].plot()

        if 'spectrogram' in plots_to_plot:
            """Some logic for correct indexing of the axs array"""
            if 'time' in plots_to_plot:
                axs_index = 1
            else:
                axs_index = 0
            if compressed_chirps:
                xextent = (-len(channel) / SAMPLE_RATE,
                           len(channel) / SAMPLE_RATE)
                spec = axs[i, axs_index].specgram(channel,
                                                  Fs=SAMPLE_RATE,
                                                  NFFT=nfft,
                                                  noverlap=(nfft // 2),
                                                  xextent=xextent)
                axs[i, axs_index].set_xlim(left=-0.005,
                                           right=(-0.005 + 0.1))
            else:
                spec = axs[i, axs_index].specgram(channel,
                                                  Fs=SAMPLE_RATE,
                                                  NFFT=nfft,
                                                  noverlap=(nfft // 2))
                # axs[i, axs_index].set_xlim(left=signal_start_seconds,
                #                            right=(signal_start_seconds +
                #                                   signal_length_seconds))
            spec[3].set_clim(to_dB(np.max(spec[0])) - dynamic_range_db,
                             to_dB(np.max(spec[0])))
            if set_index is not None:
                fig.colorbar(spec[3],
                             ax=axs[i, axs_index],
                             pad=0.2,
                             aspect=40,
                             location='bottom')
            else:
                fig.colorbar(spec[3],
                             ax=axs[i, axs_index])
            axs[i, axs_index].sharex(axs[0, 0])
            if sharey:
                axs[i, axs_index].sharey(axs[0, axs_index])
            axs[i, axs_index].axis(ymax=freq_max)
            # axs[i, axs_index].set_title(f'{channel.name}, spectrogram')
            axs[len(data) - 1, axs_index].set_xlabel('Time [s]')
            axs[i, axs_index].set_ylabel('Frequency [Hz]')
            axs[i, axs_index].plot(sharex=axs[0, 0])

        if 'fft' in plots_to_plot:
            """Some logic for correct indexing of the axs array"""
            if ('time' in plots_to_plot) and ('spectrogram' in plots_to_plot):
                axs_index = 2
            elif ('time' in plots_to_plot) ^ ('spectrogram' in plots_to_plot):
                axs_index = 1
            else:
                axs_index = 0
            data_fft = scipy.fft.fft(channel.values, axis=0)
            data_fft_dB = to_dB(np.abs(data_fft))
            fftfreq = scipy.fft.fftfreq(len(data_fft_dB),  1 / SAMPLE_RATE)
            data_fft_dB = np.fft.fftshift(data_fft_dB)[len(channel) // 2:]
            fftfreq = np.fft.fftshift(fftfreq)[len(channel) // 2:]
            axs[i, axs_index].sharex(axs[0, axs_index])
            axs[i, axs_index].sharey(axs[0, axs_index])
            axs[i, axs_index].grid()
            # axs[i, axs_index].set_title(f'{channel.name}, FFT')
            axs[len(data) - 1, axs_index].set_xlabel("Frequency [kHz]")
            axs[i, axs_index].set_ylabel("Amplitude [dB]")
            axs[i, axs_index].set_xlim(left=0,
                                       right=freq_max / 1000)
            axs[i, axs_index].set_ylim(bottom=-25,
                                       top=80)
            axs[i, axs_index].plot(fftfreq / 1000, data_fft_dB)


def wave_statistics(fig, axs, data: pd.DataFrame):
    """Plot average and variance of waveform.
    TODO:   Use confidence interval instead of variance?
    """
    chirp_range = [0,
                   len(data['Sensor 1']) - 1]
    average = average_of_signals(data, chirp_range)
    variance = variance_of_signals(data, chirp_range)
    time_axis = np.linspace(start=0,
                            stop=len(data['Sensor 1'][0]) / SAMPLE_RATE,
                            num=len(data['Sensor 1'][0]))

    # fig.suptitle('Wave statistics')
    for i, channel in enumerate(data.columns[:3]):
        axs[i].plot(time_axis, average[channel][0], label='Average')
        axs[i].plot(time_axis,
                    average[channel][0] + variance[channel][0],
                    label='Average + variance',
                    linestyle='--',
                    color='orange')
        axs[i].plot(time_axis,
                    average[channel][0] - variance[channel][0],
                    label='Average - variance',
                    linestyle='--',
                    color='orange')
        # axs[i].set_title(channel)
        axs[i].set_xlabel('Time [s]')
        axs[i].legend()
        axs[i].grid()


def spectrogram_with_lines(sensor: Sensor,
                           measurements: pd.DataFrame,
                           arrival_times: np.ndarray,
                           nfft: int = 1024,
                           dynamic_range_db: int = 40):
    """Plot the spectrograms along with lines for expected reflections"""
    fig, ax = plt.subplots(figsize=set_window_size())
    spec = plt.specgram(measurements[sensor.name],
                        Fs=SAMPLE_RATE,
                        NFFT=nfft,
                        noverlap=(nfft // 2))
    spec[3].set_clim(to_dB(np.max(spec[0])) - dynamic_range_db,
                     to_dB(np.max(spec[0])))
    # ax.set_title(f'Expected wave arrival times for {sensor.name}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_ylim(0, 5000)
    # ax.set_xlim(2.5, 2.505)
    fig.colorbar(spec[3])
    ax.axvline(arrival_times[0],
               linestyle='--',
               linewidth=2,
               color='#ED217C',
               label='Direct wave')
    [ax.axvline(line,
                linestyle='--',
                linewidth=2,
                color='#DFA06E',
                label='1st reflections')
     for line in (arrival_times[1:5])]
    [ax.axvline(line,
                linestyle='--',
                linewidth=2,
                color='#1b998b',
                label='2nd reflections')
     for line in (arrival_times[5:])]
    # """Use scientific notation"""
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # plot_legend_without_duplicates(placement='upper right')


def envelope_with_lines(sensor: Sensor,
                        measurement: np.ndarray,
                        arrival_times: np.ndarray):
    """Plot the correlation between the chirp signal and the measured signal"""
    time_axis = np.linspace(0,
                            len(measurement) / SAMPLE_RATE,
                            len(measurement))
    measurements_envelope = get_envelopes(measurement)

    _, ax = plt.subplots(figsize=set_window_size())
    ax.plot(time_axis,
            measurement)
    ax.plot(time_axis,
            (measurements_envelope))
    ax.axvline(arrival_times[0],
               linestyle='--',
               color='#ED217C',
               label='Direct wave',
               linewidth=2)
    [ax.axvline(line,
                linestyle='--',
                color='#DFA06E',
                label='1st reflections',
                linewidth=2)
     for line in (arrival_times[1:5])]
    [ax.axvline(line,
                linestyle='--',
                color='#1B998B',
                label='2nd reflections',
                linewidth=2)
     for line in (arrival_times[5:])]
    # ax.set_title(f'Expected wave arrival times for {sensor.name}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude [V]')
    # ax.set_xlim(0, 5)
    """Use scientific notation"""
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plot_legend_without_duplicates(placement='lower right')
    ax.grid()


def plot_filter_response(sos: np.ndarray,
                         cutoff_highpass: int,
                         cutoff_lowpass: int):
    w, h = scipy.signal.sosfreqz(sos, worN=2**15)
    _, ax = plt.subplots(figsize=set_window_size())
    ax.semilogx((SAMPLE_RATE * 0.5 / np.pi) * w, to_dB(abs(h)))
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')
    ax.axvline(cutoff_highpass, color='green')
    ax.axvline(cutoff_lowpass, color='green')


"""Matplotlib settings"""


def set_fontsizes():
    """Inspired by
    https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    """
    SMALL_SIZE = 13
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 18
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def set_window_size(rows: int = 1, cols: int = 1):
    """Set the window size for the plots"""
    figsize: tuple
    if rows == 1 and cols == 1:
        figsize = (5.5, 3.5)
    elif rows == 2 and cols == 1:
        figsize = (5.5, 4)
    elif rows == 3 and cols == 1:
        figsize = (5.5, 4)
    elif rows == 1 and cols == 2:
        figsize = (9, 3)
    elif cols == 2:
        figsize = (9, 3)
    elif cols == 3:
        figsize = (9, 3)
    else:
        raise ValueError('Window size not defined for given dimensions.')
    return figsize


def adjust_plot_margins():
    """Use the same plot adjustments for all figures,
    given that the figsizes are the same.
    """
    plt.subplots_adjust(left=0.175, right=0.98,
                        top=0.935, bottom=0.16,
                        hspace=0.28, wspace=0.2)


if __name__ == '__main__':
    pass
