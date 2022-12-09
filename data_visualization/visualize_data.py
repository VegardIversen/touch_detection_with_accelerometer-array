import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from setups import Setup
from global_constants import SAMPLE_RATE

from csv_to_df import csv_to_df
from data_processing.preprocessing import crop_data
from data_processing.detect_echoes import get_hilbert
from data_visualization.drawing import plot_legend_without_duplicates
from data_processing.processing import avg_waveform, var_waveform


def compare_signals(fig, axs,
                    data: list,
                    freq_max: int = 50000,
                    nfft: int = 256,
                    dynamic_range_db: int = 60,
                    log_time_signal: bool = False,
                    sharey: bool = False,
                    plots_to_plot: list = ['time', 'spectrogram', 'fft'],
                    compressed_chirps: bool = False,
                    signal_start_seconds: float = 0,
                    signal_length_seconds: float = 5):
    """Visually compare two signals, by plotting:
    time signal, spectogram, and fft.
    NOTE:   ['time', 'spectrogram', 'fft'] has to be in this order,
            but can be in any combination.
    """
    for i, channel in enumerate(data):
        """Convert to pd.Series if necessary"""
        if isinstance(channel, np.ndarray):
            channel = pd.Series(channel, name='Sensor ' + str(i + 1))

        if 'time' in plots_to_plot:
            if compressed_chirps:
                time_axis = np.linspace(start=-len(channel) / SAMPLE_RATE,
                                        stop=len(channel) / SAMPLE_RATE,
                                        num=len(channel))
                axs[i, 0].set_xlim(left=-0.005,
                                   right=(-0.005 + 0.035))
                axs[i, 0].set_ylabel('Correlation coefficient [-]')
            else:
                time_axis = np.linspace(start=0,
                                        stop=len(channel) / SAMPLE_RATE,
                                        num=len(channel))
                axs[i, 0].set_xlim(left=signal_start_seconds,
                                   right=(signal_start_seconds +
                                          signal_length_seconds))
                axs[i, 0].set_ylabel('Amplitude [V]')
            axs[i, 0].sharex(axs[0, 0])
            if sharey:
                axs[i, 0].sharey(axs[0, 0])
            axs[i, 0].grid()
            if log_time_signal:
                axs[i, 0].plot(time_axis, 10 * np.log10(channel))
                axs[i, 0].set_ylim(bottom=np.max(10 * np.log10(channel)) - 60)
            else:
                axs[i, 0].plot(time_axis, channel)
            axs[i, 0].set_title(f'{channel.name}, time signal')
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
                axs[i, axs_index].set_xlim(left=signal_start_seconds,
                                           right=(signal_start_seconds +
                                                  signal_length_seconds))
            spec[3].set_clim(10 * np.log10(np.max(spec[0])) - dynamic_range_db,
                             10 * np.log10(np.max(spec[0])))
            fig.colorbar(spec[3], ax=axs[i, axs_index])
            axs[i, axs_index].sharex(axs[0, 0])
            axs[i, axs_index].sharey(axs[0, axs_index])
            axs[i, axs_index].axis(ymax=freq_max)
            axs[i, axs_index].set_title(f'{channel.name}, spectrogram')
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
            data_fft_dB = 20 * np.log10(np.abs(data_fft))
            fftfreq = scipy.fft.fftfreq(len(data_fft_dB),  1 / SAMPLE_RATE)
            data_fft_dB = np.fft.fftshift(data_fft_dB)[len(channel) // 2:]
            fftfreq = np.fft.fftshift(fftfreq)[len(channel) // 2:]
            axs[i, axs_index].sharex(axs[0, axs_index])
            if sharey:
                axs[i, axs_index].sharey(axs[0, axs_index])
            axs[i, axs_index].grid()
            axs[i, axs_index].set_title(f'{channel.name}, FFT')
            axs[len(data) - 1, axs_index].set_xlabel("Frequency [Hz]")
            axs[i, axs_index].set_ylabel("Amplitude [dB]")
            axs[i, axs_index].set_xlim(left=0,
                                       right=freq_max)
            axs[i, axs_index].set_ylim(bottom=-70,
                                       top=120)
            axs[i, axs_index].plot(fftfreq, data_fft_dB)


def wave_statistics(fig, axs, data: pd.DataFrame):
    """Plot average and variance of waveform.
    TODO:   Use confidence interval instead of variance?
    """
    chirp_range = [0,
                   len(data['Sensor 1']) - 1]
    avg = avg_waveform(data, chirp_range)
    var = var_waveform(data, chirp_range)
    time_axis = np.linspace(start=0,
                            stop=len(data['Sensor 1'][0]) / SAMPLE_RATE,
                            num=len(data['Sensor 1'][0]))

    fig.suptitle('Wave statistics')
    for i, chan in enumerate(data.columns[:3]):
        axs[i].plot(time_axis, avg[chan][0], label='Average')
        axs[i].plot(time_axis,
                    avg[chan][0] + var[chan][0],
                    label='Average + variance',
                    linestyle='--',
                    color='orange')
        axs[i].plot(time_axis,
                    avg[chan][0] - var[chan][0],
                    label='Average - variance',
                    linestyle='--',
                    color='orange')
        axs[i].set_title(chan)
        axs[i].set_xlabel('Time [s]')
        axs[i].legend()
        axs[i].grid()


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
                         measurements: pd.DataFrame,
                         arrival_times: np.ndarray):
    """Plot the correlation between the chirp signal and the measured signal"""
    time_axis_corr = np.linspace(-1000 * len(measurements) / SAMPLE_RATE,
                                 1000 * len(measurements) / SAMPLE_RATE,
                                 (len(measurements)))
    measurements_comp_hilb = get_hilbert(measurements)

    for i, sensor in enumerate(setup.sensors):
        plt.subplot(311 + i, sharex=plt.gca())
        plt.title(f'Correlation between {setup.actuators[0]} and {sensor}')
        plt.plot(time_axis_corr,
                 measurements[sensor.name],
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


def set_fontsizes():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def subplots_adjust(signal_type: str, rows: int, columns: int):
    """Adjust the subplots for the 1 column plots.
    Insert this function before starting a new subplot
    or before the plt.show() function.
    Choose between signal_type: ['time', 'spectrogram', 'fft'].
    """
    if signal_type == 'time' and rows == 1 and columns == 1:
        plt.subplots_adjust(left=0.12, right=0.99,
                            top=0.9, bottom=0.2,
                            hspace=0.28, wspace=0.2)
    elif signal_type == 'time' and rows == 3 and columns == 1:
        plt.subplots_adjust(left=0.125, right=0.965,
                            top=0.955, bottom=0.07,
                            hspace=0.28, wspace=0.2)
    elif signal_type == 'spectrogram' and rows == 3 and columns == 1:
        plt.subplots_adjust(left=0.125, right=1.05,
                            top=0.955, bottom=0.07,
                            hspace=0.28, wspace=0.2)
    elif signal_type == 'fft' and rows == 3 and columns == 1:
        plt.subplots_adjust(left=0.125, right=0.95,
                            top=0.955, bottom=0.07,
                            hspace=0.28, wspace=0.2)


if __name__ == '__main__':
    pass
