import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from setups import Setup
from constants import SAMPLE_RATE

from csv_to_df import csv_to_df
from data_processing.preprocessing import crop_data
from data_processing.detect_echoes import get_hilbert
from data_viz_files.drawing import plot_legend_without_duplicates
from data_processing.processing import avg_waveform, var_waveform


def compare_signals(fig, axs,
                    data: list,
                    freq_max: int = 60000,
                    nfft: int = 256,
                    dynamic_range_db: int = 60,
                    log_time_signal: bool = False,
                    sharey: bool = False):
    """Visually compare two signals, by plotting:
    time signal, spectogram, fft and (optionally) difference signal
    """
    for i, channel in enumerate(data):
        """Convert to pd.Series if necessary"""
        if isinstance(data[i], np.ndarray):
            data[i] = pd.Series(data[i], name='Sensor ' + str(i + 1))

        """Time signal"""
        time_axis = np.linspace(start=0,
                                stop=len(data[i]) / SAMPLE_RATE,
                                num=len(data[i]))
        axs[i, 0].sharex(axs[0, 0])
        if sharey:
            axs[i, 0].sharey(axs[0, 0])
        axs[i, 0].grid()
        if log_time_signal:
            axs[i, 0].plot(time_axis, 10 * np.log10(data[i]))
            axs[i, 0].set_ylim(bottom=np.max(10 * np.log10(data[i])) - 60)
        else:
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
        spec[3].set_clim(10 * np.log10(np.max(spec[0])) - dynamic_range_db,
                         10 * np.log10(np.max(spec[0])))
        fig.colorbar(spec[3], ax=axs[i, 1])
        axs[i, 1].sharex(axs[0, 0])
        axs[i, 1].sharey(axs[0, 1])
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
        if sharey:
            axs[i, 2].sharey(axs[0, 2])
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


def set_window_size():
    """Set window size"""
    # Get screen size
    screen_size = plt.rcParams['figure.figsize']
    # Set window size
    plt.rcParams['figure.figsize'] = [screen_size[0] * 1.5,
                                      screen_size[1] * 1.5]


if __name__ == '__main__':
    pass
