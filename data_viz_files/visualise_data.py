import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
from scipy import signal
from csv_to_df import csv_to_df


def crop_data(data, time_start=0, time_end=5, sample_rate=150000):
    """Crop data to the range given by the
    global variables CROP_START and CROP_END.
    """
    data_cropped = data[int(time_start * sample_rate):int(time_end * sample_rate)]
    return data_cropped


def crop_data_threshold(data, threshold=0.0006):
    data_cropped = data.loc[(data > threshold).any(axis=1)]
    return data_cropped


def plot_fft(df, sample_rate=150000, window=False):

    if window:
        hamming_window = scipy.signal.hamming(len(df))
        data_fft = scipy.fft.fft(df.values * hamming_window)
    else:
        data_fft = scipy.fft.fft(df.values, axis=0)

    fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / sample_rate)
    plt.grid()
    plt.title('fft of signal')
    plt.xlabel("Frequency [hz]")
    plt.ylabel("Amplitude")
    plt.plot(fftfreq, 20 * np.log10(np.abs(data_fft)))
    ax = plt.subplot(1, 1, 1)
    # Only plot positive frequencies
    ax.set_xlim(0)
    plt.show()


def plot_fft_with_hamming(df, sample_rate=150000):
    plot_fft(df, window=True)


def plot_data(df, crop=True):
    if crop:
        df = crop_data(df)

    df.plot()
    plt.legend(df.columns)
    plt.grid()
    plt.show()


def plot_spectogram(df,
                    include_signal=True,
                    sample_rate=150000,
                    channel='channel 1',
                    freq_max=None):

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


def compare_signals(df1, df2,
                    sample_rate=150000,
                    freq_max=60000,
                    time_start=0,
                    time_end=None,
                    plot_diff=False,
                    save=False,
                    filename='compared_signal.png'):
    """Visually compare two signals, by plotting:
    time signal, spectogram, fft and (optionally) difference signal
    """
    # Time signal 1
    time_axis = np.linspace(0, len(df1) // sample_rate, num=len(df1))
    ax1 = plt.subplot(231)
    ax1.set_xlim(time_start, time_end)
    plt.grid()
    plt.plot(time_axis, df1)
    plt.title('Time signal 1')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')

    # Time signal 2
    plt.subplot(234, sharex=ax1, sharey=ax1)
    plt.grid()
    plt.plot(time_axis, df2)
    plt.title('Time signal 2')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')

    # Spectogram of signal 1
    ax3 = plt.subplot(232, sharex=ax1)
    plt.specgram(df1, Fs=sample_rate)
    plt.axis(ymax=freq_max, xmin=time_start, xmax=time_end)
    plt.title('Spectrogram of signal 1')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    # Spectogram of signal 2
    plt.subplot(235, sharex=ax3, sharey=ax3)
    plt.specgram(df2, Fs=sample_rate)
    plt.axis(ymin=0, ymax=freq_max, xmin=time_start, xmax=time_end)
    plt.title('Spectrogram of signal 2')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    # FFT of signal 1
    ax5 = plt.subplot(233)
    ax5.set_xlim(left=0, right=freq_max)
    data_fft = scipy.fft.fft(crop_data(df1.values), axis=0)
    fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / sample_rate)
    data_fft = np.fft.fftshift(data_fft)
    fftfreq = np.fft.fftshift(fftfreq)
    plt.grid()
    plt.title('FFT of signal 1')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.plot(fftfreq, 20 * np.log10(np.abs(data_fft)))

    # FFT of signal 2
    plt.subplot(236, sharex=ax5, sharey=ax5)
    data_fft = scipy.fft.fft(crop_data(df2.values), axis=0)
    fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / sample_rate)
    data_fft = np.fft.fftshift(data_fft)
    fftfreq = np.fft.fftshift(fftfreq)
    plt.grid()
    plt.title('FFT of signal 2')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.plot(fftfreq, 20 * np.log10(np.abs(data_fft)))

    # Adjust to look nice in fullscreen view
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
        # Time signal difference
        signal_diff = np.abs(df1 - df2)
        time_axis = np.linspace(0, len(df1) // sample_rate, num=len(df1))
        ax1 = plt.subplot(311)
        ax1.set_xlim(time_start, time_end)
        plt.grid()
        plt.plot(time_axis, signal_diff)
        plt.title('Difference between signals 1 and 2')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [V]')

        # Spectogram of signal difference
        ax2 = plt.subplot(312, sharex=ax1)
        plt.specgram(signal_diff, sample_rate=sample_rate)
        plt.axis(ymax=freq_max, xmin=time_start, xmax=time_end)
        plt.title('Spectrogram of the difference between signals 1 and 2')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')

        # FFT of signal difference
        ax5 = plt.subplot(313)
        ax5.set_xlim(left=0, right=freq_max)
        data_fft = scipy.fft.fft(signal_diff.values, axis=0)
        fftfreq = scipy.fft.fftfreq(len(data_fft),  1 / sample_rate)
        data_fft = np.fft.fftshift(data_fft)
        fftfreq = np.fft.fftshift(fftfreq)
        plt.grid()
        plt.title('FFT of the difference between signals 1 and 2')
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude [dB]")
        plt.plot(fftfreq, 20 * np.log10(np.abs(data_fft)))

        plt.subplots_adjust(left=0.06, right=0.985,
                            top=0.97, bottom=0.06,
                            hspace=0.3, wspace=0.2)
        plt.show()


def plot_data_vs_noiseavg(df, channel='channel 1'):
    """Plot data vs noise average
    Input:  df with all channels or channel specified by argument
    Output: Plot of data vs noise average
    """
    noise_df = csv_to_df(file_folder='base_data',
                         file_name='df_average_noise')
    compare_signals(df[channel], noise_df[channel])


def plot_data_subtracted_noise(df, channel='channel 1'):
    """Plot data subtracted by noise average
    Input:  df with all channels or channel specified by argument
    Output: Plot of data subtracted by noise average
    """
    noise_df = csv_to_df(file_folder='base_data',
                         file_name='df_average_noise')
    df_sub_noise = noise_df - df
    compare_signals(df[channel], df_sub_noise[channel])


def plot_data_sub_ffts(df, channel='channel 1', sample_rate=150000):
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
                                columns=['channel 1', 'channel 2', 'channel 3'])
    ax1 = plt.subplot(311)
    fftfreq_data = scipy.fft.fftfreq(len(df_fft),  1 / sample_rate)
    data_fft = np.fft.fftshift(df_fft)
    fftfreq_data = np.fft.fftshift(fftfreq_data)
    plt.grid()
    plt.title('data')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.plot(fftfreq_data, 20 * np.log10(np.abs(data_fft)))

    plt.subplot(312, sharey=ax1, sharex=ax1)
    fftfreq_data_noise = scipy.fft.fftfreq(len(noise_df_fft),  1 / sample_rate)
    data_noise_fft = np.fft.fftshift(noise_df_fft)
    fftfreq_data_noise = np.fft.fftshift(fftfreq_data_noise)
    plt.grid()
    plt.title('noise')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.plot(fftfreq_data_noise, 20 * np.log10(np.abs(data_noise_fft)))

    plt.subplot(313, sharey=ax1, sharex=ax1)
    fftfreq_data_sub_noise = scipy.fft.fftfreq(len(df_fft_sub_noise_fft),  1 / sample_rate)
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


if __name__ == '__main__':
    chirp_df = csv_to_df(file_folder='div_files',
                         file_name='chirp_test_fs_96000_t_max_2s_20000-60000hz_1vpp_1cyc_setup3_method_linear_v3')

    chirp_gen_df = csv_to_df(file_folder='div_files',
                             file_name='chirp_custom_fs_96000_tmax_2_20000-60000_method_linear')

    touch_df = csv_to_df(file_folder='fingernail_test_passive_setup2',
                         file_name='touch_test_fingernail_passive_setup2_place_A1_center_v2')

    stop = 400000
    print(stop)

    time_axis = np.linspace(0, 5, num=len(touch_df['channel 1']))
    b, a = scipy.signal.butter(5, 1000 / (sample_rate / 2), btype='highpass', output='ba')
    filt_touch = scipy.signal.filtfilt(b, a, touch_df['channel 1'])
    plot_spectogram(chirp_gen_df, sample_rate=96000)
    ax1 = plt.subplot((211))
    plt.plot(time_axis, touch_df['channel 1'])
    plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.plot(time_axis, filt_touch)

    # chirp_cropped = crop_data_threshold(chirp_df.iloc[:stop])
    # x = np.correlate(chirp_cropped['channel 1'], chirp_gen_df['channel 1'], 'full')
    # ax1 = plt.subplot(211)
    # plt.plot(chirp_df['channel 1'])
    # plt.subplot(212)
    # plt.plot(chirp_gen_df['channel 1'])
    # b, a = scipy.signal.butter(2,[2,5]*1000//(SAMPLE_RATE*2))
    # filt_chirp = scipy.signal.filtfilt(b,a, x)
    # ax1 = plt.subplot(211)
    # plt.plot(x)
    # plt.subplot(212, sharex=ax1, sharey=ax1)
    # plt.plot(filt_chirp)
    plt.show()
