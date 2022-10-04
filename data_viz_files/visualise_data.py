import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
from scipy import signal

DATA_DELIMITER = ","
CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3']
SAMPLE_RATE = 150000     # Hz

# Crop limits in seconds
TIME_START = 0
TIME_END = 5

def crop_data(data):
    """Crop data to the range given by the
    global variables CROP_START and CROP_END.
    """
    data_cropped = data[int(TIME_START * SAMPLE_RATE)
                        :int(TIME_END * SAMPLE_RATE)]
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


def filter_signal(sig, freqs, sample_rate=150000):
    """Input an array of frequencies <freqs> to filter out
    with a Q factor given by an array of <Qs>.
    """
    for freq in freqs:
        # We want smaller q-factors for higher frequencies
        q = freq ** (1 / 3)
        b_notch, a_notch = signal.iirnotch(freq / (0.5 * sample_rate), q)
        sig_filtered = sig

        for channel in sig_filtered:
            sig_filtered[channel] = signal.filtfilt(b_notch,
                                                    a_notch,
                                                    sig[channel].values)

    return sig_filtered


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
        plt.specgram(df[channel], sample_rate=sample_rate)
        plt.axis(ymax=freq_max)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency')
    else:
        plt.specgram(df[channel], sample_rate=sample_rate)
        plt.axis(ymax=freq_max)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency')
    plt.show()


def compare_signals(df1, df2,
                    sample_rate=150000,
                    freq_max=60000,
                    time_start=0,
                    time_end=None,
                    plot_diff=False):

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


def plot_data_vs_noiseavg(data_file, channel='channel 1'):
    noise_df = pd.read_csv(
                            f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\base_data\\df_average_noise.csv',
                            delimiter=DATA_DELIMITER)
    df = pd.read_csv(data_file, delimiter=DATA_DELIMITER, names=CHANNEL_NAMES)

    compare_signals(df[channel], noise_df[channel])


def plot_data_subtracted_noise(data_file, channel='channel 1'):
    noise_df = pd.read_csv(
                            f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\base_data\\df_average_noise.csv',
                            delimiter=DATA_DELIMITER)
    df = pd.read_csv(data_file, delimiter=DATA_DELIMITER, names=CHANNEL_NAMES)

    df_sub_noise = noise_df-df
    compare_signals(df[channel], df_sub_noise[channel])


def plot_data_sub_ffts(data_file, channel='channel 1'):
    noise_df = pd.read_csv(
                            f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\base_data\\df_average_noise.csv',
                            delimiter=DATA_DELIMITER)
    df = pd.read_csv(data_file, delimiter=DATA_DELIMITER, names=CHANNEL_NAMES)
    noise_df_fft = scipy.fft.fft(noise_df.values, axis=0)
    df_fft = scipy.fft.fft(df.values, axis=0)
    df_fft_sub_noise_fft = df_fft - noise_df_fft
    df_sub_noise = pd.DataFrame(scipy.fft.ifft(df_fft_sub_noise_fft), columns=CHANNEL_NAMES)
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


if __name__ == '__main__':

    # CONFIG
    SAMPLE_RATE = 150000     # Hz

    # Crop limits in seconds
    TIME_START = 0
    TIME_END = 5

    DATA_DELIMITER = ","

    data_folder = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
    test_file1 = data_folder + ('\\holdfinger_test_active_setup2_5\\'
                                'hold_test_B1_setup2_5_sinus_2khz_10vpp_cyclcount_1_burstp_1s_v1.csv')
    test_file2 = data_folder + ('\\holdfinger_test_active_setup2_5'
                                '\\hold_test_B1_setup2_5_sinus_2khz_10vpp_cyclcount_1_burstp_1s_v2.csv')
    data_file = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_A1_center_v2.csv'
    plot_data_subtracted_noise(data_file)
    # df1 = pd.read_csv(test_file1,
    #                   delimiter=DATA_DELIMITER,
    #                   names=['channel 1', 'channel 2', 'channel 3'])
    # df2 = pd.read_csv(test_file2,
    #                   delimiter=DATA_DELIMITER,
    #                   names=['channel 1', 'channel 2', 'channel 3'])

    # # Filter a signal
    # df1_filtered = filter_signal(df1.copy(),
    #                              freqs=[49, 150, 24000, 48000, 56000],
    #                              sample_rate=SAMPLE_RATE)
    # df2_filtered = filter_signal(df2.copy(),
    #                              freqs=[49, 150, 24000, 48000, 56000],
    #                              sample_rate=SAMPLE_RATE)

    # compare_signals(df1['channel 1'],
    #                 df2['channel 1'],
    #                 sample_rate=SAMPLE_RATE,
    #                 time_start=TIME_START,
    #                 time_end=TIME_END)
