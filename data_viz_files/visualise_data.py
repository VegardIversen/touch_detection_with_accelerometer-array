import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
from scipy import signal
import os
import seaborn as sb
# sb.set_theme(style="darkgrid")
# sb.set(font_scale=12/10)
from matplotlib.widgets import Slider, Button
from constants import *
from csv_to_df import csv_to_df
from data_processing.preprocessing import crop_data, get_phase_and_vph_of_compressed_signal, filter_general, compress_chirp, cut_out_signal
from data_processing.detect_echoes import get_travel_times, get_hilbert_envelope
from objects import Table, Actuator, Sensor
#from setups import Setup2, Setup3, Setup3_2, Setup3_4, Setup6
from data_viz_files.drawing import ax_legend_without_duplicates
import scaleogram as scg
import pywt
import tftb.processing as tfp

def plot_scaleogram(df, sample_rate=150000, channels=['channel 1']):
    chirp = df['wave_gen']
    time_axis = np.linspace(0, len(df) // sample_rate, num=len(df))
    #data_df = df.drop(columns=['wave_gen'])
    data_df = df
    #create subplots for the lenght of channels
    
    if len(channels) == 1:
        channel = channels[0]
        fig, axs = plt.subplots(1, 1, figsize=(20, 10))
        data = data_df[channel].values
        print(f'the mean of the signal is: {np.mean(data)}')
        print(scg.wfun.get_wavlist())
        freq_start = 100  # Hz
        freq_stop = 40000  # Hz
        
        # Compute the associated scale range
        period_start = 1 / freq_stop  # seconds
        period_stop = 1 / freq_start  # seconds
        #scales = scg.periods2scales(np.linspace(period_start, period_stop, num=1000))
        #freqs = np.logspace(np.log10(100), np.log10(40000), num=4000)
        #scales = scg.periods2scales(1 / freqs)
        scg.cws(time_axis, data, wavelet='morl', yaxis='frequency')
        #scg.cws(time_axis, data, wavelet='morl', yaxis='frequency', ylim=[100,50000])
        plt.show()

    else:
        fig, axs = plt.subplots(len(channels), 1, figsize=(20, 10))
        #loop over the channels
        for i, channel in enumerate(channels):
            #get the data for the channel
            data = data_df[channel]
            ax = axs[i]
            #get the scaleogram
            scg.cws(time_axis, data, scales='cmor1-1.5', ylabel=channel, xlabel='frequency [hz]', ax=ax)
            plt.show()

def psudo_wigner_ville_dist(df, channel):
    signal = df[channel].values
    window_size = 1024
    n_frequencies = 512
    n_times = 512
    segment_size = 1024

    # Segment the signal into smaller chunks
    segments = np.array_split(signal, len(signal) // segment_size)

    # Apply transform to each segment separately
    pwvd = []
    for seg in segments:
        pwvd_seg = PseudoWignerVilleDistribution(seg, window=window_size, n_frequencies=n_frequencies, n_times=n_times)
        pwvd.append(pwvd_seg)

    # Concatenate the segments to obtain the full transform
    pwvd = np.concatenate(pwvd, axis=1)
def wigner_ville_dist(df, channel):
    z = df[channel]
    z_cut = cut_out_signal(z, 150e3, 0.0012)
    window_size = 1024
    hop_size = 512
    freq_range = (100, 40000)  # Limit the frequency range to relevant frequencies

# Compute the Wigner-Ville distribution
    
    #wvd = tfp.WignerVilleDistribution(z_cut, window_size, hop_size, freq_range)
    wvd = tfp.WignerVilleDistribution(z_cut)
    #wvd.run()
    #wvd.plot(kind='contour', show_tf=True, title='Wigner-Ville Distribution')
    
def custom_wigner_ville_batch(df, channel, batch_size=100000):
    
    # Define the signal parameters
    duration = 5  # seconds
    sampling_rate = 150000  # Hz
    
    signal_data = df[channel].values
    
    # Define the WVD parameters
    window_size = 0.1  # seconds
    n_fft = int(2 ** np.ceil(np.log2(window_size * sampling_rate)))
    n_overlap = n_fft // 2

    # Calculate the number of batches
    n_samples = signal_data.shape[0]
    n_batches = (n_samples - n_fft) // batch_size + 1

    # Process each batch separately
    wvd_segments = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size + n_fft, n_samples)
        segment = signal_data[start_idx:end_idx]
        wvd_segment = signal.welch(segment, fs=sampling_rate, nperseg=n_fft, noverlap=n_overlap, nfft=n_fft, return_onesided=False, detrend=False)
        wvd_segments.append(wvd_segment[1] * np.conj(wvd_segment[1]))

    # Combine the WVD segments
    wvd = np.concatenate(wvd_segments, axis=0)

    # Reshape wvd to be a 2D array
    n_segments = len(wvd_segments)
    n_freqs = len(np.fft.fftfreq(n_fft+1, 1/sampling_rate))
    wvd_2d = np.real(wvd).reshape((n_segments, n_freqs))

    # Create the meshgrid for the WVD
    time_wvd, freq_wvd = np.meshgrid(np.linspace(0, duration, n_segments+1), np.fft.fftfreq(n_fft+1, 1/sampling_rate))

    # Plot the Wigner-Ville distribution
    plt.pcolormesh(time_wvd, freq_wvd, wvd_2d.T, shading='auto')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.show()

def plot_fft(df, sample_rate=150000, window=False):
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        print('dataframe')
        if window:
            hamming_window = scipy.signal.hamming(len(df))
            data_fft = scipy.fft.fft(df.values * hamming_window)
        else:
            data_fft = scipy.fft.fft(df.values, axis=0)
    else: 
        print('else')
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
    plt.legend(df.columns)
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
    plt.grid()
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
                    channel='channel 1',
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


def compare_signals(df1, df2, df3,
                    freq_max=40000,
                    nfft=256,
                    plot_diff=False,
                    save=False,
                    filename='compared_signal.png',
                    plot_1_name='Signal 1',
                    plot_2_name='Signal 2',
                    plot_3_name='Signal 3',
                    sync_time=False):
    """Visually compare two signals, by plotting:
    time signal, spectogram, fft and (optionally) difference signal
    """

    """Change numpy array to dataframe if needed"""
    if isinstance(df1, np.ndarray):
        df1 = pd.DataFrame(df1, columns=['channel 1'])
    if isinstance(df2, np.ndarray):
        df2 = pd.DataFrame(df2, columns=['channel 2'])
    if isinstance(df3, np.ndarray):
        df3_df = pd.DataFrame(df3, columns=['channel 3'])
        df3 = df3_df['channel 3']

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
    #vmin = 10 * np.log10(np.max(df1)) - dynamic_range_db
    c_min1 = np.min(10*np.log10(df1))
    c_max1 = np.max(10*np.log10(df1))
    ax3 = plt.subplot(332, sharex=ax1)
    #plt.specgram(df1, Fs=SAMPLE_RATE, NFFT=nfft, noverlap=(nfft // 2), vmin=vmin)
    plt.specgram(df1, Fs=SAMPLE_RATE, NFFT=nfft, noverlap=(nfft // 2))
    plt.clim(-80, -140)
    plt.axis(ymax=freq_max)
    plt.title(f'{plot_1_name}, spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()

    """Spectrogram of signal 2"""
    c_min2 = np.min(10*np.log10(df2))
    c_max2 = np.max(10*np.log10(df2))
    plt.subplot(335, sharex=ax2, sharey=ax3)
    #plt.specgram(df2, Fs=SAMPLE_RATE, NFFT=nfft, noverlap=(nfft // 2), vmin=vmin, )
    plt.specgram(df1, Fs=SAMPLE_RATE, NFFT=nfft, noverlap=(nfft // 2))
    plt.clim(-80, -140)
    plt.axis(ymax=freq_max)
    
    plt.title(f'{plot_2_name}, spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()

    """Spectrogram of signal 3"""
    c_min3 = np.min(10*np.log10(df3))
    c_max3 = np.max(10*np.log10(df3))
    plt.subplot(338, sharex=ax3, sharey=ax3)
    #plt.specgram(df3, Fs=SAMPLE_RATE, NFFT=nfft, noverlap=(nfft // 2), vmin=vmin)
    plt.specgram(df1, Fs=SAMPLE_RATE, NFFT=nfft, noverlap=(nfft // 2))
    plt.clim(-80, -140)
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


def plot_data_sub_ffts(df, channel='channel 1'):
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

def plot_vphs(
            folder, 
            setup, 
            threshold1, 
            threshold2,
            ch1='channel 1',
            ch2='channel 2',
            bandwidth=None,
            multichannel=False,
            show=False
            ):

    SETUP = setup
    files = {}
    
    print(folder)
    if isinstance(folder, str):
        for file in os.listdir(ROOT_FOLDER+folder):
            if file.endswith('.csv'):
                print(file)
                measurements = csv_to_df(
                                        file_folder=folder,
                                        file_name=file[:-4],
                                        channel_names=SETUP.get_channel_names()
                                        )
                
                plt.show()
                freq = np.fft.fftfreq(len(measurements), 1/SAMPLE_RATE)
                if bandwidth is not None:
                    measurements_filt = filter_general(
                                                        measurements,
                                                        filtertype='highpass',
                                                        cutoff_highpass=bandwidth[0],
                                                        # cutoff_lowpass=BANDWIDTH[1],
                                                        order=4)
                    files['freq'] = freq[(freq>bandwidth[0]) & (freq<bandwidth[1])]
                else:
                    measurements_filt = measurements
                    files['freq'] = freq
                measurements_filt_comp = compress_chirp(measurements_filt, custom_chirp=None)
                if multichannel:
                    print('multichannel')
                    phase1, vph1, _ = get_phase_and_vph_of_compressed_signal(
                                                                        measurements_filt_comp,
                                                                        ch1='channel 1',
                                                                        ch2='channel 2',
                                                                        bandwidth=bandwidth,
                                                                        distance=np.linalg.norm(SETUP.sensor_1.coordinates-SETUP.sensor_2.coordinates),
                                                                        threshold1=threshold1,
                                                                        threshold2=threshold2,
                                                            )
                    phase2, vph2, _ = get_phase_and_vph_of_compressed_signal(
                                                                        measurements_filt_comp,
                                                                        ch1='channel 2',
                                                                        ch2='channel 3',
                                                                        bandwidth=bandwidth,
                                                                        distance=np.linalg.norm(SETUP.sensor_2.coordinates-SETUP.sensor_3.coordinates),
                                                                        threshold1=threshold1,
                                                                        threshold2=threshold2)
                    phase3, vph3, _ = get_phase_and_vph_of_compressed_signal(
                                                                        measurements_filt_comp,
                                                                        ch1='channel 1',
                                                                        ch2='channel 3',
                                                                        bandwidth=bandwidth,
                                                                        distance=np.linalg.norm(SETUP.sensor_1.coordinates-SETUP.sensor_3.coordinates),
                                                                        threshold1=threshold1,
                                                                        threshold2=threshold2)
                    if vph1.min() < 0:
                        print(f'vp1 < 0 for file {file} and channel {ch1}')
                        vph1 = vph1*-1
                    if vph2.min() < 0:
                        print(f'vp2 < 0 for file {file} and channel {ch2}')
                        vph2 = vph2*-1
                    if vph3.min() < 0:
                        print(f'vp3 < 0 for file {file} and channel {ch2}')
                        vph3 = vph3*-1
                    files[file[:-4] + '_ch1_ch2'] = vph1
                    files[file[:-4] + '_ch2_ch3'] = vph2
                    files[file[:-4] + '_ch1_ch3'] = vph3


                else:
                    if ch1 == 'channel 1':
                        sens1_d = SETUP.sensor_1.coordinates
                    if ch2 == 'channel 2':
                        sens2_d = SETUP.sensor_2.coordinates
                    if ch1 == 'channel 2':
                        sens1_d = SETUP.sensor_2.coordinates
                    if ch2 == 'channel 1':
                        sens2_d = SETUP.sensor_1.coordinates
                    if ch2 == 'channel 3':
                        sens2_d = SETUP.sensor_3.coordinates
                    if ch1 == 'channel 3':
                        sens1_d = SETUP.sensor_3.coordinates
                    print(f'distance is: {np.linalg.norm(sens1_d - sens2_d)}')
                    phase, vph, _ = get_phase_and_vph_of_compressed_signal(
                                                                        measurements_filt_comp,
                                                                        ch1=ch1,
                                                                        ch2=ch2,
                                                                        bandwidth=bandwidth,
                                                                        distance=np.linalg.norm(sens1_d-sens2_d),
                                                                        threshold1=threshold1,
                                                                        threshold2=threshold2)
                    name = file[:-4] + f'_{ch1}_{ch2}'
                    if vph.min() < 0:
                        print(f'vp < 0 for file {file} and channel {ch1}')
                        vph = vph*-1
                    files[name] = vph
        df = pd.DataFrame.from_dict(files)
        dfm = df.melt('freq', var_name='files', value_name='vph')
        #plot = sb.lineplot(data=dfm, x='freq', y='vph', hue='files')
        if show:
            plt.show()
        return dfm
        
                
def plot_plate_speed_sliders_book():
    freqency = np.fft.fftfreq(750000, 1/SAMPLE_RATE)
    E = 3.8*10**9 #Pa
    rho_start = 650 #kg/m^3
    init_rho = 725
    rho_end = 800 #kg/m^3
    Poisson = 0.2
    #plate thickness
    h = 0.02 #m
    #loss factor
    eta_start = 10 * 10**(-3)
    eta_end = 30 * 10**(-3)
    h=0.02
    def f(freq,h, rho, Poisson,E):
        return np.sqrt(1.8*np.sqrt(E/(rho*(1-Poisson**2)))*h*freq)
    fig, ax = plt.subplots()
    line, = ax.plot(freqency[freqency>0], f(freqency[freqency>0], h, init_rho, Poisson, E), lw=2, label='spoon')
    line2, = ax.plot(freqency[freqency>0], f(freqency[freqency>0], h, 550, 0.4, 9.5*10**9), lw=2, label='furugran')
    fig.subplots_adjust(bottom=0.3)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase velocity [m/s]')
    

    # adjust the main plot to make room for the sliders

    #create a slider for the rho value in the range of rho start to rho end
    rho_slider_spoon_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])

    rho_slider_spoon = Slider(rho_slider_spoon_ax, 'rho spoon', rho_start, rho_end, valinit=init_rho)
    rho_slider_furugran_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    
    rho_slider_furugran = Slider(rho_slider_furugran_ax, 'rho furugran', 400, 700, valinit=550)
    #create slider for E value in the range of E start to E end
    E_slider_ax  = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    E_slider_furugran = Slider(E_slider_ax, 'E furugran', 7*10**9, 12*10**9, valinit=9.5*10**9)
    #update the graph when the slider is changed
    def update(val):
        line.set_ydata(f(freqency[freqency>0], h, rho_slider_spoon.val, Poisson, E))

        line2.set_ydata(f(freqency[freqency>0], h, rho_slider_furugran.val, 0.4, E_slider_furugran.val))
        fig.canvas.draw_idle()
    
    #register the update function with the slider
    rho_slider_spoon.on_changed(update)
    E_slider_furugran.on_changed(update)
    rho_slider_furugran.on_changed(update)
    fig.legend()
    plt.grid()
    plt.show()
                
def plot_estimated_reflections_with_sliders(setup, measurements_comp):
    
    setup.draw()
    actuator, sensors = setup.get_objects()
    fig, ax = plt.subplots(3, 1)
    fig.subplots_adjust(bottom=0.1)
    measurements_comp_hilb = get_hilbert_envelope(measurements_comp)
    # """Calculate wave propagation speed"""
    # prop_speed = SETUP.get_propagation_speed(measurements_comp['channel 1'],
    #                                          measurements_comp['channel 2'])
    # prop_speed *= 1.3
    #print(f'Prop speed: {prop_speed}')
    #SETUP.set_propagation_speed(avg_vph)
    
    #prop_speed = SETUP.propagation_speed
    prop_speed = 603.1585605364801
    print(f'Prop speed: {prop_speed}')
    propspeed_slider_ax  = fig.add_axes([0.25, 0.005, 0.65, 0.03])
    propspeed_slider = Slider(propspeed_slider_ax, 'propegation speed', 1, 3000, valinit=600)
    """Calculate wave arrival times"""
    def get_arrivl_times(sensors, actuator, prop_speed):
        arrival_times = np.array([])
        for sensor in sensors:
            time, _ = get_travel_times(actuator[0],
                                    sensor,
                                    prop_speed,
                                    ms=False,
                                    print_info=False,
                                    relative_first_reflection=False)
            time = time + 2.5
            arrival_times = np.append(arrival_times, time)
        """Reshape arrival_times to a 2D array with len(sensor) rows"""
        arrival_times = np.reshape(arrival_times, (len(sensors), len(arrival_times) // len(sensors)))
        arrival_times *= 1000   # Convert to ms
        return arrival_times
    time_axis_corr = np.linspace(0,
                                 1000 * len(measurements_comp) / SAMPLE_RATE,
                                 (len(measurements_comp)))
    arrival_times = get_arrivl_times(sensors, actuator, prop_speed)
    firstref_arr = []
    secondref_arr = []
    first_arrival_arr = []
    for i, sensor in enumerate(sensors):
        ax[i].set_title('Correlation between chirp and channel ' + str(i + 1))
        ax[i].plot(time_axis_corr, measurements_comp['channel ' + str(i + 1)], label='Correlation')
        ax[i].plot(time_axis_corr, measurements_comp_hilb['channel ' + str(i + 1)], label='Hilbert envelope')
        first_arrival = ax[i].axvline(arrival_times[i][0], linestyle='--', color='r', label='Direct wave')
        firstref = [ax[i].axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])]
        secondref = [ax[i].axvline(line, linestyle='--', color='purple', label='2nd reflections')  for line in (arrival_times[i][5:])]
        first_arrival_arr.append(first_arrival)
        firstref_arr.append(firstref)
        secondref_arr.append(secondref)
        ax[i].set_xlabel('Time [ms]')
        ax[i].set_ylabel('Amplitude [V]')
        ax_legend_without_duplicates(ax[i])
        ax[i].grid()

    def update(val):
        arrival_times = get_arrivl_times(sensors, actuator, propspeed_slider.val)
        for i in range(len(sensors)):
            first_arrival_arr[i].set_xdata(arrival_times[i][0])
            for idx, line1 in enumerate(arrival_times[i][1:5]):
                firstref_arr[i][idx].set_xdata(line1)
            for jdx, line2 in enumerate(arrival_times[i][5:]):
                secondref_arr[i][jdx].set_xdata(line2)
                
        fig.canvas.draw_idle()

    propspeed_slider.on_changed(update)
    plt.tight_layout(pad=0.02)
    plt.show()

def inspect_touch(df,savefig=False, file_format='png'):
    """Inspect touch data by plotting the raw from channel 1 and the spectogram of the raw data on this channel.
    share axis between the two plots"""
    fig, axs = plt.subplots(1, 2, figsize=(10, 10), sharex=True)
    #time axis
    time_axis = np.linspace(0, len(df) / SAMPLE_RATE, len(df))
    axs[0].plot(time_axis,df['channel 1'])
    axs[0].set_title('Raw data from channel 1')
    axs[0].set_ylabel('Amplitude [V]')
    axs[0].grid()
    plt.style.use('default')
    spec = axs[1].specgram(df['channel 1'], Fs=SAMPLE_RATE, NFFT=256, noverlap=128)
    spec[3].set_clim(to_dB(np.max(spec[0])) - 60,
                             to_dB(np.max(spec[0])))
    axs[1].set_title('Spectrogram of raw data from channel 1')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Frequency [Hz]')
    plt.tight_layout()
    if savefig:
        fig.savefig(f'../figures/inspect_touch.{file_format}', dpi=300)
    plt.show()

    

def plot_compare_signals_v2(
                        df: pd.DataFrame,
                        nfft: int = 256,
                        sharey: bool = False,
                        freq_max: int = 45000,
                        set_index: int = None,
                        dynamic_range_db: int = 60,
                        log_time_signal: bool = False,
                        compressed_chirps: bool = False,
                        plots_to_plot: list = ['time', 'spectrogram', 'fft']):
                        
                        FIGSIZE_ONE_COLUMN = (4.5, 3)
                        fig, axs = plt.subplots(
                                                nrows=df.shape[1],
                                                ncols=len(plots_to_plot),
                                                figsize=FIGSIZE_ONE_COLUMN,
                                                squeeze=False)
                        compare_signals_v2(fig, axs, [df[channel] for channel in df]
                        )
                        for ax in axs.flatten():
                            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                        subplots_adjust('time', rows=3, columns=1)
                        plt.show()
                    

def compare_signals_v2(fig, axs,
                    data: list,
                    nfft: int = 256,
                    sharey: bool = False,
                    freq_max: int = 45000,
                    set_index: int = None,
                    dynamic_range_db: int = 60,
                    log_time_signal: bool = False,
                    compressed_chirps: bool = False,
                    plots_to_plot: list = ['time', 'spectrogram', 'fft']):
    """Visually compare two signals, by plotting:
    time signal, spectogram, and fft.
    NOTE:   ['time', 'spectrogram', 'fft'] has to be in this order,
            but can be in any combination.
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
            data_fft_dB = to_dB(np.abs(data_fft))
            fftfreq = scipy.fft.fftfreq(len(data_fft_dB),  1 / SAMPLE_RATE)
            data_fft_dB = np.fft.fftshift(data_fft_dB)[len(channel) // 2:]
            fftfreq = np.fft.fftshift(fftfreq)[len(channel) // 2:]
            axs[i, axs_index].sharex(axs[0, axs_index])
            axs[i, axs_index].sharey(axs[0, axs_index])
            axs[i, axs_index].grid()
            axs[i, axs_index].set_title(f'{channel.name}, FFT')
            axs[len(data) - 1, axs_index].set_xlabel("Frequency [kHz]")
            axs[i, axs_index].set_ylabel("Amplitude [dB]")
            axs[i, axs_index].set_xlim(left=0,
                                       right=freq_max / 1000)
            axs[i, axs_index].set_ylim(bottom=-25,
                                       top=80)
            axs[i, axs_index].plot(fftfreq / 1000, data_fft_dB)    

#create a function to db
def to_dB(x):
    return 20 * np.log10(x)
def subplots_adjust(signal_type: list, rows: int = 1, columns: int = 1):
    """Adjust the spacing in plots, based on type of plot and number of grapgs.
    Insert this function before starting a new subplot
    or before the plt.show() function.
    signal_type can be a combination of ['time', 'spectrogram', 'fft'] that is
    defined beforehand.
    """
    if signal_type == ['time'] or ['spectrogram'] or ['fft'] and rows == 1 and columns == 1:
        """Use same spacing for all plots, possibly temporarily"""
        plt.subplots_adjust(left=0.18, right=0.971,
                            top=0.927, bottom=0.152,
                            hspace=0.28, wspace=0.2)
    elif signal_type == ['time'] and rows == 1 and columns == 1:
        plt.subplots_adjust(left=0.12, right=0.98,
                            top=0.9, bottom=0.2,
                            hspace=0.28, wspace=0.2)
    elif signal_type == ['time'] and rows == 2 and columns == 1:
        plt.subplots_adjust(left=0.153, right=0.98,
                            top=0.957, bottom=0.079,
                            hspace=0.237, wspace=0.2)
    elif signal_type == ['time'] and rows == 3 and columns == 1:
        plt.subplots_adjust(left=0.125, right=0.965,
                            top=0.955, bottom=0.07,
                            hspace=0.28, wspace=0.2)
    elif signal_type == ['spectrogram'] and rows == 1 and columns == 1:
        plt.subplots_adjust(left=0.17, right=1,
                            top=0.929, bottom=0.145,
                            hspace=0.28, wspace=0.2)
    elif signal_type == ['spectrogram'] and rows == 2 and columns == 1:
        plt.subplots_adjust(left=0.167, right=1,
                            top=0.955, bottom=0.08,
                            hspace=0.236, wspace=0.2)
    elif signal_type == ['spectrogram'] and rows == 3 and columns == 1:
        plt.subplots_adjust(left=0.125, right=1.05,
                            top=0.955, bottom=0.07,
                            hspace=0.28, wspace=0.2)
    elif signal_type == ['fft'] and rows == 1 and columns == 1:
        plt.subplots_adjust(left=0.121, right=0.98,
                            top=0.926, bottom=0.14,
                            hspace=0.28, wspace=0.15)
    elif signal_type == ['fft'] and rows == 2 and columns == 1:
        plt.subplots_adjust(left=0.125, right=0.957,
                            top=0.955, bottom=0.075,
                            hspace=0.28, wspace=0.2)
    elif signal_type == ['fft'] and rows == 3 and columns == 1:
        plt.subplots_adjust(left=0.125, right=0.95,
                            top=0.955, bottom=0.07,
                            hspace=0.28, wspace=0.2)
    elif signal_type == ['time', 'spectrogram'] and rows == 2 and columns == 1:
        plt.subplots_adjust(left=0.18, right=0.97,
                            top=0.955, bottom=0.0,
                            hspace=0.19, wspace=0.2)
    elif signal_type == ['setup']:
        plt.subplots_adjust(left=0.13, right=0.97,
                            top=0.97, bottom=0.146,
                            hspace=0.28, wspace=0.2)
    else:
        raise ValueError('Signal type or rows and columns not recognized.')

def figure_size_setup(overleaf_size=0.75):
    sb.set(font_scale=12/10)  # font size = 12pt / 10pt/scale = 1.2 times the default size

    # Calculate the column width in inches (assumes page size and margins as specified in the question)
    page_width_mm = 250
    left_margin_mm = 25
    right_margin_mm = 25
    column_width_inches = (page_width_mm - left_margin_mm - right_margin_mm) / 25.4
    # Set the figure height in inches
    figure_height_inches = 6
    # Calculate the figure width in inches as 0.75 of the column width
    
    figure_width_inches = column_width_inches * overleaf_size#0.75

    # Create the figure and set the size
    fig, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches))

    return fig, ax
    
if __name__ == '__main__':
    pass
