from scipy import signal
import pandas as pd
import numpy as np
from pathlib import Path

from constants import *


import matplotlib.pyplot as plt
import os
"""FILTERING"""


def filter_general(sig, filtertype, cutoff_highpass=20000, cutoff_lowpass=40000, order=4):
    """filtertype: 'highpass', 'lowpass' or 'bandpass"""
    if filtertype == 'highpass':
        sos = signal.butter(order,
                            cutoff_highpass / (0.5 * SAMPLE_RATE),
                            'highpass',
                            output='sos')
    elif filtertype == 'lowpass':
        sos = signal.butter(order,
                            cutoff_lowpass / (0.5 * SAMPLE_RATE),
                            'lowpass',
                            output='sos')
    elif filtertype == 'bandpass':
        sos = signal.butter(order,
                            [cutoff_highpass / (0.5 * SAMPLE_RATE),
                            cutoff_lowpass / (0.5 * SAMPLE_RATE)],
                            'bandpass',
                            output='sos')
    else:
        raise ValueError('Filtertype not recognized')

    sig_filtered = sig.copy()
    if isinstance(sig, pd.DataFrame):
        for channel in sig_filtered:
            sig_filtered[channel] = signal.sosfilt(sos,
                                                   sig[channel].values)
    else:
        sig_filtered = signal.sosfilt(sos,
                                      sig)

    return sig_filtered
def cut_out_signal(df, rate, threshold):
    """
    Inputs audio data in the form of a numpy array. Converts to pandas series
    to find the rolling average and apply the absolute value to the signal at all points.
    
    Additionally takes in the sample rate and threshold (amplitude). Data below the threshold
    will be filtered out. This is useful for filtering out environmental noise from recordings. 
    """
    mask = []
    signal = df.apply(np.abs) # Convert to series to find rolling average and apply absolute value to the signal at all points. 
    signal_mean = signal.rolling(window = int(rate/50), min_periods = 1, center = True).mean() # Take the rolling average of the series within our specified window.
    
    for mean in signal_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    mask_arr = np.array(mask)
    signal_focusing = df.loc[mask_arr]
    return signal_focusing #, mask_arr

def shift_signal(signal, index):
    return np.roll(signal, -index)

def get_first_index_above_threshold(signal, threshold):
    return signal[signal > threshold].index[0]

def match_signals(sig1, sig2, threshold=0.001):
    """Match the signals by shifting the second signal
    until the two signals have the same amplitude at the
    first point above the threshold.
    """
    # Find the first index above the threshold for each signal
    index1 = get_first_index_above_threshold(sig1, threshold)
    index2 = get_first_index_above_threshold(sig2, threshold)
    print('hello')
    print(f'First index above threshold for signal 1: {index1}')
    # Shift the second signal to match the first
    sig2_shifted = shift_signal(sig2, index2 - index1)

    return sig1, sig2_shifted

def get_envelope(df, window_size=300): #300 gives pretty detailed env, 500 gives more rough env
    upper_env = df.rolling(window=window_size).max().shift(int(-window_size/2))
    lower_env = df.rolling(window=window_size).min().shift(int(-window_size/2))
    return upper_env, lower_env

def filter_notches(sig, freqs):
    """Input an array of frequencies <freqs> to filter out
    with a Q factor given by an array of <Qs>.
    """
    for freq in freqs:
        q = freq ** (1 / 3) # We want smaller q-factors for higher frequencies
        b_notch, a_notch = signal.iirnotch(freq / (0.5 * SAMPLE_RATE), q)
        sig_filtered = sig.copy()
        for channel in sig_filtered:
            # Probably a better way to do this than a double for loop
            sig_filtered[channel] = signal.filtfilt(b_notch, a_notch,
                                                    sig[channel].values)
    return sig_filtered


# function that returns the fft of a signal
def fft(signal, sample_rate=150000, shift=True):
    """Returns the fft of a signal"""
    fft = np.fft.fft(signal, n=len(signal)*100)
    freq = np.fft.fftfreq(signal.size*100, 1/sample_rate)
    if shift:
        fft = np.fft.fftshift(fft)
        freq = np.fft.fftshift(freq)
    return fft, freq

#function that returns the inverse fft of a signal
def ifft(fft, sample_rate=150000):
    """Returns the inverse fft of a signal"""
    ifft = np.fft.ifft(fft)
    return ifft
    
"""CROPPING"""

def manual_cut_signal(signal):
    print('Find start of cut')
    plt.plot(signal)
    plt.show()
    start = int(input('Start: '))
    print('Find end of cut')
    plt.plot(signal)
    plt.show()
    end = int(input('End: '))
    print(f'Start: {start}, End: {end}')
    return start, end


def cut_out_signal(df, rate, threshold):
    """
    Inputs audio data in the form of a numpy array. Converts to pandas series
    to find the rolling average and apply the absolute value to the signal at all points.

    Additionally takes in the sample rate and threshold (amplitude). Data below the threshold
    will be filtered out. This is useful for filtering out environmental noise from recordings.
    """
    mask = []
    """Convert to series to find rolling average and apply absolute value to the signal at all points."""
    signal = df.apply(np.abs)
    """Take the rolling average of the series within our specified window."""
    signal_mean = signal.rolling(window = int(rate / 15), min_periods=1, center=True).mean()
    for mean in signal_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    mask_arr = np.array(mask)
    signal_focusing = df.loc[mask_arr]
    return signal_focusing, mask_arr

def cut_out_signal_df(df, rate, threshold):

    df_ret = pd.DataFrame()
    for channel in df:
        df_ret[channel], _ = cut_out_signal(df[channel], rate, threshold)
        
    return df_ret

def crop_data(sig, time_start=None, time_end=None, threshold=0):
    """Crop either DataFrame input, pandas series or a numpy array input.
    NOTE:   Vegard will fix this to work properly,
            but could use some logic from here.
    """
    # Some logic for assuming cropping type and length
    if (time_start or time_start == 0) and not time_end:
        time_end = len(sig) / sample_rate
    elif time_end and not (time_start or time_start == 0):
        time_start = 0

    if (time_start or time_start == 0) and time_end:
        if isinstance(sig, np.ndarray):
            data_cropped = sig[int(time_start * SAMPLE_RATE):int(time_end * SAMPLE_RATE)]
        else:
            data_cropped = sig.loc[time_start * SAMPLE_RATE:time_end * SAMPLE_RATE]
    elif not (time_start or time_start == 0) and not time_end:
        if isinstance(sig, pd.DataFrame):
            data_cropped = sig.loc[(sig > threshold).any(axis=1)]
            data_cropped = sig.loc[(sig.iloc[::-1] > threshold).any(axis=1)]
        else:
            data_cropped = sig.loc[sig > threshold]

    return data_cropped


def crop_data_threshold(data, threshold=0.0006):
    if isinstance(data, pd.DataFrame):
        data_cropped = data.loc[(data > threshold).any(axis=1)]
    else:
        data_cropped = data.loc[data > threshold]
    return data_cropped

def subtract_signals_from_eachother(df1, df2):
    """Subtracts two signals from eachother"""
    df_cut1 = cut_out_signal(df1, 150000, 0.0006)
    df_cut2 = cut_out_signal(df2, 150000, 0.0006)

    df_sub = df1 - df2
    return df_sub

def signal_to_db(df):
    """Converts a signal to decibels"""
    df_db = 20 * np.log10(df)
    return df_db
#create a function that takes in a dataframe and removes silence around the signal
def remove_silence(df, threshold=0.0006):
    """Takes in a dataframe and removes silence around the signal"""
    df = df.loc[(df > threshold).any(axis=1)]
    """Reverse df"""
    rev = df.iloc[::-1].reset_index(drop=True)
    """Remove silence from the end_fre"""
    rev = rev.loc[(rev > threshold).any(axis=1)]
    """Reverse again"""
    df_cleand = rev.iloc[::-1].reset_index(drop=True)
    return df_cleand


"""Compressing chirps"""


def compress_chirp(measurements: pd.DataFrame, custom_chirp: np.ndarray, use_recorded_chirp: bool = True):
    """Compresses a chirp with cross correlation."""
    compressed_chirp = measurements.copy()
    if 'wave_gen' in measurements.columns and use_recorded_chirp:
        for channel in measurements:
            compressed_chirp[channel] = signal.correlate(measurements[channel],
                                                         measurements['wave_gen'],
                                                         mode='same')
    else:
        for channel in measurements:
            compressed_chirp[channel] = signal.correlate(measurements[channel],
                                                         custom_chirp,
                                                         mode='same')
    return compressed_chirp


def inspect_touch_pulse(folder, length_of_touch, threshold=0.0007,
                        channel_names=['channel 1', 'channel 2', 'channel 3', 'WAVE_GEN'],
                        plot=True, number_of_channels=3, sample_rate=150000, plot_mean_only=False):
    
    data_sum = pd.DataFrame(columns=['channel 1', 'channel 2', 'channel 3'])
    data_sum_fft = pd.DataFrame(columns=['channel 1', 'channel 2', 'channel 3'])
    colors = ['red', 'blue', 'green']
    n_files = 0
    ymin=0
    ymax=0
    if plot:
    
        fig, axs = plt.subplots(nrows=number_of_channels, ncols=1, sharex=True, sharey=True)
    if isinstance(folder, str):
        for f in os.listdir(folder):

            if f.endswith(".csv"):
                print(f)
                file_path = os.path.join(folder, f)
                df = pd.read_csv(file_path, names=channel_names )
                if len(df.columns) > 3:
                    df = df.drop(columns=['WAVE_GEN'], axis=1)
                for idx, channel in enumerate(df.columns):
                    start_index = get_first_index_above_threshold(df[channel], threshold)
                    end_index = start_index + length_of_touch
                    data = df[channel][start_index:end_index].values
                    window = np.hamming(length_of_touch)
                    data = data * window
                    data_pad = np.pad(data, ((150000 - len(data))//2, (150000 - len(data))//2), 'constant', constant_values=(0, 0))
                    #print(f'data max: {np.max(data_pad)} for file: {f} on channel: {channel}')
                    if ymin>np.min(data):
                        ymin=np.min(data)
                    if ymax<np.max(data):
                        ymax=np.max(data)
                    if plot and not plot_mean_only:
                        plt.subplot(number_of_channels, 1, idx + 1)
                        axs[idx].plot(data, color=colors[idx])   
                    if data_sum[channel].empty or data_sum[channel].isnull().values.any():
                        data_sum[channel] = data
                    else: 
                        data_sum[channel] = data_sum[channel].add(data)
                    if np.isnan(data).any():
                        print('nan')
                    
            n_files += 1
        data_mean = data_sum/n_files
        #print(data_mean.min().min())
        ymin=data_mean.min().min()
        ymax=data_mean.max().max()
        print(f'ymin: {ymin}, ymax: {ymax}')
        if plot:
            for idx, channel in enumerate(data_mean.columns):
                axs[idx].set_title(f'{channel}, with threshold {threshold} and n_samples {length_of_touch}')
                axs[idx].plot(data_mean[channel],'--', color='black', label='mean')
                axs[idx].legend()
                axs[idx].set_ylim(ymin, ymax)
                
            plt.tight_layout()
            plt.show()

    #elif isinstance(folder, pathlib):

"Get phase of compressed signal"

def get_phase_and_vph_of_compressed_signal(
                                        compressed_signal,ch1='channel 1',
                                        ch2='channel 3',
                                        distance=0.1, 
                                        threshold=2.0,
                                        bandwidth=None,
                                        threshold1=None, 
                                        threshold2=None, 
                                        duration_cut=55, 
                                        n_pi=0, 
                                        detrend=False, 
                                        set_thresh_man=False, 
                                        plot=False,
                                        plot_cut=True):
    """Get phase of compressed signal"""
    if detrend:
        compressed_signal[ch1] = signal.detrend(compressed_signal[ch1])
        compressed_signal[ch2] = signal.detrend(compressed_signal[ch2])
    if threshold1 is None:
        threshold1 = threshold
    if threshold2 is None:
        threshold2 = threshold
    if set_thresh_man:
        print(f'set threshold manually for {ch1}')
        plt.plot(compressed_signal[ch1], label='ch1')
        plt.legend()
        plt.show()
        threshold1 = float(input('threshold: '))
        print(f'set threshold manually for {ch2}')
        plt.plot(compressed_signal[ch2], label='ch2')
        plt.legend()
        plt.show()
        threshold2 = float(input('threshold: '))
    start_index_ch1 = get_first_index_above_threshold(compressed_signal[ch1], threshold1)
    end_index_ch1 = start_index_ch1 + duration_cut
    start_index_ch2 = get_first_index_above_threshold(compressed_signal[ch2], threshold2)
    end_index_ch2 = start_index_ch2 + duration_cut
    if set_thresh_man:
        print(f'start_index_ch1: {start_index_ch1}, end_index_ch1: {end_index_ch1}')
        print(f'start_index_ch2: {start_index_ch2}, end_index_ch2: {end_index_ch2}')
    
    #zero pad signal but keep time position

    cut_ch1 = compressed_signal[ch1][start_index_ch1:end_index_ch1]
    cut_ch2 = compressed_signal[ch2][start_index_ch2:end_index_ch2]
    #add window to signal
    window = np.hamming(duration_cut)
    cut_ch1_win = cut_ch1 * window
    cut_ch2_win = cut_ch2 * window
    # plt.plot(cut_ch1_win, label='ch1')
    # plt.plot(cut_ch2_win*10, label='ch2')
    # plt.legend()
    # plt.show()
    if set_thresh_man or plot_cut:
        plt.plot(cut_ch1_win, label='ch1')
        plt.plot(cut_ch2_win, label='ch2')
        plt.legend()
        plt.show()
    s1t = np.zeros(len(compressed_signal[ch1]))
    s1t[start_index_ch1:end_index_ch1] = cut_ch1_win
    s2t = np.zeros(len(compressed_signal[ch2]))
    s2t[start_index_ch2:end_index_ch2] = cut_ch2_win
    S1f = np.fft.fft(s1t)
    S2f = np.fft.fft(s2t)
    freq = np.fft.fftfreq(len(s1t), 1/SAMPLE_RATE)
    
    if bandwidth is not None:
        freq_cut = freq[(freq>bandwidth[0]) & (freq<bandwidth[1])]
        phase = np.unwrap(np.angle(S2f/S1f)) +n_pi*np.pi
        phase_cut = phase[(freq>bandwidth[0]) & (freq<bandwidth[1])]
        v_ph = -2*np.pi*freq_cut*distance/(phase_cut)
        #time_delay = -phase[(freq>bandwidth[0]) & (freq<bandwidth[1])]/(2*np.pi*freq_cut)
        group_delay = -np.diff(phase_cut)/(2*np.pi*np.diff(freq_cut))
        
    else:
        phase = np.unwrap(np.angle(S2f/S1f)) +n_pi*np.pi
        v_ph = -2*np.pi*freq*distance/(phase)
        #time_delay = -phase/(2*np.pi*freq)
        group_delay = -np.diff(phase)/(2*np.pi*np.diff(freq))
    if plot:
        plt.plot(freq, phase)
        plt.ylabel(ylabel='phase')
        plt.xlabel(xlabel='frequency')
        plt.show()
        plt.plot(freq_cut, v_ph, label='phase velocity')
        plt.ylabel('velocity (m/s)')
        plt.xlabel('frequency (Hz)')
        plt.legend()
        plt.show()
        # plt.plot(freq_cut, time_delay)
        # plt.ylabel('time delay (s)')
        # plt.xlabel('frequency (Hz)')
        # plt.title('phase time delay')
        # plt.show()
        # plt.plot(freq_cut[:-1], group_delay)
        # plt.ylabel('time delay (s)')
        # plt.xlabel('frequency (Hz)')
        # plt.title('group time delay')
        # plt.show()
    if bandwidth is not None:
        return phase_cut, v_ph, freq_cut
    else:
        return phase, v_ph, freq
    

def compress_single_touch(sig, set_threshold_man=False, threshold=None,n_sampl=None, plot=False):
    """Compress signal to touch"""
    if set_threshold_man:
        if n_sampl is not None:
            plt.plot(sig)
            plt.title('set threshold manually')
            plt.show()
            threshold = float(input('threshold: '))
            start_index = get_first_index_above_threshold(sig, threshold)
            end_index = start_index + n_sampl
            direct_signal = sig[start_index:end_index]
        else:
            start, end = manual_cut_signal(sig)
            direct_signal = sig[start:end]
    else:
        if threshold is None or n_sampl is None:
            print('set threshold manually and define the number of samples')
            return -1
        else:
            start_index = get_first_index_above_threshold(sig, threshold)
            end_index = start_index + n_sampl
            direct_signal = sig[start_index:end_index]
    print(f'start index {start_index}')
    compressed = signal.correlate(sig, direct_signal, mode='same')
    if plot:
        plt.plot(compressed)
        plt.show()
    return compressed, start_index

def compress_df_touch(df, set_threshold_man=False, thresholds=[None,None,None],n_sampl=None, plot=False):
    """Compress signal to touch"""
    if len(df.columns) > 3:
        print('removing wavegen column')
        df = df.drop(columns=['wave_gen'], axis=1)
    df_compressed = df.copy()
    start_indexs = []
    for idx, col in enumerate(df.columns):
        df_compressed[col], start_index = compress_single_touch(sig=df[col], set_threshold_man=set_threshold_man, threshold=thresholds[idx], n_sampl=n_sampl, plot=plot)
        start_indexs.append(start_index)

    return df_compressed, start_indexs
        

def cut_out_pulse_wave(df,channels=['channel 1', 'channel 3'], start_stops=None, plot=False):
    if start_stops is None:
        start_stops = []
        for ch in channels:
            start, stop = manual_cut_signal(df[ch])
            start_stops.append((start, stop))
    start1, end1 = start_stops[0]
    start2, end2 = start_stops[1]
    temp_arr = np.zeros((len(df[channels[0]]),len(channels)))
    #filling the array
    temp_arr[start1:end1,0] = df[channels[0]].iloc[start1:end1]
    temp_arr[start2:end2,1] = df[channels[1]].iloc[start2:end2]
    df_sig_only = pd.DataFrame(temp_arr, columns=channels)
    if plot:
        plt.plot(df_sig_only)
        plt.show()
    return df_sig_only

if __name__ == '__main__':
    pass
