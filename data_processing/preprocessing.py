from scipy import signal
import pandas as pd
import numpy as np
from pathlib import Path

from constants import *


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
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.size, 1/sample_rate)
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
    signal_mean = signal.rolling(window = int(rate / 50), min_periods=1, center=True).mean()

    for mean in signal_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    mask_arr = np.array(mask)
    signal_focusing = df.loc[mask_arr]
    return signal_focusing  # , mask_arr


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


def compress_chirp(measurements: pd.DataFrame, custom_chirp: np.ndarray):
    """Compresses a chirp with cross correlation."""
    compressed_chirp = measurements.copy()
    if 'chirp' in measurements.columns:
        for channel in measurements:
            compressed_chirp[channel] = signal.correlate(measurements[channel],
                                                         measurements['chirp'],
                                                         mode='same')
    else:
        for channel in measurements:
            compressed_chirp[channel] = signal.correlate(measurements[channel],
                                                         custom_chirp,
                                                         mode='same')
    return compressed_chirp


if __name__ == '__main__':
    pass
