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


def compress_chirp(measurements: pd.DataFrame):
    """Compresses a chirp with cross correlation."""
    compressed_chirp = measurements.copy()
    for channel in measurements:
        compressed_chirp[channel] = signal.correlate(measurements[channel],
                                                     measurements['chirp'],
                                                     mode='same')
    return compressed_chirp


if __name__ == '__main__':
    pass
