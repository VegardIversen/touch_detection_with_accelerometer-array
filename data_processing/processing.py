"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import interpolate

from global_constants import CHIRP_CHANNEL_NAMES, INTERPOLATION_FACTOR


def average_of_signals(measurements: pd.DataFrame,
                       chirp_range: list) -> pd.DataFrame:
    """Find the average waveforms"""
    signals_average = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES,
                                   data=np.empty((1, 4), np.ndarray))
    for channel in signals_average:
        signals_average.at[0, channel] = np.empty(measurements.at[0, channel].size)

    for channel in measurements:
        chirps = np.empty((chirp_range[1], measurements.at[0, channel].size))
        for i, chirp in enumerate(range(chirp_range[0], chirp_range[1])):
            chirps[i] = measurements.at[chirp, channel]
        signals_average.at[0, channel] = np.mean(chirps, axis=0)

    return signals_average


def variance_of_signals(measurements: pd.DataFrame,
                        chirp_range: list) -> pd.DataFrame:
    """Find the variance of the waveforms"""
    signals_variance = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES,
                                    data=np.empty((1, 4), np.ndarray))
    for chan in signals_variance:
        signals_variance.at[0, chan] = np.empty(measurements.at[0, chan].size)

    for chan in measurements:
        chirps = np.empty((chirp_range[1], measurements.at[0, chan].size))
        for i, chirp in enumerate(range(chirp_range[0], chirp_range[1])):
            chirps[i] = measurements.at[chirp, chan]
        signals_variance.at[0, chan] = np.var(chirps, axis=0)

    return signals_variance


def normalize(signals: np.ndarray or pd.DataFrame) -> np.ndarray or pd.DataFrame:
    """Normalize array to be between t_min and t_max"""
    if isinstance(signals, pd.DataFrame):
        for channel in signals:
            signals.at[0, channel] = normalize(signals.at[0, channel])
            signals.at[0, channel] = signal.detrend(signals.at[0, channel])
    else:
        signals = signals - np.min(signals)
        if np.max(signals) != 0:
            signals = signals / np.max(signals)
        else:
            signals = np.zeros(signals.size)
            return signals
        signals = signal.detrend(signals)
    return signals


def interpolate_waveform(signals: pd.DataFrame) -> pd.DataFrame:
    """Interpolate waveform to have new_length with numpy"""
    new_length = signals.shape[0] * INTERPOLATION_FACTOR
    # measurements_interp = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES,
    #                                    data=np.empty((new_length, measurements.shape[1]), np.ndarray))
    signals_interpolated = pd.DataFrame(data=np.empty((new_length,
                                                       signals.shape[1]),
                                                      np.ndarray),
                                        columns=signals.columns,
                                        index=range(new_length))
    for channel in signals:
        old_length = signals[channel].size
        x = np.linspace(0, old_length, old_length)
        f = interpolate.interp1d(x, signals[channel], kind='cubic')
        x_new = np.linspace(0, old_length, new_length)
        signals_interpolated[channel] = f(x_new)
    return signals_interpolated


def correct_drift(data_split: pd.DataFrame,
                  data_to_sync_with: pd.DataFrame) -> pd.DataFrame:
    """Align each chrip in a recording better when split into equal lengths.
    NOTE:   Also normalizes output.
    """
    for channel in data_split:
        for chirp in range(len(data_split['Sensor 1'])):
            corr = signal.correlate(data_to_sync_with[channel][0],
                                    data_split[channel][chirp],
                                    mode='same')
            delays = np.linspace(start=-0.5 * len(corr),
                                 stop=0.5 * len(corr),
                                 num=len(corr))
            delay = delays[np.argmax(corr)]
            SHIFT_BY = (np.rint(delay)).astype(int)
            data_split.at[chirp, channel] = np.roll(data_split.at[chirp, channel],
                                                    SHIFT_BY)
            data_split.at[chirp, channel] = normalize(data_split.at[chirp, channel])
    return data_split


def to_dB(measurements: pd.DataFrame or np.ndarray):
    """Converts measurements to dB"""
    measurements_dB = 10 * np.log10(measurements)
    return measurements_dB
