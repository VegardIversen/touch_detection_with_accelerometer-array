"""Author: Niklas Strømsnes,
           filter_general() co-authored with Vegard Iversen
Date: 2022-01-09
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

from utils.global_constants import SAMPLE_RATE
from utils.data_visualization.visualize_data import (plot_filter_response)


"""FILTERING"""


def filter_general(signals: pd.DataFrame or np.ndarray,
                   filtertype: str,
                   critical_frequency: int,
                   q: float = 0.05,
                   order: int = 4,
                   plot_response: bool = False):
    """filtertype: 'highpass', 'lowpass' or 'bandpass.
    NOTE:   q is a value that determines the width of the flat bandpass,
            and is a value between 0 and 1. order determines the slope.
    """
    if filtertype == 'highpass':
        sos = signal.butter(order,
                            critical_frequency / (0.5 * SAMPLE_RATE),
                            'highpass',
                            output='sos')
    elif filtertype == 'lowpass':
        sos = signal.butter(order,
                            critical_frequency / (0.5 * SAMPLE_RATE),
                            'lowpass',
                            output='sos')
    elif filtertype == 'bandpass':
        sos = signal.butter(order,
                            [critical_frequency * (1 - q) / (0.5 * SAMPLE_RATE),
                             critical_frequency * (1 + q) / (0.5 * SAMPLE_RATE)],
                            'bandpass',
                            output='sos')
    else:
        raise ValueError('Filtertype not recognized')

    signals_filtered = signals.copy()
    if isinstance(signals, pd.DataFrame):
        for channel in signals_filtered:
            signals_filtered[channel] = signal.sosfilt(sos,
                                                       signals[channel].values)
    else:
        signals_filtered = signal.sosfilt(sos,
                                          signals)

    if plot_response:
        plot_filter_response(sos, critical_frequency, critical_frequency)

    return signals_filtered


def compress_chirps(measurements: pd.DataFrame,
                    custom_reference: np.ndarray = None):
    """Compresses a chirp with cross correlation"""
    compressed_chirps = measurements.copy()
    if custom_reference is None:
        for channel in measurements:
            compressed_chirps[channel] = signal.correlate(measurements[channel],
                                                          measurements['Actuator'],
                                                          mode='same')
        return compressed_chirps

    for channel in measurements:
        compressed_chirps[channel] = signal.correlate(measurements[channel],
                                                      custom_reference,
                                                      mode='same')
    return compressed_chirps


"""CROPPING"""


def crop_data(signals: pd.DataFrame or np.ndarray,
              time_start: float = None,
              time_end: float = None):
    """Crop either DataFrame input, pandas series or a numpy array input."""
    if isinstance(signals, np.ndarray):
        signals_cropped = signals[int(time_start * SAMPLE_RATE):
                                  int(time_end * SAMPLE_RATE)]
    else:
        signals_cropped = signals.loc[time_start * SAMPLE_RATE:
                                      time_end * SAMPLE_RATE]
    return signals_cropped


def crop_measurement_to_signal(signal: np.ndarray,
                               std_threshold_multiplier: float = 8):
    """Crop the signal to the first and last value
    above a threshold given by the standard deviation.
    """
    # Find the first index where the signal is higher than five times the standard deviation
    std = np.std(signal)
    threshold = std_threshold_multiplier * std
    start_index = np.argmax(signal > threshold)

    # Find the last index where the signal is higher than five times the standard deviation
    end_index = len(signal) - np.argmax(signal[::-1] > threshold) - 1

    # Add 10% of the signal length to the start and end index
    start_index = int(start_index - 0.1 * (end_index - start_index))
    end_index = int(end_index + 0.1 * (end_index - start_index))

    # Crop the signal
    signal = signal[start_index:end_index]

    return signal


def window_signals(signals: pd.DataFrame,
                   length_of_signal_seconds: float,
                   window_function: str = 'tukey',
                   window_parameter: float = None):
    """Takes in a dataframe and set silence around the signal to zero."""
    length_of_signal_samples = int(length_of_signal_seconds * SAMPLE_RATE)
    peak_index = np.argmax(signals)
    if window_parameter:
        window = signal.get_window((window_function,
                                    window_parameter),
                                   length_of_signal_samples)
    else:
        window = signal.get_window(window_function, length_of_signal_samples)
    window = np.pad(window,
                    (peak_index - int(length_of_signal_samples / 2),
                     len(signals) - peak_index - int(length_of_signal_samples / 2)),
                    'edge')
    """Plot the window function"""
    # _, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(np.linspace(0, len(window) / SAMPLE_RATE, len(window)), window)
    # ax.set_title('Window function')
    # ax.set_xlabel('Samples')
    # ax.grid()
    """Correct rounding errors from padding"""
    if len(window) > len(signals):
        window = np.delete(window, -1)
    elif len(window) < len(signals):
        window = np.append(window, 0)
    signals = signals.multiply(window, axis=0)
    return signals


if __name__ == '__main__':
    pass