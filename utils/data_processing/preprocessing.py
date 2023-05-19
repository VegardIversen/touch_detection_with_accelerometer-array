"""Author: Niklas Strømsnes,
           filter_general() co-authored with Vegard Iversen
Date: 2022-01-09
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from utils.data_processing.detect_echoes import get_envelopes
from utils.data_processing.processing import get_noise_max_value

from utils.global_constants import ORIGINAL_SAMPLE_RATE, SAMPLE_RATE
from utils.data_visualization.visualize_data import plot_filter_response


"""FILTERING"""


def filter_signal(
    signals: pd.DataFrame or np.ndarray,
    filtertype: str,
    critical_frequency: int,
    q: float = 0.05,
    order: int = 4,
    plot_response: bool = False,
    sample_rate: int = ORIGINAL_SAMPLE_RATE,
):
    """filtertype: 'highpass', 'lowpass' or 'bandpass.
    NOTE:   q is a value that determines the width of the flat bandpass,
            and is a value between 0 and 1. order determines the slope.
    """
    if filtertype == "highpass" or filtertype == "lowpass":
        sos = signal.butter(
            order,
            critical_frequency / (0.5 * sample_rate),
            filtertype,
            output="sos",
        )
    elif filtertype == "bandpass":
        sos = signal.butter(
            order,
            [
                critical_frequency * (1 - q) / (0.5 * sample_rate),
                critical_frequency * (1 + q) / (0.5 * sample_rate),
            ],
            "bandpass",
            output="sos",
        )
    else:
        raise ValueError("Filtertype not recognized")

    if isinstance(signals, pd.DataFrame):
        signals_filtered = pd.DataFrame(
            index=signals.index,
            columns=signals.columns,
            data=signal.sosfiltfilt(sos, signals.values, axis=0)
        )
    else:
        signals_filtered = signal.sosfiltfilt(sos, signals, axis=0)

    if plot_response:
        plot_filter_response(sos, critical_frequency, critical_frequency)

    return signals_filtered


def compress_chirps(
    measurements: pd.DataFrame or np.ndarray, custom_reference: np.ndarray = None
):
    """Compresses a chirp with cross correlation"""
    if isinstance(measurements, np.ndarray):
        # Add the measurements array and the actuator signal as columns to the measurements
        measurements = pd.DataFrame(measurements, columns=["Sensor 1"])

    compressed_chirps = measurements.copy()
    if custom_reference is None:
        for channel in measurements:
            compressed_chirps[channel] = signal.correlate(
                measurements[channel], measurements["Actuator"], mode="same"
            )
        return compressed_chirps

    for channel in measurements:
        compressed_chirps[channel] = signal.correlate(
            measurements[channel], custom_reference, mode="same"
        )
    return compressed_chirps


"""CROPPING"""


def crop_data(
    signals: pd.DataFrame or np.ndarray,
    time_start: float = None,
    time_end: float = None,
    sample_rate: int = SAMPLE_RATE,
    apply_window_function: bool = False,
):
    """Crop either DataFrame input, pandas series or a numpy array input."""
    if isinstance(signals, np.ndarray):
        signals_cropped = signals[
            int(time_start * sample_rate) : int(time_end * sample_rate)
        ]
        if apply_window_function:
            window = signal.windows.tukey(len(signals_cropped), alpha=0.05)
            # window = signal.windows.hamming(len(signals_cropped))
    elif isinstance(signals, pd.DataFrame):
        signals_cropped = signals.loc[time_start * sample_rate : time_end * sample_rate]
        if apply_window_function:
            window = signal.windows.tukey(len(signals_cropped), alpha=0.05)
            # window = signal.windows.hamming(len(signals_cropped))
            _, ax = plt.subplots()
            ax.plot(window)
            for channel in signals_cropped:
                signals_cropped[channel] = signals_cropped[channel].values * window
    else:
        # Not implemented for other types
        raise NotImplementedError
    return signals_cropped


def crop_to_signal(
    measurements: pd.DataFrame or np.ndarray,
    padding_percent: float = 0.05,
    threshold: float = None,
):
    """Crop the signal to the first and last value
    above a threshold given by the max value in the noise.
    """
    if isinstance(measurements, pd.DataFrame):
        _, start_index, end_index = crop_to_signal(
            measurements["Sensor 1"],
            padding_percent,
            threshold,
        )
        cropped_measurements = measurements.iloc[start_index:end_index, :]
        cropped_measurements.reset_index(drop=True, inplace=True)
        return cropped_measurements
    # Find the first index where the signal is higher than the threshold
    envelope = get_envelopes(measurements)
    if threshold is None:
        noise_max_value = get_noise_max_value(
            envelope,
            time_window_percentage=0.02,
        )
        threshold = 2 * noise_max_value
    # Find the first index in the signal where the signal is higher than the threshold
    start_index = np.argmax(np.abs(envelope) > threshold)

    # Find the last index where the signal is higher than the threshold
    end_index = len(measurements) - np.argmax(np.abs(envelope)[::-1] > (threshold))

    # Add padding to the signal
    signal_length = end_index - start_index
    start_index -= int(signal_length * padding_percent)
    if start_index < 0:
        start_index = 0
    end_index += int(signal_length * padding_percent)
    if end_index > len(measurements):
        end_index = len(measurements)

    # Crop the signal
    signal = measurements[start_index:end_index]

    return signal, start_index, end_index


def window_signals(
    signals: pd.DataFrame,
    length_of_signal_seconds: float,
    window_function: str = "tukey",
    window_parameter: float = None,
    peak_index: int = None,
    sample_rate: int = SAMPLE_RATE,
):
    """Takes in a dataframe and set silence around the signal to zero."""
    length_of_signal_samples = int(length_of_signal_seconds * sample_rate)
    if peak_index is None:
        peak_index = np.argmax(signals)
    if window_parameter:
        window = signal.get_window(
            (window_function, window_parameter), length_of_signal_samples
        )
    else:
        window = signal.get_window(window_function, length_of_signal_samples)
    window = np.pad(
        window,
        (
            peak_index - int(length_of_signal_samples / 2),
            len(signals) - peak_index - int(length_of_signal_samples / 2),
        ),
        "edge",
    )
    """Plot the window function"""
    # _, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(np.linspace(0, len(window) / sample_rate, len(window)), window)
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


if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be ran from CLI")
