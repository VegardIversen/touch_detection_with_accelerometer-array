"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import interpolate
from utils.data_processing.detect_echoes import get_envelopes
from utils.global_constants import (
    CHIRP_CHANNEL_NAMES,
    INTERPOLATION_FACTOR,
    SAMPLE_RATE,
)


def average_of_signals(measurements: pd.DataFrame, chirp_range: list) -> pd.DataFrame:
    """Find the average waveforms"""
    signals_average = pd.DataFrame(
        columns=CHIRP_CHANNEL_NAMES, data=np.empty((1, 4), np.ndarray)
    )
    for channel in signals_average:
        signals_average.at[0, channel] = np.empty(measurements.at[0, channel].size)

    for channel in measurements:
        chirps = np.empty((chirp_range[1], measurements.at[0, channel].size))
        for i, chirp in enumerate(range(chirp_range[0], chirp_range[1])):
            chirps[i] = measurements.at[chirp, channel]
        signals_average.at[0, channel] = np.mean(chirps, axis=0)

    return signals_average


def variance_of_signals(measurements: pd.DataFrame, chirp_range: list) -> pd.DataFrame:
    """Find the variance of the waveforms"""
    signals_variance = pd.DataFrame(
        columns=CHIRP_CHANNEL_NAMES, data=np.empty((1, 4), np.ndarray)
    )
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
            signals[channel] = normalize(signals[channel])
        return signals

    signals = signals - np.min(signals)
    if np.max(signals) != 0:
        signals = signals / np.max(signals)
    else:
        signals = np.zeros(signals.size)
        return signals
    signals = signal.detrend(signals)
    return signals


def interpolate_signal(
    signals: pd.DataFrame or np.ndarray,
) -> pd.DataFrame or np.ndarray:
    """Interpolate waveform to have new_length with numpy"""
    new_length = signals.shape[0] * INTERPOLATION_FACTOR

    if isinstance(signals, np.ndarray):
        sample_axis_original = np.linspace(
            0,
            signals.size,
            signals.size,
        )
        interpolation_function = interpolate.interp1d(
            sample_axis_original,
            signals,
            kind="cubic",
        )
        sample_axis_interpolated = np.linspace(
            0,
            signals.size,
            new_length,
        )
        return interpolation_function(sample_axis_interpolated)

    signals_interpolated = pd.DataFrame(
        data=np.empty((new_length, signals.shape[1]), np.ndarray),
        columns=signals.columns,
        index=range(new_length),
    )
    for channel in signals:
        old_length = signals[channel].size
        sample_axis_original = np.linspace(
            0,
            old_length,
            old_length,
        )
        interpolation_function = interpolate.interp1d(
            sample_axis_original,
            signals[channel],
            kind="cubic",
        )
        sample_axis_interpolated = np.linspace(
            0,
            old_length,
            new_length,
        )
        signals_interpolated[channel] = interpolation_function(sample_axis_interpolated)
    return signals_interpolated


def align_signals_by_correlation(
    signals: pd.DataFrame,
    signals_to_align_with: pd.DataFrame,
    align_to_first_channel: bool = False,
) -> pd.DataFrame:
    """Align signals to signals_to_align_with using their correlation"""
    shifted_signals = signals.copy()
    for channel in signals:
        corr = signal.correlate(
            signals_to_align_with[channel], signals[channel], mode="same"
        )
        delays = np.linspace(
            start=-0.5 * len(corr), stop=0.5 * len(corr), num=len(corr)
        )
        delay = delays[np.argmax(corr)]
        SHIFT_BY = (np.rint(delay)).astype(int)
        shifted_signals[channel] = np.roll(signals[channel], SHIFT_BY)
        shifted_signals[channel] = normalize(shifted_signals[channel])
    return shifted_signals


def align_signals_by_max_value(
    signals: pd.DataFrame,
    signals_to_align_with: pd.DataFrame,
):
    """Align signals to signals_to_align with using their max values"""
    shifted_signals = signals.copy()
    for channel in signals:
        max_index_in_signal1 = np.argmax(signals[channel])
        max_index_in_signal2 = np.argmax(signals_to_align_with[channel])
        SHIFT_BY = (np.rint(max_index_in_signal2 - max_index_in_signal1)).astype(int)
        shifted_signals[channel] = np.roll(signals[channel], SHIFT_BY)
        # Normalize and match the amplitude of signals_to_align_with
        shifted_signals[channel] = normalize(shifted_signals[channel])
        shifted_signals[channel] = (
            shifted_signals[channel]
            * np.max(signals_to_align_with[channel])
            / np.max(shifted_signals[channel])
        )
    return shifted_signals


def align_signals_by_first_peak(
    signals: pd.DataFrame,
    signals_to_align_with: pd.DataFrame,
):
    """Align signals to signals_to_align with using their first peak above 33% of the max value"""
    shifted_signals = signals.copy()
    threshold_signal1 = 0.33 * np.max(signals["Sensor 1"])
    threshold_signal2 = 0.33 * np.max(signals_to_align_with["Sensor 2"])
    first_peak_above_threshold_signal1 = signal.find_peaks(
        signals["Sensor 1"], height=threshold_signal1
    )[0][0]
    first_peak_above_threshold_signal2 = signal.find_peaks(
        signals_to_align_with["Sensor 2"], height=threshold_signal2
    )[0][0]
    SHIFT_BY = (
        np.rint(first_peak_above_threshold_signal2 - first_peak_above_threshold_signal1)
    ).astype(int)
    # Use the same normalization factor for all channels
    normalization_factor = np.max(signals_to_align_with["Sensor 1"]) / np.max(
        shifted_signals["Sensor 1"]
    )
    for channel in signals:
        shifted_signals[channel] = np.roll(signals[channel], SHIFT_BY)
        # Normalize and match the amplitude of signals_to_align_with
        shifted_signals[channel] = shifted_signals[channel] * normalization_factor
    return shifted_signals


def get_noise_max_value(
    envelope: np.ndarray,
    time_window_percentage: float = 0.02,
):
    """Characterize the noise of the measurement
    in terms of the standard deviation and max value
    based on the first 0.1 seconds of the measurement"""
    window_index_end = int(time_window_percentage * len(envelope))
    noise_max_value = np.max(envelope[0:window_index_end])
    return noise_max_value
