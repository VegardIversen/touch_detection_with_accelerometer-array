import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

from global_constants import SAMPLE_RATE


"""FILTERING"""


def compress_chirps(measurements: pd.DataFrame):
    """Compresses a chirp with cross correlation."""
    compressed_chirp = measurements.copy()
    for channel in measurements:
        compressed_chirp[channel] = signal.correlate(measurements[channel],
                                                     measurements['Actuator'],
                                                     mode='same')
    return compressed_chirp


"""CROPPING"""


def crop_data(signals: pd.DataFrame or np.ndarray,
              time_start: float = None,
              time_end: float = None):
    """Crop either DataFrame input, pandas series or a numpy array input.
    NOTE:   Not really finished testing yet.
    """
    """Some logic for assuming cropping type and length"""
    if isinstance(signals, np.ndarray):
        signals_cropped = signals[int(time_start * SAMPLE_RATE):
                                  int(time_end * SAMPLE_RATE)]
    else:
        signals_cropped = signals.loc[time_start * SAMPLE_RATE:
                                      time_end * SAMPLE_RATE]
    return signals_cropped


def window_signals(signals: pd.DataFrame,
                   length_of_signal_seconds: float,
                   threshold: float):
    """Takes in a dataframe and set silence around the signal to zero.
    TODO:   Use an actual window function, e.g. hamming or tukey.
    """
    """Find the index of the beginning of the signal"""
    signal_start_samples = signals.loc[(np.abs(signals) >
                                        threshold).any(axis=1)].index[0]
    length_of_signal_samples = int(length_of_signal_seconds * SAMPLE_RATE)
    with_window = False
    if with_window:
        window = signal.tukey(signal_start_samples + length_of_signal_samples)
        window = np.pad(window, (0, signals.shape[0] - len(window)), 'constant')
        _, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(window)
        ax.set_title('Window function')
        ax.set_xlabel('Samples')
        for channel in signals:
            signals[channel] = signals[channel] * window
    else:
        """Set everything before the signal to zero"""
        signals.loc[:signal_start_samples] = 0
        """Set everything after signal_start + length_of_signal to zero"""
        signals.loc[(signal_start_samples + length_of_signal_samples):] = 0

    signal_start_seconds = signal_start_samples / SAMPLE_RATE
    return signals, signal_start_seconds


if __name__ == '__main__':
    pass
