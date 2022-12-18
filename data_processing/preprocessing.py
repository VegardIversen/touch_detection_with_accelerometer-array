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
                   length_of_signal_seconds: float):
    """Takes in a dataframe and set silence around the signal to zero"""
    """Find the index of the beginning of the signal"""
    length_of_signal_samples = int(length_of_signal_seconds * SAMPLE_RATE)
    peak_index = np.argmax(signals)
    window = signal.tukey(length_of_signal_samples, alpha=0.1)
    window = np.pad(window,
                    (peak_index - int(length_of_signal_samples / 2),
                     len(signals) - peak_index - int(length_of_signal_samples / 2)),
                    'edge')
    _, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.linspace(0, len(window) / SAMPLE_RATE, len(window)), window)
    ax.set_title('Window function')
    ax.set_xlabel('Samples')
    ax.grid()
    signals = signals.multiply(window, axis=0)
    return signals


if __name__ == '__main__':
    pass
