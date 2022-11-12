import numpy as np
import pandas as pd
import scipy.signal as signal

from constants import CHIRP_CHANNEL_NAMES


def avg_waveform(measurements: pd.DataFrame,
                 chirp_range: list) -> pd.DataFrame:
    """Find the average waveforms"""
    avg_waveforms = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES,
                                 data=np.empty((1, 4), np.ndarray))
    for chan in avg_waveforms:
        avg_waveforms.at[0, chan] = np.empty(18750)

    for chan in measurements:
        chirps = np.empty((chirp_range[1], 18750))
        for i, chirp in enumerate(range(chirp_range[0], chirp_range[1])):
            chirps[i] = measurements.at[chirp, chan]
        avg_waveforms.at[0, chan] = np.mean(chirps, axis=0)

    return avg_waveforms


def normalize(data: np.ndarray or pd.DataFrame,
              min: float = 0,
              max: float = 1) -> np.ndarray or pd.DataFrame:
    """Normalize array to be between t_min and t_max"""
    if isinstance(data, pd.DataFrame):
        for chan in data:
            data.at[0, chan] = normalize(data.at[0, chan], min, max)
            data.at[0, chan] = signal.detrend(data.at[0, chan])

    else:
        data = data - np.min(data)
        data = data / np.max(data)
        data = data * (max - min)
        data = data + min
        data = signal.detrend(data)
    return data
