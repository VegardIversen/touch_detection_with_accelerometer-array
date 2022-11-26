import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import interpolate

from constants import CHIRP_CHANNEL_NAMES


def avg_waveform(measurements: pd.DataFrame,
                 chirp_range: list) -> pd.DataFrame:
    """Find the average waveforms"""
    avg_waveforms = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES,
                                 data=np.empty((1, 4), np.ndarray))
    for chan in avg_waveforms:
        avg_waveforms.at[0, chan] = np.empty(measurements.at[0, chan].size)

    for chan in measurements:
        chirps = np.empty((chirp_range[1], measurements.at[0, chan].size))
        for i, chirp in enumerate(range(chirp_range[0], chirp_range[1])):
            chirps[i] = measurements.at[chirp, chan]
        avg_waveforms.at[0, chan] = np.mean(chirps, axis=0)

    return avg_waveforms


def var_waveform(measurements: pd.DataFrame,
                 chirp_range: list) -> pd.DataFrame:
    """Find the variance of the waveforms"""
    var_waveforms = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES,
                                 data=np.empty((1, 4), np.ndarray))
    for chan in var_waveforms:
        var_waveforms.at[0, chan] = np.empty(measurements.at[0, chan].size)

    for chan in measurements:
        chirps = np.empty((chirp_range[1], measurements.at[0, chan].size))
        for i, chirp in enumerate(range(chirp_range[0], chirp_range[1])):
            chirps[i] = measurements.at[chirp, chan]
        var_waveforms.at[0, chan] = np.var(chirps, axis=0)

    return var_waveforms


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
        if np.max(data) != 0:
            data = data / np.max(data)
        else:
            data = np.zeros(data.size)
            return data
        data = data * (max - min)
        data = data + min
        data = signal.detrend(data)
    return data


def interpolate_waveform(waveform: np.ndarray,
                         new_length: int) -> np.ndarray:
    """Interpolate waveform to have new_length with numpy"""
    old_length = waveform.size
    x = np.linspace(0, old_length, old_length)
    f = interpolate.interp1d(x, waveform)
    x_new = np.linspace(0, old_length, new_length)
    waveform = f(x_new)
    return waveform


def correct_drift(data_split: pd.DataFrame,
                  data_to_sync_with: pd.DataFrame,
                  n_interp: int) -> pd.DataFrame:
    for chan in data_split:
        for chirp in range(len(data_split['Sensor 1'])):
            """Interpolate the signals for better resolution and drift correction"""
            data_split.at[chirp, chan] = interpolate_waveform(data_split.at[chirp, chan],
                                                              new_length=n_interp)
            corr = signal.correlate(data_to_sync_with[chan][0],
                                    data_split[chan][chirp],
                                    mode='same')
            delay_arr = np.linspace(start=-0.5 * n_interp,
                                    stop=0.5 * n_interp,
                                    num=n_interp)
            delay = delay_arr[np.argmax(corr)]
            SHIFT_BY = (np.rint(delay)).astype(int)
            data_split.at[chirp, chan] = np.roll(data_split.at[chirp, chan],
                                                 SHIFT_BY)
            data_split.at[chirp, chan] = normalize(data_split.at[chirp, chan], 0, 1)
    return data_split
