import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import interpolate

from global_constants import CHIRP_CHANNEL_NAMES, INTERPOLATION_FACTOR


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


def normalize(data: np.ndarray or pd.DataFrame) -> np.ndarray or pd.DataFrame:
    """Normalize array to be between t_min and t_max"""
    if isinstance(data, pd.DataFrame):
        for chan in data:
            data.at[0, chan] = normalize(data.at[0, chan])
            data.at[0, chan] = signal.detrend(data.at[0, chan])
    else:
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        else:
            data = np.zeros(data.size)
            return data
        data = signal.detrend(data)
    return data


def interpolate_waveform(measurements: pd.DataFrame) -> pd.DataFrame:
    """Interpolate waveform to have new_length with numpy"""
    new_length = measurements.shape[0] * INTERPOLATION_FACTOR
    measurements_interp = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES,
                                       data=np.empty((new_length, measurements.shape[1]), np.ndarray))
    for chan in measurements:
        old_length = measurements[chan].size
        x = np.linspace(0, old_length, old_length)
        f = interpolate.interp1d(x, measurements[chan], kind='cubic')
        x_new = np.linspace(0, old_length, new_length)
        measurements_interp[chan] = f(x_new)
    return measurements_interp


def correct_drift(data_split: pd.DataFrame,
                  data_to_sync_with: pd.DataFrame) -> pd.DataFrame:
    """Align each chrip in a recording better when split into equal lengths.
    NOTE:   Also normalizes output and crops each chirp for faster processing.
    """
    for chan in data_split:
        for chirp in range(len(data_split['Sensor 1'])):
            corr = signal.correlate(data_to_sync_with[chan][0],
                                    data_split[chan][chirp],
                                    mode='same')
            delay_arr = np.linspace(start=-0.5 * len(corr),
                                    stop=0.5 * len(corr),
                                    num=len(corr))
            delay = delay_arr[np.argmax(corr)]
            SHIFT_BY = (np.rint(delay)).astype(int)
            data_split.at[chirp, chan] = np.roll(data_split.at[chirp, chan],
                                                 SHIFT_BY)
            data_split.at[chirp, chan] = normalize(data_split.at[chirp, chan])
    return data_split


def to_dB(measurements: pd.DataFrame or np.ndarray):
    """Converts measurements to dB"""
    measurements_dB = 10 * np.log10(measurements)
    return measurements_dB
