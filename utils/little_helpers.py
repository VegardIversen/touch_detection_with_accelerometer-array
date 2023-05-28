import numpy as np
import pandas as pd


def to_dB(input_signal: pd.DataFrame or np.ndarray or float):
    """Converts measurements to dB"""
    if isinstance(input_signal, float):
        input_signal = np.array([input_signal])
    # Avoid division by zero
    input_signal[input_signal == 0] = 1e-10
    measurements_dB = 20 * np.log10(input_signal)
    return measurements_dB


def to_linear(input_signals: pd.DataFrame or np.ndarray or float):
    """Converts measurements in dB to linear"""
    if isinstance(input_signals, float):
        input_signals = np.array([input_signals])
    measurements_linear = 10 ** (input_signals / 20)
    return measurements_linear


def distance_between(object1_coordinates: np.ndarray, object2_coordinates: np.ndarray):
    """Calculate the distance between two objects"""
    distance = np.linalg.norm(object1_coordinates - object2_coordinates)
    return distance
