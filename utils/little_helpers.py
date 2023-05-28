import numpy as np
import pandas as pd


def to_dB(input: pd.DataFrame or np.ndarray or float):
    """Converts measurements to dB"""
    if isinstance(input, float):
        input = np.array([input])
    # Avoid division by zero
    input[input == 0] = 1e-10
    measurements_dB = 20 * np.log10(input)
    return measurements_dB


def distance_between(object1_coordinates: np.ndarray, object2_coordinates: np.ndarray):
    """Calculate the distance between two objects"""
    distance = np.linalg.norm(object1_coordinates - object2_coordinates)
    return distance
