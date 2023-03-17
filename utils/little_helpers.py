import numpy as np
import pandas as pd


def to_dB(measurements: pd.DataFrame or np.ndarray):
    """Converts measurements to dB"""
    measurements_dB = 10 * np.log10(measurements)
    return measurements_dB


def distance_between(object1_coordinates: np.ndarray, object2_coordinates: np.ndarray):
    """Calculate the distance between two objects"""
    distance = np.linalg.norm(object1_coordinates - object2_coordinates)
    return distance
