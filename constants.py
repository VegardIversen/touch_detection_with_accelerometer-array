"""GLOBAL CONSTANTS"""
import numpy as np

"""Settings"""
INTERP_FACTOR = 8
SAMPLE_RATE = 150000 * INTERP_FACTOR
CHANNEL_NAMES = np.array(['Sensor 1', 'Sensor 2', 'Sensor 3'])
CHIRP_CHANNEL_NAMES = np.array(['Actuator', 'Sensor 1', 'Sensor 2', 'Sensor 3'])
