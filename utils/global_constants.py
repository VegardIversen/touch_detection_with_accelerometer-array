"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

"""GLOBAL CONSTANTS"""

"""Settings"""
ORIGINAL_SAMPLE_RATE = 150000
INTERPOLATION_FACTOR = 6
SAMPLE_RATE = ORIGINAL_SAMPLE_RATE * INTERPOLATION_FACTOR
CHANNEL_NAMES = ['Sensor 1', 'Sensor 2', 'Sensor 3']
CHIRP_CHANNEL_NAMES = ['Actuator', 'Sensor 1', 'Sensor 2', 'Sensor 3']

"""Sensor int values, e.g. for use with sensors[SENSOR_1]"""
ACTUATOR_1 = 0
SENSOR_1 = 0
SENSOR_2 = 1
SENSOR_3 = 2

"""Saving figures"""
FIGURES_SAVE_PATH = '/home/niklast/Documents/GitHub/Specialization-Project-LaTeX/Figures/'
