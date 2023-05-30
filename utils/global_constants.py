"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

"""GLOBAL CONSTANTS"""
ASSERTIONS_SHOULD_BE_INCLUDED = False

"""Settings"""
# COMSOL sample rate is 500 kHz, measurements at 150     kHz
ORIGINAL_SAMPLE_RATE = 500e3
INTERPOLATION_FACTOR = 4
SAMPLE_RATE = ORIGINAL_SAMPLE_RATE * INTERPOLATION_FACTOR
CHANNEL_NAMES = ["Sensor 1", "Sensor 2", "Sensor 3"]
CHANNEL_NAMES_WITH_ACTUATOR = ["Actuator", "Sensor 1", "Sensor 2", "Sensor 3"]

"""Sensor int values, e.g. for use with sensors[SENSOR_1]"""
ACTUATOR_1 = 0
SENSOR_1 = 0
SENSOR_2 = 1
SENSOR_3 = 2

"""Coordinates"""
x = 0
y = 1

"""Saving figures"""
FIGURES_SAVE_PATH = (
    "/home/niklast/Documents/GitHub/Master-Thesis-NTNU/figures/python_figures"
)
