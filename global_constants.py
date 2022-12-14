"""GLOBAL CONSTANTS"""

"""Settings"""
INTERPOLATION_FACTOR = 8
SAMPLE_RATE = 150000 * INTERPOLATION_FACTOR
CHANNEL_NAMES = ['Sensor 1', 'Sensor 2', 'Sensor 3']
CHIRP_CHANNEL_NAMES = ['Actuator', 'Sensor 1', 'Sensor 2', 'Sensor 3']

"""Figure sizes"""
FIGSIZE_ONE_SIGNAL = (8, 4)
FIGSIZE_ONE_COLUMN = (8, 9)
FIGSIZE_THREE_COLUMNS = (20, 9)

"""Sensor int values, primarily for use with sensors[SENSOR_*]"""
ACTUATOR_1 = 0
SENSOR_1 = 0
SENSOR_2 = 1
SENSOR_3 = 2
