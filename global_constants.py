"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

"""GLOBAL CONSTANTS"""

"""Settings"""
INTERPOLATION_FACTOR = 8
SAMPLE_RATE = 150000 * INTERPOLATION_FACTOR
CHANNEL_NAMES = ['Sensor 1', 'Sensor 2', 'Sensor 3']
CHIRP_CHANNEL_NAMES = ['Actuator', 'Sensor 1', 'Sensor 2', 'Sensor 3']

"""Figure sizes
TODO:   Replace with set_window_size() in data_visualization.visualize_data.
"""
FIGSIZE_ONE_COLUMN = (4.5, 3)
FIGSIZE_TWO_COLUMNS = (4.5, 4)
FIGSIZE_THREE_COLUMNS = (4.5, 5)

"""Sensor int values, e.g. for use with sensors[SENSOR_1]"""
ACTUATOR_1 = 0
SENSOR_1 = 0
SENSOR_2 = 1
SENSOR_3 = 2

"""Saving figures"""
FIGURES_SAVE_PATH = '/home/niklast/Documents/GitHub/Specialization-Project-LaTeX/Figures/'
