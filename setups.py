"""Each setup has its own child class of the general Setup class.
TODO:   - Add remaining setups.
        - Expand propagation speed function to use all options
          for better estimation.
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

from constants import SAMPLE_RATE
from objects import MirroredSensor, MirroredSource, Table, Actuator, Sensor

from data_viz_files.drawing import plot_legend_without_duplicates
from data_processing.detect_echoes import find_mirrored_source, flip_sensors, flip_sources


class Setup:
    table = Table()
    actuators = np.array([])
    sensors = np.array([])

    def __init__(self):
        raise NotImplementedError("Setup version needs to be specified, e.g. Setup2")

    def draw(self):
        plt.axes()
        self.table.draw()
        [actuator.draw() for actuator in self.actuators]
        [sensor.draw() for sensor in self.sensors]
        plt.axis('scaled')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plot_legend_without_duplicates()
        plt.show()

    def get_objects(self):
        return self.actuators, self.sensors


class Setup2(Setup):
    """Sensors in an 8 cm edge triangle in C2"""
    actuators = np.empty(shape=1, dtype=Actuator)
    sensors = np.empty(shape=3, dtype=Sensor)

    def __init__(self):
        self.actuators[0] = Actuator(coordinates=np.array([1 / 2 * self.table.LENGTH,
                                                           1 / 9 * self.table.WIDTH]))
        self.sensors[0] = Sensor(coordinates=np.array([self.table.LENGTH / 2,
                                                       self.table.WIDTH - 0.082]),
                                 name='Sensor 2')
        SENSOR_1_OFFSET = np.array([-0.08 / 2, -(np.sqrt(0.08 ** 2 - 0.04 ** 2))])
        self.sensors[1] = Sensor(coordinates=(self.sensors[0].coordinates + SENSOR_1_OFFSET),
                                 name='Sensor 1')
        SENSOR_3_OFFSET = np.array([0.08 / 2, -(np.sqrt(0.08 ** 2 - 0.04 ** 2))])
        self.sensors[2] = Sensor(coordinates=(self.sensors[0].coordinates + SENSOR_3_OFFSET),
                                 name='Sensor 3')


    def get_propagation_speed(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        n = len(df1)
        """Convert to df if np.ndarray"""
        if type(df1) == np.ndarray:
            df1 = pd.DataFrame(df1)
        if type(df2) == np.ndarray:
            df2 = pd.DataFrame(df2)

        corr = signal.correlate(df1, df2, mode='same') \
               / np.sqrt(signal.correlate(df2, df2, mode='same')[int(n / 2)]
               * signal.correlate(df1, df1, mode='same')[int(n / 2)])

        delay_arr = np.linspace(-0.5 * n / SAMPLE_RATE, 0.5 * n / SAMPLE_RATE, n)
        delay = delay_arr[np.argmax(corr)]
        distance = np.abs(self.sensor_1.y - self.sensor_2.y)
        propagation_speed = np.round(np.abs(distance / delay), decimals=2)
        return propagation_speed


class Setup3_2(Setup):
    """Sensors in a straight line across the full table"""
    actuators = np.empty(shape=1, dtype=Actuator)
    sensors = np.empty(shape=3, dtype=Sensor)

    def __init__(self):
        self.sensors[0] = Sensor(coordinates=np.array([0.135, 0.305]),
                                 name='Sensor 1')
        self.sensors[1] = Sensor(coordinates=(self.sensors[0].coordinates + np.array([0.267, 0])),
                                 name='Sensor 2')
        self.sensors[2] = Sensor(self.sensors[1].coordinates + np.array([0.267, 0]),
                                 name='Sensor 3')
        self.actuators[0] = Actuator(np.array([self.sensors[0].x / 2,
                                               self.sensors[0].y]))


class Setup3_4(Setup3_2):
    def __init__(self):
        super().__init__()
        """Exact location of the actuator is not remembered,
        but it is one of these two.
        """
        # self.actuator.set_coordinates(np.array([0.01, self.actuator.y]))
        self.sensor_3.set_coordinates(np.array([self.table.LENGTH - 0.009,
                                                self.sensor_3.y]))


class Setup6(Setup):
    def __init__(self):
        self.actuators[0] = Actuator(coordinates=np.array([self.table.LENGTH / 2,
                                                           self.table.WIDTH / 2]),
                                     name='Actuator')
        self.sensors[0] = Sensor(coordinates=np.array([0.489, 0.242]),
                                 name='Sensor 1')


    def get_propagation_speed(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        n = len(df1)
        """Convert to df if np.ndarray"""
        if type(df1) == np.ndarray:
            df1 = pd.DataFrame(df1)
        if type(df2) == np.ndarray:
            df2 = pd.DataFrame(df2)

        corr = signal.correlate(df1, df2, mode='same') \
               / np.sqrt(signal.correlate(df2, df2, mode='same')[int(n / 2)]
               * signal.correlate(df1, df1, mode='same')[int(n / 2)])

        delay_arr = np.linspace(-0.5 * n / SAMPLE_RATE, 0.5 * n / SAMPLE_RATE, n)
        delay = delay_arr[np.argmax(corr)]
        distance = np.linalg.norm(self.actuators[0].coordinates - self.sensors[0].coordinates)
        propagation_speed = np.abs(distance / delay)
        return propagation_speed