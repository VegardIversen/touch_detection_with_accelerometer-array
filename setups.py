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
from objects import Table, Actuator, Sensor

from data_viz_files.drawing import plot_legend_without_duplicates


class Setup:
    table = Table()
    actuators: np.ndarray
    sensors: np.ndarray

    def __init__(self):
        raise NameError("Setup version needs to be specified")

    def draw(self):
        plt.axes()
        self.table.draw()
        [actuator.draw() for actuator in self.actuators]
        [sensor.draw() for sensor in self.sensors]
        plt.axis('scaled')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plot_legend_without_duplicates()
        # plt.show()

    def get_objects(self):
        return self.actuators, self.sensors


class Setup2(Setup):
    """Sensors in an 8 cm edge triangle in C2"""
    actuators = np.empty(shape=1, dtype=Actuator)
    sensors = np.empty(shape=3, dtype=Sensor)
    actuators[0] = Actuator(coordinates=np.array([1 / 2 * Table.LENGTH,
                                                  1 / 9 * Table.WIDTH]))
    sensors[1] = Sensor(coordinates=np.array([Table.LENGTH / 2,
                                              Table.WIDTH - 0.082]),
                        name='Sensor 2')
    SENSOR_1_OFFSET = np.array([-0.08 / 2, -(np.sqrt(0.08 ** 2 - 0.04 ** 2))])
    sensors[0] = Sensor(coordinates=(sensors[1].coordinates + SENSOR_1_OFFSET),
                        name='Sensor 1')
    SENSOR_3_OFFSET = np.array([0.08 / 2, -(np.sqrt(0.08 ** 2 - 0.04 ** 2))])
    sensors[2] = Sensor(coordinates=(sensors[1].coordinates + SENSOR_3_OFFSET),
                        name='Sensor 3')

    def __init__(self):
        pass

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

        corr = signal.correlate(df1, df2, mode='same') / \
            np.sqrt(signal.correlate(df2, df2, mode='same')[int(n / 2)] *
                    signal.correlate(df1, df1, mode='same')[int(n / 2)])

        delay_arr = np.linspace(start=-0.5 * n / SAMPLE_RATE,
                                stop=0.5 * n / SAMPLE_RATE,
                                num=n)
        delay = delay_arr[np.argmax(corr)]
        distance = np.abs(self.actuators[0].y - self.sensors[1].y)
        propagation_speed = np.round(np.abs(distance / delay), decimals=2)
        return propagation_speed


class Setup3(Setup):
    actuators = np.empty(shape=1, dtype=Actuator)
    sensors = np.empty(shape=3, dtype=Sensor)
    sensors[2] = Sensor(coordinates=np.array([Table.WIDTH - 0.212, 0.235]),
                        name='Sensor 3')
    sensors[0] = Sensor(sensors[2].coordinates + np.array([-0.101, 0.008]),
                        name='Sensor 1')
    actuators[0] = Actuator(np.array([sensors[0].x - 0.10, 0.248]))
    sensors[1] = Sensor(np.array([actuators[0].x, actuators[0].y +
                                  actuators[0].RADIUS + 0.013 / 2]),
                        name='Sensor 2')

    def __init__(self):
        pass

    def get_propagation_speed(self, measurements: pd.DataFrame):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        object_1 = self.sensors[0]
        object_2 = self.sensors[2]
        n = len(measurements[object_1.name])
        corr = signal.correlate(measurements[object_1.name],
                                measurements[object_2.name],
                                mode='same')
        delay_arr = np.linspace(start=-0.5 * n / SAMPLE_RATE,
                                stop=0.5 * n / SAMPLE_RATE,
                                num=n)
        delay = delay_arr[np.argmax(corr)]
        distance = np.linalg.norm(object_1.coordinates - object_2.coordinates)
        propagation_speed = np.abs(distance / delay)
        return propagation_speed


class Setup3_2(Setup):
    """Sensors in a straight line across the full table"""
    actuators = np.empty(shape=1, dtype=Actuator)
    sensors = np.empty(shape=3, dtype=Sensor)

    sensors[0] = Sensor(coordinates=np.array([0.135, 0.305]),
                        name='Sensor 1')
    sensors[1] = Sensor(coordinates=(sensors[0].coordinates + np.array([0.267, 0])),
                        name='Sensor 2')
    sensors[2] = Sensor(sensors[1].coordinates + np.array([0.267, 0]),
                        name='Sensor 3')
    actuators[0] = Actuator(np.array([sensors[0].x / 2,
                                      sensors[0].y]))

    def __init__(self):
        pass

    def get_propagation_speed(self, measurements: pd.DataFrame):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        n = len(measurements[self.sensors[0].name])
        corr = signal.correlate(measurements[self.sensors[1].name],
                                measurements[self.sensors[0].name],
                                mode='same')
        delay_arr = np.linspace(start=-0.5 * n / SAMPLE_RATE,
                                stop=0.5 * n / SAMPLE_RATE,
                                num=n)
        delay = delay_arr[np.argmax(corr)]
        distance = np.abs(self.sensors[0].x - self.sensors[1].x)
        propagation_speed = np.round(np.abs(distance / delay), decimals=2)
        return propagation_speed


class Setup3_2_without_sensor2(Setup3_2):
    sensors = np.empty(shape=2, dtype=Sensor)
    sensors[0] = Sensor(coordinates=np.array([0.135, 0.305]),
                        name='Sensor 1')
    sensors[1] = Sensor(sensors[0].coordinates + np.array([2 * 0.267, 0]),
                        name='Sensor 3')


class Setup3_4(Setup3_2):
    """Sensor 3 even closer to the edge of the table"""
    def __init__(self):
        super().__init__()
        self.sensors[2].set_coordinates(np.array([self.table.LENGTH - 0.009,
                                                  self.sensors[2].y]))


class Setup4_5(Setup2):
    def __init__(self):
        self.actuators[0].set_coordinates(Table.C1)

    def get_propagation_speed(self, measurements: pd.DataFrame):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        """Choose two objects to calculate speed between"""
        object_1 = self.actuators[0]
        object_2 = self.sensors[0]
        n = len(measurements[object_1.name])
        corr = signal.correlate(measurements[object_1.name],
                                measurements[object_2.name],
                                mode='same')
        delay_arr = np.linspace(start=-0.5 * n / SAMPLE_RATE,
                                stop=0.5 * n / SAMPLE_RATE,
                                num=n)
        delay = delay_arr[np.argmax(corr)]
        distance = np.linalg.norm(object_1.coordinates - object_2.coordinates)
        propagation_speed = np.round(np.abs(distance / delay), decimals=2)
        return propagation_speed


class Setup6(Setup):
    """Actuator in the middle of the table, sensor
    placed approx. 16 cm towardds one of the corners
    """
    actuators = np.empty(shape=1, dtype=Actuator)
    sensors = np.empty(shape=1, dtype=Sensor)
    actuators[0] = Actuator(coordinates=np.array([Table.LENGTH / 2,
                                                  Table.WIDTH / 2]),
                            name='Actuator')
    sensors[0] = Sensor(coordinates=np.array([0.489, 0.242]),
                        name='Sensor 1')

    def __init__(self):
        pass

    def get_propagation_speed(self, measurements: pd.DataFrame):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        """Choose two objects to calculate speed between"""
        object_1 = self.actuators[0]
        object_2 = self.sensors[0]
        n = len(measurements[object_1.name])
        corr = signal.correlate(measurements[object_1.name],
                                measurements[object_2.name],
                                mode='same')
        delay_arr = np.linspace(start=-0.5 * n / SAMPLE_RATE,
                                stop=0.5 * n / SAMPLE_RATE,
                                num=n)
        delay = delay_arr[np.argmax(corr)]
        distance = np.linalg.norm(object_1.coordinates -
                                  object_2.coordinates)
        propagation_speed = np.abs(distance / delay)
        return propagation_speed


class Setup7(Setup):
    """Sensors in a straight line instead of a triangle in C2,
    actuator placed in front of the sensors
    """
    actuators = np.empty(shape=1, dtype=Actuator)
    sensors = np.empty(shape=3, dtype=Sensor)
    actuators[0] = Actuator(coordinates=(np.array([1 / 3 * Table.LENGTH,
                                                   5 / 6 * Table.WIDTH])))
    sensors[0] = Sensor(coordinates=(Table.C2 + np.array([-0.035, 0])),
                        name='Sensor 1')
    sensors[1] = Sensor(coordinates=Table.C2,
                        name='Sensor 2')
    sensors[2] = Sensor(coordinates=(Table.C2 + np.array([0.03, 0])),
                        name='Sensor 3')

    def __init__(self):
        pass

    def get_propagation_speed(self, measurements: pd.DataFrame):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        object_1 = self.sensors[0]
        object_2 = self.sensors[2]
        n = len(measurements[object_1.name])
        corr = signal.correlate(measurements[object_1.name],
                                measurements[object_2.name],
                                mode='same')
        delay_arr = np.linspace(start=-0.5 * n / SAMPLE_RATE,
                                stop=0.5 * n / SAMPLE_RATE,
                                num=n)
        delay = delay_arr[np.argmax(corr)]
        distance = np.linalg.norm(object_1.coordinates - object_2.coordinates)
        propagation_speed = np.abs(distance / delay)
        return propagation_speed
