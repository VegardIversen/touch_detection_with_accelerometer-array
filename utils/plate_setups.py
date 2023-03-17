"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.data_processing.preprocessing import window_signals

from utils.global_constants import SAMPLE_RATE, ACTUATOR_1, SENSOR_1, SENSOR_2, SENSOR_3
from utils.objects import Table, Plate, Actuator, Sensor
from utils.little_helpers import distance_between
from utils.data_visualization.drawing import plot_legend_without_duplicates
from utils.data_processing.detect_echoes import find_first_peak_index, get_envelopes


class Setup:
    surface = Plate()
    actuators: np.ndarray
    sensors: np.ndarray

    def __init__(self):
        raise NameError("Setup version needs to be specified")

    def draw(self):
        plt.axes()
        plt.gcf().set_size_inches(5.5, 3.5)
        self.surface.draw()
        [actuator.draw() for actuator in self.actuators]
        [sensor.draw() for sensor in self.sensors if sensor.plot]
        plt.axis("scaled")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plot_legend_without_duplicates()

    def get_propagation_speed(
        self,
        measurements: pd.DataFrame,
        object_1: Sensor or Actuator,
        object_2: Sensor or Actuator,
    ):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        n = len(measurements[object_1.name])
        corr = signal.correlate(
            measurements[object_1.name], measurements[object_2.name], mode="same"
        )
        delay_arr = np.linspace(
            start=-0.5 * n / SAMPLE_RATE, stop=0.5 * n / SAMPLE_RATE, num=n
        )
        delay = delay_arr[np.argmax(corr)]
        distance = np.linalg.norm(object_1.coordinates - object_2.coordinates)
        propagation_speed = np.abs(distance / delay)
        return propagation_speed


class Setup1(Setup):
    """Sensors in the middle of the table to
    separate direct signal and reflections.
    """

    actuators = np.empty(shape=1, dtype=Actuator)
    sensors = np.empty(shape=3, dtype=Sensor)
    actuators[ACTUATOR_1] = Actuator(
        coordinates=np.array([1 / 2 * Plate.LENGTH - 0.15, 1 / 2 * Plate.WIDTH])
    )
    sensors[SENSOR_1] = Sensor(
        coordinates=(actuators[ACTUATOR_1].coordinates + np.array([0.10, 0])),
        name="Sensor 1",
    )
    sensors[SENSOR_2] = Sensor(
        coordinates=(sensors[SENSOR_1].coordinates + np.array([0.10, 0])),
        name="Sensor 2",
    )
    sensors[SENSOR_3] = Sensor(
        coordinates=(sensors[SENSOR_2].coordinates + np.array([0.10, 0])),
        name="Sensor 3",
    )

    def __init__(self):
        pass

    def get_propagation_speed(
        self, measurements: pd.DataFrame, prominence: float = 0.001
    ):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        object_1 = self.sensors[SENSOR_1]
        object_2 = self.sensors[SENSOR_3]
        # n = len(measurements[object_1.name])
        # corr = signal.correlate(measurements[object_1.name],
        #                         measurements[object_2.name],
        #                         mode='same')
        # delay_arr = np.linspace(start=-0.5 * n / SAMPLE_RATE,
        #                         stop=0.5 * n / SAMPLE_RATE,
        #                         num=n)
        # delay = delay_arr[np.argmax(corr)]
        # distance = np.linalg.norm(object_1.coordinates - object_2.coordinates)
        # propagation_speed = np.abs(distance / delay)
        """Alternatively:"""
        # peak_object1 = np.argmax(np.abs(signal.hilbert(measurements[object_1.name])))
        # peak_object2 = np.argmax(np.abs(signal.hilbert(measurements[object_2.name])))
        # delay = (peak_object2 - peak_object1) / SAMPLE_RATE
        # distance = np.linalg.norm(object_1.coordinates - object_2.coordinates)
        # propagation_speed = np.abs(distance / delay)
        """Or even alternativelier:"""
        measurements = get_envelopes(measurements)
        _, ax = plt.subplots()
        first_peak_object1 = find_first_peak_index(measurements[object_1.name], ax=ax)
        first_peak_object2 = find_first_peak_index(measurements[object_2.name], ax=ax)
        delay = (first_peak_object2 - first_peak_object1) / SAMPLE_RATE
        distance = np.linalg.norm(object_1.coordinates - object_2.coordinates)
        propagation_speed = np.abs(distance / delay)
        return propagation_speed


class Setup2(Setup):
    """Sensors in a straight line across the full table"""

    actuators = np.empty(shape=1, dtype=Actuator)
    sensors = np.empty(shape=3, dtype=Sensor)

    sensors[SENSOR_1] = Sensor(coordinates=np.array([0.135, 0.305]), name="Sensor 1")
    sensors[SENSOR_2] = Sensor(
        coordinates=(sensors[SENSOR_1].coordinates + np.array([0.267, 0])),
        name="Sensor 2",
    )
    sensors[SENSOR_3] = Sensor(
        sensors[SENSOR_2].coordinates + np.array([0.267, 0]), name="Sensor 3"
    )
    actuators[0] = Actuator(np.array([sensors[SENSOR_1].x / 2, sensors[SENSOR_1].y]))

    def __init__(self):
        pass

    def get_propagation_speed(self, measurements: pd.DataFrame):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        n = len(measurements[self.sensors[SENSOR_1].name])
        corr = signal.correlate(
            measurements[self.sensors[SENSOR_2].name],
            measurements[self.sensors[SENSOR_1].name],
            mode="same",
        )
        delay_arr = np.linspace(
            start=-0.5 * n / SAMPLE_RATE, stop=0.5 * n / SAMPLE_RATE, num=n
        )
        delay = delay_arr[np.argmax(corr)]
        distance = np.abs(self.sensors[SENSOR_1].x - self.sensors[SENSOR_2].x)
        propagation_speed = np.round(np.abs(distance / delay), decimals=2)
        return propagation_speed


class Setup3(Setup):
    """Inspecting touch signal right under the touch location
    and over and under at"""

    actuators = np.empty(shape=1, dtype=Actuator)
    sensors = np.empty(shape=3, dtype=Sensor)
    actuators[ACTUATOR_1] = Actuator(coordinates=(np.array([0.45, Plate.WIDTH / 2])))
    sensors[SENSOR_1] = Sensor(
        coordinates=actuators[ACTUATOR_1].coordinates, name="Sensor 1"
    )
    sensors[SENSOR_2] = Sensor(
        coordinates=(Plate.LENGTH / 2, Plate.WIDTH / 2), name="Sensor 2"
    )
    sensors[SENSOR_3] = Sensor(
        coordinates=(Plate.LENGTH / 2, Plate.WIDTH / 2), name="Sensor 3"
    )

    def __init__(self):
        pass

    def get_propagation_speed(self, measurements: pd.DataFrame):
        object_1 = self.sensors[SENSOR_1]
        object_2 = self.sensors[SENSOR_3]
        envelopes = get_envelopes(measurements)
        _, ax = plt.subplots()
        first_peak_object1 = find_first_peak_index(envelopes[object_1.name], ax=ax)
        measurements = window_signals(
            measurements,
            length_of_signal_seconds=(0.384 - 0.378),
            peak_index=first_peak_object1,
        )
        n = len(measurements[object_1.name])
        corr = signal.correlate(
            measurements[object_1.name], measurements[object_2.name], mode="same"
        )
        delay_arr = np.linspace(
            start=-0.5 * n / SAMPLE_RATE, stop=0.5 * n / SAMPLE_RATE, num=n
        )
        delay = delay_arr[np.argmax(corr)]
        distance = np.linalg.norm(object_1.coordinates - object_2.coordinates)
        propagation_speed = np.abs(distance / delay)
        return propagation_speed


class Setup_3x3(Setup):
    actuators = np.empty(shape=1, dtype=Actuator)
    sensors = np.empty(shape=9, dtype=Sensor)
    actuators[ACTUATOR_1] = Actuator(
        coordinates=np.array([1 / 2 * Plate.LENGTH - 0.15, 1 / 2 * Plate.WIDTH])
    )
    sensors[1] = Sensor(
        coordinates=np.array([Plate.LENGTH / 2, Plate.WIDTH - 0.05]), name="Sensor 2"
    )
    sensors[0] = Sensor(
        coordinates=(sensors[1].x - 0.02, sensors[1].y), name="Sensor 1"
    )
    sensors[2] = Sensor(
        coordinates=(sensors[1].x + 0.02, sensors[1].y), name="Sensor 3"
    )
    sensors[3] = Sensor(
        coordinates=(sensors[1].x - 0.02, sensors[1].y - 0.02), name="Sensor 4"
    )
    sensors[4] = Sensor(
        coordinates=(sensors[1].x, sensors[1].y - 0.02), name="Sensor 5"
    )
    sensors[5] = Sensor(
        coordinates=(sensors[1].x + 0.02, sensors[1].y - 0.02), name="Sensor 6"
    )
    sensors[6] = Sensor(
        coordinates=(sensors[1].x - 0.02, sensors[1].y - 0.04), name="Sensor 7"
    )
    sensors[7] = Sensor(
        coordinates=(sensors[1].x, sensors[1].y - 0.04), name="Sensor 8"
    )
    sensors[8] = Sensor(
        coordinates=(sensors[1].x + 0.02, sensors[1].y - 0.04), name="Sensor 9"
    )

    def __init__(self):
        pass

    def get_propagation_speed(
        self, measurements: pd.DataFrame, prominence: float = 0.001
    ):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        object_1 = self.sensors[SENSOR_1]
        object_2 = self.sensors[SENSOR_3]
        # n = len(measurements[object_1.name])
        # corr = signal.correlate(measurements[object_1.name],
        #                         measurements[object_2.name],
        #                         mode='same')
        # delay_arr = np.linspace(start=-0.5 * n / SAMPLE_RATE,
        #                         stop=0.5 * n / SAMPLE_RATE,
        #                         num=n)
        # delay = delay_arr[np.argmax(corr)]
        # distance = np.linalg.norm(object_1.coordinates - object_2.coordinates)
        # propagation_speed = np.abs(distance / delay)
        """Alternatively:"""
        # peak_object1 = np.argmax(np.abs(signal.hilbert(measurements[object_1.name])))
        # peak_object2 = np.argmax(np.abs(signal.hilbert(measurements[object_2.name])))
        # delay = (peak_object2 - peak_object1) / SAMPLE_RATE
        # distance = np.linalg.norm(object_1.coordinates - object_2.coordinates)
        # propagation_speed = np.abs(distance / delay)
        """Or even alternativelier:"""
        measurements = get_envelopes(measurements)
        _, ax = plt.subplots()
        first_peak_object1 = find_first_peak_index(measurements[object_1.name], ax=ax)
        first_peak_object2 = find_first_peak_index(measurements[object_2.name], ax=ax)
        delay = (first_peak_object2 - first_peak_object1) / SAMPLE_RATE
        distance = np.linalg.norm(object_1.coordinates - object_2.coordinates)
        propagation_speed = np.abs(distance / delay)
        return propagation_speed


class Setup_Linear_Array(Setup):
    """A line of number_of_sensors sensors,
    with an actuator given by actuator_coordinates.
    """

    def __init__(
        self,
        number_of_sensors: int,
        actuator_coordinates: np.ndarray,
        array_start_coordinates: np.ndarray,
        array_spacing_m: float,
    ):
        self.number_of_sensors = number_of_sensors
        self.actuator_coordinates = actuator_coordinates
        self.array_start_coordinates = array_start_coordinates
        self.array_spacing = array_spacing_m
        self.actuators = np.empty(shape=1, dtype=Actuator)
        self.sensors = np.empty(shape=number_of_sensors, dtype=Sensor)
        self.actuators[ACTUATOR_1] = Actuator(coordinates=actuator_coordinates)
        for i in range(number_of_sensors):
            self.sensors[i] = Sensor(
                coordinates=np.array(
                    [
                        array_start_coordinates[0] + i * array_spacing_m,
                        array_start_coordinates[1],
                    ]
                ),
                name=f"Sensor {i + 1}",
            )


class Setup4(Setup_Linear_Array):
    # Setup_Linear_Array with arugments for a 3 sensor array
    def __init__(self, actuator_coordinates: np.ndarray):
        super().__init__(
            number_of_sensors=3,
            actuator_coordinates=actuator_coordinates,
            array_start_coordinates=np.array([0.45, 0.65]),
            array_spacing_m=0.01,
        )

    def get_propagation_speed(
        self, measurements: pd.DataFrame, prominence: float = 0.001
    ):
        """Use the cross correlation between the two channels
        to find the propagation speed.
        """
        object1 = self.sensors[SENSOR_1]
        object2 = self.sensors[SENSOR_2]
        # noise_max_value_object1 = get_noise_max_value(measurements[object1.name],
        #                                               time_window_s=0.1,)
        # noise_max_value_object2 = get_noise_max_value(measurements[object2.name],
        #                                               time_window_s=0.1,)
        # first_rise_object1 = (
        #     measurements[object1.name] > (2 * noise_max_value_object1)).idxmax()
        # first_rise_object2 = (
        #     measurements[object2.name] > (2 * noise_max_value_object2)).idxmax()
        # first_rise_object1 = (
        #     measurements[object1.name] > (np.max(measurements[object1.name]) / 2)).idxmax()
        # first_rise_object2 = (
        #     measurements[object2.name] > (np.max(measurements[object2.name]) / 2)).idxmax()
        # delay = (first_rise_object2 - first_rise_object1) / SAMPLE_RATE
        # assert delay != 0, f"""Delay is 0: First rise indices are {first_rise_object1} and {first_rise_object2}"""
        distance = np.linalg.norm(object1.coordinates - object2.coordinates)
        # propagation_speed = np.abs(distance / delay)
        # Or by max value:
        peak_object1 = np.argmax(np.abs(signal.hilbert(measurements[object1.name])))
        peak_object2 = np.argmax(np.abs(signal.hilbert(measurements[object2.name])))
        delay = (peak_object2 - peak_object1) / SAMPLE_RATE
        souce_distance_cat = np.abs(self.actuators[0].x - object1.x)
        source_distance_hyp = distance_between(
            self.actuators[0].coordinates, object1.coordinates
        )
        angle = np.arccos((souce_distance_cat / source_distance_hyp))
        propagation_speed = np.abs(distance / (delay / np.cos(angle)))
        return propagation_speed
