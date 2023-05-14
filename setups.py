"""Each setup has its own child class of the general Setup class.
TODO:   - Add remaining setups.
        - Expand propagation speed function to use all options
          for better estimation.
"""
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

from global_constants import (SAMPLE_RATE,
                              ACTUATOR_1,
                              SENSOR_1,
                              SENSOR_2,
                              SENSOR_3)
import objects as obj
from objects import Table, Actuator, Sensor, SimulatedPlate
from data_viz_files.visualise_data import figure_size_setup
from data_viz_files.drawing import plot_legend_without_duplicates
import results_thesis as rt
import data_processing.detect_echoes as de
import data_processing.wave_properties as wp

class Setup:
    table = obj.Table()
    actuators: np.ndarray
    sensors: np.ndarray
    

    def __init__(self):
        raise NameError("Setup version needs to be specified")

    def draw(self, save_fig=False, fig_name=None, file_format='png', actuator_show=True, show_tab=False):
        plt.axes()
        self.table.draw()
        if actuator_show:
            [actuator.draw() for actuator in self.actuators]
        [sensor.draw() for sensor in self.sensors if sensor.plot]
        plt.axis('scaled')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plot_legend_without_duplicates()
        if save_fig:
            plt.savefig(f'{fig_name}.{file_format}',dpi=300, format=file_format)
        if show_tab:
            plt.show()

    def get_propagation_speed(self,
                              measurements: pd.DataFrame,
                              object_1: Sensor,
                              object_2: Sensor):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
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

    def get_objects(self):
        return self.actuators, self.sensors

class SimulatedSetup():
    plate = obj.SimulatedPlate()
    actuators = np.empty(shape=1, dtype=obj.Actuator)
    sensors: np.ndarray
    def __init__(self, comsol_file=9, positions=[], SAMPLE_RATE=500000, center_freq=15000):
        self.wave_data, self.x_pos, self.y_pos, self.z_pos, self.time_axis = rt.get_comsol_data(comsol_file)
        self.positions = positions
        if not positions:
            self.positions = [0]
        n_sensors = len(self.x_pos)
        self.signal_length = len(self.wave_data[0])
        self.SAMPLE_RATE = SAMPLE_RATE
        self.freq_vel = np.fft.rfftfreq(self.signal_length, d=1/SAMPLE_RATE)
        self.sensors = np.empty(shape=n_sensors, dtype=obj.Sensor)
        self.actuators[0] = obj.Actuator(coordinates=np.array([self.x_pos[0], self.y_pos[0]]))
        self.v_gr_calc, self.v_ph_calc = wp.theoretical_group_phase_vel(self.freq_vel, material='LDPE_tonni20mm')
        self.velocities = rt.get_velocity_at_freq(freq=center_freq)
        self.v_ph_center_freq = self.velocities['A0']['phase_velocity']
        if positions:
            self.sensors = np.empty(shape=len(positions), dtype=obj.Sensor)
            for idx, pos in enumerate(positions):
                self.sensors[idx] = obj.Sensor(coordinates=np.array([self.x_pos[pos], self.y_pos[pos]]), name=f'Sensor {pos+1} (+1)')
        else:
            for i in range(n_sensors):
                self.sensors[i] = obj.Sensor(coordinates=np.array([self.x_pos[i], self.y_pos[i]]), name=f'Sensor {i+1}')

    
    
    def draw(self, save_fig=False, fig_name=None, file_format='png', actuator_show=True, show_tab=True):
        plt.axes()
        self.plate.draw()
        if actuator_show:
            [actuator.draw() for actuator in self.actuators]
        [sensor.draw() for sensor in self.sensors if sensor.plot]
        plt.axis('scaled')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plot_legend_without_duplicates()
        if save_fig:
            plt.savefig(f'{fig_name}.{file_format}',dpi=300, format=file_format)
        if show_tab:
            plt.show()

    def get_diagonal_distance(self, positions):
        distance = np.sqrt((self.x_pos[positions[1]] - self.x_pos[positions[0]])**2 + (self.y_pos[positions[1]] - self.y_pos[positions[0]])**2)
        print(f'distance: {distance}')
        return distance
    
    def reflections(self):
        attenuation_dBpm = 0
        travel_distances = []
        #NEED TO FIX THIS IF I EVER USE MORE THAN 1 SENSOR!!
        for sensor_i in range(len(self.sensors)):
            measurement_i = np.zeros(len(self.wave_data[0]))
            arrival_times, distances = de.get_travel_times(
                self.actuators[0],
                self.sensors[sensor_i],
                self.v_ph_center_freq,
                surface=self.plate,
                milliseconds=False,
                relative_first_reflection=False,
                print_info=False,
            )
            
            
            #print(f'distances: {distances}')
            #print(f'arrival_times: {arrival_times}')
            # # Hardcode the arrival times to be only the indices 0, 3, 4, and 11 of arrival_times
            # #arrival_times = arrival_times[[0, 3, 4, 11]]
            # for arrival_time in arrival_times:
            #     arrival_time_index = int(arrival_time * self.SAMPLE_RATE)
            #     travel_distance_m = arrival_time * self.v_ph_center_freq
            #     measurement_i[
            #         arrival_time_index : arrival_time_index + self.signal_length
            #     ] += self.wave_data[self.positions[0]] * 10 ** (-attenuation_dBpm * travel_distance_m / 20)
            # sensor_measurements[f"Sensor {sensor_i + 1}"] = measurement_i
            # travel_distances.append(distances[:2])
        return arrival_times, distances
        
class Setup2(Setup):
    """Sensors in an 8 cm edge triangle in C2"""
    actuators = np.empty(shape=1, dtype=obj.Actuator)
    sensors = np.empty(shape=3, dtype=obj.Sensor)
    actuators[0] = obj.Actuator(coordinates=np.array([1 / 2 * Table.LENGTH,
                                                  1 / 9 * Table.WIDTH]))
    sensors[SENSOR_2] = obj.Sensor(coordinates=np.array([Table.LENGTH / 2,
                                                     Table.WIDTH - 0.082]),
                               name='Sensor 2')
    SENSOR_1_OFFSET = np.array([-0.08 / 2, -(np.sqrt(0.08 ** 2 - 0.04 ** 2))])
    sensors[SENSOR_1] = obj.Sensor(coordinates=(sensors[SENSOR_2].coordinates + SENSOR_1_OFFSET),
                               name='Sensor 1')
    SENSOR_3_OFFSET = np.array([0.08 / 2, -(np.sqrt(0.08 ** 2 - 0.04 ** 2))])
    sensors[SENSOR_3] = obj.Sensor(coordinates=(sensors[SENSOR_2].coordinates + SENSOR_3_OFFSET),
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
        distance = np.abs(self.actuators[0].y - self.sensors[SENSOR_2].y)
        propagation_speed = np.round(np.abs(distance / delay), decimals=2)
        return propagation_speed


class Setup3(Setup):
    actuators = np.empty(shape=1, dtype=obj.Actuator)
    sensors = np.empty(shape=3, dtype=obj.Sensor)
    sensors[SENSOR_3] = obj.Sensor(coordinates=np.array([Table.WIDTH - 0.212, 0.235]),
                               name='Sensor 3')
    sensors[SENSOR_1] = obj.Sensor(sensors[SENSOR_3].coordinates + np.array([-0.101, 0.008]),
                               name='Sensor 1')
    actuators[0] = obj.Actuator(np.array([sensors[SENSOR_1].x - 0.10, 0.248]))
    sensors[SENSOR_2] = obj.Sensor(np.array([actuators[0].x, actuators[0].y +
                                  actuators[0].RADIUS + 0.013 / 2]),
                               name='Sensor 2')

    def __init__(self):
        pass

    def get_propagation_speed(self, measurements: pd.DataFrame):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        object_1 = self.sensors[SENSOR_1]
        object_2 = self.sensors[SENSOR_3]
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
    actuators = np.empty(shape=1, dtype=obj.Actuator)
    sensors = np.empty(shape=3, dtype=obj.Sensor)

    sensors[SENSOR_1] = obj.Sensor(coordinates=np.array([0.135, 0.305]),
                               name='Sensor 1')
    sensors[SENSOR_2] = obj.Sensor(coordinates=(sensors[SENSOR_1].coordinates + np.array([0.267, 0])),
                               name='Sensor 2')
    sensors[SENSOR_3] = obj.Sensor(sensors[SENSOR_2].coordinates + np.array([0.267, 0]),
                               name='Sensor 3')
    actuators[0] = obj.Actuator(np.array([sensors[SENSOR_1].x / 2,
                                      sensors[SENSOR_1].y]))

    def __init__(self):
        pass

    def get_propagation_speed(self, measurements: pd.DataFrame):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        n = len(measurements[self.sensors[SENSOR_1].name])
        corr = signal.correlate(measurements[self.sensors[SENSOR_2].name],
                                measurements[self.sensors[SENSOR_1].name],
                                mode='same')
        delay_arr = np.linspace(start=-0.5 * n / SAMPLE_RATE,
                                stop=0.5 * n / SAMPLE_RATE,
                                num=n)
        delay = delay_arr[np.argmax(corr)]
        distance = np.abs(self.sensors[SENSOR_1].x - self.sensors[SENSOR_2].x)
        propagation_speed = np.round(np.abs(distance / delay), decimals=2)
        return propagation_speed


class Setup3_2_without_sensor2(Setup3_2):
    sensors = np.empty(shape=2, dtype=obj.Sensor)
    sensors[SENSOR_1] = obj.Sensor(coordinates=np.array([0.135, 0.305]),
                        name='Sensor 1')
    sensors[SENSOR_2] = obj.Sensor(sensors[SENSOR_1].coordinates + np.array([2 * 0.267, 0]),
                        name='Sensor 3')


class Setup3_4(Setup3_2):
    """Sensor 3 even closer to the edge of the table"""
    def __init__(self):
        super().__init__()
        self.sensors[SENSOR_3].set_coordinates(np.array([self.table.LENGTH - 0.009,
                                                  self.sensors[SENSOR_3].y]))


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
        object_2 = self.sensors[SENSOR_1]
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
    actuators = np.empty(shape=1, dtype=obj.Actuator)
    sensors = np.empty(shape=1, dtype=obj.Sensor)
    actuators[0] = obj.Actuator(coordinates=np.array([Table.LENGTH / 2,
                                                  Table.WIDTH / 2]),
                            name='Actuator')
    sensors[SENSOR_1] = obj.Sensor(coordinates=np.array([0.489, 0.242]),
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
        object_2 = self.sensors[SENSOR_1]
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
    actuators = np.empty(shape=1, dtype=obj.Actuator)
    sensors = np.empty(shape=3, dtype=obj.Sensor)
    actuators[0] = Actuator(coordinates=(np.array([1 / 3 * Table.LENGTH,
                                                   5 / 6 * Table.WIDTH])))
    sensors[SENSOR_1] = obj.Sensor(coordinates=(Table.C2 + np.array([-0.035, 0])),
                        name='Sensor 1')
    sensors[SENSOR_2] = obj.Sensor(coordinates=Table.C2,
                        name='Sensor 2')
    sensors[SENSOR_3] = obj.Sensor(coordinates=(Table.C2 + np.array([0.03, 0])),
                        name='Sensor 3')

    def __init__(self):
        pass

    def get_propagation_speed(self, measurements: pd.DataFrame):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        object_1 = self.sensors[SENSOR_1]
        object_2 = self.sensors[SENSOR_3]
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


class Setup9(Setup):
    """Sensors in the middle of the table to
    separate direct signal and reflections.
    NOTE:   Sensor 2 is not used in this setup,
            but is included to make the code more
            consistent with the measurement channels.
    """
    actuators = np.empty(shape=1, dtype=obj.Actuator)
    sensors = np.empty(shape=3, dtype=obj.Sensor)
    actuators[ACTUATOR_1] = obj.Actuator(coordinates=[1 / 2 * Table.LENGTH - 0.10,
                                                  1 / 2 * Table.WIDTH])
    sensors[SENSOR_1] = obj.Sensor(coordinates=(actuators[ACTUATOR_1].coordinates + np.array([0.10, 0])),
                               name='Sensor 1')
    sensors[SENSOR_2] = obj.Sensor(coordinates=np.array([0, 0]),
                               name='Sensor 2',
                               plot=False)
    sensors[SENSOR_3] = obj.Sensor(coordinates=(sensors[SENSOR_1].coordinates + np.array([0.10, 0])),
                               name='Sensor 3')

    def __init__(self):
        pass

    def get_propagation_speed(self, measurements: pd.DataFrame):
        """Use the cross correlation between the two channels
        to find the propagation speed. Based on:
        https://stackoverflow.com/questions/41492882/find-time-shift-of-two-signals-using-cross-correlation
        """
        object_1 = self.sensors[SENSOR_1]
        object_2 = self.sensors[SENSOR_2]
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
