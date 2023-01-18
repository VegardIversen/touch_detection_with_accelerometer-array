"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

from utils.global_constants import SAMPLE_RATE, SENSOR_1, SENSOR_2, SENSOR_3
from utils.objects import MirroredSensor, MirroredSource, Table, Actuator, Sensor


def find_indices_of_peaks(signal, height, plot=False, hilbert=True):
    """Find the indices of the peaks in the signal."""
    peak_indices = pd.DataFrame(columns=['Sensor 1',
                                         'Sensor 2',
                                         'Sensor 3',
                                         'Actuator'])

    if not hilbert:
        # Find the peaks of the square of the signal
        signal_squared = np.power(signal, 2)
        peak_indices, _ = signal.find_peaks(signal_squared, height)
    else:
        # Find the peaks of the Hilbert envelope of the signal
        signal_filtered_hilbert = get_envelope(signal)
        peak_indices, _ = signal.find_peaks(signal_filtered_hilbert, height)

    if peak_indices.size == 0:
        raise ValueError('No peaks found!')

    if plot:
        time_axis = np.linspace(0, len(signal), num=len(signal))
        fig, ax = plt.subplots(nrows=1)
        if not hilbert:
            ax.plot(time_axis / SAMPLE_RATE, signal_squared, label='sqrd')  # type: ignore
            ax.plot(time_axis[peak_indices] / SAMPLE_RATE,
                    signal_squared[peak_indices],
                    'x',
                    label='peaks')

        else:
            ax.plot(time_axis / SAMPLE_RATE,
                    signal_filtered_hilbert,
                    label='filtered hilbert')
            ax.plot(time_axis[peak_indices] / SAMPLE_RATE,
                    signal_filtered_hilbert[peak_indices],
                    'x',
                    label='peaks')
        ax.set_xlabel("Time [s]")
        ax.legend()
        fig.tight_layout()
        plt.grid()
        plt.show()

    return peak_indices


def get_envelope(signals: pd.DataFrame or pd.Series or np.ndarray):
    """Get the Hilbert envelope for all channels in df"""
    envelopes_of_signals = signals.copy()
    if isinstance(signals, np.ndarray) or isinstance(signals, pd.Series):
        envelopes_of_signals = np.abs(signal.hilbert(signals))
        return envelopes_of_signals
    elif isinstance(signals, pd.DataFrame):
        for channel in envelopes_of_signals:
            envelope = signal.hilbert(signals[channel])
            envelopes_of_signals[channel] = np.abs(envelope)
    return envelopes_of_signals


def find_first_peak(signals: pd.DataFrame, prominence: float = 0.1):
    """Return the index of the first peak of sig_np"""
    signals_values = signals.values
    peaks, _ = signal.find_peaks(signals_values, prominence=prominence)
    if peaks.size == 0:
        raise ValueError('No peaks found!')
        # return 0
    peak_index = peaks[0]
    return peak_index


def get_expected_reflections_pos(speed, peak, s=[0.26, 0.337, 0.386, 0.41]):
    t = s / speed
    n = t * SAMPLE_RATE + peak
    return n.tolist()


def find_mirrored_source(actuator: Actuator,
                         edges_to_bounce_on: np.ndarray):
    """Calculate the coordinate of the mirrored source
    to be used to find the wave travel distance.
    """
    NO_REFLECTION = 0
    mirrored_source = MirroredSource(actuator.coordinates)
    for edge in edges_to_bounce_on:
        if edge == NO_REFLECTION:
            continue
        elif edge == Table.TOP_EDGE:
            MIRRORED_SOURCE_OFFSET = np.array([0, 2 * (Table.WIDTH -
                                                       mirrored_source.y)])
            mirrored_source.set_coordinates(mirrored_source.coordinates +
                                            MIRRORED_SOURCE_OFFSET)
        elif edge == Table.RIGHT_EDGE:
            MIRRORED_SOURCE_OFFSET = np.array([2 * (Table.LENGTH -
                                                    mirrored_source.x), 0])
            mirrored_source.set_coordinates(mirrored_source.coordinates +
                                            MIRRORED_SOURCE_OFFSET)
        elif edge == Table.BOTTOM_EDGE:
            MIRRORED_SOURCE_OFFSET = np.array([0, -2 * mirrored_source.y])
            mirrored_source.set_coordinates(mirrored_source.coordinates +
                                            MIRRORED_SOURCE_OFFSET)
        elif edge == Table.LEFT_EDGE:
            MIRRORED_SOURCE_OFFSET = np.array([-2 * mirrored_source.x, 0])
            mirrored_source.set_coordinates(mirrored_source.coordinates +
                                            MIRRORED_SOURCE_OFFSET)
    return mirrored_source


def flip_sources(sources: np.ndarray,
                 edges_to_flip_around: np.ndarray):
    """Draw the sources in the flipped positions.
    The table edges are numbered as:

         _______1_______
        |               |
      4 |               | 2
        |               |
        |_______________|
                3

    """

    for edge in edges_to_flip_around:
        for source in sources:
            new_source = MirroredSource(source.coordinates)
            if edge == Table.TOP_EDGE:
                new_source.set_coordinates(new_source.coordinates +
                                           np.array([0, 2 * (Table.WIDTH -
                                                             new_source.y)]))
                sources = np.append(sources, new_source)
            elif edge == Table.RIGHT_EDGE:
                new_source.set_coordinates(new_source.coordinates +
                                           np.array([2 * (Table.LENGTH -
                                                          new_source.x), 0]))
                sources = np.append(sources, new_source)
            elif edge == Table.BOTTOM_EDGE:
                new_source.set_coordinates(new_source.coordinates +
                                           np.array([0, -2 * new_source.y]))
                sources = np.append(sources, new_source)
            elif edge == Table.LEFT_EDGE:
                new_source.set_coordinates(new_source.coordinates +
                                           np.array([-2 * new_source.x, 0]))
                sources = np.append(sources, new_source)
    return sources


def flip_sensors(sensors: np.ndarray,
                 edges_to_flip_around: np.ndarray):
    """Draw the sensors in the flipped positions.
    The table edges are numbered as:

         _______1_______
        |               |
      4 |               | 2
        |               |
        |_______________|
                3

    """
    for edge in edges_to_flip_around:
        for sensor in sensors:
            new_sensor = MirroredSensor(sensor.coordinates, sensor.name)
            if edge == Table.TOP_EDGE:
                new_sensor.set_coordinates(sensor.coordinates +
                                           np.array([0, 2 * (Table.WIDTH -
                                                             sensor.y)]))
                sensors = np.append(sensors, new_sensor)
            elif edge == Table.RIGHT_EDGE:
                new_sensor.set_coordinates(sensor.coordinates +
                                           np.array([2 * (Table.LENGTH -
                                                          sensor.x), 0]))
                sensors = np.append(sensors, new_sensor)
            elif edge == Table.BOTTOM_EDGE:
                new_sensor.set_coordinates(sensor.coordinates +
                                           np.array([0, -2 * sensor.y]))
                sensors = np.append(sensors, new_sensor)
            elif edge == Table.LEFT_EDGE:
                new_sensor.set_coordinates(sensor.coordinates +
                                           np.array([-2 * sensor.x, 0]))
                sensors = np.append(sensors, new_sensor)
    return sensors


def get_travel_times(actuator: Actuator,
                     sensor: Sensor,
                     propagation_speed: float,
                     print_info: bool = False,
                     milliseconds: bool = False,
                     relative_first_reflection: bool = True):
    """Get the travel distance from first and second reflections.
    TODO:   Add logic for not calculating physically impossible reflections.
            This is not necessary for predicting WHEN the reflections will
            arrive, but to visualise which reflections we are seeing.
    """
    arrival_times = np.array([])
    travel_distances = np.array([])

    """Calculate the direct wave travel time"""
    direct_travel_distance = np.linalg.norm(actuator.coordinates -
                                            sensor.coordinates)
    travel_distances = np.append(travel_distances, direct_travel_distance)
    direct_travel_time = direct_travel_distance / propagation_speed
    arrival_times = np.append(arrival_times, direct_travel_time)

    if print_info and not relative_first_reflection:
        print(f"\nDirect travel distance: \
              {np.round(direct_travel_distance, 5)} m")
        print(f"\nDirect travel time: {np.round(direct_travel_time, 6)} s")

    EDGES = np.array([1, 2, 3, 4])
    # Iterate thorugh all combinations of edges to reflect from
    for edge_1 in range(0, EDGES.size + 1):
        for edge_2 in range(0, EDGES.size + 1):
            if edge_1 == edge_2:
                # Can't reflect from the same edge twice
                continue
            elif edge_1 and not edge_2:
                # To avoid repeating first reflection calculations
                continue
            mirrored_source = find_mirrored_source(actuator,
                                                   np.array([edge_1, edge_2]))
            distance_to_sensor = np.linalg.norm(mirrored_source.coordinates -
                                                sensor.coordinates)
            if relative_first_reflection:
                distance_to_sensor -= direct_travel_distance
            time_to_sensors = distance_to_sensor / propagation_speed
            if not edge_1:
                if print_info:
                    print(f'\nReflecting from {edge_2}: \t \
                              Distance: {np.round(distance_to_sensor, 5)} m \t\
                              Time: {np.round(time_to_sensors, 6)} s')
            else:
                if print_info:
                    print(f'\nReflecting from {edge_1}, then {edge_2}:      \
                            \t Distance: {np.round(distance_to_sensor, 5)} m\
                            \t Time: {np.round(time_to_sensors, 6)} s')
            travel_distances = np.append(travel_distances, distance_to_sensor)
            arrival_times = np.append(arrival_times, time_to_sensors)

    if milliseconds:
        arrival_times *= 1000
    if relative_first_reflection:
        travel_distances[0] = 0
        arrival_times[0] = 0

    return arrival_times, travel_distances


if __name__ == '__main__':
    pass
