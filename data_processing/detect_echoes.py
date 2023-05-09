import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd

from objects import MirroredSensor, MirroredSource, Table, Actuator, Sensor
from constants import *

from csv_to_df import csv_to_df
from data_processing.preprocessing import filter_general, crop_data


def find_indices_of_peaks(sig_np, height, plot=False, hilbert=True):
    """Find the indices of the peaks in the signal."""
    peak_indices = pd.DataFrame(columns=['channel 1', 'channel 2', 'channel 3', 'chirp'])

    if not hilbert:
        # Find the peaks of the square of the signal
        signal_sqr = np.power(sig_np, 2)
        peak_indices, _ = signal.find_peaks(signal_sqr, height)
    else:
        # Find the peaks of the Hilbert envelope of the signal
        sig_np_filtered_hilbert = get_hilbert_envelope(sig_np)
        peak_indices, _ = signal.find_peaks(sig_np_filtered_hilbert, height)

    if peak_indices.size == 0:
        raise ValueError('No peaks found!')

    if plot:
        time_axis = np.linspace(0, len(sig_np), num=len(sig_np))
        fig, ax0 = plt.subplots(nrows=1)
        # ax0.plot(time_axis, sig_np, label='signal')
        # ax0.plot(time_axis / 150000, sig_np_filtered_hilbert, label='filtered')
        if not hilbert:
            ax0.plot(time_axis / SAMPLE_RATE, signal_sqr, label='sqrd')
            ax0.plot(time_axis[peak_indices] / SAMPLE_RATE, signal_sqr[peak_indices], 'x', label='peaks')

        else:
            ax0.plot(time_axis / SAMPLE_RATE, sig_np_filtered_hilbert, label='filtered hilbert')
            ax0.plot(time_axis[peak_indices] / SAMPLE_RATE, sig_np_filtered_hilbert[peak_indices], 'x', label='peaks')
        ax0.set_xlabel("Time [s]")
        ax0.legend()
        fig.tight_layout()
        plt.grid()
        plt.show()

    return peak_indices


def get_hilbert_envelope(sig):
    """Get the Hilbert envelope for all channels in df"""
    sig_hilb = sig.copy()
    if isinstance(sig, np.ndarray) or isinstance(sig, pd.Series):
        sig_hilb = np.abs(signal.hilbert(sig))
        return sig_hilb
    elif isinstance(sig, pd.DataFrame):
        for channel in sig_hilb:
            sig_hilb[channel] = np.abs(signal.hilbert(sig[channel]))
    return sig_hilb


def find_first_peak(sig_df, height):
    """Return the index of the first peak of sig_np"""
    sig_np = sig_df.values
    peaks, _ = signal.find_peaks(sig_np, height)
    if peaks.size == 0:
        raise ValueError('No peaks found!')
        # return 0
    peak_index = peaks[0]
    return peak_index


def get_expected_reflections_pos(speed, peak, s=[0.26, 0.337, 0.386, 0.41]):
    t = s / speed
    n = t * SAMPLE_RATE + peak
    return n.tolist()


def find_mirrored_source(
    actuator: Actuator, edges_to_bounce_on: np.ndarray, surface: Table or Plate
):
    """Calculate the coordinate of the mirrored source
    to be used to find the wave travel distance.
    """
    NO_REFLECTION = 0
    mirrored_source = MirroredSource(actuator.coordinates)
    for edge in edges_to_bounce_on:
        if edge == NO_REFLECTION:
            continue
        elif edge == surface.TOP_EDGE:
            MIRRORED_SOURCE_OFFSET = np.array(
                [0, 2 * (surface.WIDTH - mirrored_source.y)]
            )
            mirrored_source.set_coordinates(
                mirrored_source.coordinates + MIRRORED_SOURCE_OFFSET
            )
        elif edge == surface.RIGHT_EDGE:
            MIRRORED_SOURCE_OFFSET = np.array(
                [2 * (surface.LENGTH - mirrored_source.x), 0]
            )
            mirrored_source.set_coordinates(
                mirrored_source.coordinates + MIRRORED_SOURCE_OFFSET
            )
        elif edge == surface.BOTTOM_EDGE:
            MIRRORED_SOURCE_OFFSET = np.array([0, -2 * mirrored_source.y])
            mirrored_source.set_coordinates(
                mirrored_source.coordinates + MIRRORED_SOURCE_OFFSET
            )
        elif edge == surface.LEFT_EDGE:
            MIRRORED_SOURCE_OFFSET = np.array([-2 * mirrored_source.x, 0])
            mirrored_source.set_coordinates(
                mirrored_source.coordinates + MIRRORED_SOURCE_OFFSET
            )
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
                new_source.set_coordinates(new_source.coordinates + np.array([0, 2 * (Table.WIDTH - new_source.y)]))
                sources = np.append(sources, new_source)
            elif edge == Table.RIGHT_EDGE:
                new_source.set_coordinates(new_source.coordinates + np.array([2 * (Table.LENGTH - new_source.x), 0]))
                sources = np.append(sources, new_source)
            elif edge == Table.BOTTOM_EDGE:
                new_source.set_coordinates(new_source.coordinates + np.array([0, -2 * new_source.y]))
                sources = np.append(sources, new_source)
            elif edge == Table.LEFT_EDGE:
                new_source.set_coordinates(new_source.coordinates + np.array([-2 * new_source.x, 0]))
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
            new_sensor = MirroredSensor(sensor.coordinates)
            if edge == Table.TOP_EDGE:
                new_sensor.set_coordinates(sensor.coordinates + np.array([0, 2 * (Table.WIDTH - sensor.y)]))
                sensors = np.append(sensors, new_sensor)
            elif edge == Table.RIGHT_EDGE:
                new_sensor.set_coordinates(sensor.coordinates + np.array([2 * (Table.LENGTH - sensor.x), 0]))
                sensors = np.append(sensors, new_sensor)
            elif edge == Table.BOTTOM_EDGE:
                new_sensor.set_coordinates(sensor.coordinates + np.array([0, -2 * sensor.y]))
                sensors = np.append(sensors, new_sensor)
            elif edge == Table.LEFT_EDGE:
                new_sensor.set_coordinates(sensor.coordinates + np.array([-2 * sensor.x, 0]))
                sensors = np.append(sensors, new_sensor)
    return sensors

def ignore_edge_combination(edge_1, edge_2):
    if edge_1 == edge_2:
        # Can't reflect from the same edge twice
        return True
    elif edge_1 and not edge_2:
        # To avoid repeating first reflection calculations
        return True
    elif (edge_1 - edge_2) == 1 or (edge_1 == 4 and edge_2 == 1):
        # Check if edges are adjacent to remove unphysical combinations
        return True
    else:
        return False

def get_travel_times(
    actuator: Actuator,
    sensor: Sensor,
    propagation_speed: float,
    surface: Table or Plate,
    print_info: bool = False,
    milliseconds: bool = False,
    relative_first_reflection: bool = True,
):
    """Get the travel distance from first and second reflections.
    The table edges are numbered as:
         _______1_______
        |               |
      4 |               | 2
        |               |
        |_______________|
                3
    """
    arrival_times = np.array([])
    travel_distances = np.array([])

    """Calculate the direct wave travel time"""
    direct_travel_distance = np.linalg.norm(actuator.coordinates - sensor.coordinates)
    travel_distances = np.append(travel_distances, direct_travel_distance)
    direct_travel_time = direct_travel_distance / propagation_speed
    arrival_times = np.append(arrival_times, direct_travel_time)

    if print_info and not relative_first_reflection:
        print_direct_travel_info(direct_travel_distance, direct_travel_time)

    EDGES = np.array([1, 2, 3, 4])
    # Iterate thorugh all combinations of edges to reflect from
    for edge_1 in range(0, EDGES.size + 1):
        for edge_2 in range(0, EDGES.size + 1):
            if ignore_edge_combination(edge_1, edge_2):
                continue
            mirrored_source = find_mirrored_source(
                actuator, np.array([edge_1, edge_2]), surface
            )
            distance_to_sensor = np.linalg.norm(
                mirrored_source.coordinates - sensor.coordinates
            )
            if relative_first_reflection:
                distance_to_sensor -= direct_travel_distance
            time_to_sensors = distance_to_sensor / propagation_speed
            print_reflections_info(
                print_info, edge_1, edge_2, distance_to_sensor, time_to_sensors
            )
            travel_distances = np.append(travel_distances, distance_to_sensor)
            arrival_times = np.append(arrival_times, time_to_sensors)

    if milliseconds:
        arrival_times *= 1000
    if relative_first_reflection:
        travel_distances[0] = 0
        arrival_times[0] = 0

    return arrival_times, travel_distances

def print_reflections_info(
    print_info, edge_1, edge_2, distance_to_sensor, time_to_sensors
):
    if not edge_1:
        if print_info:
            print(
                f"\nReflecting from {edge_2}: \t \
                              Distance: {np.round(distance_to_sensor, 5)} m \t\
                              Time: {np.round(time_to_sensors, 6)} s"
            )
    else:
        if print_info:
            print(
                f"\nReflecting from {edge_1}, then {edge_2}:      \
                            \t Distance: {np.round(distance_to_sensor, 5)} m\
                            \t Time: {np.round(time_to_sensors, 6)} s"
            )


def print_direct_travel_info(direct_travel_distance, direct_travel_time):
    print(
        f"\nDirect travel distance: \
              {np.round(direct_travel_distance, 5)} m"
    )
    print(f"\nDirect travel time: {np.round(direct_travel_time, 6)} s")

if __name__ == '__main__':
    signal_df = csv_to_df(file_folder='first_test_touch_passive_setup2',
                          file_name='touch_test_passive_setup2_place_C3_center_v2')
    find_indices_of_peaks(signal_df)
