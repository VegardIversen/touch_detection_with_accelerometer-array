import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd

from csv_to_df import csv_to_df
from data_processing.preprocessing import filter_general, crop_data
from objects import MirroredSensor, MirroredSource, Table, Actuator, Sensor


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
            ax0.plot(time_axis / 150000, signal_sqr, label='sqrd')
            ax0.plot(time_axis[peak_indices] / 150000, signal_sqr[peak_indices], 'x', label='peaks')

        else:
            ax0.plot(time_axis / 150000, sig_np_filtered_hilbert, label='filtered hilbert')
            ax0.plot(time_axis[peak_indices] / 150000, sig_np_filtered_hilbert[peak_indices], 'x', label='peaks')
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


def get_expected_reflections_pos(speed, peak, s=[0.26, 0.337, 0.386, 0.41], Fs=150000):
    t = s / speed
    n = t * Fs + peak
    return n.tolist()


def find_mirrored_source(actuator: Actuator,
                         edges_to_bounce_on: np.array):
    """Calculate the coordinate of the mirrored source
    to be used to find the wave travel distance.
    """
    NO_REFLECTION = 0
    mirrored_source = MirroredSource(actuator.coordinates)
    for edge in edges_to_bounce_on:
        if edge == NO_REFLECTION:
            continue
        elif edge == Table.TOP_EDGE:
            MIRRORED_SOURCE_OFFSET = np.array([0, 2 * (Table.WIDTH - mirrored_source.y)])
            mirrored_source.set_coordinates(mirrored_source.coordinates + MIRRORED_SOURCE_OFFSET)
        elif edge == Table.RIGHT_EDGE:
            MIRRORED_SOURCE_OFFSET = np.array([2 * (Table.LENGTH - mirrored_source.x), 0])
            mirrored_source.set_coordinates(mirrored_source.coordinates + MIRRORED_SOURCE_OFFSET)
        elif edge == Table.BOTTOM_EDGE:
            MIRRORED_SOURCE_OFFSET = np.array([0, -2 * mirrored_source.y])
            mirrored_source.set_coordinates( mirrored_source.coordinates + MIRRORED_SOURCE_OFFSET)
        elif edge == Table.LEFT_EDGE:
            MIRRORED_SOURCE_OFFSET = np.array([-2 * mirrored_source.x, 0])
            mirrored_source.set_coordinates(mirrored_source.coordinates + MIRRORED_SOURCE_OFFSET)
    return mirrored_source


def flip_sources(sources: np.array,
                 edges_to_flip_around: np.array):
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


def flip_sensors(sensors: np.array,
                 edges_to_flip_around: np.array):
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


def get_travel_distances_firsts(actuator_coord: np.array,
                                sensor_coord: np.array):
    """Get the travel distance of a mirrored source.
    See figure in "Predicting reflection times" on Notion.
    NOTE:   These are currently only for the four direct reflections.
    TODO:   Add secondary reflections and add option for outputting
            the n first reflections along with their travel paths.
    """
    # Collection of calcualted distances
    # s_norms = pd.DataFrame()
    s_norms = np.array([])

    # Vector between actuator and sensor:
    s_0 = sensor_coord - actuator_coord
    s_0_norm = np.linalg.norm(s_0)

    # Vector from actuator to edge:
    d = actuator_coord - np.array([actuator_coord[0], Table.WIDTH])
    # Vector from mirrored source to sensor:
    s_1_norm = np.linalg.norm(2 * d + s_0)
    # s_norms['s_1'] = [s_1_norm, d]
    s_norms = np.append(s_norms, s_1_norm)

    # Use vector to different edge:
    d = actuator_coord - np.array([Table.LENGTH, actuator_coord[1]])
    # Vector from mirrored source to sensor:
    s_2_norm = np.linalg.norm(2 * d + s_0)
    # s_norms['s_2'] = [s_2_norm, d]
    s_norms = np.append(s_norms, s_2_norm)

    # Use vector to different edge:
    d = actuator_coord - np.array([actuator_coord[0], 0])
    # Vector from mirrored source to sensor:
    s_3_norm = np.linalg.norm(2 * d + s_0)
    # s_norms['s_3'] = [s_3_norm, d]
    s_norms = np.append(s_norms, s_3_norm)

    # Use vector to different edge:
    d = actuator_coord - np.array([0, actuator_coord[1]])
    # Vector from mirrored source to sensor:
    s_4_norm = np.linalg.norm(2 * d + s_0)
    # s_norms['s_4'] = [s_4_norm, d]
    s_norms = np.append(s_norms, s_4_norm)

    return s_norms


def get_travel_distances(actuator: Actuator,
                         sensor: Sensor,
                         print_distances: bool = False):
    """Get the travel distance from first and second reflections.
    TODO:   Add logic for not calculating physically impossible reflections.
            This not necessary for predicting WHEN the reflections will arrive,
            but to visualise which reflections we are using.
    """
    travel_distances = np.array([])
    # Direct travel distance:
    direct_travel_distance = np.linalg.norm(actuator.coordinates - sensor.coordinates)
    travel_distances = np.append(travel_distances, direct_travel_distance)
    print(f"\nDirect travel distance: {direct_travel_distance}")

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
            mirrored_source = find_mirrored_source(actuator, np.array([edge_1, edge_2]))
            distance_to_sensor = np.linalg.norm(mirrored_source.coordinates - sensor.coordinates)
            if not edge_1:
                if print_distances:
                    print(f'\nReflecting from {edge_2}. Distance: {distance_to_sensor}')
            else:
                if print_distances:
                    print(f'\nReflecting from {edge_1}, then {edge_2}. Distance: {distance_to_sensor}')
            travel_distances = np.append(travel_distances, distance_to_sensor)

    return travel_distances


if __name__ == '__main__':
    signal_df = csv_to_df(file_folder='first_test_touch_passive_setup2',
                          file_name='touch_test_passive_setup2_place_C3_center_v2')
    find_indices_of_peaks(signal_df)
