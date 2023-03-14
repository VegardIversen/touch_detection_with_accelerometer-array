import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from main_scripts.generate_ideal_signal import compare_to_ideal_signal
from utils.csv_to_df import csv_to_df
from utils.data_processing.detect_echoes import get_envelopes
from utils.data_processing.preprocessing import (crop_data,
                                                 crop_dataframe_to_signals,
                                                 filter)
from utils.data_processing.processing import interpolate_waveform
from utils.data_visualization.drawing import plot_legend_without_duplicates
from utils.data_visualization.visualize_data import compare_signals
from utils.global_constants import ASSERTIONS, ORIGINAL_SAMPLE_RATE, SAMPLE_RATE, SENSOR_2
from utils.little_helpers import distance_between
from utils.plate_setups import Setup4


def linear_array_by_edge():
    SETUP = Setup4(actuator_coordinates=np.array([0.35, 0.35]))

    FILE_FOLDER = 'Plate_10mm/Setup4/'
    FILE_NAME = 'nik_touch_35_35_v1'
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    measurements['Actuator'] = 0
    # Swap the data in column 'Sensor 1' with the data in column 'Sensor 3'
    measurements['Sensor 1'], measurements['Sensor 3'] = measurements['Sensor 3'], measurements['Sensor 1'].copy()
    measurements = crop_data(measurements,
                             time_start=1.135,
                             time_end=1.15,
                             sample_rate=ORIGINAL_SAMPLE_RATE,)
    # measurements = crop_dataframe_to_signals(measurements)
    measurements = interpolate_waveform(measurements)
    CRITICAL_FREQUENCY = 6000
    measurements = filter(measurements,
                          filtertype='highpass',
                          critical_frequency=CRITICAL_FREQUENCY,
                          plot_response=False,
                          order=6)
    # CRITICAL_FREQUENCY = 45000
    measurements = filter(measurements,
                          filtertype='lowpass',
                          critical_frequency=CRITICAL_FREQUENCY,
                          plot_response=False,
                          order=6)
    _, distances = compare_to_ideal_signal(SETUP,
                                           measurements,
                                           attenuation_dBpm=20,
                                           chirp_length_s=0.125,
                                           frequency_start=1,
                                           frequency_stop=1 * CRITICAL_FREQUENCY,)
    calculate_source_location(measurements, distances, SETUP)
    return 0


def calculate_source_location(measurements, distances, setup):
    """Based the direct wave and first reflection"""
    envelopes = get_envelopes(measurements)
    fig, ax = plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    time_axis_s = np.arange(len(envelopes['Sensor 1'])) / SAMPLE_RATE
    peaks_indices = np.empty((0, 2))

    for i, channel in enumerate(['Sensor 3', 'Sensor 2']):
        color = plt.cm.get_cmap('tab10')(i)
        ax.plot(time_axis_s, envelopes[channel], label=channel, color=color)
        peak_indices, _ = signal.find_peaks(envelopes[channel], distance=0.0001 * SAMPLE_RATE)
        peak_heights = envelopes[channel][peak_indices]
        sorted_peaks = sorted(zip(peak_indices, peak_heights),
                              key=lambda x: x[1], reverse=True)
        highest_peaks = sorted_peaks[:2]
        highest_peaks = np.array(highest_peaks)
        peaks_indices = np.vstack((peaks_indices, highest_peaks[:, 0]))
        for peak in highest_peaks:
            peak_x = peak[0] / SAMPLE_RATE
            ax.axvline(x=peak_x, color=color, linestyle='--',
                       label=f'{channel} peaks')
    plot_legend_without_duplicates(placement='upper right')
    ax.grid()

    peaks_indices = peaks_indices / SAMPLE_RATE
    propagation_speed_sensor1 = np.abs((distances[0][1] - distances[0][0]) /
                                       ((peaks_indices[0][0] - peaks_indices[0][1])))
    print(f'Propagation speed sensor 1: {propagation_speed_sensor1:.2f} m/s')
    propagation_speed_sensor2 = np.abs((distances[1][1] - distances[1][0]) /
                                       ((peaks_indices[1][0] - peaks_indices[1][1])))
    print(f'Propagation speed sensor 2: {propagation_speed_sensor2:.2f} m/s')
    propagation_speed_mean = (
        propagation_speed_sensor1 + propagation_speed_sensor2) / 2
    print(f'Propagation speed mean: {propagation_speed_mean:.2f} m/s')
    propagation_speed = propagation_speed_mean

    # Calculate phi, the angle between source and sensors
    phi_time_difference = (peaks_indices[0][0] - peaks_indices[1][0])
    phi_arccos_argument = (phi_time_difference * propagation_speed) / 0.01
    phi_rad = np.arccos(phi_arccos_argument)
    phi_deg = np.rad2deg(phi_rad)
    print(f'phi: {phi_rad:.2f} rad, or {phi_deg:.2f} deg')

    # Calculate theta, the angle between mirrored source and sensors
    theta_time_difference = (peaks_indices[0][1] - peaks_indices[1][1])
    theta_arccos_argument = (theta_time_difference * propagation_speed) / 0.01
    if ASSERTIONS:
        assert ((np.abs(phi_arccos_argument) <= 1) and (np.abs(theta_arccos_argument) <= 1)), \
            f'Phi arccos argument is {phi_arccos_argument:.2f}, theta arccos argument is {theta_arccos_argument:.2f}.'
    theta_rad = np.arccos(theta_arccos_argument)
    theta_deg = np.rad2deg(theta_rad)
    print(f'theta: {theta_rad:.2f} rad, or {theta_deg:.2f} deg')

    distance_to_source = np.abs((2 * 0.05) /
                                ((np.tan(theta_rad) - np.tan(phi_rad)) * np.cos(phi_rad)))
    print(f'Distance to source: {distance_to_source:.2f} m')
    plt.figure()
    setup.draw()
    estimated_source_x = setup.sensors[SENSOR_2].x - \
        distance_to_source * np.cos(phi_rad)
    estimated_source_y = setup.sensors[SENSOR_2].y - \
        distance_to_source * np.sin(phi_rad)
    plt.scatter(estimated_source_x,
                estimated_source_y,
                label='Estimated source',
                color='#D4434A',
                marker='x',)
    plot_legend_without_duplicates(placement='lower right')


if __name__ == '__main__':
    raise RuntimeError('This file is not meant to be run directly.')
