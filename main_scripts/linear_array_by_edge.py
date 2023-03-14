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
    SETUP.draw()

    FILE_FOLDER = 'Plate_10mm/Setup4/'
    FILE_NAME = 'nik_touch_35_35_v2'
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    measurements['Actuator'] = 0
    measurements = crop_data(measurements,
                             time_start=1.0475,
                             time_end=1.0475 + 0.0025,
                             sample_rate=ORIGINAL_SAMPLE_RATE,)
    # measurements = crop_dataframe_to_signals(measurements)
    measurements = interpolate_waveform(measurements)
    CUTOFF_FREQUENCY = 50000
    measurements = filter(measurements,
                          filtertype='highpass',
                          critical_frequency=CUTOFF_FREQUENCY,
                          plot_response=False,)
    _, distances = compare_to_ideal_signal(SETUP,
                                           measurements,
                                           attenuation_dBpm=25,
                                           chirp_length_s=0.125,
                                           frequency_start=1,
                                           frequency_stop=0.7 * CUTOFF_FREQUENCY,)
    envelopes = get_envelopes(measurements)
    fig, ax = plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title('Measurements with Peaks')
    time_axis_s = np.arange(len(envelopes['Sensor 1'])) / SAMPLE_RATE
    peaks_indices = []

    for i, channel in enumerate(['Sensor 1', 'Sensor 2']):
        color = plt.cm.get_cmap('tab10')(i)
        ax.plot(time_axis_s, envelopes[channel], label=channel, color=color)
        peak_indices, _ = signal.find_peaks(envelopes[channel])
        peak_heights = envelopes[channel][peak_indices]
        sorted_peaks = sorted(zip(peak_indices, peak_heights),
                              key=lambda x: x[1], reverse=True)
        highest_peaks = sorted_peaks[:2]
        highest_peaks = np.array(highest_peaks)
        peaks_indices.append(highest_peaks[:, 0])
        for peak in highest_peaks:
            peak_x = peak[0] / SAMPLE_RATE
            ax.axvline(x=peak_x, color=color, linestyle='--',
                       label=f'{channel} peaks')
    plot_legend_without_duplicates()
    ax.grid()

    propagation_speed_sensor1 = (distances[0][1] - distances[0][0]) / \
        ((peaks_indices[0][0] - peaks_indices[0][1]) / SAMPLE_RATE)
    print(f'Propagation speed sensor 1: {np.abs(propagation_speed_sensor1):.2f} m/s')
    propagation_speed_sensor2 = (distances[1][1] - distances[1][0]) / \
        ((peaks_indices[1][0] - peaks_indices[1][1]) / SAMPLE_RATE)
    print(f'Propagation speed sensor 2: {np.abs(propagation_speed_sensor2):.2f} m/s')
    propagation_speed_mean = (
        propagation_speed_sensor1 + propagation_speed_sensor2) / 2
    print(f'Propagation speed mean: {np.abs(propagation_speed_mean):.2f} m/s')

if __name__ == '__main__':
    raise RuntimeError('This file is not meant to be run directly.')
