import matplotlib.pyplot as plt
from constants import CHIRP_CHANNEL_NAMES
from csv_to_df import csv_to_df
from data_processing.processing import (interpolate_waveform)
from data_processing.preprocessing import (window_signals)
from data_viz_files.visualise_data import (set_fontsizes)
from setups import Setup3_2_without_sensor2
from generate_results import (plot_time_signals,
                              plot_spectrogram_signals,
                              plot_fft_signals)


def main():
    """Run some general commands for all functions:
        - Choose file and open it
        - Channel selection
        - Choose setup and draw it
        - Interpolation
        - Generate results from functions
    """

    """Choose file"""
    FILE_FOLDER = 'prop_speed_files/setup3_2'
    FILE_NAME = 'prop_speed_chirp3_setup3_2_v1'
    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Delete sensor 2 as it doesn't have the required bandwidth"""
    measurements = measurements.drop(['Sensor 2'], axis='columns')
    CHIRP_CHANNEL_NAMES.remove('Sensor 2')

    """Choose setup"""
    SETUP = Setup3_2_without_sensor2()
    """Draw setup"""
    SETUP.draw()

    """Pyplot adjustments"""
    set_fontsizes()

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Set everything but the signal to zero"""
    signal_length_seconds = 2 + 0.05  # Length of chirp + time for sensor 3 to die down
    threshold = 0.001  # Determine empirically
    measurements, signal_start_seconds = window_signals(measurements,
                                                        signal_length_seconds,
                                                        threshold)

    """Run functions from generate_results.py"""
    FIGSIZE_ONE_COLUMN = (8, 9)
    plot_time_signals(measurements,
                      signal_start_seconds,
                      signal_length_seconds,
                      FIGSIZE_ONE_COLUMN)
    plot_spectrogram_signals(measurements,
                             signal_start_seconds,
                             signal_length_seconds,
                             FIGSIZE_ONE_COLUMN)
    plot_fft_signals(measurements,
                     signal_start_seconds,
                     signal_length_seconds,
                     FIGSIZE_ONE_COLUMN)
    plt.show()


if __name__ == '__main__':
    main()
