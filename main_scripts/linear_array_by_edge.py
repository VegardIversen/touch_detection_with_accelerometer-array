

from matplotlib import pyplot as plt
import numpy as np
from main_scripts.generate_ideal_signal import compare_to_ideal_signal
from utils.csv_to_df import csv_to_df
from utils.data_processing.detect_echoes import get_envelopes
from utils.data_processing.preprocessing import (filter,
                                                 crop_dataframe_to_signals)
from utils.data_processing.processing import interpolate_waveform
from utils.data_visualization.visualize_data import compare_signals
from utils.plate_setups import Setup4


def linear_array_by_edge():
    SETUP = Setup4(actuator_coordinates=np.array([0.35, 0.35]))
    SETUP.draw()

    FILE_FOLDER = 'Plate_10mm/Setup4/'
    FILE_NAME = 'nik_touch_35_35_v1'
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    measurements = interpolate_waveform(measurements)
    CUTOFF_FREQUENCY = 1
    # measurements = filter(measurements,
    #                       filtertype='highpass',
    #                       critical_frequency=CUTOFF_FREQUENCY,)
    envelopes = get_envelopes(measurements)
    # crop_dataframe_to_signals(measurements, threshold_parameter=0.0001)
    PLOTS_TO_PLOT = ['time']
    fig, axs = plt.subplots(nrows=3,
                            ncols=len(PLOTS_TO_PLOT),
                            sharex=True,
                            sharey=True,
                            squeeze=False,)
    compare_signals(fig, axs,
                    [measurements['Sensor 1'],
                     measurements['Sensor 2'],
                     measurements['Sensor 3']],
                    plots_to_plot=PLOTS_TO_PLOT,
                    nfft=2**10,
                    freq_max=10000)
    compare_signals(fig, axs,
                    [envelopes['Sensor 1'],
                     envelopes['Sensor 2'],
                     envelopes['Sensor 3']],
                    plots_to_plot=PLOTS_TO_PLOT,
                    nfft=2**10,
                    freq_max=10000)
    [ax.grid() for ax in axs[:, 0]]
    compare_to_ideal_signal(SETUP,
                            measurements,
                            attenuation_dBpm=7,
                            chirp_length_s=0.125,
                            frequency_start=CUTOFF_FREQUENCY,
                            frequency_stop=3000,)


if __name__ == '__main__':
    raise RuntimeError('This file is not meant to be run directly.')
