from matplotlib import pyplot as plt
import numpy as np
from main_scripts.generate_ideal_signal import compare_to_ideal_signal
from utils.csv_to_df import csv_to_df
from utils.data_processing.detect_echoes import get_envelopes
from utils.data_processing.preprocessing import crop_to_signal, filter
from utils.data_processing.processing import interpolate_waveform
from utils.data_visualization.visualize_data import compare_signals
from utils.global_constants import SAMPLE_RATE
from utils.plate_setups import Setup3


def inspect_touch():
    SETUP = Setup3()
    SETUP.draw()

    FILE_FOLDER = "Plate_10mm/Setup3/touch"
    FILE_NAME = "nik_touch_v1"
    measurements = csv_to_df(file_folder=FILE_FOLDER, file_name=FILE_NAME)
    measurements = crop_to_signal(measurements)
    CUTOFF_FREQUENCY = 5000
    measurements = filter(
        measurements,
        filtertype="highpass",
        critical_frequency=CUTOFF_FREQUENCY,
        order=4,
    )
    measurements = interpolate_waveform(measurements)
    envelopes = get_envelopes(measurements)
    PLOTS_TO_PLOT = ["time", "spectrogram"]
    fig, axs = plt.subplots(
        nrows=3,
        ncols=len(PLOTS_TO_PLOT),
        squeeze=False,
    )
    compare_signals(
        fig,
        axs,
        [measurements["Sensor 1"], measurements["Sensor 2"], measurements["Sensor 3"]],
        plots_to_plot=PLOTS_TO_PLOT,
        nfft=2**14,
        freq_max=10000,
    )
    # compare_signals(fig, axs,
    #                 [envelopes['Sensor 1'],
    #                  envelopes['Sensor 2'],
    #                  envelopes['Sensor 3']],
    #                 plots_to_plot=PLOTS_TO_PLOT,
    #                 nfft=2**10,
    #                 freq_max=10000)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(PLOTS_TO_PLOT),
        squeeze=False,
    )
    time_axis = np.linspace(
        0, measurements.shape[0] / SAMPLE_RATE, measurements.shape[0]
    )
    axs[0, 0].plot(time_axis, measurements["Sensor 1"])
    # axs[0, 0].plot(time_axis, measurements['Sensor 2'])
    # axs[0, 0].plot(time_axis, measurements['Sensor 3'])
    [ax.grid() for ax in axs[:, 0]]
    [ax.set_xlabel("Time [s]") for ax in axs[:, 0]]
    [ax.set_ylabel("Amplitude") for ax in axs[:, 0]]
    axs[0, 0].legend(["Sensor 1"], loc="upper right")
    compare_to_ideal_signal(
        SETUP,
        measurements,
        attenuation_dBpm=15,
        chirp_length_s=0.125,
        frequency_start=1,
        frequency_stop=40000,
    )

    # Set the title to the actuator coordinates
    # [ax.set_ylabel('') for ax in axs[:, 0]]


if __name__ == "__main__":
    raise RuntimeError("This script is not meant to run standalone.")
