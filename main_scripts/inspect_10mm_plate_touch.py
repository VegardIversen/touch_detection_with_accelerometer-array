from matplotlib import pyplot as plt
import numpy as np
from main_scripts.generate_ideal_signal import compare_to_ideal_signal
from utils.csv_to_df import make_dataframe_from_csv
from utils.data_processing.detect_echoes import get_envelopes
from utils.data_processing.preprocessing import crop_data, crop_to_signal, filter_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals
from utils.global_constants import ORIGINAL_SAMPLE_RATE, SAMPLE_RATE
from utils.plate_setups import Setup3


def inspect_touch():
    SETUP = Setup3()
    SETUP.draw()

    FILE_FOLDER = "Plate_10mm/Setup3/touch"
    FILE_NAME = "nik_touch_v1"
    measurements = make_dataframe_from_csv(file_folder=FILE_FOLDER, file_name=FILE_NAME)
    measurements = crop_to_signal(measurements)
    CRITICAL_FREQUENCY = 15000
    measurements = filter_signal(
        measurements,
        filtertype="highpass",
        critical_frequency=CRITICAL_FREQUENCY,
        order=4,
    )
    measurements = interpolate_signal(measurements)
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
        nfft=2**5,
        freq_max=40000,
        dynamic_range_db=14,
    )
    fig.suptitle("Measurements")
    compare_to_ideal_signal(
        SETUP,
        measurements,
        attenuation_dBpm=15,
        cutoff_frequency=CRITICAL_FREQUENCY,
    )

if __name__ == "__main__":
    raise RuntimeError("This script is not meant to run standalone.")
