import matplotlib.pyplot as plt

from utils.csv_to_df import make_dataframe_from_csv
from utils.data_processing.preprocessing import crop_to_signal, filter_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals


def inspect_touch_release():
    FILE_FOLDER = "Plate_10mm/Setup4/swipe"
    FILE_NAME = "nik_swipe_35to50_center_v3"
    measurements = make_dataframe_from_csv(file_folder=FILE_FOLDER, file_name=FILE_NAME)

    measurements["Actuator"] = 0
    measurements = crop_to_signal(measurements)
    measurements = filter_signal(
        measurements,
        filtertype="highpass",
        critical_frequency=30000,
        order=8,
    )
    measurements = filter_signal(
        measurements,
        filtertype="lowpass",
        critical_frequency=35000,
        order=8,
    )
    measurements = interpolate_signal(measurements)

    fig, axs = plt.subplots(
        nrows=3,
        ncols=2,
        squeeze=False,
    )
    compare_signals(
        fig,
        axs,
        [measurements["Sensor 1"], measurements["Sensor 2"], measurements["Sensor 3"]],
        plots_to_plot=["time", "spectrogram"],
        nfft=2**12,
        freq_max=50000
    )


if __name__ == "__main__":
    raise RuntimeError("This script is not meant to run standalone.")
