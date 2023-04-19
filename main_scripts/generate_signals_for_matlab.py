import numpy as np
from matplotlib import pyplot as plt

from utils.csv_to_df import make_dataframe_from_csv
from utils.data_processing.detect_echoes import get_analytic_signal, get_envelopes
from utils.data_processing.preprocessing import crop_data, crop_to_signal, filter_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals
from utils.global_constants import ORIGINAL_SAMPLE_RATE
from utils.plate_setups import Setup4


def generate_signals_for_matlab():
    SETUP = Setup4(actuator_coordinates=np.array([0.35, 0.35]))

    FILE_FOLDER = "Plate_10mm/Setup4/touch"
    FILE_NAME = "nik_touch_35_35_v1"
    measurements = make_dataframe_from_csv(file_folder=FILE_FOLDER, file_name=FILE_NAME)

    measurements["Actuator"] = 0
    # measurements = crop_to_signal(measurements)
    measurements = crop_data(
        measurements,
        time_start=1.14,
        time_end=1.14072,
        sample_rate=ORIGINAL_SAMPLE_RATE,
    )
    CRITICAL_FREQUENCY = 30000
    measurements = filter_signal(
        measurements,
        filtertype="bandpass",
        critical_frequency=CRITICAL_FREQUENCY,
        plot_response=False,
        order=2,
        q=0.05,
    )
    # CRITICAL_FREQUENCY = 6050
    # measurements = filter_signal(
    #     measurements,
    #     filtertype="highpass",
    #     critical_frequency=CRITICAL_FREQUENCY,
    #     plot_response=False,
    #     order=8,
    # )
    # measurements = crop_to_signal(measurements)
    measurements = measurements.drop(columns=["Actuator"])
    measurements = interpolate_signal(measurements)

    # Plot the signals
    fig, axs = plt.subplots(1, 2, squeeze=False)
    compare_signals(
        fig,
        axs,
        [measurements["Sensor 1"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [measurements["Sensor 2"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [measurements["Sensor 3"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )
    axs[0, 0].legend(["Sensor 1", "Sensor 2", "Sensor 3"], loc="upper right")

    analytic_signals = get_analytic_signal(measurements)
    analytic_signals.to_csv(
        "./matlab_signals.csv",
        index=False,
        header=False,
    )

    envelopes = get_envelopes(measurements)
    fig, axs = plt.subplots(1, 1, squeeze=False)
    compare_signals(
        fig,
        axs,
        [envelopes["Sensor 1"]],
        plots_to_plot=["time"],
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [envelopes["Sensor 2"]],
        plots_to_plot=["time"],
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [envelopes["Sensor 3"]],
        plots_to_plot=["time"],
        sharey=True,
    )
