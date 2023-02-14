import scipy
import scipy.signal as signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.global_constants import (
    SAMPLE_RATE,
    ACTUATOR_1,
    SENSOR_1,
    SENSOR_2,
    SENSOR_3,
    FIGURES_SAVE_PATH,
)
from utils.csv_to_df import csv_to_df
from utils.data_processing.detect_echoes import (
    get_travel_times,
    find_first_peak_index,
)
from utils.data_processing.preprocessing import (
    filter_general,
    crop_measurement_to_signal_ndarray,
    crop_measurements_to_signals_dataframe,
)
from utils.data_processing.processing import (
    interpolate_waveform,
    normalize,
)
from utils.data_visualization.visualize_data import (
    compare_signals,
    envelope_with_lines,
)
from utils.plate_setups import Setup, Setup1, Setup2, Setup3


def dispersive_filter():
    # Define the length of the signal in seconds
    LENGTH = 0.1
    SAMPLE_RATE = 1e7

    # Calculate the number of samples
    N = int(SAMPLE_RATE * LENGTH)

    # Generate a chirp with bandwidth 40 kHz
    chirp = signal.chirp(np.linspace(0, LENGTH, N), 0, LENGTH, 40000)
    impulse = signal.correlate(chirp, chirp, mode="same")

    # Calculate the FFT
    fft = np.fft.fft(impulse)
    fft_frequencies = np.fft.fftfreq(N, 1 / SAMPLE_RATE)
    # Crop to 40 kHz
    fft = fft[np.abs(fft_frequencies) < 40000]
    fft_frequencies = fft_frequencies[np.abs(fft_frequencies) < 40000]

    phase_parameter = 0.01  #
    # A linear graph providing the phase offset
    phase_offset = np.linspace(phase_parameter, -phase_parameter, len(fft))
    # Do an fft shift on the phase offset
    phase_offset = np.fft.fftshift(phase_offset)
    phase_offset = phase_parameter - np.abs(phase_offset) + 0.05

    # Plot the phase offset
    _, ax = plt.subplots()
    ax.plot(fft_frequencies, phase_offset)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase offset (rad)")
    ax.set_title("Phase response of the dispersive filter, +0.5 for time offset")
    ax.grid()

    new_chirp = np.fft.ifft(fft * np.exp(-1j * phase_offset * fft_frequencies))

    new_chirp = signal.resample(new_chirp, len(chirp))

    # Compansate for the insane amplitude boost happening somewhere
    impulse = normalize(impulse)
    new_chirp = normalize(new_chirp) / 2

    # Calculate the time values for each sample
    time = np.linspace(0, LENGTH, N)
    cropped_time = np.linspace(0, LENGTH, len(new_chirp))

    # Plot the signal
    _, ax = plt.subplots()
    ax.plot(time, impulse, label="Original impulse chirp")
    ax.plot(cropped_time, new_chirp, label="Phase-altered impulse")
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Dispersion simulation on a 0-40kHz compressed chirp")
    ax.grid()

    return 0


def correct_dispersion():
    setup = Setup1()
    setup.draw()
    FILE_FOLDER = "Plate_10mm/Setup1/touch"
    FILE_NAME = "nik_touch_v2"
    measurements = csv_to_df(file_folder=FILE_FOLDER, file_name=FILE_NAME)

    measurements = measurements.drop(columns=["Actuator"])

    measurements = interpolate_waveform(measurements)

    BANDWIDTH = [4000, 40000]
    measurements = filter_general(
        measurements,
        filtertype="highpass",
        critical_frequency=BANDWIDTH[0],
        order=2**5,
        plot_response=False,
    )

    measurements = crop_measurements_to_signals_dataframe(
        measurements,
        std_threshold_multiplier=10,
    )

    # measurements = crop_measurements_to_signals_dataframe(measurements,
    #                                                       std_threshold_multiplier=7,)

    # measurement = crop_measurement_to_signal_ndarray(measurements['Sensor 3'],
    #                                                  std_threshold_multiplier=1,)

    # Only look at one sensor for now
    measurement = measurements["Sensor 1"]
    measurement = np.pad(measurement.values, (len(measurement) // 2, 0))

    # Calculate the FFT
    fft = np.fft.fft(measurement)
    fft_frequencies = np.fft.fftfreq(len(measurement), 1 / SAMPLE_RATE)
    # Crop to an upper limit
    fft = fft[np.abs(fft_frequencies) < BANDWIDTH[1]]
    fft_frequencies = fft_frequencies[np.abs(fft_frequencies) < BANDWIDTH[1]]

    # A linear graph providing the phase offset, with some trickery for correct lengths
    phase_response = np.linspace(0, -40, len(fft_frequencies) // 2 + 1) - 10 * np.pi
    # phase_response = np.pad(phase_response, (len(fft_frequencies) - len(phase_response), 0))
    # phase_response = np.fft.fftshift(phase_response)
    phase_offsets = -(
        2
        * np.pi
        * (fft_frequencies[len(fft_frequencies) - len(phase_response) :])
        * 0.1
        / phase_response
    )
    # Concatenate phase_offset so that is mirrored around 0
    phase_offsets = np.concatenate((-phase_offsets, phase_offsets[-1:0:-1]))

    phase_parameter = 0.2
    phase_offsets = phase_offsets * phase_parameter

    # phase_offsets = -1 / fft_frequencies
    # phase_offsets[:int(len(phase_offsets) / 2)] = -phase_offsets[:int(len(phase_offsets) / 2)]
    # phase_offsets = (optimal_phase_parameter / (fft_frequencies))
    # # Set values between -2000Hz and 2000 Hz to zero

    # Plot the phase offset
    _, ax = plt.subplots()
    ax.plot(fft_frequencies, phase_offsets)
    # ax.plot(fft_frequencies, phase_response)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase offset (rad)")
    ax.grid()

    # Correct the phase offset
    corrected_ffts = fft * np.exp(-1j * phase_offsets)

    # Calculate the inverse FFT
    corrected_measurement = np.fft.ifft(corrected_ffts)

    corrected_measurement = signal.resample(corrected_measurement, len(measurement))

    corrected_measurement = crop_measurement_to_signal_ndarray(corrected_measurement)

    # # Plot the corrected measurements
    fig, ax = plt.subplots(2, 1, squeeze=False)
    compare_signals(
        fig, ax, [measurement, np.real(corrected_measurement)], plots_to_plot=["time"]
    )
    compare_signals(
        fig,
        ax,
        [
            np.abs(signal.hilbert(np.real(measurement))),
            np.abs(signal.hilbert(np.real(corrected_measurement))),
        ],
        plots_to_plot=["time"],
    )
    ax[0, 0].set_title("Original")
    ax[1, 0].set_title("Corrected")
    ax[0, 0].grid()
    ax[1, 0].grid()

    propagation_speed = setup.get_propagation_speed(measurements)
    # propagation_speed = 75
    arrival_times, _ = get_travel_times(
        setup.actuators[ACTUATOR_1],
        setup.sensors[SENSOR_1],
        propagation_speed,
        milliseconds=False,
        relative_first_reflection=True,
        print_info=False,
    )
    # max_index = np.argmax(np.abs(signal.hilbert(np.real(corrected_measurement))))
    max_index = find_first_peak_index(
        np.abs(signal.hilbert(np.real(corrected_measurement)))
    )
    correction_offset = max_index
    correction_offset_time = correction_offset / SAMPLE_RATE
    arrival_times += correction_offset_time
    envelope_with_lines(setup.sensors[SENSOR_1], corrected_measurement, arrival_times)

    fig, ax = plt.subplots(2, 3, squeeze=False)
    compare_signals(
        fig,
        ax,
        [measurement, corrected_measurement],
        plots_to_plot=["time", "spectrogram", "fft"],
        dynamic_range_db=40,
        nfft=2**8,
        freq_max=20000,
    )

    return 0


def get_prominences_of_peaks(measurement: np.ndarray):
    # Get the standard deviation
    std = np.std(measurement)

    # Get the peaks
    peaks, _ = signal.find_peaks(measurement, height=2 * std, prominence=std)

    # Get the prominence of the peaks
    prominence = signal.peak_prominences(measurement, peaks)[0]

    # Plot the peaks
    # _, ax = plt.subplots()
    # ax.plot(measurement)
    # ax.plot(peaks, measurement[peaks], 'x')
    # ax.set_xlabel('Sample')
    # ax.set_ylabel('Amplitude')
    # ax.grid()

    return prominence
