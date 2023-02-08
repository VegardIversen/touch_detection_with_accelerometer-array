import scipy
import scipy.signal as signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.global_constants import (CHIRP_CHANNEL_NAMES,
                                    SAMPLE_RATE,
                                    ACTUATOR_1,
                                    SENSOR_1,
                                    SENSOR_2,
                                    SENSOR_3,
                                    FIGURES_SAVE_PATH)
from utils.csv_to_df import csv_to_df
from utils.simulations import simulated_phase_velocities
from utils.data_processing.detect_echoes import (get_envelopes,
                                                 get_travel_times,
                                                 find_first_peak_index)
from utils.data_processing.preprocessing import (compress_chirps,
                                                 crop_data,
                                                 window_signals,
                                                 filter_general)
from utils.data_processing.processing import (average_of_signals,
                                              interpolate_waveform,
                                              normalize,
                                              variance_of_signals,
                                              correct_drift,
                                              to_dB)
from utils.data_visualization.visualize_data import (compare_signals,
                                                     wave_statistics,
                                                     envelope_with_lines,
                                                     spectrogram_with_lines,
                                                     set_window_size,
                                                     adjust_plot_margins)
from main_scripts.correlation_bandpassing import (make_gaussian_cosine)

from utils.setups import (Setup,
                          Setup1,
                          Setup2,
                          Setup3)


def dispersive_filter():
    # Define the length of the signal in seconds
    LENGTH = 0.1
    SAMPLE_RATE = 1e7

    # Calculate the number of samples
    N = int(SAMPLE_RATE * LENGTH)

    # Generate a chirp with bandwidth 40 kHz
    chirp = signal.chirp(np.linspace(0, LENGTH, N), 0, LENGTH, 40000)
    impulse = signal.correlate(chirp, chirp, mode='same')

    # Calculate the FFT
    impulse_fft = np.fft.fft(impulse)
    impulse_fft_frequencies = np.fft.fftfreq(N, 1 / SAMPLE_RATE)
    # Crop to 40 kHz
    impulse_fft = impulse_fft[impulse_fft_frequencies < 40000]
    impulse_fft_frequencies = impulse_fft_frequencies[impulse_fft_frequencies < 40000]

    # A linear graph providing the phase offset
    phase_offset = np.linspace(0.05, -0.05, len(impulse_fft))
    # Do an fft shift on the phase offset
    phase_offset = np.fft.fftshift(phase_offset)

    # Plot the phase offset
    _, ax = plt.subplots()
    ax.plot(impulse_fft_frequencies, phase_offset)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase offset (rad)')
    ax.grid()

    new_chirp = np.fft.ifft(
        impulse_fft * np.exp(-1j * phase_offset * impulse_fft_frequencies))

    # Set the values before time 0 in new_chirp to 0
    new_chirp[:len(new_chirp) // 2] = 0

    # Calculate the time values for each sample
    time = np.linspace(0, LENGTH, N)
    cropped_time = np.linspace(0, LENGTH, len(new_chirp))

    # Plot the signal
    _, ax = plt.subplots()
    ax.plot(time, impulse, label='Original')
    ax.plot(cropped_time, new_chirp, label='New chirp')
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid()

    return 0


def correct_dispersion():
    setup = Setup1()
    # Open file
    FILE_FOLDER = 'Setup_1/touch'
    FILE_NAME = 'touch_v1'
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    # Interpolate waveforms
    measurements = interpolate_waveform(measurements)

    measurement = measurements['Sensor 3']

    # Calculate the FFT
    ffts = np.fft.fft(measurement)
    fft_frequencies = np.fft.fftfreq(len(measurement), 1 / SAMPLE_RATE)
    # Crop to 40 kHz
    ffts = ffts[np.abs(fft_frequencies) < 40000]
    fft_frequencies = fft_frequencies[np.abs(fft_frequencies) < 40000]

    FIND_PHASE_PARAMETER = True
    if FIND_PHASE_PARAMETER:
        PHASE_PARAMETERS = np.linspace(-100, 100, 500)
        average_prominences = []
        highest_values = []

        for phase_parameter in PHASE_PARAMETERS:
            # A linear graph providing the phase offset
            phase_offset = np.linspace(phase_parameter, -phase_parameter, len(ffts))
            # Do an fft shift on the phase offset
            phase_offset = np.fft.fftshift(phase_offset)
            phase_offset = phase_parameter + np.abs(phase_offset)

            # Correct the phase offset
            corrected_ffts = ffts * np.exp(-1j * phase_offset * fft_frequencies)

            # Calculate the inverse FFT
            corrected_chirp = np.fft.ifft(corrected_ffts)

            corrected_chirp = signal.resample(corrected_chirp, len(measurement))

            # Get the prominence of the peaks
            average_prominence = np.average(get_prominences_of_peaks(
                np.abs(signal.hilbert(np.real(corrected_chirp)))))
            average_prominences.append(average_prominence)
            highest_values.append(np.max(np.abs(signal.hilbert(np.real(corrected_chirp)))))

        # Plot the average prominence
        _, ax = plt.subplots()
        ax.plot(PHASE_PARAMETERS, average_prominences)
        ax.set_xlabel('Phase parameter')
        ax.set_ylabel('Average prominence')
        ax.grid()

        # Plot the highest value
        _, ax = plt.subplots()
        ax.plot(PHASE_PARAMETERS, highest_values)
        ax.set_xlabel('Phase parameter')
        ax.set_ylabel('Highest value')
        ax.grid()

        # Choose which criteria to pick the phase parameter
        optimal_phase_parameter = PHASE_PARAMETERS[np.argmax(average_prominences)]
        # optimal_phase_parameter = PHASE_PARAMETERS[np.argmax(highest_values)]
    else:
        # Set phase_parameter manually
        optimal_phase_parameter = -7.1
    # A linear graph providing the phase offset
    phase_offset = np.linspace(optimal_phase_parameter, -optimal_phase_parameter, len(ffts))
    # Do an fft shift on the phase offset
    phase_offset = np.fft.fftshift(phase_offset)
    phase_offset = optimal_phase_parameter + np.abs(phase_offset)

    # Plot the phase offset
    _, ax = plt.subplots()
    ax.plot(fft_frequencies, phase_offset)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase offset (rad)')
    ax.grid()

    # Correct the phase offset
    corrected_ffts = ffts * np.exp(-1j * phase_offset * fft_frequencies)

    # Calculate the inverse FFT
    corrected_chirp = np.fft.ifft(corrected_ffts)

    corrected_chirp = signal.resample(corrected_chirp, len(measurement))

    # # Plot the corrected measurements
    fig, ax = plt.subplots(2, 1, squeeze=False)
    compare_signals(fig, ax,
                    [measurement,
                     np.real(corrected_chirp)],
                    plots_to_plot=['time'])
    compare_signals(fig, ax,
                    [np.abs(signal.hilbert(np.real(measurement))),
                     np.abs(signal.hilbert(np.real(corrected_chirp)))],
                    plots_to_plot=['time'])
    ax[0, 0].set_title('Original')
    ax[1, 0].set_title('Corrected')
    ax[0, 0].grid()
    ax[1, 0].grid()

    propagation_speed = 75
    arrival_times, _ = get_travel_times(setup.actuators[ACTUATOR_1],
                                        setup.sensors[SENSOR_3],
                                        propagation_speed,
                                        milliseconds=False,
                                        relative_first_reflection=True,
                                        print_info=False)
    max_index = np.argmax(np.abs(signal.hilbert(np.real(corrected_chirp))))
    # zero_index = corrected_chirp.shape[0] / 2
    correction_offset = max_index
    correction_offset_time = correction_offset / SAMPLE_RATE
    arrival_times += correction_offset_time
    envelope_with_lines(setup.sensors[SENSOR_3],
                        corrected_chirp,
                        arrival_times)

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
