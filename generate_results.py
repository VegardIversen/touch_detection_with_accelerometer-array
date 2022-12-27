"""Generate results for the paper.
This will be a collection of functions that will be called
and print and plot different results with a common configuration.
"""
import scipy
import scipy.signal as signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from global_constants import (CHIRP_CHANNEL_NAMES,
                              SAMPLE_RATE,
                              ACTUATOR_1,
                              SENSOR_1,
                              SENSOR_2,
                              SENSOR_3,
                              FIGURES_SAVE_PATH)
from csv_to_df import csv_to_df
from simulations import simulated_phase_velocities
from data_processing.detect_echoes import (get_envelope,
                                           get_travel_times)
from data_processing.preprocessing import (compress_chirps,
                                           crop_data,
                                           window_signals,
                                           filter_general)
from data_processing.processing import (average_of_signals,
                                        interpolate_waveform,
                                        normalize,
                                        variance_of_signals,
                                        correct_drift,
                                        to_dB)
from data_visualization.visualize_data import (compare_signals,
                                               wave_statistics,
                                               envelope_with_lines,
                                               spectrogram_with_lines,
                                               set_window_size,
                                               subplots_adjust)
from setups import (Setup,
                    Setup3_2,
                    Setup7,
                    Setup9)


"""Setup 3_2"""


def results_setup3_2():
    """Run some general commands for all functions:
        - Choose file and open it
        - Channel selection
        - Choose setup and draw it
        - Interpolation
        - Generate results from functions
    """
    """Choose file"""
    FILE_FOLDER = 'prop_speed_files/setup3_2'
    FILE_NAME = 'prop_speed_chirp3_setup3_2_v2'
    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Delete sensor 2 as it doesn't have the required bandwidth"""
    measurements = measurements.drop(['Sensor 2'], axis='columns')
    CHIRP_CHANNEL_NAMES.remove('Sensor 2')

    """Choose setup"""
    SETUP = Setup3_2()
    """Draw setup"""
    subplots_adjust('setup', rows=1, columns=1)
    SETUP.draw()

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Set everything but the signal to zero"""
    direct_signal_seconds_sensor3 = 2 + 0.05  # Length of chirp + time for sensor 3 to die down
    threshold_amplitude = 0.0015  # Determine empirically
    measurements, signal_start_seconds = window_signals(measurements,
                                                        direct_signal_seconds_sensor3,
                                                        threshold_amplitude)
    """Compress chirps"""
    measurements = compress_chirps(measurements)

    """Crop the compressed chirps to be only the direct signal"""
    # measurements.iloc[:(int(SAMPLE_RATE * 0.005))] = 0
    direct_signal_seconds_sensor1 = 0.0003
    threshold_amplitude = 1000  # Determine empirically
    measurements['Sensor 1'], _ = window_signals(pd.DataFrame(measurements['Sensor 1']),
                                                              direct_signal_seconds_sensor1,
                                                              threshold_amplitude)
    # direct_signal_seconds_sensor2 = 0.00025
    # threshold_amplitude = 500  # Determine empirically
    # measurements['Sensor 2'], _ = window_signals(pd.DataFrame(measurements['Sensor 2']),
    #                                                           direct_signal_seconds_sensor2,
    #                                                           threshold_amplitude)
    direct_signal_seconds_sensor3 = 0.00035
    threshold_amplitude = 50  # Determine empirically
    measurements['Sensor 3'], _ = window_signals((pd.DataFrame(measurements['Sensor 3'])),
                                                               direct_signal_seconds_sensor3,
                                                               threshold_amplitude)
    """Run functions to generate results
    NOTE:   Only change the functions below this line
    """
    find_and_plot_power_loss(measurements,
                             signal_start_seconds,
                             direct_signal_seconds_sensor3,
                             SETUP)
    transfer_function_setup3_2(measurements,
                               SETUP)


def plot_time_signals_setup3_2(measurements: pd.DataFrame,
                               signal_start_seconds: float,
                               signal_length_seconds: float):
    """SETTINGS FOR PLOTTING"""
    plots_to_plot = ['time']

    """Plot the raw measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=set_window_size(rows=measurements.shape[1],
                                                    cols=len(plots_to_plot)),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    subplots_adjust('time', rows=3, columns=1)

    """Compress chirps"""
    measurements = compress_chirps(measurements)

    """Plot the measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=plots_to_plot,
                    compressed_chirps=True)

    """Use scientific notation"""
    for ax in axs.flatten():
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    """Adjust for correct spacing in plot"""
    subplots_adjust('time', rows=3, columns=1)


def plot_spectrogram_signals_setup3_2(measurements: pd.DataFrame,
                                      signal_start_seconds: float,
                                      signal_length_seconds: float):
    """SETTINGS FOR PLOTTING"""
    NFFT = 2048
    PLOTS_TO_PLOT = ['spectrogram']

    """Plot the raw measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    nfft=NFFT,
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    subplots_adjust('spectrogram', rows=3, columns=1)

    """Compress chirps"""
    measurements = compress_chirps(measurements)

    """Plot the measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    nfft=NFFT,
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=True)

    """Adjust for correct spacing in plot"""
    subplots_adjust('spectrogram', rows=3, columns=1)


def plot_fft_signals_setup3_2(measurements: pd.DataFrame,
                              signal_start_seconds: float,
                              signal_length_seconds: float):
    """SETTINGS FOR PLOTTING"""
    PLOTS_TO_PLOT = ['fft']

    """Plot the raw measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    subplots_adjust('fft', rows=3, columns=1)

    """Limit y axis"""
    for ax in axs.flatten():
        ax.set_ylim(bottom=-70, top=120)

    """Compress chirps"""
    measurements = compress_chirps(measurements)

    """Plot the measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=True)

    """Adjust for correct spacing in plot"""
    subplots_adjust('fft', rows=3, columns=1)

    """Limit y axis"""
    for ax in axs.flatten():
        ax.set_ylim(bottom=-50, top=230)


def transfer_function_setup3_2(measurements: pd.DataFrame,
                               SETUP: Setup):
    """Calculate the transfer function of the table
    by dividing the FFT of the direct signal on the output
    by the FFT of the direct signal on the input,
    H(f) = Y(f)/X(f),
    where the signal on sensor 1 is the input and
    the signal on sensor 3 is the output."""

    """Plot the cropped signal"""
    PLOTS_TO_PLOT = ['time', 'spectrogram', 'fft']
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=FIGSIZE_THREE_COLUMNS,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=True,
                    nfft=128)

    """Find the FFT of the input signal"""
    input_signal = measurements['Sensor 1'].values
    fft_input = scipy.fft.fft(input_signal)
    fft_frequencies = scipy.fft.fftfreq(fft_input.size,
                                        d=1 / SAMPLE_RATE)
    fft_input = np.fft.fftshift(fft_input)
    fft_frequencies = np.fft.fftshift(fft_frequencies)

    """Find the FFT of the output signal"""
    output_signal = measurements['Sensor 3'].values
    fft_output = scipy.fft.fft(output_signal)
    fft_frequencies = scipy.fft.fftfreq(fft_output.size,
                                        d=1 / SAMPLE_RATE)
    fft_output = np.fft.fftshift(fft_output)
    fft_frequencies = np.fft.fftshift(fft_frequencies)

    """Find the transfer function"""
    transfer_function = fft_input / fft_output

    """Limit the frequency range to 0-50000 Hz"""
    transfer_function = transfer_function[(fft_frequencies >= 0) &
                                          (fft_frequencies <= 45000)]
    fft_frequencies = fft_frequencies[(fft_frequencies >= 0) &
                                      (fft_frequencies <= 45000)]

    """Find the amplitude and phase of the transfer function"""
    transfer_function_amplitude = np.abs(transfer_function)
    transfer_function_phase = np.unwrap(np.angle(transfer_function)) - 1 * np.pi

    """Find the phase velocity"""
    distance_between_sensors = np.linalg.norm(SETUP.sensors[SENSOR_3].coordinates -
                                              SETUP.sensors[SENSOR_1].coordinates)
    phase_velocities = (2 * np.pi * fft_frequencies * distance_between_sensors /
                        transfer_function_phase)

    """Plot the amplitude and the phase of fft_input"""
    fig, axs = plt.subplots(nrows=2,
                            ncols=1,
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False,
                            sharex=True)
    axs[0, 0].plot(fft_frequencies,
                   transfer_function_phase)
    axs[0, 0].set_title('Phase of the transfer function')
    axs[0, 0].set_xlabel('Frequency [Hz]')
    axs[0, 0].set_ylabel('Phase [rad]')
    axs[0, 0].set_ylim(bottom=-200, top=500)
    axs[0, 0].grid()
    axs[1, 0].plot(fft_frequencies,
                   phase_velocities,
                   label='Measured phase velocities')
    axs[1, 0].set_title('Phase velocity')
    axs[1, 0].set_xlabel('Frequency [Hz]')
    axs[1, 0].set_ylabel('v_phase [m/s]')
    axs[1, 0].set_ylim(bottom=0, top=1000)
    axs[1, 0].grid()

    (simulated_phase_velocities_uncorrected,
     simulated_phase_velocities_corrected,
     phase_velocity_shear) = simulated_phase_velocities(fft_frequencies)
    axs[1, 0].plot(fft_frequencies,
                   simulated_phase_velocities_uncorrected,
                   label='Simulated phase velocities',
                   linestyle='--')
    axs[1, 0].plot(fft_frequencies,
                   simulated_phase_velocities_corrected,
                   label='Simulated corrected phase velocities',
                   linestyle='--')
    axs[1, 0].plot(fft_frequencies,
                   np.ones_like(fft_frequencies) * phase_velocity_shear,
                   label='Simulated shear phase velocities',
                   linestyle='--')
    axs[1, 0].legend()


def find_and_plot_power_loss(measurements: pd.DataFrame,
                             signal_start_seconds: float,
                             signal_length_seconds: float,
                             SETUP: Setup):
    """Find the power of the signal with RMS"""
    signal_rms = np.sqrt(np.mean(measurements ** 2))
    signal_power = signal_rms ** 2
    power_loss_sensor1_to_sensor_3 = signal_power[SENSOR_3] / signal_power[SENSOR_1]
    power_loss_sensor1_to_sensor_3_dB = to_dB(power_loss_sensor1_to_sensor_3)
    distance_between_sensors = np.linalg.norm(SETUP.sensors[SENSOR_3].coordinates -
                                              SETUP.sensors[SENSOR_1].coordinates)
    print(f'\nPower loss sensor 1 to sensor 3, for 100 Hz to 40 kHz: {power_loss_sensor1_to_sensor_3_dB:.2f} dB')
    print(f'Power loss per meter: {(power_loss_sensor1_to_sensor_3_dB / distance_between_sensors):.2f} dB/m\n')

    """Compress chirps"""
    measurements = compress_chirps(measurements)

    """Plot the measurements"""
    PLOTS_TO_PLOT = ['fft']
    fig, axs = plt.subplots(nrows=1,
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=FIGSIZE_ONE_SIGNAL,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements['Sensor 1']],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    PLOTS_TO_PLOT=PLOTS_TO_PLOT,
                    compressed_chirps=True)
    compare_signals(fig, axs,
                    [measurements['Sensor 3']],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    PLOTS_TO_PLOT=PLOTS_TO_PLOT,
                    compressed_chirps=True)
    for ax in axs.flatten():
        ax.grid()
    axs[0, 0].set_title('FFT of sensor 1 and sensor 3, ' +
                        f'{distance_between_sensors} meters apart')
    axs[0, 0].legend(['Sensor 1', 'Sensor 3'])
    axs[0, 0].set_ylim(bottom=-25, top=80)

    """Adjust for correct spacing inz plot"""
    subplots_adjust('fft', rows=1, columns=1)


"""Setup 7"""


def results_setup7():
    """Run some general commands for all functions:
        - Choose file and open it
        - Channel selection
        - Choose setup and draw it
        - Interpolation
        - Generate results from functions
    """

    """Choose file"""
    FILE_FOLDER = 'setup7'
    FILE_NAME = 'notouch_20to40khz_1s_10vpp_v1'

    """Choose crop times"""
    TIME_START = 0.114 + 0.0036  # s
    TIME_END = 0.246 - 0.0035  # s

    """Choose setup"""
    SETUP = Setup7()
    """Draw setup"""
    subplots_adjust(['setup'])
    SETUP.draw()
    plt.savefig(FIGURES_SAVE_PATH + 'setup7_draw.pdf',
                format='pdf')

    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME,
                             channel_names=CHIRP_CHANNEL_NAMES)

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Crop data"""
    measurements = crop_data(measurements,
                             time_start=TIME_START,
                             time_end=TIME_END)

    # plot_raw_time_signal_setup7(measurements)
    scattering_setup7(SETUP)



def plot_raw_time_signal_setup7(measurements: pd.DataFrame):
    """Plot raw signal"""
    time_axis = np.linspace(start=0,
                            stop=1000 * len(measurements) / SAMPLE_RATE,
                            num=len(measurements))
    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=set_window_size())
    ax.set_title('Chirp from 20 khz to 40 kHz in 125 ms')
    ax.plot(time_axis, measurements['Sensor 1'], label='Sensor 1')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (mV)')
    ax.legend()
    ax.grid()

    """Adjust for correct spacing in plot"""
    subplots_adjust(['time'], rows=1, columns=1)


def scattering_setup7(setup: Setup):
    """Open file"""
    FILE_FOLDER = 'setup7'
    FILE_NAME = 'notouchThenHoldB2_20to40khz_125ms_10vpp_v1'
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Plot measurements"""
    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    fig, axs = plt.subplots(nrows=1, ncols=1,
                            sharex=True, sharey=True,
                            figsize=set_window_size(rows=1,
                                                    cols=1),
                            squeeze=False)
    time_axis = np.linspace(0, 5, measurements.shape[0])
    for channel in CHANNELS_TO_PLOT:
        axs[0, 0].plot(time_axis, measurements[channel], label=channel)
    axs[0, 0].set_title('Shifted measurements 1')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Amplitude [V]')
    axs[0, 0].legend()
    axs[0, 0].grid()

    """Shift to align chirps better with their time intervals"""
    SHIFT_BY = int(0.0877 * SAMPLE_RATE)
    for channel in measurements:
        measurements[channel] = np.roll(measurements[channel],
                                        -SHIFT_BY)
        measurements[channel][-SHIFT_BY:] = 0

    """Compress chirps"""
    CHIRP_LENGTH = int(0.125 * SAMPLE_RATE)
    PRE_PAD_LENGTH = int(2.5 * SAMPLE_RATE)
    POST_PAD_LENGTH = measurements.shape[0] - (PRE_PAD_LENGTH + CHIRP_LENGTH) - 1
    chirp = measurements['Actuator'][0:CHIRP_LENGTH + 1]
    reference_chirp = np.pad(chirp,
                             (PRE_PAD_LENGTH, POST_PAD_LENGTH),
                             mode='constant')
    _, ax = plt.subplots(1, 1)
    ax.plot(np.linspace(-2.5, 2.5, measurements.shape[0]), reference_chirp)
    for channel in measurements:
        measurements[channel] = signal.correlate(measurements[channel],
                                                 reference_chirp,
                                                 mode='same')

    """Split the channels into arrays of length 125 ms"""
    measurements_split = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES)
    for channel in measurements_split:
        measurements_split[channel] = np.split(measurements[channel],
                                               indices_or_sections=40)

    """Plot the shifted measurements"""
    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    fig, axs = plt.subplots(nrows=1, ncols=1,
                            sharex=True, sharey=True,
                            figsize=set_window_size(rows=1,
                                                    cols=1),
                            squeeze=False)
    time_axis = np.linspace(0, 5, measurements.shape[0])
    for channel in CHANNELS_TO_PLOT:
        axs[0, 0].plot(time_axis, measurements[channel], label=channel)
    axs[0, 0].set_title('Shifted measurements 1')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Amplitude [V]')
    axs[0, 0].legend()
    axs[0, 0].grid()
    """Use scientific notation"""
    for ax in axs.flatten():
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    subplots_adjust(['time'])

    """Make up for drift in signal generator"""
    measurements_split = correct_drift(measurements_split,
                                       data_to_sync_with=measurements_split)

    """Plot all chirps on top of each other"""
    time_axis = np.linspace(start=0,
                            stop=len(measurements_split['Sensor 1'][0]) / SAMPLE_RATE,
                            num=len(measurements_split['Sensor 1'][0]))
    fig, axs = plt.subplots(nrows=3, ncols=1,
                            sharex=True, sharey=True,
                            figsize=set_window_size(rows=2))
    for i, _ in enumerate(measurements_split['Sensor 1'][0:39]):
        axs[0].plot(time_axis, measurements_split['Sensor 1'][i])
        axs[1].plot(time_axis, measurements_split['Sensor 3'][i])
        axs[2].plot(time_axis, measurements_split['Actuator'][i])
    axs[0].set_title('39 chirps, sensor 1')
    axs[1].set_title('39 chirps, sensor 3')
    axs[1].set_xlabel('Time [s]')
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    subplots_adjust(['time'], rows=2)


"""Setup 9"""


def results_setup9():
    """Choose setup"""
    SETUP = Setup9()
    """Draw setup"""
    subplots_adjust(['setup'])
    SETUP.draw()
    plt.savefig(FIGURES_SAVE_PATH + 'setup1_draw.pdf',
                format='pdf')

    """Run functions to generate results"""
    plot_touch_signals_setup9()
    # plot_chirp_signals_setup9()
    # transfer_function_setup9(SETUP)
    # scattering_setup9(SETUP)
    # predict_reflections_setup9(SETUP)


def plot_touch_signals_setup9():
    # TODO: Make the other plotting functions to behave like this one,
    # !      i.e. plot individual channels and plot types in for loops.
    """Choose file"""
    FILE_FOLDER = 'setup9_10cm/touch'
    FILE_NAME = 'touch_v1'
    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Crop data to the full touch signal"""
    measurements_full_touch = crop_data(measurements,
                                        time_start=2.05,
                                        time_end=2.5575)

    """Filter signals to remove 50 Hz and other mud"""
    measurements_full_touch = filter_general(measurements_full_touch,
                                             filtertype='highpass',
                                             cutoff_highpass=200)

    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    for channel in CHANNELS_TO_PLOT:
        plot_touch_time_signal_setup9(measurements_full_touch, [channel])
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup9_touch_time_{channel_file_name}_full.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    for channel in CHANNELS_TO_PLOT:
        plot_touch_spectrogram_setup9(measurements_full_touch,
                                      [channel],
                                      dynamic_range_db=60)
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup9_touch_spectrogram_{channel_file_name}_full.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    """Crop data to the beginning of the touch signal"""
    measurements_beginning_of_touch = crop_data(measurements_full_touch,
                                                time_start=(2.05 + 0.0538),
                                                time_end=(2.05 + 0.07836))
    for channel in CHANNELS_TO_PLOT:
        plot_touch_time_signal_setup9(measurements_beginning_of_touch, [channel])
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup9_touch_time_{channel_file_name}_beginning.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    for channel in CHANNELS_TO_PLOT:
        plot_touch_spectrogram_setup9(measurements_beginning_of_touch,
                                      [channel],
                                      dynamic_range_db=60,
                                      nfft=1024)
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup9_touch_spectrogram_{channel_file_name}_beginning.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    fig, axs = plt.subplots(nrows=1, ncols=1,
                            figsize=set_window_size(),
                            squeeze=False)

    CHANNELS_TO_PLOT = ['Sensor 1']
    for channel in CHANNELS_TO_PLOT:
        plot_touch_fft_setup9(measurements_beginning_of_touch,
                              fig, axs,
                              [channel])
    # axs[0, 0].legend(CHANNELS_TO_PLOT)
    # axs[0, 0].grid()
    channel_file_name = channel.replace(" ", "").lower()
    file_name = 'setup9_touch_fft_beginning.pdf'
    plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')


def plot_chirp_signals_setup9():
    """Choose file"""
    FILE_FOLDER = 'setup9_10cm/chirp/100Hz_to_40kHz_single_chirp'
    FILE_NAME = 'chirp_v1'
    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Crop data to the full touch signal"""
    measurements_raw = crop_data(measurements,
                                 time_start=1.38,
                                 time_end=3.531)

    plot_chirp_time_signal_setup9(measurements_raw)
    file_name = 'setup9_chirp_time_raw.pdf'
    # plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    plot_chirp_spectrogram_setup9(measurements_raw,
                                  dynamic_range_db=60)
    file_name = 'setup9_chirp_spectrogram_raw.pdf'
    # plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    plot_chirp_fft_setup9(measurements_raw)
    for ax in plt.gcf().axes:
        ax.set_ylim([-20, 60])
    file_name = 'setup9_chirp_fft_raw.pdf'
    # plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    """Crop data to the compressed chirp signal"""
    measurements_compressed = compress_chirps(measurements)
    measurements_compressed = measurements_compressed.loc[2.499 * SAMPLE_RATE:
                                                          2.51 * SAMPLE_RATE]

    plot_chirp_time_signal_setup9(measurements_compressed,
                                  envelope=True)
    """Use scientific notation for y-axis"""
    for ax in plt.gcf().axes:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    file_name = 'setup9_chirp_time_compressed.pdf'
    # plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    plot_chirp_spectrogram_setup9(measurements_compressed,
                                  dynamic_range_db=40,
                                  nfft=512)
    file_name = 'setup9_chirp_spectrogram_compressed.pdf'
    # plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    plot_chirp_fft_setup9(measurements_compressed)
    for ax in plt.gcf().axes:
        ax.set_ylim([0, 100])
    file_name = 'setup9_chirp_fft_compressed.pdf'
    # plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')


def plot_touch_time_signal_setup9(measurements: pd.DataFrame,
                                  channels_to_plot: list = ['Actuator',
                                                            'Sensor 1',
                                                            'Sensor 3']):
    """Plot the time signal full touch signal"""
    PLOTS_TO_PLOT = ['time']
    fig, axs = plt.subplots(nrows=len(channels_to_plot),
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(rows=len(channels_to_plot),
                                                    cols=len(PLOTS_TO_PLOT)),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in channels_to_plot],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    subplots_adjust(PLOTS_TO_PLOT,
                    rows=len(channels_to_plot),
                    columns=len(PLOTS_TO_PLOT))


def plot_touch_spectrogram_setup9(measurements: pd.DataFrame,
                                  channels_to_plot: list = ['Sensor 1',
                                                            'Sensor 3'],
                                  dynamic_range_db: int = 60,
                                  nfft: int = 4096):
    """Plot the spectrogram of a touch signal"""
    PLOTS_TO_PLOT = ['spectrogram']
    fig, axs = plt.subplots(nrows=len(channels_to_plot),
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(rows=len(channels_to_plot),
                                                    cols=len(PLOTS_TO_PLOT)),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in channels_to_plot],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False,
                    dynamic_range_db=dynamic_range_db,
                    nfft=nfft)

    """Adjust for correct spacing in plot"""
    subplots_adjust(PLOTS_TO_PLOT,
                    rows=len(channels_to_plot),
                    columns=len(PLOTS_TO_PLOT))


def plot_touch_time_signal_and_spectrogram_setup9(measurements: pd.DataFrame,
                                                  nfft: int = 4096):
    """Plot the full touch signal"""
    CHANNELS_TO_PLOT = ['Sensor 1']
    PLOTS_TO_PLOT = ['time', 'spectrogram']
    fig, axs = plt.subplots(nrows=len(CHANNELS_TO_PLOT),
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(rows=len(CHANNELS_TO_PLOT),
                                                    cols=len(PLOTS_TO_PLOT)),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in CHANNELS_TO_PLOT],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    # subplots_adjust(PLOTS_TO_PLOT,
    #                 rows=len(PLOTS_TO_PLOT),
    #                 columns=len(CHANNELS_TO_PLOT))

    """Move the xlabel from the first row to the second"""
    # axs[0, 0].set_xlabel('')
    # axs[1, 0].set_xlabel('Time (s)')


def plot_touch_fft_setup9(measurements: pd.DataFrame,
                          fig, axs,
                          channels_to_plot = ['Sensor 1', 'Sensor 3']):
    """Plot the FFT a touch signal"""
    PLOTS_TO_PLOT = ['fft']
    compare_signals(fig, axs,
                    [measurements[channel] for channel in channels_to_plot],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False)
    """Limit to +25 dB and -25 dB"""
    axs[0, 0].set_ylim([-25, 25])

    """Adjust for correct spacing in plot"""
    subplots_adjust(PLOTS_TO_PLOT,
                    rows=len(channels_to_plot),
                    columns=len(PLOTS_TO_PLOT))


def plot_chirp_time_signal_setup9(measurements: pd.DataFrame,
                                  envelope: bool = False):
    """Plot the time signal full chirp signal"""
    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    PLOTS_TO_PLOT = ['time']
    fig, axs = plt.subplots(nrows=len(CHANNELS_TO_PLOT),
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(rows=len(CHANNELS_TO_PLOT),
                                                    cols=len(PLOTS_TO_PLOT)),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in CHANNELS_TO_PLOT],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False)
    if envelope:
        compare_signals(fig, axs,
                        [get_envelope(measurements[channel]) for channel in CHANNELS_TO_PLOT],
                        plots_to_plot=PLOTS_TO_PLOT,
                        compressed_chirps=False)
        """Re-add grid"""
        for ax in axs.flatten():
            ax.grid()
    """Adjust for correct spacing in plot"""
    subplots_adjust(PLOTS_TO_PLOT,
                    rows=len(CHANNELS_TO_PLOT),
                    columns=len(PLOTS_TO_PLOT))


def plot_chirp_spectrogram_setup9(measurements: pd.DataFrame,
                                  dynamic_range_db: int = 60,
                                  nfft: int = 4096):
    """Plot the spectrogram of beginning of the chirp signal"""
    CHANNELS_TO_PLOT = ['Actuator', 'Sensor 1']
    PLOTS_TO_PLOT = ['spectrogram']
    fig, axs = plt.subplots(nrows=len(CHANNELS_TO_PLOT),
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(rows=len(CHANNELS_TO_PLOT),
                                                    cols=len(PLOTS_TO_PLOT)),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in CHANNELS_TO_PLOT],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False,
                    dynamic_range_db=dynamic_range_db,
                    nfft=nfft)

    """Adjust for correct spacing in plot"""
    subplots_adjust(PLOTS_TO_PLOT,
                    rows=len(CHANNELS_TO_PLOT),
                    columns=len(PLOTS_TO_PLOT))


def plot_chirp_time_signal_and_spectrogram_setup9(measurements: pd.DataFrame,
                                                  dynamic_range_db: int = 60,
                                                  nfft: int = 4096):
    """Plot the time signal full chirp signal"""
    # ! Look at later maybe
    CHANNELS_TO_PLOT = ['Actuator', 'Sensor 1']
    PLOTS_TO_PLOT = ['time', 'spectrogram']
    fig, axs = plt.subplots(nrows=len(CHANNELS_TO_PLOT),
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(rows=len(CHANNELS_TO_PLOT),
                                                    cols=len(PLOTS_TO_PLOT)),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in CHANNELS_TO_PLOT],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in CHANNELS_TO_PLOT],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False,
                    dynamic_range_db=dynamic_range_db,
                    nfft=nfft)

    """Adjust for correct spacing in plot"""
    subplots_adjust(PLOTS_TO_PLOT,
                    rows=len(CHANNELS_TO_PLOT),
                    columns=len(PLOTS_TO_PLOT))


def plot_chirp_fft_setup9(measurements: pd.DataFrame):
    """Plot the FFT a chirp signal"""
    CHANNELS_TO_PLOT = ['Actuator', 'Sensor 1']
    PLOTS_TO_PLOT = ['fft']
    fig, axs = plt.subplots(nrows=len(CHANNELS_TO_PLOT),
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(rows=len(CHANNELS_TO_PLOT),
                                                    cols=len(PLOTS_TO_PLOT)),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in CHANNELS_TO_PLOT],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    subplots_adjust(PLOTS_TO_PLOT,
                    rows=len(CHANNELS_TO_PLOT),
                    columns=len(PLOTS_TO_PLOT))


def transfer_function_setup9(setup: Setup):
    """Calculate the transfer function of the table
    by dividing the FFT of the direct signal on the output
    by the FFT of the direct signal on the input,
    H(f) = Y(f)/X(f),
    where the signal on sensor 1 is the input and
    the signal on sensor 3 is the output."""

    """Choose file"""
    FILE_FOLDER = 'setup9_10cm/chirp/100Hz_to_40kHz_single_chirp'
    FILE_NAME = 'chirp_v1'

    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Compress chirps"""
    measurements = compress_chirps(measurements)

    """Crop the compressed chirps to be only the direct signal"""
    direct_signal_seconds_sensor1 = 0.0012
    measurements['Sensor 1'] = window_signals(pd.DataFrame(measurements['Sensor 1']),
                                                           direct_signal_seconds_sensor1,
                                                           window_function='tukey')
    direct_signal_seconds_sensor3 = 0.00085
    measurements['Sensor 3'] = window_signals(pd.DataFrame(measurements['Sensor 3']),
                                                           direct_signal_seconds_sensor3,
                                                           window_function='tukey')

    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    PLOTS_TO_PLOT = ['time']
    fig, axs = plt.subplots(nrows=len(CHANNELS_TO_PLOT),
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=FIGSIZE_ONE_COLUMN,
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in CHANNELS_TO_PLOT],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=True,
                    nfft=256)
    compare_signals(fig, axs,
                    [get_envelope(measurements[channel]) for channel in CHANNELS_TO_PLOT],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=True,
                    dynamic_range_db=15,
                    nfft=32)
    """Adjust for correct spacing in plot"""
    subplots_adjust(PLOTS_TO_PLOT,
                    rows=len(CHANNELS_TO_PLOT),
                    columns=len(PLOTS_TO_PLOT))

    """Find the FFT of the input signal"""
    input_signal = measurements['Sensor 1'].values
    fft_input = scipy.fft.fft(input_signal)
    fft_frequencies = scipy.fft.fftfreq(fft_input.size,
                                        d=1 / SAMPLE_RATE)
    fft_input = np.fft.fftshift(fft_input)
    fft_frequencies = np.fft.fftshift(fft_frequencies)

    """Find the FFT of the output signal"""
    output_signal = measurements['Sensor 3'].values
    fft_output = scipy.fft.fft(output_signal)
    fft_output = np.fft.fftshift(fft_output)

    """Find the transfer function"""
    transfer_function = fft_output / fft_input

    """Limit the frequency range to 1-40000 Hz"""
    transfer_function = transfer_function[(fft_frequencies >= 1) &
                                          (fft_frequencies <= 40000)]
    fft_frequencies = fft_frequencies[(fft_frequencies >= 1) &
                                      (fft_frequencies <= 40000)]

    """Find the amplitude and phase of the transfer function"""
    amplitude_response = np.abs(transfer_function)
    phase_response = np.unwrap(np.angle(transfer_function)) - 2 * np.pi

    """Find the phase velocity"""
    distance_between_sensors = np.linalg.norm(setup.sensors[SENSOR_3].coordinates -
                                              setup.sensors[SENSOR_1].coordinates)
    phase_velocities = - (2 * np.pi * fft_frequencies * distance_between_sensors /
                          phase_response)

    plt.close('all')
    """Plot the amplitude response"""
    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=FIGSIZE_ONE_SIGNAL,
                            squeeze=False)
    axs[0, 0].plot(fft_frequencies / 1000,
                   to_dB(amplitude_response))
    axs[0, 0].set_title('Amplitude response')
    axs[0, 0].set_xlabel('Frequency [kHz]')
    axs[0, 0].set_ylabel('Amplitude [dB]')
    axs[0, 0].set_ylim(bottom=-20, top=20)
    axs[0, 0].grid()
    subplots_adjust(['fft'])
    plt.savefig(FIGURES_SAVE_PATH + 'setup9_amplitude_response.pdf',
                format='pdf')

    """Plot the phase response"""
    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=FIGSIZE_ONE_SIGNAL,
                            squeeze=False)
    axs[0, 0].plot(fft_frequencies / 1000,
                   phase_response)
    axs[0, 0].set_title('Phase response')
    axs[0, 0].set_xlabel('Frequency [kHz]')
    axs[0, 0].set_ylabel('Phase [rad]')
    axs[0, 0].set_ylim(bottom=-50, top=50)
    axs[0, 0].grid()
    subplots_adjust(['fft'])
    plt.savefig(FIGURES_SAVE_PATH + 'setup9_phase_response.pdf',
                format='pdf')

    (simulated_phase_velocities_uncorrected,
     simulated_phase_velocities_corrected,
     phase_velocity_shear) = simulated_phase_velocities(fft_frequencies)

    """Plot the phase velocities"""
    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=FIGSIZE_ONE_SIGNAL,
                            squeeze=False,
                            sharex=True)
    axs[0, 0].plot(fft_frequencies / 1000,
                   simulated_phase_velocities_uncorrected,
                   label='Simulated phase velocities',
                   linestyle='--')
    axs[0, 0].plot(fft_frequencies / 1000,
                   simulated_phase_velocities_corrected,
                   label='Simulated corrected phase velocities',
                   linestyle='--')
    # axs[0, 0].plot(fft_frequencies / 1000,
    #                np.ones_like(fft_frequencies) * phase_velocity_shear,
    #                label='Simulated shear phase velocities',
    #                linestyle='--')
    axs[0, 0].set_title('Phase velocities')
    axs[0, 0].set_xlabel('Frequency kHz]')
    axs[0, 0].set_ylabel('Phase velocity [m/s]')
    axs[0, 0].set_ylim(bottom=0, top=2500)
    axs[0, 0].legend()
    axs[0, 0].grid()
    subplots_adjust(['fft'])
    plt.savefig(FIGURES_SAVE_PATH + 'simulated_phase_velocities.pdf',
                format='pdf')
    axs[0, 0].plot(fft_frequencies / 1000,
                   phase_velocities,
                   label='Measured phase velocities')
    axs[0, 0].set_ylim(bottom=0, top=1000)
    axs[0, 0].legend()
    plt.savefig(FIGURES_SAVE_PATH + 'setup9_phase_velocities.pdf',
                format='pdf')
    plt.show()


def scattering_setup9(setup: Setup):
    """Open file"""
    FILE_FOLDER = 'setup9_10cm/scattering_tests/15kHz_to_40kHz_125ms'
    FILE_NAME = 'no_touch_v1'
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Shift to align chirps better with their time intervals"""
    SHIFT_BY = int((0.0877 + 0.0388) * SAMPLE_RATE)
    for channel in measurements:
        measurements[channel] = np.roll(measurements[channel],
                                        -SHIFT_BY)
        measurements[channel][-SHIFT_BY:] = 0

    """Compress chirps"""
    CHIRP_LENGTH = int(0.125 * SAMPLE_RATE)
    PRE_PAD_LENGTH = int(2.5 * SAMPLE_RATE)
    POST_PAD_LENGTH = measurements.shape[0] - (PRE_PAD_LENGTH + CHIRP_LENGTH) - 1
    chirp = measurements['Actuator'][0:CHIRP_LENGTH + 1]
    reference_chirp = np.pad(chirp,
                             (PRE_PAD_LENGTH, POST_PAD_LENGTH),
                             mode='constant')
    _, ax = plt.subplots(1, 1)
    ax.plot(np.linspace(-2.5, 2.5, measurements.shape[0]), reference_chirp)
    for channel in measurements:
        measurements[channel] = signal.correlate(measurements[channel],
                                                 reference_chirp,
                                                 mode='same')

    """Split the channels into arrays of length 125 ms"""
    measurements_split = pd.DataFrame(columns=CHIRP_CHANNEL_NAMES)
    for channel in measurements_split:
        measurements_split[channel] = np.split(measurements[channel],
                                               indices_or_sections=40)

    """Plot the shifted measurements"""
    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    fig, axs = plt.subplots(nrows=1, ncols=1,
                            sharex=True, sharey=True,
                            figsize=set_window_size(rows=1,
                                                    cols=1),
                            squeeze=False)
    time_axis = np.linspace(0, 5, measurements.shape[0])
    for channel in CHANNELS_TO_PLOT:
        axs[0, 0].plot(time_axis, measurements[channel], label=channel)
    axs[0, 0].set_title('Shifted measurements 1')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Amplitude [V]')
    axs[0, 0].legend()
    axs[0, 0].grid()
    """Use scientific notation"""
    for ax in axs.flatten():
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    subplots_adjust(['time'])

    """Make up for drift in signal generator"""
    measurements_split = correct_drift(measurements_split,
                                       data_to_sync_with=measurements_split)

    """Plot all chirps on top of each other"""
    time_axis = np.linspace(start=0,
                            stop=len(measurements_split['Sensor 1'][0]) / SAMPLE_RATE,
                            num=len(measurements_split['Sensor 1'][0]))
    fig, axs = plt.subplots(nrows=3, ncols=1,
                            sharex=True, sharey=True,
                            figsize=set_window_size(rows=2))
    for i, _ in enumerate(measurements_split['Sensor 1'][0:39]):
        axs[0].plot(time_axis, measurements_split['Sensor 1'][i])
        axs[1].plot(time_axis, measurements_split['Sensor 3'][i])
        axs[2].plot(time_axis, measurements_split['Actuator'][i])
    axs[0].set_title('39 chirps, sensor 1')
    axs[1].set_title('39 chirps, sensor 3')
    axs[1].set_xlabel('Time [s]')
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    subplots_adjust(['time'], rows=2)


def predict_reflections_setup9(setup: Setup):
    """Open file"""
    FILE_FOLDER = 'setup9_10cm/chirp/100Hz_to_40kHz_single_chirp'
    FILE_NAME = 'chirp_v1'
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Filter signals"""
    # measurements = filter_general(measurements,
    #                               filtertype='highpass',
    #                               cutoff_highpass=10000)

    """Compress chirps"""
    measurements = compress_chirps(measurements)

    """Calculate propagation speed based on """

    predict_reflections_in_envelopes_setup9(setup, measurements)
    predict_reflections_in_spectrograms_setup9(setup, measurements)


def predict_reflections_in_envelopes_setup9(setup: Setup,
                                            measurements: pd.DataFrame):
    """Calculate arrival times for sensor 1"""
    arrival_times, _ = get_travel_times(setup.actuators[ACTUATOR_1],
                                        setup.sensors[SENSOR_1],
                                        propagation_speed=300,
                                        ms=True,
                                        relative_first_reflection=False)
    """Plot lines for expected arrival times"""
    envelope_with_lines(setup.sensors[SENSOR_1],
                        measurements,
                        arrival_times)
    subplots_adjust(['time'])
    plt.savefig(FIGURES_SAVE_PATH + 'predicted_arrivals_envelope_sensor1_setup9.pdf',
                format='pdf')

    """Calculate arrival times for sensor 3"""
    arrival_times, _ = get_travel_times(setup.actuators[ACTUATOR_1],
                                        setup.sensors[SENSOR_3],
                                        propagation_speed=290,
                                        ms=True,
                                        relative_first_reflection=False)
    envelope_with_lines(setup.sensors[SENSOR_3],
                        measurements,
                        arrival_times)
    subplots_adjust(['time'])
    plt.savefig(FIGURES_SAVE_PATH + 'predicted_arrivals_envelope_sensor3_setup9.pdf',
                format='pdf')


def predict_reflections_in_spectrograms_setup9(setup: Setup,
                                               measurements: pd.DataFrame,
                                               nfft=128,
                                               dynamic_range_db=15):
    """Calculate arrival times for sensor 1"""
    arrival_times, _ = get_travel_times(setup.actuators[ACTUATOR_1],
                                        setup.sensors[SENSOR_1],
                                        propagation_speed=580,
                                        ms=False,
                                        relative_first_reflection=False)
    """Fit arrival times to compressed chrip times"""
    arrival_times += 2.5
    """Plot lines for expected arrival times"""
    spectrogram_with_lines(setup.sensors[SENSOR_1],
                           measurements,
                           arrival_times,
                           nfft,
                           dynamic_range_db=25)
    subplots_adjust(['spectrogram'])
    plt.savefig(FIGURES_SAVE_PATH + 'predicted_arrivals_spectrogram_sensor1_setup9.pdf',
                format='pdf')

    """Calculate arrival times for sensor 3"""
    arrival_times, _ = get_travel_times(setup.actuators[ACTUATOR_1],
                                        setup.sensors[SENSOR_3],
                                        propagation_speed=580,
                                        ms=False,
                                        relative_first_reflection=False)
    """Fit arrival times to compressed chrip times"""
    arrival_times += 2.5
    spectrogram_with_lines(setup.sensors[SENSOR_3],
                           measurements,
                           arrival_times,
                           nfft,
                           dynamic_range_db)
    subplots_adjust(['spectrogram'])
    plt.savefig(FIGURES_SAVE_PATH + 'predicted_arrivals_spectrogram_sensor3_setup9.pdf',
                format='pdf')


"""Generate custom graphs"""


def custom_plots():
    """Generate custom graphs"""
    scattering_figure_3()


def scattering_figure_3():
    """Use coordinates of points along a graph to make a plot"""
    # ! Not finished yet, polar plot is messed up
    ka_1 = csv_to_df(file_folder='Data visualisation/Figure datasets/scattering_figure_3',
                     file_name='ka_1',
                     channel_names=['r', 'theta'])
    """Interpolate points"""
    # ka_1 = interpolate_waveform(ka_1)

    """Plot"""
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(ka_1.theta,
            ka_1.r,
            label='Rigid inclusion')
    ax.legend()
    ax.grid()

    """Adjust for correct spacing in plot"""
    subplots_adjust('fft', rows=1, columns=1)


def scattering_figure_5():
    """Use coordinates of points along a graph to make a plot"""

    rigid_inlcusion = csv_to_df(file_folder='Data visualisation/Figure datasets/scattering_figure_5',
                                file_name='rigid_inclusion',
                                channel_names=['x', 'y'])
    steel_thickness_50in = csv_to_df(file_folder='Data visualisation/Figure datasets/scattering_figure_5',
                                     file_name='steel_thickness_50in',
                                     channel_names=['x', 'y'])
    hole = csv_to_df(file_folder='Data visualisation/Figure datasets/scattering_figure_5',
                     file_name='hole',
                     channel_names=['x', 'y'])
    bonded_steel_inclusion_thickness_0_5in = csv_to_df(file_folder='Data visualisation/Figure datasets/scattering_figure_5',
                                                       file_name='bonded_steel_inclusion_thickness_0_5in',
                                                       channel_names=['x', 'y'])
    """Interpolate points"""
    rigid_inlcusion = interpolate_waveform(rigid_inlcusion)
    steel_thickness_50in = interpolate_waveform(steel_thickness_50in)
    hole = interpolate_waveform(hole)
    bonded_steel_inclusion_thickness_0_5in = interpolate_waveform(bonded_steel_inclusion_thickness_0_5in)

    """Plot"""
    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=FIGSIZE_ONE_SIGNAL)
    ax.plot(bonded_steel_inclusion_thickness_0_5in.x,
            bonded_steel_inclusion_thickness_0_5in.y,
            label='Bonded steel inclusion, thickness 0.5 in')
    ax.plot(steel_thickness_50in.x,
            steel_thickness_50in.y,
            label='Heavy, bonded steel inclusion, thickness 50 in')
    ax.plot(hole.x,
            hole.y,
            label='Limiting case of a hole')
    ax.plot(rigid_inlcusion.x,
            rigid_inlcusion.y,
            label='Rigid inclusion')
    ax.set_xlabel('ka')
    ax.set_ylabel(r'$|f(\pi)|/\sqrt{a}$')
    ax.set_xlim(left=0, right=5)
    ax.set_ylim(bottom=0, top=4)
    ax.legend()
    ax.grid()

    """Adjust for correct spacing in plot"""
    subplots_adjust('fft', rows=1, columns=1)


if __name__ == '__main__':
    raise RuntimeError('This file is not meant to be run directly.')
