"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

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
                                                 filter)
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

from utils.table_setups import (Setup,
                                Setup1,
                                Setup2,
                                Setup3)


"""Setup 1"""


def setup1_results():
    print('Setup 1')

    """Choose setup"""
    SETUP = Setup1()
    SETUP.draw()
    adjust_plot_margins()
    plt.savefig(FIGURES_SAVE_PATH + 'setup1_draw.pdf',
                format='pdf')

    choice = ''
    while choice not in ['1', '2', '3', '4', '5']:
        print('\nWhich results do you want to generate?')
        print('1: Plot touch signals')
        print('2: Plot chirp signals')
        print('3: Transfer function and phase velocities')
        print('4: Scattering')
        print('5: Predict reflections')
        choice = input('Enter your choice: ')
        if choice == '1':
            setup1_plot_touch_signals()
        elif choice == '2':
            setup1_plot_chirp_signals()
        elif choice == '3':
            setup1_transfer_function(SETUP)
        elif choice == '4':
            setup1_scattering()
        elif choice == '5':
            setup1_predict_reflections(SETUP)
        else:
            print('Please choose a number between 1 and 5.')


def setup1_plot_touch_signals():
    print('Plotting touch signals')

    """Choose file"""
    FILE_FOLDER = 'Setup_1/touch'
    FILE_NAME = 'touch_v1'
    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Crop data to the full touch signal"""
    measurements_full_touch = crop_data(measurements,
                                        time_start=2.05,
                                        time_end=2.5075)

    """Filter signals to remove 50 Hz and other mud"""
    measurements_full_touch = filter_general(measurements_full_touch,
                                             filtertype='highpass',
                                             critical_frequency=100)

    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    for channel in CHANNELS_TO_PLOT:
        setup1_plot_touch_time_signal(measurements_full_touch, [channel])
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup1_touch_time_{channel_file_name}_full.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    for channel in CHANNELS_TO_PLOT:
        setup1_plot_touch_spectrogram(measurements_full_touch,
                                      [channel],
                                      dynamic_range_db=60)
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup1_touch_spectrogram_{channel_file_name}_full.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    """Crop data to the beginning of the touch signal"""
    measurements_beginning_of_touch = crop_data(measurements_full_touch,
                                                time_start=(2.05 + 0.0538),
                                                time_end=(2.05 + 0.07836))
    for channel in CHANNELS_TO_PLOT:
        setup1_plot_touch_time_signal(
            measurements_beginning_of_touch, [channel])
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup1_touch_time_{channel_file_name}_beginning.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    for channel in CHANNELS_TO_PLOT:
        setup1_plot_touch_spectrogram(measurements_beginning_of_touch,
                                      [channel],
                                      dynamic_range_db=60,
                                      nfft=1024)
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup1_touch_spectrogram_{channel_file_name}_beginning.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    fig, axs = plt.subplots(nrows=1, ncols=1,
                            figsize=set_window_size(),
                            squeeze=False)

    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    for channel in CHANNELS_TO_PLOT:
        setup1_plot_touch_fft(measurements_beginning_of_touch,
                              fig, axs,
                              [channel])
    axs[0, 0].legend(CHANNELS_TO_PLOT)
    axs[0, 0].grid()
    # axs[0, 0].set_title('')
    channel_file_name = channel.replace(" ", "").lower()
    file_name = 'setup1_touch_fft_beginning.pdf'
    plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')


def setup1_plot_chirp_signals():
    print('Plotting chirp signals')

    """Choose file"""
    FILE_FOLDER = 'Setup_1/chirp/100Hz_to_40kHz_single_chirp'
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

    CHANNELS_TO_PLOT = ['Actuator', 'Sensor 1', 'Sensor 3']
    for channel in CHANNELS_TO_PLOT:
        setup1_plot_chirp_time_signal(measurements_raw, [channel])
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup1_chirp_time_{channel_file_name}_raw.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    for channel in CHANNELS_TO_PLOT:
        setup1_plot_chirp_spectrogram(measurements_raw,
                                      [channel],
                                      dynamic_range_db=60)
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup1_chirp_spectrogram_{channel_file_name}_raw.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=set_window_size(),
                            squeeze=False)
    for channel in CHANNELS_TO_PLOT:
        setup1_plot_chirp_fft(measurements_raw,
                              fig, axs,
                              [channel])
        channel_file_name = channel.replace(" ", "").lower()
    axs[0, 0].set_ylim([-20, 100])
    axs[0, 0].legend(CHANNELS_TO_PLOT)
    file_name = 'setup1_chirp_fft_raw.pdf'
    plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')
    for ax in plt.gcf().axes:
        ax.set_ylim([-20, 60])

    """Crop data to the compressed chirp signal"""
    measurements_compressed = compress_chirps(measurements)
    measurements_compressed = measurements_compressed.loc[2 * SAMPLE_RATE:
                                                          3 * SAMPLE_RATE]

    for channel in CHANNELS_TO_PLOT:
        setup1_plot_chirp_time_signal(measurements_compressed, [channel])
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup1_chirp_time_{channel_file_name}_compressed.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')
    """Use scientific notation for y-axis"""
    for ax in plt.gcf().axes:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    for channel in CHANNELS_TO_PLOT:
        setup1_plot_chirp_spectrogram(measurements_compressed,
                                      [channel],
                                      dynamic_range_db=60,
                                      nfft=512)
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup1_chirp_spectrogram_{channel_file_name}_compressed.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=set_window_size(),
                            squeeze=False)
    for channel in CHANNELS_TO_PLOT:
        setup1_plot_chirp_fft(measurements_compressed,
                              fig, axs,
                              [channel])
        channel_file_name = channel.replace(" ", "").lower()
    axs[0, 0].set_ylim([-20, 100])
    axs[0, 0].legend(CHANNELS_TO_PLOT)
    # axs[0, 0].set_title('')
    file_name = 'setup1_chirp_fft_compressed.pdf'
    plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')


def setup1_plot_touch_time_signal(measurements: pd.DataFrame,
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
    adjust_plot_margins()


def setup1_plot_touch_spectrogram(measurements: pd.DataFrame,
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
    adjust_plot_margins()


def setup1_plot_touch_fft(measurements: pd.DataFrame,
                          fig, axs,
                          channels_to_plot=['Sensor 1', 'Sensor 3']):
    """Plot the FFT a touch signal"""
    PLOTS_TO_PLOT = ['fft']
    compare_signals(fig, axs,
                    [measurements[channel] for channel in channels_to_plot],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False)
    """Limit to +25 dB and -25 dB"""
    axs[0, 0].set_ylim([-25, 25])

    """Adjust for correct spacing in plot"""
    adjust_plot_margins()


def setup1_plot_chirp_time_signal(measurements: pd.DataFrame,
                                  channels_to_plot: list = ['Actuator',
                                                            'Sensor 1',
                                                            'Sensor 3'],
                                  envelope: bool = False):
    """Plot the time signal full chirp signal"""
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
    if envelope:
        compare_signals(fig, axs,
                        [get_envelopes(measurements[channel])
                         for channel in channels_to_plot],
                        plots_to_plot=PLOTS_TO_PLOT,
                        compressed_chirps=False)
        """Re-add grid"""
        for ax in axs.flatten():
            ax.grid()
    """Adjust for correct spacing in plot"""
    adjust_plot_margins()


def setup1_plot_chirp_spectrogram(measurements: pd.DataFrame,
                                  channels_to_plot: list = ['Actuator',
                                                            'Sensor 1',
                                                            'Sensor 3'],
                                  dynamic_range_db: int = 60,
                                  nfft: int = 4096):
    """Plot the spectrogram of beginning of the chirp signal"""
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
    adjust_plot_margins()


def setup1_plot_chirp_fft(measurements: pd.DataFrame,
                          fig, axs,
                          channels_to_plot: list = ['Actuator',
                                                    'Sensor 1',
                                                    'Sensor 3'],):
    """Plot the FFT a chirp signal"""
    PLOTS_TO_PLOT = ['fft']
    compare_signals(fig, axs,
                    [measurements[channel] for channel in channels_to_plot],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    adjust_plot_margins()


def setup1_transfer_function(setup: Setup):
    """Calculate the transfer function of the table
    by dividing the FFT of the direct signal on the output
    by the FFT of the direct signal on the input,
    H(f) = Y(f)/X(f),
    where the signal on sensor 1 is the input and
    the signal on sensor 3 is the output."""
    print('Calculating transfer function')

    """Choose file"""
    FILE_FOLDER = 'Setup_1/chirp/100Hz_to_40kHz_single_chirp'
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
                                              window_function='tukey',
                                              window_parameter=0.1)
    direct_signal_seconds_sensor3 = 0.00085
    measurements['Sensor 3'] = window_signals(pd.DataFrame(measurements['Sensor 3']),
                                              direct_signal_seconds_sensor3,
                                              window_function='tukey',
                                              window_parameter=0.1)

    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    PLOTS_TO_PLOT = ['time']
    fig, axs = plt.subplots(nrows=len(CHANNELS_TO_PLOT),
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in CHANNELS_TO_PLOT],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=True,
                    nfft=256)
    compare_signals(fig, axs,
                    [get_envelopes(measurements[channel])
                     for channel in CHANNELS_TO_PLOT],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=True,
                    dynamic_range_db=15,
                    nfft=32)
    """Adjust for correct spacing in plot"""
    adjust_plot_margins()

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

    """Plot the amplitude response"""
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=set_window_size())
    ax.plot(fft_frequencies / 1000, to_dB(amplitude_response))
    # axs[0, 0].set_title('Amplitude response')
    ax.set_xlabel('Frequency [kHz]')
    ax.set_ylabel('Amplitude [dB]')
    ax.set_ylim(bottom=-20, top=20)
    ax.grid()
    adjust_plot_margins()
    FILE_NAME = 'setup1_amplitude_response.pdf'
    plt.savefig(FIGURES_SAVE_PATH + FILE_NAME, format='pdf')

    """Find the power of the signal with RMS"""
    signal_rms = np.sqrt(np.mean(measurements ** 2))
    signal_power = signal_rms ** 2  # yes it just undoes the square root
    power_loss_sensor1_to_sensor_3 = signal_power[SENSOR_3] / \
        signal_power[SENSOR_1]
    power_loss_sensor1_to_sensor_3_dB = to_dB(power_loss_sensor1_to_sensor_3)
    distance_between_sensors = np.linalg.norm(setup.sensors[SENSOR_3].coordinates -
                                              setup.sensors[SENSOR_1].coordinates)
    print('\nPower loss sensor 1 to sensor 3, for 100 Hz to 40 kHz: '
          f'{power_loss_sensor1_to_sensor_3_dB:.2f} dB')
    print('Power loss per meter: '
          f'{(power_loss_sensor1_to_sensor_3_dB / distance_between_sensors):.2f} dB/m\n')

    """Plot the phase response"""
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=set_window_size())
    ax.plot(fft_frequencies / 1000, phase_response)
    # ax.set_title('Phase response')
    ax.set_xlabel('Frequency [kHz]')
    ax.set_ylabel('Phase [rad]')
    ax.set_ylim(bottom=-50, top=50)
    ax.grid()
    adjust_plot_margins()
    FILE_NAME = 'setup1_phase_response.pdf'
    plt.savefig(FIGURES_SAVE_PATH + FILE_NAME, format='pdf')

    (simulated_phase_velocities_uncorrected,
     simulated_phase_velocities_corrected,
     phase_velocity_shear) = simulated_phase_velocities(fft_frequencies)

    """Plot the phase velocities"""
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=set_window_size(),
                           sharex=True)
    ax.plot(fft_frequencies / 1000,
            simulated_phase_velocities_uncorrected,
            label='Simulated velocities',
            linestyle='--')
    ax.plot(fft_frequencies / 1000,
            simulated_phase_velocities_corrected,
            label='Simulated corrected velocities',
            linestyle='--')
    # ax.plot(fft_frequencies / 1000,
    #                np.ones_like(fft_frequencies) * phase_velocity_shear,
    #                label='Simulated shear phase velocities',
    #                linestyle='--')
    # ax.set_title('Phase velocities')
    ax.set_xlabel('Frequency [kHz]')
    ax.set_ylabel('Phase velocity [m/s]')
    ax.set_ylim(bottom=0, top=2500)
    ax.legend()
    ax.grid()
    adjust_plot_margins()
    FILE_NAME = 'simulated_phase_velocities.pdf'
    plt.savefig(FIGURES_SAVE_PATH + FILE_NAME, format='pdf')
    ax.plot(fft_frequencies / 1000,
            phase_velocities,
            label='Measured velocities')
    ax.set_ylim(bottom=0, top=1000)
    ax.legend()
    FILE_NAME = 'setup1_phase_velocities.pdf'
    plt.savefig(FIGURES_SAVE_PATH + FILE_NAME, format='pdf')


def setup1_scattering():
    print('Scattering')
    """Open file"""
    FILE_FOLDER = 'Setup_1/scattering_tests/15kHz_to_40kHz_125ms'
    FILE_NAME = 'no_touch_v1'
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    measurements = interpolate_waveform(measurements)

    """Plot measurements"""
    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    fig, axs = plt.subplots(nrows=1, ncols=1,
                            sharex=True, sharey=True,
                            figsize=set_window_size(),
                            squeeze=False)
    time_axis = np.linspace(0, 5, measurements.shape[0])
    for channel in CHANNELS_TO_PLOT:
        axs[0, 0].plot(time_axis, measurements[channel], label=channel)
    # axs[0, 0].set_title('Shifted measurements')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Amplitude [V]')
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].grid()
    adjust_plot_margins()

    """Shift to align chirps better with their time intervals"""
    SHIFT_BY = int(0.0877 * SAMPLE_RATE)
    for channel in measurements:
        measurements[channel] = np.roll(measurements[channel],
                                        -SHIFT_BY)
        measurements[channel][-SHIFT_BY:] = 0

    """Compress chirps"""
    CHIRP_LENGTH = int(0.125 * SAMPLE_RATE)
    PRE_PAD_LENGTH = int(2.5 * SAMPLE_RATE)
    POST_PAD_LENGTH = measurements.shape[0] - \
        (PRE_PAD_LENGTH + CHIRP_LENGTH) - 1
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
                            figsize=set_window_size(),
                            squeeze=False)
    time_axis = np.linspace(0, 5, measurements.shape[0])
    for channel in CHANNELS_TO_PLOT:
        axs[0, 0].plot(time_axis, measurements[channel], label=channel)
    # axs[0, 0].set_title('Shifted measurements 1')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Amplitude [V]')
    axs[0, 0].set_xlim(0, 0.002)
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].grid()
    """Use scientific notation"""
    for ax in axs.flatten():
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    adjust_plot_margins()

    """Make up for drift in signal generator"""
    measurements_split = correct_drift(measurements_split,
                                       data_to_sync_with=measurements_split)

    """Plot all chirps on top of each other"""
    time_axis = np.linspace(start=0,
                            stop=len(
                                measurements_split['Sensor 1'][0]) / SAMPLE_RATE,
                            num=len(measurements_split['Sensor 1'][0]))
    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    for channel in CHANNELS_TO_PLOT:
        fig, ax = plt.subplots(nrows=1, ncols=1,
                               figsize=set_window_size())
        for i, _ in enumerate(measurements_split[channel][0:39]):
            ax.plot(1000 * time_axis, measurements_split[channel][i])
        # ax.set_title(f'{i} chirps, {channel}')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Normalized \n correlation factor [-]')
        ax.set_xlim(0, 1)
        ax.grid()
        plt.subplots_adjust(left=0.19, right=0.967,
                            top=0.921, bottom=0.155,
                            hspace=0.28, wspace=0.2)
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup1_scattering_stacked_chirps_{channel_file_name}.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')


def setup1_predict_reflections(setup: Setup):
    print('Predicting reflections')
    """Open file"""
    FILE_FOLDER = 'Setup_1/touch'
    FILE_NAME = 'touch_v1'
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Compress chirps"""
    # measurements = compress_chirps(measurements)

    """Filter signals by a correlation based bandpass filter"""
    # measurements = filter_general(measurements,
    #                               filtertype='highpass',
    #                               cutoff_highpass=10000)
    cosine = make_gaussian_cosine(frequency=4000,
                                  num_of_samples=measurements.shape[0])
    measurements = compress_chirps(measurements, cosine)

    setup1_predict_reflections_in_envelopes(setup, measurements)
    setup1_predict_reflections_in_spectrograms(setup, measurements)


def setup1_predict_reflections_in_envelopes(setup: Setup,
                                            measurements: pd.DataFrame):
    """Calculate arrival times for sensor 1"""
    propagation_speed = setup.get_propagation_speed(measurements)
    print(f'Propagation speed: {propagation_speed}')
    arrival_times, _ = get_travel_times(setup.actuators[ACTUATOR_1],
                                        setup.sensors[SENSOR_1],
                                        propagation_speed,
                                        milliseconds=False,
                                        relative_first_reflection=True,
                                        print_info=False)
    max_index = np.argmax(np.abs(signal.hilbert(measurements['Sensor 1'])))
    zero_index = measurements.shape[0] / 2
    correction_offset = max_index - zero_index
    correction_offset_time = 2 * correction_offset / SAMPLE_RATE
    arrival_times += correction_offset_time
    """Plot lines for expected arrival times"""
    envelope_with_lines(setup.sensors[SENSOR_1],
                        measurements,
                        arrival_times)
    adjust_plot_margins()
    file_name = 'predicted_arrivals_envelope_sensor1_setup1.pdf'
    plt.savefig(FIGURES_SAVE_PATH + file_name,
                format='pdf')

    """Calculate arrival times for sensor 3"""
    arrival_times, _ = get_travel_times(setup.actuators[ACTUATOR_1],
                                        setup.sensors[SENSOR_3],
                                        propagation_speed,
                                        milliseconds=False,
                                        relative_first_reflection=True,
                                        print_info=False)
    max_index = np.argmax(np.abs(signal.hilbert(measurements['Sensor 3'])))
    zero_index = measurements.shape[0] / 2
    correction_offset = max_index - zero_index
    correction_offset_time = 2 * correction_offset / SAMPLE_RATE
    arrival_times += correction_offset_time
    envelope_with_lines(setup.sensors[SENSOR_3],
                        measurements,
                        arrival_times)
    adjust_plot_margins()
    file_name = 'predicted_arrivals_envelope_sensor3_setup1.pdf'
    plt.savefig(FIGURES_SAVE_PATH + file_name,
                format='pdf')


def setup1_predict_reflections_in_spectrograms(setup: Setup,
                                               measurements: pd.DataFrame,
                                               nfft=1024,
                                               dynamic_range_db=60):
    propagation_speed = setup.get_propagation_speed(measurements,
                                                    prominence=0.2)
    """Calculate arrival times for sensor 1"""
    arrival_times, _ = get_travel_times(setup.actuators[ACTUATOR_1],
                                        setup.sensors[SENSOR_1],
                                        propagation_speed,
                                        milliseconds=False,
                                        relative_first_reflection=False)
    """Fit arrival times to compressed chrip times"""
    arrival_times += 2.5
    """Plot lines for expected arrival times"""
    spectrogram_with_lines(setup.sensors[SENSOR_1],
                           measurements,
                           arrival_times,
                           nfft,
                           dynamic_range_db)
    adjust_plot_margins()
    plt.savefig(FIGURES_SAVE_PATH + 'predicted_arrivals_spectrogram_sensor1_setup1.pdf',
                format='pdf')

    """Calculate arrival times for sensor 3"""
    arrival_times, _ = get_travel_times(setup.actuators[ACTUATOR_1],
                                        setup.sensors[SENSOR_3],
                                        propagation_speed,
                                        milliseconds=False,
                                        relative_first_reflection=False)
    """Fit arrival times to compressed chrip times"""
    arrival_times += 2.5
    spectrogram_with_lines(setup.sensors[SENSOR_3],
                           measurements,
                           arrival_times,
                           nfft,
                           dynamic_range_db)
    adjust_plot_margins()
    plt.savefig(FIGURES_SAVE_PATH + 'predicted_arrivals_spectrogram_sensor3_setup1.pdf',
                format='pdf')


"""Setup 2"""


def setup2_results():
    """Run some general commands for all functions"""
    print('Setup 2')
    """Choose file"""
    FILE_FOLDER = 'Setup_2'
    FILE_NAME = 'prop_speed_chirp3_setup3_2_v2'
    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Delete sensor 2 as it doesn't have the required bandwidth"""
    measurements = measurements.drop(['Sensor 2'], axis='columns')
    CHIRP_CHANNEL_NAMES.remove('Sensor 2')

    """Choose setup"""
    SETUP = Setup2()
    SETUP.draw()
    adjust_plot_margins()
    plt.savefig(FIGURES_SAVE_PATH + 'setup2_draw.pdf',
                format='pdf')

    setup2_transfer_function(SETUP)


def setup2_plot_time_signals(measurements: pd.DataFrame,
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
    adjust_plot_margins('time', rows=3, columns=1)

    """Compress chirps"""
    measurements = compress_chirps(measurements)

    """Plot the measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(plots_to_plot),
                            figsize=set_window_size(),
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
    adjust_plot_margins('time', rows=3, columns=1)


def setup2_plot_spectrogram_signals(measurements: pd.DataFrame,
                                    signal_start_seconds: float,
                                    signal_length_seconds: float):
    """SETTINGS FOR PLOTTING"""
    NFFT = 2048
    PLOTS_TO_PLOT = ['spectrogram']

    """Plot the raw measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    nfft=NFFT,
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    adjust_plot_margins('spectrogram', rows=3, columns=1)

    """Compress chirps"""
    measurements = compress_chirps(measurements)

    """Plot the measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    nfft=NFFT,
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=True)

    """Adjust for correct spacing in plot"""
    adjust_plot_margins('spectrogram', rows=3, columns=1)


def setup2_plot_fft_signals(measurements: pd.DataFrame,
                            signal_start_seconds: float,
                            signal_length_seconds: float):
    """SETTINGS FOR PLOTTING"""
    PLOTS_TO_PLOT = ['fft']

    """Plot the raw measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=False)

    """Adjust for correct spacing in plot"""
    adjust_plot_margins('fft', rows=3, columns=1)

    """Limit y axis"""
    for ax in axs.flatten():
        ax.set_ylim(bottom=-70, top=120)

    """Compress chirps"""
    measurements = compress_chirps(measurements)

    """Plot the measurements"""
    fig, axs = plt.subplots(nrows=measurements.shape[1],
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in measurements],
                    signal_start_seconds=signal_start_seconds,
                    signal_length_seconds=signal_length_seconds,
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=True)

    """Adjust for correct spacing in plot"""
    adjust_plot_margins('fft', rows=3, columns=1)

    """Limit y axis"""
    for ax in axs.flatten():
        ax.set_ylim(bottom=-50, top=230)


def setup2_transfer_function(setup: Setup):
    """Calculate the transfer function of the table
    by dividing the FFT of the direct signal on the output
    by the FFT of the direct signal on the input,
    H(f) = Y(f)/X(f),
    where the signal on sensor 1 is the input and
    the signal on sensor 3 is the output."""
    print('Calculating transfer function')

    """Choose file"""
    FILE_FOLDER = 'Setup_2'
    FILE_NAME = 'prop_speed_chirp3_setup3_2_v1'

    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Compress chirps"""
    measurements = compress_chirps(measurements)

    """Crop the compressed chirps to be only the direct signal"""
    direct_signal_seconds_sensor1 = 0.01
    measurements['Sensor 1'] = window_signals(pd.DataFrame(measurements['Sensor 1']),
                                              direct_signal_seconds_sensor1,
                                              window_function='tukey',
                                              window_parameter=0.1)
    direct_signal_seconds_sensor3 = 0.01
    measurements['Sensor 3'] = window_signals(pd.DataFrame(measurements['Sensor 3']),
                                              direct_signal_seconds_sensor3,
                                              window_function='tukey',
                                              window_parameter=0.1)

    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    PLOTS_TO_PLOT = ['time']
    fig, axs = plt.subplots(nrows=len(CHANNELS_TO_PLOT),
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(),
                            squeeze=False)
    compare_signals(fig, axs,
                    [measurements[channel] for channel in CHANNELS_TO_PLOT],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=True,
                    nfft=256)
    compare_signals(fig, axs,
                    [get_envelopes(measurements[channel])
                     for channel in CHANNELS_TO_PLOT],
                    plots_to_plot=PLOTS_TO_PLOT,
                    compressed_chirps=True,
                    dynamic_range_db=15,
                    nfft=32)
    """Adjust for correct spacing in plot"""
    adjust_plot_margins()

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

    """Plot the amplitude response"""
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=set_window_size())
    ax.plot(fft_frequencies / 1000, to_dB(amplitude_response))
    # ax.set_title('Amplitude response')
    ax.set_xlabel('Frequency [kHz]')
    ax.set_ylabel('Amplitude [dB]')
    ax.set_ylim(bottom=-35, top=20)
    ax.grid()
    adjust_plot_margins()
    FILE_NAME = 'setup2_amplitude_response.pdf'
    plt.savefig(FIGURES_SAVE_PATH + FILE_NAME, format='pdf')

    """Find the power of the signal with RMS"""
    signal_rms = np.sqrt(np.mean(measurements ** 2))
    signal_power = signal_rms ** 2  # yes it just undoes the square root
    power_loss_sensor1_to_sensor_3 = signal_power[SENSOR_3] / \
        signal_power[SENSOR_1]
    power_loss_sensor1_to_sensor_3_dB = to_dB(power_loss_sensor1_to_sensor_3)
    distance_between_sensors = np.linalg.norm(setup.sensors[SENSOR_3].coordinates -
                                              setup.sensors[SENSOR_1].coordinates)
    print(
        f'\nPower loss sensor 1 to sensor 3, for 100 Hz to 40 kHz: {power_loss_sensor1_to_sensor_3_dB:.2f} dB')
    print(
        f'Power loss per meter: {(power_loss_sensor1_to_sensor_3_dB / distance_between_sensors):.2f} dB/m\n')


def find_and_plot_power_loss(measurements: pd.DataFrame,
                             signal_start_seconds: float,
                             signal_length_seconds: float,
                             setup: Setup):
    """Find the power of the signal with RMS"""
    signal_rms = np.sqrt(np.mean(measurements ** 2))
    signal_power = signal_rms ** 2  # yes it just undoes the square root
    power_loss_sensor1_to_sensor_3 = signal_power[SENSOR_3] / \
        signal_power[SENSOR_1]
    power_loss_sensor1_to_sensor_3_dB = to_dB(power_loss_sensor1_to_sensor_3)
    distance_between_sensors = np.linalg.norm(setup.sensors[SENSOR_3].coordinates -
                                              setup.sensors[SENSOR_1].coordinates)
    print(
        f'\nPower loss sensor 1 to sensor 3, for 100 Hz to 40 kHz: {power_loss_sensor1_to_sensor_3_dB:.2f} dB')
    print(
        f'Power loss per meter: {(power_loss_sensor1_to_sensor_3_dB / distance_between_sensors):.2f} dB/m\n')

    """Compress chirps"""
    measurements = compress_chirps(measurements)

    """Plot the measurements"""
    PLOTS_TO_PLOT = ['fft']
    fig, axs = plt.subplots(nrows=1,
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(),
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
    # axs[0, 0].set_title('FFT of sensor 1 and sensor 3, ' +
        # f'{distance_between_sensors} meters apart')
    axs[0, 0].legend(['Sensor 1', 'Sensor 3'])
    axs[0, 0].set_ylim(bottom=-25, top=80)

    """Adjust for correct spacing inz plot"""
    adjust_plot_margins('fft', rows=1, columns=1)


"""Setup 3"""


def setup3_results():
    """Run some general commands for all functions."""
    print('Setup 3')

    """Choose setup"""
    SETUP = Setup3()
    SETUP.draw()
    adjust_plot_margins()
    plt.savefig(FIGURES_SAVE_PATH + 'setup3_draw.pdf',
                format='pdf')

    setup3_scattering()


def setup3_plot_raw_time_signal():
    """Open file"""
    FILE_FOLDER = 'Setup_3'
    FILE_NAME = 'notouch_20to40khz_1s_10vpp_v1'
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME,
                             channel_names=CHIRP_CHANNEL_NAMES)

    """Choose crop times"""
    TIME_START = 0.114 + 0.0036  # s
    TIME_END = 0.246 - 0.0035  # s

    measurements = interpolate_waveform(measurements)
    measurements = crop_data(measurements,
                             time_start=TIME_START,
                             time_end=TIME_END)

    """Plot raw signal"""
    time_axis = np.linspace(start=0,
                            stop=1000 * len(measurements) / SAMPLE_RATE,
                            num=len(measurements))
    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=set_window_size())
    # ax.set_title('Chirp from 20 khz to 40 kHz in 125 ms')
    ax.plot(time_axis, measurements['Sensor 1'], label='Sensor 1')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (mV)')
    ax.legend()
    ax.grid()

    """Adjust for correct spacing in plot"""
    adjust_plot_margins()


def setup3_scattering():
    """Open file"""
    FILE_FOLDER = 'Setup_3'
    FILE_NAME = 'notouchThenHoldB2_20to40khz_125ms_10vpp_v1'
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    measurements = interpolate_waveform(measurements)

    """Plot measurements"""
    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    fig, axs = plt.subplots(nrows=1, ncols=1,
                            sharex=True, sharey=True,
                            figsize=set_window_size(),
                            squeeze=False)
    time_axis = np.linspace(0, 5, measurements.shape[0])
    for channel in CHANNELS_TO_PLOT:
        axs[0, 0].plot(time_axis, measurements[channel], label=channel)
    # axs[0, 0].set_title('Shifted measurements')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Amplitude [V]')
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].grid()
    adjust_plot_margins()
    file_name = 'setup3_scattering_shifted_measurements_raw.pdf'
    plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    """Shift to align chirps better with their time intervals"""
    SHIFT_BY = int(0.0877 * SAMPLE_RATE)
    for channel in measurements:
        measurements[channel] = np.roll(measurements[channel],
                                        -SHIFT_BY)
        measurements[channel][-SHIFT_BY:] = 0

    """Compress chirps"""
    CHIRP_LENGTH = int(0.125 * SAMPLE_RATE)
    PRE_PAD_LENGTH = int(2.5 * SAMPLE_RATE)
    POST_PAD_LENGTH = measurements.shape[0] - \
        (PRE_PAD_LENGTH + CHIRP_LENGTH) - 1
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
                            figsize=set_window_size(),
                            squeeze=False)
    time_axis = np.linspace(0, 5, measurements.shape[0])
    for channel in CHANNELS_TO_PLOT:
        axs[0, 0].plot(time_axis, measurements[channel], label=channel)
    # axs[0, 0].set_title('Shifted measurements 1')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Amplitude [V]')
    axs[0, 0].set_xlim(0, 0.002)
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].grid()
    """Use scientific notation"""
    for ax in axs.flatten():
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    adjust_plot_margins()

    """Make up for drift in signal generator"""
    measurements_split = correct_drift(measurements_split,
                                       data_to_sync_with=measurements_split)

    """Plot all chirps on top of each other"""
    time_axis = np.linspace(start=0,
                            stop=len(
                                measurements_split['Sensor 1'][0]) / SAMPLE_RATE,
                            num=len(measurements_split['Sensor 1'][0]))
    CHANNELS_TO_PLOT = ['Sensor 1', 'Sensor 3']
    for channel in CHANNELS_TO_PLOT:
        fig, ax = plt.subplots(nrows=1, ncols=1,
                               figsize=set_window_size())
        for i, _ in enumerate(measurements_split[channel][0:39]):
            ax.plot(1000 * time_axis, measurements_split[channel][i])
        # ax.set_title(f'{i} chirps, {channel}')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Normalized \ncorrelation factor [-]')
        ax.set_xlim(0, 1)
        ax.grid()
        plt.subplots_adjust(left=0.19, right=0.967,
                            top=0.921, bottom=0.155,
                            hspace=0.28, wspace=0.2)
        channel_file_name = channel.replace(" ", "").lower()
        file_name = f'setup3_scattering_stacked_chirps_{channel_file_name}.pdf'
        plt.savefig(FIGURES_SAVE_PATH + file_name, format='pdf')

    """Separate chirps into with and without touch"""
    # notouch_average = average_of_signals(measurements_split,
    #                                      chirp_range=[0, 16])
    # touch_average = average_of_signals(measurements_split,
    #                                    chirp_range=[25, 39])
    # # Plot it all
    # fig, ax = plt.subplots(nrows=1, ncols=1,
    #                        figsize=set_window_size())
    # time_axis = np.linspace(start=0,
    #                         stop=len(touch_average['Sensor 1'][0]) / SAMPLE_RATE,
    #                         num=len(touch_average['Sensor 1'][0]))
    # ax.plot(1000 * time_axis, notouch_average['Sensor 1'],
    #         label='No touch average')
    # ax.plot(1000 * time_axis, touch_average['Sensor 1'],
    #         label='Touch average')
    # # ax.set_title('Sensor 1')
    # ax.set_xlabel('Time [ms]')
    # ax.set_ylabel('Normalized \ncorrelation factor [-]')
    # ax.set_xlim(0, 1)
    # ax.legend(loc='upper right')
    # ax.grid()
    # plt.subplots_adjust(left=0.19, right=0.967,
    #                     top=0.921, bottom=0.155,
    #                     hspace=0.28, wspace=0.2)


"""Generate custom graphs"""


def custom_plots():
    """Generate custom graphs"""
    print('Generate custom graphs')
    scattering_figure_3()


def scattering_figure_3():
    """Use coordinates of points along a graph to make a plot"""
    print('Scattering figure 3')
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
    adjust_plot_margins('fft', rows=1, columns=1)


def scattering_figure_5():
    """Use coordinates of points along a graph to make a plot"""
    print('Scattering figure 5')
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
    bonded_steel_inclusion_thickness_0_5in = interpolate_waveform(
        bonded_steel_inclusion_thickness_0_5in)

    """Plot"""
    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=set_window_size())
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
    adjust_plot_margins('fft', rows=1, columns=1)


if __name__ == '__main__':
    raise RuntimeError('This file is not meant to be run directly.')
