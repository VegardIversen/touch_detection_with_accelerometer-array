import scipy.signal as signal
import scipy
from scipy.optimize import minimize
from scipy import interpolate
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.widgets import Slider, Button
from pathlib import Path
from objects import Table, Actuator, Sensor
from setups import Setup2, Setup3, Setup3_2, Setup3_4, Setup6, Setup9, SimulatedSetup
from constants import SAMPLE_RATE, CHANNEL_NAMES, CHIRP_CHANNEL_NAMES
from data_processing import cross_correlation_position as ccp
from csv_to_df import csv_to_df, csv_to_df_thesis
from data_viz_files.visualise_data import (
    compare_signals,
    plot_vphs,
    plot_fft,
    plot_plate_speed_sliders_book,
    plot_estimated_reflections_with_sliders,
    compare_signals_v2,
    plot_compare_signals_v2,
)
from data_processing.preprocessing import (
    get_first_index_above_threshold,
    interpolate_waveform,
    crop_data,
    filter_general,
    compress_chirp,
    get_phase_and_vph_of_compressed_signal,
    cut_out_signal,
    manual_cut_signal,
    compress_df_touch,
    cut_out_pulse_wave,
    shift_signal,
)
from data_processing.detect_echoes import (
    find_first_peak,
    get_hilbert_envelope,
    get_travel_times,
)
from data_processing.find_propagation_speed import find_propagation_speed_with_delay
from data_viz_files.drawing import plot_legend_without_duplicates
import timeit
import data_processing.wave_properties as wp
import data_processing.sensor_testing as st
from data_viz_files.visualise_data import (
    inspect_touch,
    figure_size_setup,
    to_dB,
    figure_size_setup_thesis,
)
import data_processing.wave_properties as wp
import data_processing.sensor_testing as st
import data_processing.preprocessing as pp
from matplotlib import style
import data_viz_files.visualise_data as vd
from data_processing.dispersion_comp import dispersion_compensation
import os

# import mph


def results_setup1():
    ## Results for phase velocity test in the beginning of the thesis.
    custom_chirp = csv_to_df(
        file_folder="Measurements\\div_files",
        file_name="chirp_custom_fs_150000_tmax_2_100-40000_method_linear",
        channel_names=CHIRP_CHANNEL_NAMES,
    )
    # df1 = csv_to_df('Measurements\\setup3_0\\', 'chirp_100_40000_2s_v1')
    # phase10, freq10  = wp.phase_plotting(df1, chirp=custom_chirp, use_recorded_chirp=True,start_stops=[(241000,508570),(241000,508570)], BANDWIDTH=[100,40000], save_fig=False, file_name='phase_plot_10cm0_45.svg', file_format='svg',figsize=0.45, n_pi=1)
    # wp.plot_velocities(phase10, freq10, 0.10, savefig=False, filename='phase_velocity_10cm.svg', file_format='svg')
    df_PE = csv_to_df_thesis("plate20mm\\setup1\\chirp", "chirp_100_40000_2s_v1")
    df_teflon = csv_to_df_thesis("plate10mm\\setup1\\chirp", "chirp_100_40000_2s_v1")
    # filter signal
    df_PE_filt = filter_general(
        df_PE,
        filtertype="bandpass",
        cutoff_highpass=10000,
        cutoff_lowpass=15000,
        order=4,
    )
    # df_teflon_filt = filter_general(df_teflon, filtertype='bandpass', cutoff_highpass=100, cutoff_lowpass=40000, order=4)
    phase_PE, freq_PE = wp.phase_plotting_chirp(
        df_PE,
        BANDWIDTH=[100, 40000],
        save_fig=False,
        file_name="phase_plot_PE_45.svg",
        file_format="svg",
        figsize=0.45,
        n_pi=1,
    )
    # phase_teflon, freq_teflon  = wp.phase_plotting_chirp(df_teflon, BANDWIDTH=[5000,40000], save_fig=False, file_name='phase_plot_teflon_45.svg', file_format='svg',figsize=0.45, n_pi=1)
    # wp.plot_velocities(phase_PE, freq_PE, 0.10, savefig=False, filename='phase_velocity_PE.svg', file_format='svg')
    # wp.plot_velocities(phase_teflon, freq_teflon, 0.10, savefig=False, filename='phase_velocity_teflon.svg', file_format='svg')
    # wp.plot_velocities(phase_PE, freq_PE, 0.10, material='HDPE', savefig=False, filename='phase_velocity_teflon.svg', file_format='svg')
    # wp.plot_velocities(phase_PE, freq_PE, 0.10, material='LDPE', savefig=False, filename='phase_velocity_teflon.svg', file_format='svg')
    print(wp.max_peak_velocity(df_PE, material="HDPE"))
    # print(wp.max_peak_velocity(df_teflon))
    # print(wp.max_peak_velocity(df_PE))


def data_viz(viz_type, folder, filename, semester="thesis", channel="wave_gen"):
    if semester == "thesis":
        df = csv_to_df_thesis(folder, filename)
    else:
        df = csv_to_df(folder, filename)
    if viz_type == "scaleogram":
        vd.plot_scaleogram(df, channels=["wave_gen"])
    elif viz_type == "wvd":
        vd.wigner_ville_dist(df, channel)
    elif viz_type == "custom_wvd":
        vd.custom_wigner_ville_batch(df, channel)
    elif viz_type == "ssq":
        vd.ssqueeze_spectrum(df, channel)


def load_simulated_data1():
    signals = ["sg8.txt", "sg28.txt", "sg108.txt"]
    freq_signals = ["wt8.txt", "wt28.txt", "wt108.txt"]
    channels = []
    for sig in signals:
        path = os.path.join(
            r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm",
            sig,
        )
        data = np.loadtxt(path)
        channels.append(data)
    return channels


def simulated_data_vel():
    distances = [0.173, 0.41, 1.359]
    channels = load_simulated_data1()
    for idx, ch in enumerate(channels):
        plt.plot(ch, label=f"channel {idx+1}")
    plt.legend()
    plt.show()
    # Compute FFTs of the three data sets
    data8 = channels[0]
    data28 = channels[1]
    data108 = channels[2]
    fft8 = np.fft.fft(data8)
    fft28 = np.fft.fft(data28)
    fft108 = np.fft.fft(data108)

    # Create frequency axis
    freq = np.fft.fftfreq(data8.size, d=1 / 1000)

    # Select positive-frequency values
    # pos_freq = freq > 0
    # freq = freq[pos_freq]
    # fft8 = np.abs(fft8[pos_freq]) / data8.size
    # fft28 = np.abs(fft28[pos_freq]) / data28.size
    # fft108 = np.abs(fft108[pos_freq]) / data108.size

    # Convert to dB scale
    fft8_db = 20 * np.log10(fft8)
    fft28_db = 20 * np.log10(fft28)
    fft108_db = 20 * np.log10(fft108)

    # Plot FFTs of the three data sets in dB scale
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    axs[0].plot(freq, fft8_db)
    axs[1].plot(freq, fft28_db)
    axs[2].plot(freq, fft108_db)

    # Set title and axis labels
    axs[0].set_title("FFT for punkt 8")
    axs[1].set_title("FFT for punkt 28")
    axs[2].set_title("FFT for punkt 108")
    for ax in axs:
        ax.set_xlabel("Frekvens (kHz)")
        ax.set_ylabel("Amplitude (dB)")
    plt.tight_layout()
    plt.show()
    phase = wp.phase_difference_div(channels[0], channels[1])
    phase1 = wp.phase_difference_div(channels[0], channels[2])
    # Compute complex transfer functions
    tf28_8 = fft28 / fft8
    tf108_8 = fft108 / fft8
    print(f"the length of fft28 is {len(fft28)}")
    # Compute phase differences between transfer functions
    phase_diff_28_8 = np.unwrap(np.angle(tf28_8))
    phase_diff_108_8 = np.unwrap(np.angle(tf108_8))

    # Create frequency axis
    # freq = np.fft.fftfreq(data8.size, d=1/1000)

    # Select positive-frequency values
    # pos_freq = freq >= 0
    # freq = freq[pos_freq]
    # phase_diff_28_8 = phase_diff_28_8[pos_freq]
    # phase_diff_108_8 = phase_diff_108_8[pos_freq]

    # Plot phase differences
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(freq, phase)
    axs[0].set_title("Phase difference tf28/tf8")
    axs[1].plot(freq, phase1)
    axs[1].set_title("Phase difference tf108/tf8")
    for ax in axs:
        ax.set_xlabel("Frekvens (kHz)")
        ax.set_ylabel("Faseforskyvning (rad)")
    plt.tight_layout()
    plt.show()


def un_reassigned_spectrogram():
    import libtfr
    from librosa import display
    import librosa

    position = 10
    # Generate sample data
    wave_data, x_pos, y_pos, z_pos, time_axis = get_comsol_data()
    sr = 501000
    # Compute the Un-reassigned spectrogram
    x = wave_data[position]
    # Compute the Un-reassigned spectrogram
    nfft = 4096
    Np = nfft
    shift = nfft / 16
    K = 6
    tm = 6.0
    flock = 0.01
    tlock = 5
    S = np.abs(
        libtfr.tfr_spec(
            s=x, N=nfft, step=shift, Np=Np, K=K, tm=tm, flock=flock, tlock=tlock
        )
    )
    print(np.shape(S), np.max(S), np.min(S))
    S = librosa.amplitude_to_db(S, ref=np.max, top_db=100)
    print(np.shape(S), np.max(S), np.min(S))
    fig, ax = plt.subplots(figsize=(20, 5))
    display.specshow(D, y_axis="log", cmap="viridis")


def wave_number_graph(number=1):
    # Given data points
    # Given data points

    fs = 501000
    data, x_pos, y_pos, z_pos, time_axis = get_comsol_data(number)
    phase = wp.phase_difference_div(data[10], data[20], fs, pos_only=True)
    freq = np.fft.fftfreq(data[0].size, 1 / fs)
    freq = freq[freq > 0]
    fig, ax = plt.subplots()
    ax.plot(freq, phase)
    ax.set_xlabel("Frekvens (Hz)")
    ax.set_ylabel("Faseforskyvning (rad)")
    ax.set_title("Faseforskyvning mellom punkt 10 og 20")
    plt.show()
    distance = x_pos[20] - x_pos[10]
    # phase_vel = wp.phase_velocity(phase, freq, distance)
    wp.plot_velocities(phase, freq, distance, material="LDPE_tonni7mm")


def draw_all_plates():
    for i in range(1, 11):
        setup = SimulatedSetup(comsol_file=i, all_sensors=True)
        setup.draw(
            save_fig=True,
            fig_name=f"allsensors_plate_data_{i}",
            file_format="png",
            actuator_show=True,
            show_tab=False,
        )
        setup.draw(
            save_fig=True,
            fig_name=f"allsensors_plate_data_{i}",
            file_format="svg",
            actuator_show=True,
            show_tab=False,
        )


def draw_simulated_plate(position=35, comsolefile=9, velocity=None):
    # position = 35
    alpha = 0.4
    setup = SimulatedSetup(comsol_file=comsolefile)
    setup.draw()
    if velocity is not None:
        print(f"setting velocity: {velocity}")
        setup.set_propagation_vel(velocity)
    print(setup.get_propagation_vel())
    distances = setup.get_travel_distances_at_position(position)
    arrival_times = setup.get_arrival_times_at_position(position)
    print(f"arrival times: {sorted(arrival_times)}")
    distances = sorted(distances)
    print(f"distances: {distances}")
    return distances, arrival_times
    # theoretical_distances = setup.get_diagonal_distance(positions=[position, position+10])
    # print(f'distances from setup: {theoretical_distances}')
    # print(f'Arrial times: {sorted(arrival_times)}')
    # print(f'distances from reflections: {distances}')
    # h_x, x = dispersion_compensation_Wilcox(position=position)
    # #h_x1, x_1  = dispersion_compensation_Wilcox(position=position+10)
    # h_x1, x_1  = dispersion_compensation_Wilcox(position=position, pertubation=True, alpha=alpha)
    # #dx = 0.00018507876875628913
    # #x = np.arange(len(h_x))*dx
    # #plot h(x) and vertical lines at the distances
    # analytic_signal = signal.hilbert(h_x)
    # envelope = np.abs(analytic_signal)
    # peaks, _ = signal.find_peaks(envelope, prominence=0.3*envelope.max())
    # analytic_signal1 = signal.hilbert(h_x1)
    # envelope1 = np.abs(analytic_signal1)
    # peaks1, _ = signal.find_peaks(envelope1, prominence=0.3*envelope1.max())

    # plt.plot(x, h_x, label=f'h(x) position: {position}')
    # #plt.plot(x_1, h_x1, label=f'h(x) postition: {position+10}')
    # plt.plot(x_1, h_x1, label=f'h(x) postition: {position} pertubation with alpha={alpha}')
    # #plt.plot(x, envelope, label='envelope')
    # #plt.plot(x[peaks], envelope[peaks], "x", label=f'peaks position: {position}')
    # #plt.plot(x_1, envelope1, label='envelope')
    # #plt.plot(x_1[peaks1], envelope1[peaks1], "x", label=f'peaks position: {position+10}')
    # print(x[peaks])
    # print(f'distances: {sorted(distances)}')
    # #print(f'x: {x}')
    # plt.vlines(distances[:7], ymin=0, ymax=1, label='reflections', colors='r')
    # plt.xlabel(xlabel='x (m)')
    # plt.legend()
    # plt.show()
    # peak_diff = x[peaks1[0]]-x[peaks[0]]
    # print(f'Distance between the peaks are {peak_diff} m')
    # print(f'theoretical distances: {theoretical_distances} m, peak distance: {peak_diff} m')
    # print(f'difference between the physical distance and the peak distance: {peak_diff-theoretical_distances} m')


def testing_wilcox_disp(
    file_n=2,
    position=25,
    fs=500000,
    dx=0.0001,
    pertubation=False,
    alpha=0.2,
    center_pulse=True,
):
    wave_data, x_pos, y_pos, z_pos, time_axis = get_comsol_data(
        9
    )  # fetches data from comsol files
    if center_pulse:
        time_axis = time_axis - 133e-6
    source_pos = [x_pos[0], y_pos[0]]
    # print(f'time_axis: {time_axis}')
    # print(f'source_pos: {source_pos}')
    distanse = np.sqrt(
        (x_pos[position] - source_pos[0]) ** 2 + (y_pos[position] - source_pos[1]) ** 2
    )
    signal = wave_data[position]
    oversampling_factor = 8
    n_fft = signal.size * oversampling_factor
    freq_vel = np.fft.fftfreq(n_fft, 1 / fs)
    freq_vel = freq_vel[: int(n_fft / 2)]
    v_gr, v_ph = wp.theoretical_group_phase_vel(
        freq_vel, material="LDPE_tonni20mm", plot=False
    )
    k_mark = freq_vel / v_ph
    # replace nan value with 0
    k_mark = np.nan_to_num(k_mark)
    print(signal.shape)
    # signal = signal.reshape(1,501)
    # print(signal.shape)
    d_step, hx = dispersion_compensation(
        time_axis,
        signal,
        freq_vel,
        k_mark,
        truncate=True,
        oversampling_factor=8,
        interpolation_method="linear",
    )
    plt.plot(d_step, hx)
    plt.show()


def test_dispersion_compensation_gen_signals():
    # load in data from tonni files
    data = load_simulated_data1()
    df = 1e3  # Frequency resolution (1kHz)
    f = np.arange(1, 50e3 + df, df)  # Frequencies from 1kHz to 50kHz
    v_gr, v_ph = wp.theoretical_group_phase_vel(f, material="LDPE_tonni7mm", plot=True)
    k_mark = f / v_ph

    signal = data[0]
    print(f"signal: {signal}")
    print(f"signal shape: {signal.shape}")

    signal_length = signal.size
    dt = 1 / (df * signal_length)  # Time resolution
    t = np.arange(0, signal_length) * dt  # Time axis
    plt.plot(t, data[0])
    plt.show()

    d_step, hx = dispersion_compensation(
        t,
        signal,
        f,
        k_mark,
        truncate=True,
        oversampling_factor=8,
        interpolation_method="linear",
    )
    # do this for every channel in data
    d_step1, hx1 = dispersion_compensation(
        t,
        data[1],
        f,
        k_mark,
        truncate=True,
        oversampling_factor=8,
        interpolation_method="linear",
    )
    d_step2, hx2 = dispersion_compensation(
        t,
        data[2],
        f,
        k_mark,
        truncate=True,
        oversampling_factor=8,
        interpolation_method="linear",
    )

    # plot signal and dispersion compensated signal together in a subfigure
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(t, signal, label="signal 17.3mm", color="r")
    axs[0].plot(t, data[1], label="signal 41mm", color="g")
    axs[0].plot(t, data[2], label="signal 135.9mm", color="b")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_title("Signal")
    axs[1].plot(d_step, hx, label="dispersion compensated signal 17.3mm")
    axs[1].plot(d_step1, hx1, label="dispersion compensated signal 41mm")
    axs[1].plot(d_step2, hx2, label="dispersion compensated signal 135.9mm")
    # plotting vertical lines at the distances, with different colors
    axs[1].vlines(0.0173, ymin=0, ymax=hx.max(), label="reflections", colors="r")
    axs[1].vlines(0.041, ymin=0, ymax=hx.max(), label="reflections", colors="g")
    axs[1].vlines(0.1359, ymin=0, ymax=hx.max(), label="reflections", colors="b")
    axs[1].set_title("Dispersion compensated signal")
    axs[1].set_xlabel("Distance (m)")
    plt.title(label="Signal and dispersion compensated signal, 17.3mm")
    plt.tight_layout()
    plt.show()


def dispersion_compensation_Wilcox(
    file_n=2, position=25, fs=500000, dx=0.0001, pertubation=False, alpha=0.2
):
    """
    Performs dispersion compensation on the input signal.

    Args:
        signal (ndarray): The input time-domain signal.
        postion (int): The postion we want to use.
        dx (float): Distance step of the final distance-trace.
        mode (float): Desired mode of the guided wave.
        vmax (float): Maximum group velocity of the guided wave.

    Returns:
        ndarray: The dispersion compensated distance-trace.


    problems:
    - I just assume dx is 1mm, not sure if i have to calculate it. But it fulfills the equations.
    - Havent quite understood what the result is. In the paper it returns to a distance trace. But what does that mean?
    - Not quite sure if I can change the frequency like I have done now. The paper doesnt mention anything about tak
    """

    def get_k_value(freq):
        v_freq = get_velocity_at_freq(freq)["A0"][
            "phase_velocity"
        ]  # fetches the velocity at the upper frequency
        k_freq = (2 * np.pi * freq) / v_freq
        return k_freq

    upper_freq = 60000
    lower_freq = 0
    new_fs = 80000
    tolerance = 1e-9
    alpha = 0.9
    # xmax = 1
    # frequency axis from 0 to 40kHz with 80000 samples
    # freq = np.linspace(lower_freq, upper_freq, new_fs)

    print("###############################################")
    wave_data_top, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(
        9
    )  # fetches data from comsol files
    (
        wave_data_bottom,
        x_pos_bottom,
        y_pos_bottom,
        z_pos_bottom,
        time_axis_bottom,
    ) = get_comsol_data(
        10
    )  # fetches data from comsol files
    print(f"x_pos_top: {x_pos_top[position]}")
    print(f"y_pos_top: {y_pos_top[position]}")
    source_pos = [x_pos_top[0], y_pos_top[0]]
    print(f"source_pos: {source_pos}")
    distanse = np.sqrt(
        (x_pos_bottom[position] - source_pos[0]) ** 2
        + (y_pos_bottom[position] - source_pos[1]) ** 2
    )
    print(f"distance: {distanse}")
    # Plots raw data
    # plt.title(f'Raw data for top and bottom at position: {position}')
    # plt.plot(wave_data_top[position], label='top')
    # plt.plot(wave_data_bottom[position], label='bottom')
    # plt.legend()
    # plt.show()
    signal = wave_data_top[position]
    # signal = (wave_data_top[position]+wave_data_bottom[position])/2 #A0 mode
    # #plotting signal mode
    # plt.plot(signal, label='A0 mode')
    # plt.title('A0 mode')
    # plt.legend()
    # plt.show()

    n_times = 16  # number of points that is an integral power of two and at least eight times as many as in the original signal.
    m = len(signal)
    n_fft = 2 ** int(np.ceil(np.log2(n_times * m)))
    print(
        f"length of signal before padding: {m}, length of signal after padding {n_fft}"
    )
    # padding signal with zeros, new length is n_fft
    signal_padded = np.pad(signal, (0, n_fft - m), mode="constant")
    print(f"shape of signal_padded: {signal_padded.shape}")
    # plotting padded signal
    # plt.plot(signal_padded)
    # plt.title('Padded signal')
    # plt.show()

    # computing fft of padded signal
    G_w = np.fft.fft(signal_padded)
    print(f"shape of G_w: {G_w.shape}")
    dt = 1 / fs  # 1/501000
    print(f"dt: {dt}")
    # computing frequency axis, will have length G_w.size
    freq_vel = np.fft.fftfreq(G_w.size, dt)
    # plotting fft of padded signal
    # plt.plot(freq_vel, np.abs(G_w))
    # plt.title('Fourier transform of padded signal')
    # plt.show()
    print(f"shape of freq_vel: {freq_vel.shape}")

    # freq_range = (freq_vel>=lower_freq) & (freq_vel<=upper_freq)
    # freq_vel = freq_vel[freq_range]
    # f_nyq = freq_vel[-1]/2
    # G_w = G_w[freq_range]

    # only looking at positive frequencies
    # G_w = G_w[freq_vel>0]
    G_w = G_w[: int(n_fft / 2)]
    print(f"length of positive G_w: {G_w.shape}")
    # freq_vel = freq_vel[freq_vel>0]
    freq_vel = freq_vel[: int(n_fft / 2)]
    print(f"length of positive freq_vel: {freq_vel.shape}")
    f_nyq = fs / 2
    print(f"f_nyq: {f_nyq}")
    print(f"last element in freq_vel: {freq_vel[-1]}")
    # plotting fft of padded signal after frequency range
    # plt.plot(freq_vel, np.abs(G_w))
    # plt.title('Fourier transform of padded signal after frequency range')
    # plt.show()
    print(f"freq_vel: {freq_vel.shape}")
    # dt = 1/upper_freq
    v_gr, v_ph = wp.theoretical_group_phase_vel(
        freq_vel, material="LDPE_tonni20mm", plot=True
    )  # group and phase velocity with the same length as freq_vel
    if pertubation:
        v_ph = (1 + alpha) * v_ph
        v_gr_old = v_gr
        v_gr = wp.group_velocity_phase(v_ph, freq_vel)
        plt.plot(freq_vel, v_gr_old, label="v_gr old")
        plt.plot(freq_vel, v_gr, label="v_gr new")
        plt.legend()
        plt.show()

    print(f"max of v_gr: {np.max(v_gr)} and max of v_ph: {np.max(v_ph)}")
    v_max = np.max(v_gr)
    max_distance_wave = m * dt * np.max(v_gr)
    print(f"max distance wave: {max_distance_wave} m")
    # k = (2*np.pi*freq_vel)/v_ph #same length as freq_vel. Wavenumber domain
    k = np.where(abs(v_ph) > tolerance, (2 * np.pi * freq_vel) / v_ph, 0)

    # Save the velocity dictionary in MATLAB format
    # savemat('velocity_data_7mm.mat', velocity_dict)
    print(f"k: {k.shape}")
    v_nyq = get_velocity_at_freq(f_nyq)["A0"][
        "phase_velocity"
    ]  # fetches the velocity at the nyquist frequency
    # print(f'k_max = {k[-1]}')
    # v_max = get_velocity_at_freq(upper_freq)['A0']['phase_velocity'] #fetches the velocity at the upper frequency
    k_nyq = get_k_value(f_nyq)
    print(f"k_nyq: {k_nyq}")
    print(f"max k: {np.max(k)}")
    n = len(k)
    print(f"length of k: {n}")
    dx = 1 / (2 * k_nyq)
    # dx = 0.00121357285
    # dx = 0.001
    print(f"dx: {dx}")
    k_min = 1 / (2 * dx * fs)
    # dk = 1/(xmax)
    # dk = 1/(n*dx)
    # dk = 1/(n_fft*dx)
    # k_max = 2*k_nyq
    # k_max = 1/(2*dx)
    k_max = np.max(k)
    dk = k_max / (len(k) - 1)

    # k_max = k[-1] #doesnt matter if i use this or this 2*np.pi*upper_freq/v_max since both are equal or 2 times k_nyq
    print(f"k_nyq: {k_nyq}, kmax: {k_max}")
    w = 2 * np.pi * freq_vel
    print(f"shape of w: {w.shape}")
    # print(f'altnerative length of k: {int(np.ceil(2 * f_nyq / (1 / (dx * m))))}')
    # plotting wavenumber vs frequency
    plt.plot(k, freq_vel)
    plt.xlabel("Wavenumber")
    plt.ylabel("Frequency")
    plt.title("Wavenumber vs frequency")
    plt.show()
    # Perform FFT on the padded signal

    # print(len(k))

    # Calculate wavenumber step and number of points in the wavenumber domain
    print(
        f"Checking if n*delta_x is larger than m*delta_t*v_max. n*delta_x is {n_fft*dx}, m*delta_t*v_max is {m*dt*np.max(v_gr)}"
    )
    # k_nyq = #k[round(1/(2*dt))]
    print(
        f"Checking if Delta x is less or equal to 1/(2k_nyq). Delta x is {dx}, 1/(2k_nyq) is {1/(2*k_nyq)}"
    )
    # dk1 = 1 / (n_fft * dx) #wavenumber step
    # creating new k axis
    # k_new = np.arange(k_min, k_max, dk)
    # k_new = np.arange(0, k_max + dk, dk)
    k_new = np.arange(0, k_max, dk)
    print(f"shape of k new: {k_new.shape}")
    # print(f'dk1: {dk1}, dk: {dk}')
    # print(f'shape of x: {x.shape}')
    # print(f'max of x: {x[-1]}')
    print(
        f"n should be larger than 2 * k_nyq / dk, n is {n_fft}, 2 * k_nyq / dk is {2 * k_nyq / dk}"
    )
    # print(f'this number of points in the wavenumber domain is {n}')

    # Interpolate the FFT to equally spaced k values
    # print(f'shape of k: {k.shape}, k_new: {k_new.shape}, freq_vel: {freq_vel.shape}')
    plt.plot(k_new, label="k_new")
    plt.plot(k, label="k")
    # plt.xlabel('Wavenumber')
    # plt.ylabel('Frequency')
    plt.xlabel("sample")
    plt.title("k_new vs k")
    plt.legend()
    plt.show()
    # Interpolate G(w) to find G(k)
    G_interp = interpolate.interp1d(
        k, G_w, kind="linear", bounds_error=False, fill_value=0
    )(k_new)
    plt.plot(k_new, G_interp.real, label="interpolated G(k)")
    plt.plot(k, G_w.real, label="G(k)")
    plt.xlabel("Wavenumber")
    plt.ylabel("Amplitude")
    plt.title("Interpolated G(k)")
    plt.legend()
    plt.show()
    print("G_interp created")
    # Calculate the group velocity of the guided wave mode at the wavenumber points
    v_gr_interp = interpolate.interp1d(
        k, v_gr, kind="linear", bounds_error=False, fill_value=0
    )(k_new)

    print(f"shape of G_interp: {G_interp.shape}")
    print(f"shape of v_gr_interp: {v_gr_interp.shape}")

    plt.plot(k_new, v_gr_interp, label="v_gr_interp")
    plt.plot(k, v_gr, label="v_gr")
    plt.xlabel("Wavenumber")
    plt.ylabel("Velocity")
    plt.title("Interpolated v_gr")
    plt.legend()
    plt.show()

    # Compute H(k) = G(k) * vgr(k)
    H_k = G_interp * (v_gr_interp)
    plt.plot(k_new, H_k, label="H_k")
    plt.xlabel("Wavenumber")
    plt.ylabel("Amplitude")
    plt.title("H(k)")
    plt.legend()
    plt.show()
    # Apply inverse FFT to H(k) to obtain the dispersion compensated distance-trace
    h_x = np.fft.ifft(H_k)
    print(f"shape of h_x: {h_x.shape} before removing zero-padding")
    # Remove zero-padding from the compensated signal
    h_x_padd = h_x
    print(h_x_padd)
    plt.plot(h_x_padd, label="h_x_padd")
    plt.legend()
    plt.show()
    h_x = h_x[:m]
    # x = np.arange(len(h_x))*dx
    # normalize h_x and signal
    h_x = h_x / np.max(h_x)
    signal = signal / np.max(signal)
    xmax = 1 / (dk) - dx
    # x = np.arange(0, xmax, dx)
    x = time_axis_top * 1e-6 * v_max
    print(f"delta x in array is {x[1]-x[0]}")
    print(f"shape of h_x: {h_x.shape}")
    print(f"shape of x: {x.shape}")
    print(f"max of x: {x[-1]}")
    print(f"shape of signal: {signal.shape}")
    # Create a subplot with the dispersion compensated signal and the original signal
    # plotting the results
    plt.subplot(2, 1, 1)
    plt.plot(x, h_x.real, label="Dispersion compensated signal")
    # plt.plot(h_x.real, label='Dispersion compensated signal')
    plt.xlabel("distance [m]")
    # plt.xlabel('sample')
    plt.ylabel("Amplitude")
    plt.title("Dispersion compensated signal")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time_axis_top, signal, label="Original signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Original signal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # G_k = np.fft.fft(h_x)
    # g_disp = np.real(np.fft.ifft(G_k))
    # plt.subplot(2, 1, 1)
    # plt.plot(time_axis_top, g_disp, label='Dispersion compensated signal')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Amplitude')
    # plt.title('Dispersion compensated signal')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.plot(time_axis_top,signal, label='Original signal')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Amplitude')
    # plt.title('Original signal')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plot fft of dispersion compensated signal and original signal with the same frequency axis and padded
    # G_k_padded = np.pad(h_x, (0, n_fft - m), mode='constant')
    # print(f'shape of G_k_padded: {G_k_padded.shape}')
    # G_k_padded = np.fft.fft(h_x_padd)
    # freq_gk = np.fft.fftfreq(h_x_padd.size, dt)
    # G_k_padded = G_k_padded[freq_gk>0]
    # freq_gk = freq_gk[freq_gk>0]
    # plt.subplot(2, 1, 1)
    # plt.plot(freq_gk, np.abs(G_k_padded), label='Dispersion compensated signal')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('Amplitude')
    # plt.title('Dispersion compensated signal')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.plot(freq_vel, np.abs(G_w), label='Original signal')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('Amplitude')
    # plt.title('Original signal')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    print("###############################################")
    return h_x.real, x


def dispersion_compensation_Wilcox_ref(file_n=2, postion=25, fs=500000, dx=0.0001):
    """
    Performs dispersion compensation on the input signal.

    Args:
        signal (ndarray): The input time-domain signal.
        postion (int): The postion we want to use.
        dx (float): Distance step of the final distance-trace.
        mode (float): Desired mode of the guided wave.
        vmax (float): Maximum group velocity of the guided wave.

    Returns:
        ndarray: The dispersion compensated distance-trace.


    problems:
    - I just assume dx is 1mm, not sure if i have to calculate it. But it fulfills the equations.
    - Havent quite understood what the result is. In the paper it returns to a distance trace. But what does that mean?
    - Not quite sure if I can change the frequency like I have done now. The paper doesnt mention anything about tak
    """

    def get_k_value(freq):
        v_freq = get_velocity_at_freq(freq)["A0"][
            "phase_velocity"
        ]  # fetches the velocity at the upper frequency
        k_freq = (2 * np.pi * freq) / v_freq
        return k_freq

    upper_freq = 60000
    lower_freq = 0
    new_fs = 80000
    # xmax = 1
    # frequency axis from 0 to 40kHz with 80000 samples
    # freq = np.linspace(lower_freq, upper_freq, new_fs)

    wave_data_top, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(
        9
    )  # fetches data from comsol files
    (
        wave_data_bottom,
        x_pos_bottom,
        y_pos_bottom,
        z_pos_bottom,
        time_axis_bottom,
    ) = get_comsol_data(
        10
    )  # fetches data from comsol files
    print(f"x_pos_top: {x_pos_top[postion]}")
    print(f"y_pos_top: {y_pos_top[postion]}")
    source_pos = [x_pos_top[0], y_pos_top[0]]
    distanse = np.sqrt(
        (x_pos_bottom[postion] - source_pos[0]) ** 2
        + (y_pos_bottom[postion] - source_pos[1]) ** 2
    )
    print(f"distance: {distanse}")
    # Plots raw data
    plt.title(f"Raw data for top and bottom at postion: {postion}")
    plt.plot(wave_data_top[postion], label="top")
    plt.plot(wave_data_bottom[postion], label="bottom")
    plt.legend()
    plt.show()
    # signal = wave_data_top[postion]
    signal = (wave_data_top[postion] + wave_data_bottom[postion]) / 2  # A0 mode
    # plotting signal mode
    plt.plot(signal, label="A0 mode")
    plt.title("A0 mode")
    plt.legend()
    plt.show()

    n_times = 16  # number of points that is an integral power of two and at least eight times as many as in the original signal.
    m = len(signal)
    n_fft = 2 ** int(np.ceil(np.log2(n_times * m)))
    print(
        f"length of signal before padding: {m}, length of signal after padding {n_fft}"
    )
    # padding signal with zeros, new length is n_fft
    signal_padded = np.pad(signal, (0, n_fft - m), mode="constant")
    print(f"shape of signal_padded: {signal_padded.shape}")
    # plotting padded signal
    plt.plot(signal_padded)
    plt.title("Padded signal")
    plt.show()

    # computing fft of padded signal
    G_w = np.fft.fft(signal_padded)
    print(f"shape of G_w: {G_w.shape}")
    dt = 1 / fs  # 1/501000
    print(f"dt: {dt}")
    # computing frequency axis, will have length G_w.size
    freq_vel = np.fft.fftfreq(G_w.size, dt)
    # plotting fft of padded signal
    plt.plot(freq_vel, np.abs(G_w))
    plt.title("Fourier transform of padded signal")
    plt.show()
    print(f"shape of freq_vel: {freq_vel.shape}")

    # freq_range = (freq_vel>lower_freq) & (freq_vel<upper_freq)
    # freq_vel = freq_vel[freq_range]
    # f_nyq = freq_vel[-1]/2
    # G_w = G_w[freq_range]

    # only looking at positive frequencies
    G_w = G_w[freq_vel > 0]
    print(f"length of positive G_w: {G_w.shape}")
    freq_vel = freq_vel[freq_vel > 0]
    print(f"length of positive freq_vel: {freq_vel.shape}")
    f_nyq = fs / 2
    print(f"f_nyq: {f_nyq}")
    print(f"last element in freq_vel: {freq_vel[-1]}")
    # plotting fft of padded signal after frequency range
    plt.plot(freq_vel, np.abs(G_w))
    plt.title("Fourier transform of padded signal after frequency range")
    plt.show()
    print(f"freq_vel: {freq_vel.shape}")
    # dt = 1/upper_freq
    v_gr, v_ph = wp.theoretical_group_phase_vel(
        freq_vel, material="LDPE_tonni20mm", plot=True
    )  # group and phase velocity with the same length as freq_vel
    print(f"v_gr: {v_gr.shape}")
    print(f"v_ph: {v_ph.shape}")
    print(f"max of v_gr: {np.max(v_gr)} and max of v_ph: {np.max(v_ph)}")
    max_distance_wave = m * dt * np.max(v_gr)
    print(f"max distance wave: {max_distance_wave} m")
    k = (2 * np.pi * freq_vel) / v_ph  # same length as freq_vel. Wavenumber domain
    print(f"k: {k.shape}")
    v_nyq = get_velocity_at_freq(f_nyq)["A0"][
        "phase_velocity"
    ]  # fetches the velocity at the nyquist frequency
    # print(f'k_max = {k[-1]}')
    # v_max = get_velocity_at_freq(upper_freq)['A0']['phase_velocity'] #fetches the velocity at the upper frequency
    k_nyq = get_k_value(f_nyq)
    print(f"k_nyq: {k_nyq}")
    print(f"max k: {np.max(k)}")
    n = len(k)
    print(f"length of k: {n}")
    dx = 1 / (2 * k_nyq)
    # dx = 0.001
    print(f"dx: {dx}")
    k_min = 1 / (2 * dx * fs)
    # dk = 1/(xmax)
    dk = 1 / (n * dx)
    # dk = 1/(n_fft*dx)
    # k_max = 2*k_nyq
    k_max = 1 / (2 * dx)
    # k_max = k[-1] #doesnt matter if i use this or this 2*np.pi*upper_freq/v_max since both are equal or 2 times k_nyq
    print(f"k_nyq: {k_nyq}, kmax: {k_max}")
    w = 2 * np.pi * freq_vel
    print(f"shape of w: {w.shape}")
    # print(f'altnerative length of k: {int(np.ceil(2 * f_nyq / (1 / (dx * m))))}')
    # plotting wavenumber vs frequency
    plt.plot(k, freq_vel)
    plt.xlabel("Wavenumber")
    plt.ylabel("Frequency")
    plt.title("Wavenumber vs frequency")
    plt.show()
    # Perform FFT on the padded signal

    # print(len(k))

    # Calculate wavenumber step and number of points in the wavenumber domain
    print(
        f"Checking if n*delta_x is larger than m*delta_t*v_max. n*delta_x is {n_fft*dx}, m*delta_t*v_max is {m*dt*np.max(v_gr)}"
    )
    # k_nyq = #k[round(1/(2*dt))]
    print(
        f"Checking if Delta x is less or equal to 1/(2k_nyq). Delta x is {dx}, 1/(2k_nyq) is {1/(2*k_nyq)}"
    )
    dk1 = 1 / (n_fft * dx)  # wavenumber step
    # creating new k axis
    # k_new = np.arange(k_min, k_max, dk)
    k_new = np.arange(0, k_max + dk, dk)

    print(f"shape of k new: {k_new.shape}")
    print(f"dk1: {dk1}, dk: {dk}")
    # print(f'shape of x: {x.shape}')
    # print(f'max of x: {x[-1]}')
    print(
        f"n should be larger than 2 * k_nyq / dk, n is {n_fft}, 2 * k_nyq / dk is {2 * k_nyq / dk}"
    )
    # print(f'this number of points in the wavenumber domain is {n}')

    # Interpolate the FFT to equally spaced k values
    # print(f'shape of k: {k.shape}, k_new: {k_new.shape}, freq_vel: {freq_vel.shape}')
    plt.plot(k_new, label="k_new")
    plt.plot(k, label="k")
    # plt.xlabel('Wavenumber')
    # plt.ylabel('Frequency')
    plt.xlabel("sample")
    plt.title("k_new vs k")
    plt.legend()
    plt.show()
    # Interpolate G(w) to find G(k)
    G_interp = interpolate.interp1d(
        k, G_w, kind="linear", bounds_error=False, fill_value=0
    )(k_new)
    plt.plot(k_new, G_interp.real, label="interpolated G(k)")
    plt.plot(k, G_w.real, label="G(k)")
    plt.xlabel("Wavenumber")
    plt.ylabel("Amplitude")
    plt.title("Interpolated G(k)")
    plt.show()
    print("G_interp created")
    # Calculate the group velocity of the guided wave mode at the wavenumber points
    v_gr_interp = interpolate.interp1d(
        k, v_gr, kind="linear", bounds_error=False, fill_value=0
    )(k_new)

    print(f"shape of G_interp: {G_interp.shape}")
    print(f"shape of v_gr_interp: {v_gr_interp.shape}")

    # Compute H(k) = G(k) * vgr(k)
    H_k = G_interp * v_gr_interp

    # Apply inverse FFT to H(k) to obtain the dispersion compensated distance-trace
    h_x = np.fft.ifft(H_k)
    print(f"shape of h_x: {h_x.shape} before removing zero-padding")
    # Remove zero-padding from the compensated signal
    h_x_padd = h_x
    h_x = h_x[:m]
    # x = np.arange(len(h_x))*dx
    # normalize h_x and signal
    h_x = h_x / np.max(h_x)
    signal = signal / np.max(signal)
    xmax = 1 / (dk) - dx
    x = np.arange(0, xmax, dx)
    print(f"shape of h_x: {h_x.shape}")
    print(f"shape of x: {x.shape}")
    print(f"max of x: {x[-1]}")
    print(f"shape of signal: {signal.shape}")
    # Create a subplot with the dispersion compensated signal and the original signal
    # plotting the results
    plt.subplot(2, 1, 1)
    plt.plot(x[:m], h_x.real, label="Dispersion compensated signal")
    # plt.plot(h_x.real, label='Dispersion compensated signal')
    plt.xlabel("distance [m]")
    # plt.xlabel('sample')
    plt.ylabel("Amplitude")
    plt.title("Dispersion compensated signal")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time_axis_top, signal, label="Original signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Original signal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # G_k = np.fft.fft(h_x)
    # g_disp = np.real(np.fft.ifft(G_k))
    # plt.subplot(2, 1, 1)
    # plt.plot(time_axis_top, g_disp, label='Dispersion compensated signal')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Amplitude')
    # plt.title('Dispersion compensated signal')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.plot(time_axis_top,signal, label='Original signal')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Amplitude')
    # plt.title('Original signal')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plot fft of dispersion compensated signal and original signal with the same frequency axis and padded
    # G_k_padded = np.pad(h_x, (0, n_fft - m), mode='constant')
    # print(f'shape of G_k_padded: {G_k_padded.shape}')
    G_k_padded = np.fft.fft(h_x_padd)
    freq_gk = np.fft.fftfreq(h_x_padd.size, dt)
    G_k_padded = G_k_padded[freq_gk > 0]
    freq_gk = freq_gk[freq_gk > 0]
    plt.subplot(2, 1, 1)
    plt.plot(freq_gk, np.abs(G_k_padded), label="Dispersion compensated signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("Dispersion compensated signal")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(freq_vel, np.abs(G_w), label="Original signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("Original signal")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return h_x.real


def read_DC_files(file_n=1):
    """
    Reads the dispersion compensated files.

    Args:
        file_n (int): The file number.

    Returns:
        ndarray: The dispersion compensated distance-trace.
    """
    if file_n == 1:
        path_A = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\LDPE20_A_Lamb.xlsx"
        path_S = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\LDPE20_S_Lamb.xlsx"

    elif file_n == 2:
        path_A = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\LDPE_disp_fxthickness_A_Lamb.xlsx"
        path_S = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\LDPE_disp_fxthickness_S_Lamb.xlsx"
    elif file_n == 3:
        path_A = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\COMSOL_PLATE_20_A_Lamb.xlsx"
        path_S = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\COMSOL_PLATE_20_S_Lamb.xlsx"
    elif file_n == 4:
        path_A = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\COMSOL_PLATE_20_long_A_Lamb.xlsx"
        path_S = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\COMSOL_PLATE_20_long_S_Lamb.xlsx"
    elif file_n == 5:
        path_A = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\REAL_PLATE_20_long_A_Lamb.xlsx"
        path_S = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\REAL_PLATE_20_long_S_Lamb.xlsx"
    elif file_n == 6:
        path_A = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\REAL_PLATE_250_A_Lamb.xlsx"
        path_S = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\REAL_PLATE_250_S_Lamb.xlsx"
    dc_A0 = pd.read_excel(path_A)
    dc_S0 = pd.read_excel(path_S)
    return dc_A0, dc_S0


def get_wavelength_DC(plot=True):
    A0, S0 = read_DC_files()
    wavelength_A0 = A0["A0 Wavelength (mm)"]
    wavelength_S0 = S0["S0 Wavelength (mm)"]
    freq = A0["A0 f (kHz)"]
    if plot:
        plt.plot(freq, wavelength_A0, label="A0")
        plt.plot(freq, wavelength_S0, label="S0")
        plt.xlabel("Frequency [kHz]")
        plt.ylabel("Wavelength [mm]")
        plt.title("Wavelength of A0 and S0 mode")
        plt.legend()
        plt.show()
    return wavelength_A0, wavelength_S0


def get_velocity_at_freq(freq, meter_per_second=True):
    """
    This returns the group and phase velocity of the A0 and S0 mode at a given frequency.
    The velocities are in m/ms.

    """
    A0, S0 = read_DC_files(4)
    idx = (A0["A0 f (kHz)"] - freq).abs().idxmin()
    group_velocity_A0 = A0.loc[idx, "A0 Energy velocity (m/ms)"]
    phase_velocity_A0 = A0.loc[idx, "A0 Phase velocity (m/ms)"]
    group_velocity_S0 = S0.loc[idx, "S0 Energy velocity (m/ms)"]
    phase_velocity_S0 = S0.loc[idx, "S0 Phase velocity (m/ms)"]
    if meter_per_second:
        group_velocity_A0 = group_velocity_A0 * 1000
        phase_velocity_A0 = phase_velocity_A0 * 1000
        group_velocity_S0 = group_velocity_S0 * 1000
        phase_velocity_S0 = phase_velocity_S0 * 1000
    velocities = {
        "A0": {
            "group_velocity": group_velocity_A0,
            "phase_velocity": phase_velocity_A0,
        },
        "S0": {
            "group_velocity": group_velocity_S0,
            "phase_velocity": phase_velocity_S0,
        },
    }
    # print(velocities)
    return velocities


def find_signal_start(signal, threshold):
    """Find the index where signal exceeds a threshold."""
    return next((i for i, val in enumerate(signal) if abs(val) > threshold), None)


def all_calculate_phase_velocities(size=0.75, save=False, theoretical=True):
    # Constants
    plate_width = 0.35  # meters
    plate_height = 0.25  # meters
    plate_thickness = 0.007  # meters
    samplerate = 500000  # Hz
    num_samples = 501
    delta_i = plate_height / (num_samples - 1)
    freq = 35000  # Hz
    cut_length = 120
    threshold = 0.01  # Threshold for signal start. Adjust according to your data

    types = "A0_only"
    size_str = str(size).replace(".", "_")
    # Get data for all sensors
    wave_top, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(number=9)
    (
        wave_bottom,
        x_pos_bottom,
        y_pos_bottom,
        z_pos_bottom,
        time_axis_bottom,
    ) = get_comsol_data(number=10)

    # Initialize result lists
    phase_velocities_A0 = []
    phase_velocities_S0 = []
    num_of_sensors = 50
    step_size = 16
    # Loop over sensor pairs
    for i in range(
        7, num_of_sensors - step_size
    ):  # Using 7 as a start point and 99 as the end to consider the pairs correctly
        j = i + step_size
        distances = np.sqrt(
            (x_pos_bottom[j] - x_pos_bottom[i]) ** 2
            + (y_pos_bottom[j] - y_pos_bottom[i]) ** 2
        )
        # print(f"Sensor pair {i} and {j} with distance {distances} m")
        S0 = (wave_top - wave_bottom) / 2
        A0 = (wave_top + wave_bottom) / 2

        # Find start of the significant signal
        start_index_A0_i = find_signal_start(A0[i], threshold=threshold)
        start_index_A0_i_plus = find_signal_start(A0[j], threshold=threshold)
        start_index_S0_i = find_signal_start(S0[i], threshold=threshold)
        start_index_S0_i_plus = find_signal_start(S0[j], threshold=threshold)

        # Slice and process the signals
        A0_cut_i = A0[i][start_index_A0_i : start_index_A0_i + cut_length] * np.hamming(
            cut_length
        )
        A0_cut_i_plus = A0[j][
            start_index_A0_i_plus : start_index_A0_i_plus + cut_length
        ] * np.hamming(cut_length)
        S0_cut_i = S0[i][start_index_S0_i : start_index_S0_i + cut_length] * np.hamming(
            cut_length
        )
        S0_cut_i_plus = S0[j][
            start_index_S0_i_plus : start_index_S0_i_plus + cut_length
        ] * np.hamming(cut_length)

        # Pad cut signals to original length
        extra_padding = 8
        A0_padded_i = np.zeros(num_samples * extra_padding)
        A0_padded_i_plus = np.zeros(num_samples * extra_padding)
        S0_padded_i = np.zeros(num_samples * extra_padding)
        S0_padded_i_plus = np.zeros(num_samples * extra_padding)
        A0_padded_i[start_index_A0_i : start_index_A0_i + cut_length] = A0_cut_i
        A0_padded_i_plus[
            start_index_A0_i_plus : start_index_A0_i_plus + cut_length
        ] = A0_cut_i_plus
        S0_padded_i[start_index_S0_i : start_index_S0_i + cut_length] = S0_cut_i
        S0_padded_i_plus[
            start_index_S0_i_plus : start_index_S0_i_plus + cut_length
        ] = S0_cut_i_plus

        # Calculate phase difference and phase velocity for both modes
        phase_AO, freq_AO = wp.phase_difference_plot(
            A0_padded_i, A0_padded_i_plus, SAMPLE_RATE=samplerate, BANDWIDTH=[0, freq]
        )
        phase_SO, freq_SO = wp.phase_difference_plot(
            S0_padded_i, S0_padded_i_plus, SAMPLE_RATE=samplerate, BANDWIDTH=[0, freq]
        )
        # print(f"min freq_AO: {min(freq_AO)}")
        phase_velocity_A0 = wp.phase_velocity(phase_AO, freq_AO, distance=distances)
        phase_velocity_S0 = wp.phase_velocity(phase_SO, freq_SO, distance=distances)

        # Append the results
        # print(f"shape of phase_velocity_A0: {phase_velocity_A0.shape}")
        phase_velocities_A0.append(phase_velocity_A0)
        phase_velocities_S0.append(phase_velocity_S0)

    max_velocity = 2000  # Set this to a value that makes sense for your data
    # convert frequenncy from Hz to kHz
    freq_AO = freq_AO / 1000
    freq_SO = freq_SO / 1000
    # Filter out the phase velocities with spikes
    phase_velocities_A0_clean = [
        v
        for v in phase_velocities_A0
        if all(abs(velocity) <= max_velocity for velocity in v)
    ]
    phase_velocities_S0_clean = [
        v
        for v in phase_velocities_S0
        if all(abs(velocity) <= max_velocity for velocity in v)
    ]
    mean_velocities_A0 = np.mean(phase_velocities_A0_clean, axis=0)
    mean_velocities_S0 = np.mean(phase_velocities_S0_clean, axis=0)
    # Compute standard deviations
    std_velocities_A0 = np.std(phase_velocities_A0_clean, axis=0)
    std_velocities_S0 = np.std(phase_velocities_S0_clean, axis=0)

    fig, ax = figure_size_setup_thesis(size)
    ax.plot(freq_AO, mean_velocities_A0, label="A0 mean")
    ax.fill_between(
        freq_AO,
        mean_velocities_A0 - std_velocities_A0,
        mean_velocities_A0 + std_velocities_A0,
        color="b",
        alpha=0.1,
    )
    if theoretical:
        types = "theoretical"
        A0_t, S0_t = read_DC_files(3)
        A0_t_phase = A0_t["A0 Phase velocity (m/ms)"]
        A0_t_freq = A0_t["A0 f (kHz)"]
        S0_t_phase = S0_t["S0 Phase velocity (m/ms)"]
        S0_t_freq = S0_t["S0 f (kHz)"]
        # convert from m/ms to m/s
        A0_t_phase = A0_t_phase * 1000
        S0_t_phase = S0_t_phase * 1000
        # interpolate the theoretical values to the same frequency as the measured values
        ax.plot(A0_t_freq, A0_t_phase, label="A0 theoretical")

        # (
        #     phase_velocities_flexural,
        #     corrected_phase_velocities,
        #     phase_velocity_shear,
        #     material,
        # ) = wp.theoretical_velocities(freq_AO, material="LDPE_tonni20mm")
        # ax.plot(freq_AO, corrected_phase_velocities, label="A0 theoretical")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Phase velocity (m/s)")
    # ax.set_title("Mean phase velocity and Standard Deviation for A0 mode")
    ax.legend()
    if save:
        fig.savefig(
            f"phase_velocity_A0_{step_size}_{size_str}_{types}.png",
            dpi=300,
            # bbox_inches="tight",
        )
        fig.savefig(
            f"phase_velocity_A0_{step_size}_{size_str}_{types}.svg",
            # bbox_inches="tight",
            dpi=300,
        )
    plt.show()

    fig, ax = figure_size_setup_thesis(size)
    ax.plot(freq_SO, mean_velocities_S0, label="S0 mean")
    ax.fill_between(
        freq_SO,
        mean_velocities_S0 - std_velocities_S0,
        mean_velocities_S0 + std_velocities_S0,
        color="b",
        alpha=0.1,
    )
    if theoretical:
        ax.plot(S0_t_freq, S0_t_phase, label="S0 theoretical")

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Phase velocity (m/s)")
    # ax.set_title("Mean phase velocity and Standard Deviation for S0 mode")
    ax.legend()
    if save:
        fig.savefig(
            f"phase_velocity_S0_{step_size}_{size_str}_{types}.png",
            dpi=300,
            # bbox_inches="tight",
        )
        fig.savefig(
            f"phase_velocity_S0_{step_size}_{size_str}_{types}.svg",
            # bbox_inches="tight",
            dpi=300,
        )
    plt.show()
    return mean_velocities_A0, mean_velocities_S0, freq_AO


def velocites_modes():
    plate_width = 0.35  # meters
    plate_height = 0.25  # meters
    plate_thickness = 0.007  # meters
    samplerate = 500000  # Hz
    num_samples = 501
    delta_i = plate_height / (num_samples - 1)
    freq = 35000  # Hz

    # Generate example signal data
    wave_top, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(number=9)
    (
        wave_bottom,
        x_pos_bottom,
        y_pos_bottom,
        z_pos_bottom,
        time_axis_bottom,
    ) = get_comsol_data(number=10)
    wavelength = get_velocity_at_freq(freq=freq)["A0"]["phase_velocity"] / freq
    print(wavelength)
    distances = np.sqrt(
        (x_pos_bottom[19] - x_pos_bottom[10]) ** 2
        + (y_pos_bottom[19] - y_pos_bottom[10]) ** 2
    )
    print(distances)
    S0 = (wave_top - wave_bottom) / 2
    A0 = (wave_top + wave_bottom) / 2
    plt.plot(S0[19], label="S0_19")
    plt.plot(S0[10], label="S0_10")
    plt.legend()
    plt.show()
    plt.plot(A0[19], label="A0_19")
    plt.plot(A0[10], label="A0_10")
    plt.legend()
    plt.show()
    A0_19, A0_10 = A0[19], A0[10]
    S0_19, S0_10 = S0[19], S0[10]
    A0_10_cutted = A0_10[25:142]
    A0_19_cutted = A0_19[36:153]
    S0_10_cutted = S0_10[28:137]
    S0_19_cutted = S0_19[34:153]
    window_A0_10 = np.hamming(len(A0_10_cutted))
    window_A0_19 = np.hamming(len(A0_19_cutted))
    window_S0_10 = np.hamming(len(S0_10_cutted))
    window_S0_19 = np.hamming(len(S0_19_cutted))
    A0_10_cutted = A0_10_cutted * window_A0_10
    A0_19_cutted = A0_19_cutted * window_A0_19
    S0_10_cutted = S0_10_cutted * window_S0_10
    S0_19_cutted = S0_19_cutted * window_S0_19

    S0_cut_10 = np.zeros(len(S0_10))
    S0_cut_19 = np.zeros(len(S0_19))
    A0_cut_10 = np.zeros(len(A0_10))
    A0_cut_19 = np.zeros(len(A0_19))
    S0_cut_10[28:137] = S0_10_cutted
    S0_cut_19[34:153] = S0_19_cutted
    A0_cut_10[25:142] = A0_10_cutted
    A0_cut_19[36:153] = A0_19_cutted

    S0_cut_10 = S0_10
    S0_cut_19 = S0_19
    A0_cut_10 = A0_10
    A0_cut_19 = A0_19
    # plot the cut
    plt.plot(S0_cut_19, label="S0_19")
    plt.plot(S0_cut_10, label="S0_10")
    plt.legend()
    plt.show()
    plt.plot(A0_cut_19, label="A0_19")
    plt.plot(A0_cut_10, label="A0_10")
    plt.legend()
    plt.show()
    print(len(A0_cut_19), len(A0_cut_10), len(S0_cut_19), len(S0_cut_10))
    phase_AO, freq_AO = wp.phase_difference_plot(
        A0_cut_19,
        A0_cut_10,
        SAMPLE_RATE=501000,
        BANDWIDTH=[0, freq],
        title=f"A0 Phase difference, distance: {distances} ",
    )
    phase_SO, freq_SO = wp.phase_difference_plot(
        S0_cut_19,
        S0_cut_10,
        SAMPLE_RATE=501000,
        BANDWIDTH=[0, freq],
        title=f"S0 Phase difference, distance: {distances} ",
    )
    print(f"shape of freq_AO: {freq_AO.shape}")
    print(f"shape of freq_SO: {freq_SO.shape}")
    phase_velocity_A0 = wp.phase_velocity(
        phase_AO,
        freq_AO,
        distance=distances,
        plot=True,
        title=f"A0 Phase velocity, distance: {distances} ",
    )
    phase_velocity_S0 = wp.phase_velocity(
        phase_SO,
        freq_SO,
        distance=distances,
        plot=True,
        title=f"S0 Phase velocity, distance: {distances} ",
    )


def wave_number_to_omega():
    plate_width = 0.35  # meters
    plate_height = 0.25  # meters
    plate_thickness = 0.007  # meters
    samplerate = 501000  # Hz
    num_samples = 501
    delta_i = plate_height / (num_samples - 1)

    # Generate example signal data
    wave_top, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(number=9)
    (
        wave_bottom,
        x_pos_bottom,
        y_pos_bottom,
        z_pos_bottom,
        time_axis_bottom,
    ) = get_comsol_data(number=10)

    # Calculate S0 and A0 modes
    S0 = (wave_top - wave_bottom) / 2
    A0 = (wave_top + wave_bottom) / 2

    # Perform 2D FFT
    S0_fft = np.fft.fft2(S0)
    A0_fft = np.fft.fft2(A0)

    # Obtain frequency and wavenumber information
    sampling_period = 1 / samplerate
    omega = (2 * np.pi / sampling_period) * np.arange(num_samples)
    k_expected = (
        omega / v_s
    )  # Expected wavenumber using phase velocity (v_s) of S0 mode

    # Calculate wavenumbers from 2D FFT
    k_fft = np.fft.fftshift(np.fft.fftfreq(num_samples, delta_i)) * 2 * np.pi

    # Plot Omega to K (Wavenumber) relationship for S0 mode
    plt.plot(omega, k_expected, label="Expected")
    plt.plot(omega, k_fft, label="FFT")
    plt.xlabel("Angular Frequency (omega)")
    plt.ylabel("Wavenumber (k)")
    plt.title("Omega to K (Wavenumber) Plot - S0 Mode")
    plt.legend()
    plt.grid(True)
    plt.show()


def warping_map():
    # Given data points
    fs = 501000
    min_freq = 0
    max_freq = 30000
    x_line_data = [2, 8]
    positions = [12, 30]
    file_number = 2
    data, x_pos, y_pos, z_pos, time_axis = get_comsol_data(file_number)

    phase, freq = wp.phase_difference_div_improved(
        data[positions[0]], data[positions[1]], fs, pos_only=True, n_pi=-1
    )
    # phase = wp.phase_difference_div(data[positions[0]], data[positions[1]], pos_only=True)
    # freq = np.fft.fftfreq(data[positions[0]].size, 1/fs)
    slices = (freq > min_freq) & (freq < max_freq)
    # freq = freq[slices]
    # phase = phase[slices]
    plt.plot(freq, phase)
    plt.xlabel("Frekvens (Hz)")
    plt.ylabel("Faseforskyvning (rad)")
    plt.show()
    if file_number in x_line_data:
        distance = y_pos[positions[1]] - y_pos[positions[0]]
        print("x_line_data")
        print(f"distance is {distance}")
    else:
        distance = x_pos[positions[1]] - x_pos[positions[0]]
        print("y_line_data")
        print(f"distance is {distance}")
    phase_vel = wp.phase_velocity(phase, freq, distance, plot=True)
    (
        phase_velocities_flexural,
        corrected_phase_velocities,
        phase_velocity_shear,
        material,
    ) = wp.theoretical_velocities(freq, material="LDPE_tonni7mm")
    vg_theoretical = wp.group_velocity_theoretical(freq, material=material)
    vg = wp.group_velocity_phase(phase_vel, freq, distance)
    wavenumber_measured = 2 * np.pi * freq / phase_vel
    wavenumber_theoretical = 2 * np.pi * freq / corrected_phase_velocities
    # Define the normalization parameter
    K_measured = 0.5 * (1 / (np.max(freq) * vg))
    K_theoretical = 0.5 * (1 / (np.max(freq) * vg_theoretical))
    w_inv_measured = freq / phase_vel
    w_inv_theoretical = freq / corrected_phase_velocities
    plt.plot(freq, phase_vel, label="Measured")
    plt.plot(freq, corrected_phase_velocities, label="Theoretical")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase Velocity (m/s)")
    plt.title("Phase Velocity (Unknown Wavenumbers)")
    plt.grid(True)
    plt.legend()
    plt.show()
    warping_map_measured = K_measured * w_inv_measured
    warping_map_theoretical = K_theoretical * w_inv_theoretical
    plt.plot(warping_map_measured, freq, label="Measured")
    plt.plot(warping_map_theoretical, freq, label="Theoretical")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Warping Map")
    plt.title("Warping Map (Unknown Wavenumbers)")
    plt.grid(True)
    plt.legend()
    plt.show()


def get_comsol_data(number=2):
    if number == 1:
        path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\Disp_on_plate_top_case3_15kHz_pulse.txt"  # y=0.2 this not acceleration data
        data_nodes = 200
    elif number == 2:
        path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\az_on_plate_top_case3_15kHz_pulse_line3.txt"  # x=0.075, starting in the middle of the source
        data_nodes = 50
    elif number == 3:
        path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\az_on_plate_top_case3_15kHz_pulse_line2.txt"  # y=0.1
        data_nodes = 50
    elif number == 4:
        path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\az_on_plate_top_case3_15kHz_pulse_line1.txt"  # y=0.2
        data_nodes = 50
    elif number == 5:
        path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_bottom_LDPE20mm_15kHz_pulse_line1.txt"  # y=0.2
        data_nodes = 50
    elif number == 6:
        path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_top_LDPE20mm_15kHz_pulse_line1.txt"  # y=0.2
        data_nodes = 50
    elif number == 7:
        path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_top_LDPE20mm_15kHz_pulse_line2.txt"  # y=0.1
        data_nodes = 50
    elif number == 8:
        path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_top_LDPE20mm_15kHz_pulse_line3.txt"  # x=0.075
        data_nodes = 50
    elif number == 9:
        path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_top_LDPE20mm_15kHz_pulse_diagonal.txt"  # diagonal
        data_nodes = 100
    elif number == 10:
        path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_bottom_LDPE20mm_15kHz_pulse_diagonal.txt"  # diagonal
        data_nodes = 100
    elif number == 11:
        path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_top_LDPE20mm_3kHz_pulse_diagonal.txt"  # diagonal
        data_nodes = 100
    elif number == 12:
        path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_bottom_LDPE20mm_3kHz_pulse_diagonal.txt"  # diagonal
        data_nodes = 100

    with open(path, "r") as f1:
        i = 0
        if number == 11 or number == 12:
            time_axis = np.linspace(0, 3000e-6, num=1501)
        else:
            time_axis = np.linspace(0, 1000e-6, num=501)
        x_pos = np.zeros(data_nodes)
        y_pos = np.zeros(data_nodes)
        z_pos = np.zeros(data_nodes)
        if number == 11 or number == 12:
            wave_data = np.zeros((data_nodes, 1501))
        else:
            wave_data = np.zeros((data_nodes, 501))

        for idx, line in enumerate(f1):
            tmp = line.split()
            if tmp[0] != "%":
                wave_data[i] = tmp[3:]
                x_pos[i] = float(tmp[0])
                y_pos[i] = float(tmp[1])
                z_pos[i] = float(tmp[2])
                i += 1
    return wave_data, x_pos, y_pos, z_pos, time_axis


def touch_signal_plot_hold():
    sample_rate = 150e3
    save = True
    size = 0.45
    size_name = "045"
    time_type = "new"
    folder = "plate20mm\\setup1_vegard\\touch"
    file_name = "touch_hold_v1"
    file_name2 = "touch_hold_v2"
    file_name3 = "touch_hold_v3"

    # fig1, ax1 = figure_size_setup(0.45)
    # data = csv_to_df_thesis(folder, file_name)
    # data2 = csv_to_df_thesis(folder, file_name2)
    data3 = csv_to_df_thesis(folder, file_name3)

    time_axis = np.linspace(0, len(data3) // sample_rate, num=len(data3))
    # drop wave_gen channel
    # data = data.drop(columns=["channel 3", "channel 2", "wave_gen"])
    # data2 = data2.drop(columns=["channel 3", "channel 2", "wave_gen"])
    data3 = data3.drop(columns=["channel 3", "channel 2", "wave_gen"])
    # spectromgram_touch(data, f"spectrogram_touch_hold_v1_fullsignal_{size_name}", size)
    # spectromgram_touch(data2, f"spectrogram_touch_hold_v2_fullsignal_{size_name}", size)
    spectromgram_touch(
        data3, f"spectrogram_touch_hold_v3_fullsignal_{size_name}", size, save=save
    )

    fig, ax = figure_size_setup_thesis(size)
    data_long3 = data3.iloc[int(1.7868e5) : int(1.9359e5)]
    if time_type == "regular":
        time_axis3 = time_axis[int(1.7868e5) : int(1.9359e5)]
    else:
        time_axis3 = np.linspace(0, len(data_long3) / sample_rate, num=len(data_long3))
    # ax.plot(time_axis3, data_long3, label=data3.columns)
    ax.plot(time_axis3, data_long3, label="channel 1")
    # ax.plot(data3, label='v3')
    # ax.title('v3')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Accleration ($\mathrm{m/s^2}$)")
    ax.legend()
    if save:
        fig.savefig(
            f"entire_signal_v3_touch_hold_ch1_{time_type}time_{size_name}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"entire_signal_v3_touch_hold_ch1_{time_type}time_{size_name}.svg",
            dpi=300,
            format="svg",
        )
    plt.show()
    spectromgram_touch(
        data_long3,
        f"spectrogram_entire_signal_v3_touch_hold_ch1_regtime_{size_name}",
        size,
        save=save,
    )

    fig, ax = figure_size_setup_thesis(size)
    data_short3 = data3.iloc[int(179193) : int(180280)]
    if time_type == "regular":
        time_axis3 = time_axis[int(179193) : int(180280)]
    else:
        time_axis3 = np.linspace(
            0, len(data_short3) / sample_rate, num=len(data_short3)
        )
    # ax.plot(time_axis3, data_short3, label=data3.columns)
    ax.plot(time_axis3, data_short3, label="channel 1")
    # ax.plot(data3, label='v3')
    # ax.title('v3')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Accleration ($\mathrm{m/s^2}$)")
    ax.legend()
    if save:
        fig.savefig(
            f"start_signal_v3_touch_hold_ch1_{time_type}time_{size_name}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"start_signal_v3_touch_hold_ch1_{time_type}time_{size_name}.svg",
            dpi=300,
            format="svg",
        )

    plt.show()
    spectromgram_touch(
        data_short3,
        f"spectrogram_start_signal_v3_touch_hold_ch1_{time_type}time_{size_name}",
        size,
        save=save,
    )


def touch_signal_plot():
    save = True
    sample_rate = 150e3
    size = 0.45
    size_name = "045"
    time_type = "new"
    folder = "plate20mm\\setup3_vegard\\touch"
    # file_name = "touch_v1"
    file_name2 = "touch_hold_ca2_4s_v1"
    # file_name3 = "touch_v5"

    # fig1, ax1 = figure_size_setup(0.45)
    # data = csv_to_df_thesis(folder, file_name)
    data2 = csv_to_df_thesis(folder, file_name2)
    # data3 = csv_to_df_thesis(folder, file_name3)

    time_axis = np.linspace(0, len(data2) // sample_rate, num=len(data2))
    # drop wave_gen channel
    # data = data.drop(columns=["channel 3", "channel 2", "wave_gen"])
    data2 = data2.drop(columns=["wave_gen"])

    fig, ax = figure_size_setup_thesis(size)

    # ax.plot(time_axis, data2, label="channel 1")'
    ax.plot(time_axis, data2, label=data2.columns)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Accleration ($\mathrm{m/s^2}$)")
    ax.legend()
    if save:
        fig.savefig(
            f"touchholdsetup3_signal_ca24v1_fullsignal_{size_name}_newsize.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"touchholdsetup3_signal_ca24v1_fullsignal_{size_name}_newsize.svg",
            dpi=300,
            format="svg",
        )
    plt.show()
    channel = data2["channel 3"]
    data_fft = scipy.fft.fft(channel.values, axis=0)
    data_fft_dB = 20 * np.log10(np.abs(data_fft))
    fftfreq = scipy.fft.fftfreq(len(data_fft_dB), 1 / sample_rate)
    data_fft_dB = np.fft.fftshift(data_fft_dB)[len(channel) // 2 :]
    fftfreq = np.fft.fftshift(fftfreq)[len(channel) // 2 :]
    fig, ax = figure_size_setup_thesis(size)
    # only use the positive frequencies and plot in db
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Amplitude [dB]")

    ax.set_xlim(left=0, right=10000 / 1000)
    ax.set_ylim(bottom=-25, top=80)
    ax.plot(fftfreq / 1000, data_fft_dB)
    if save:
        fig.savefig(
            f"touchholdsetup3_signal_ca24v1_fullsignal_fftch3_{size_name}_newsize.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"touchholdsetup3_signal_ca24v1_fullsignal_fftch3_{size_name}_newsize.svg",
            dpi=300,
            format="svg",
        )
    plt.show()
    exit()
    spectromgram_touch(
        data2,
        f"spectrogram_touchholdsetup3_signal_ca24v1_fullsignal_{size_name}_newsize",
        size,
        save=save,
    )

    fig, ax = figure_size_setup_thesis(size)
    data_long2 = data2.iloc[int(1.8744e5) : int(2.0358e5)]
    if time_type == "regular":
        time_axis2 = time_axis[int(1.8744e5) : int(2.0358e5)]
    else:
        time_axis2 = np.linspace(0, len(data_long2) / sample_rate, num=len(data_long2))
    # ax.plot(time_axis2, data_long2, label=data2.columns)
    ax.plot(time_axis2, data_long2, label="channel 1")
    # ax.plot(data2, label='v2')
    # plt.title('v2')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Accleration ($\mathrm{m/s^2}$)")
    ax.legend()
    if save:
        fig.savefig(
            f"entire_signal_v4_touch_ch1_{time_type}time_{size_name}_newsize.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"entire_signal_v4_touch_ch1_{time_type}time_{size_name}_newsize.svg",
            dpi=300,
            format="svg",
        )

    plt.show()
    spectromgram_touch(
        data_long2,
        f"spectrogram_entire_signal_v4_touch_ch1_{time_type}time_{size_name}_newsize",
        size,
        save=save,
    )

    fig, ax = figure_size_setup_thesis(size)
    data_short2 = data2.iloc[int(188530) : int(189575)]
    if time_type == "regular":
        time_axis2 = time_axis[int(188530) : int(189575)]
    else:
        time_axis2 = np.linspace(
            0, len(data_short2) / sample_rate, num=len(data_short2)
        )
    # ax.plot(time_axis2, data_short2, label=data2.columns)
    ax.plot(time_axis2, data_short2, label="channel 1")
    # ax.plot(data2, label='v2')
    # plt.title('v2')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Accleration ($\mathrm{m/s^2}$)")
    ax.legend()
    if save:
        fig.savefig(
            f"start_signal_v4_touch_ch1_{time_type}time_{size_name}_newsize.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"start_signal_v4_touch_ch1_{time_type}time_{size_name}_newsize.svg",
            dpi=300,
            format="svg",
        )

    plt.show()
    spectromgram_touch(
        data_short2,
        f"spectrogram_start_signal_v4_touch_ch1_{time_type}time_{size_name}_newsize",
        size,
        save=save,
    )


def chirp_signal():
    sample_rate = 150e3
    save = True
    size = 0.45
    size_name = "045"
    time_type = "new"
    folder = "plate20mm\\setup1_vegard\\chirp"
    file_name = "chirp_v1"
    file_name2 = "chirp_v2"
    file_name3 = "chirp_v3"
    data = csv_to_df_thesis(folder, file_name)
    time_axis = np.linspace(0, len(data) // sample_rate, num=len(data))
    # store wave_gen channel
    wave_gen = data["wave_gen"]
    # drop wave_gen channel
    data = data.drop(columns=["channel 3", "channel 2", "wave_gen"])
    fig, ax = figure_size_setup_thesis(size)
    ax.plot(time_axis, data, label="channel 1")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Accleration ($\mathrm{m/s^2}$)")
    ax.legend()
    if save:
        fig.savefig(
            f"chirp_signal_v1_fullsignal_{size_name}_newsize.png", dpi=300, format="png"
        )
        fig.savefig(
            f"chirp_signal_v1_fullsignal_{size_name}_newsize.svg", dpi=300, format="svg"
        )
    plt.show()
    # plot the fft of the chirp signal
    print()
    N = len(data)
    # fft = np.fft.fft(np.squeeze(data.values))
    # spectrum_db = 20 * np.log10(np.abs(fft[0 : N // 2 + 1]))
    # freq = np.fft.fftfreq(N, 1 / sample_rate)[0 : N // 2 + 1]
    channel = data["channel 1"]
    data_fft = scipy.fft.fft(channel.values, axis=0)
    data_fft_dB = 20 * np.log10(np.abs(data_fft))
    fftfreq = scipy.fft.fftfreq(len(data_fft_dB), 1 / sample_rate)
    data_fft_dB = np.fft.fftshift(data_fft_dB)[len(channel) // 2 :]
    fftfreq = np.fft.fftshift(fftfreq)[len(channel) // 2 :]
    fig, ax = figure_size_setup_thesis(size)
    # only use the positive frequencies and plot in db
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Amplitude [dB]")

    ax.set_xlim(left=0, right=50000 / 1000)
    ax.set_ylim(bottom=-25, top=80)
    ax.plot(fftfreq / 1000, data_fft_dB)

    if save:
        fig.savefig(
            f"chirp_signal_v1_fullsignal_fft_{size_name}_newsize.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"chirp_signal_v1_fullsignal_fft_{size_name}_newsize.svg",
            dpi=300,
            format="svg",
        )
    plt.show()


def swipe_signal():
    sample_rate = 150e3
    save = True
    size = 0.75
    size_name = "075"
    time_type = "new"
    folder = "plate20mm\\setup1_vegard\\swipe"
    file_name = "left_right_swipe_v1"
    file_name2 = "touch_hold_v2"
    file_name3 = "right_left_swipe_hold_v1"

    # fig1, ax1 = figure_size_setup(0.45)
    # data = csv_to_df_thesis(folder, file_name)
    # data2 = csv_to_df_thesis(folder, file_name2)
    data3 = csv_to_df_thesis(folder, file_name3)

    time_axis = np.linspace(0, len(data3) // sample_rate, num=len(data3))
    # drop wave_gen channel
    # data = data.drop(columns=["channel 3", "channel 2", "wave_gen"])
    # data2 = data2.drop(columns=["channel 3", "channel 2", "wave_gen"])
    data3 = data3.drop(columns=["channel 3", "channel 2", "wave_gen"])
    fig, ax = figure_size_setup_thesis(size)
    ax.plot(time_axis, data3, label="channel 1")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Accleration ($\mathrm{m/s^2}$)")
    ax.legend()
    if save:
        fig.savefig(
            f"entire_signal_v1_swipe_holdrl_ch1_{time_type}time_{size_name}_newsize.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"entire_signal_v1_swipe_holdrl_ch1_{time_type}time_{size_name}_newsize.svg",
            dpi=300,
            format="svg",
        )
    plt.show()
    exit()
    # spectromgram_touch(data, f"spectrogram_touch_hold_v1_fullsignal_{size_name}", size)
    # spectromgram_touch(data2, f"spectrogram_touch_hold_v2_fullsignal_{size_name}", size)
    spectromgram_touch(
        data3, f"spectrogram_touch_hold_v3_fullsignal_{size_name}", size, save=save
    )

    fig, ax = figure_size_setup_thesis(size)
    data_long3 = data3.iloc[int(1.7868e5) : int(1.9359e5)]
    if time_type == "regular":
        time_axis3 = time_axis[int(1.7868e5) : int(1.9359e5)]
    else:
        time_axis3 = np.linspace(0, len(data_long3) / sample_rate, num=len(data_long3))
    # ax.plot(time_axis3, data_long3, label=data3.columns)
    ax.plot(time_axis3, data_long3, label="channel 1")
    # ax.plot(data3, label='v3')
    # ax.title('v3')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Accleration ($\mathrm{m/s^2}$)")
    ax.legend()
    if save:
        fig.savefig(
            f"entire_signal_v3_touch_hold_ch1_{time_type}time_{size_name}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"entire_signal_v3_touch_hold_ch1_{time_type}time_{size_name}.svg",
            dpi=300,
            format="svg",
        )
    plt.show()
    spectromgram_touch(
        data_long3,
        f"spectrogram_entire_signal_v3_touch_hold_ch1_regtime_{size_name}",
        size,
        save=save,
    )

    fig, ax = figure_size_setup_thesis(size)
    data_short3 = data3.iloc[int(179193) : int(180280)]
    if time_type == "regular":
        time_axis3 = time_axis[int(179193) : int(180280)]
    else:
        time_axis3 = np.linspace(
            0, len(data_short3) / sample_rate, num=len(data_short3)
        )
    # ax.plot(time_axis3, data_short3, label=data3.columns)
    ax.plot(time_axis3, data_short3, label="channel 1")
    # ax.plot(data3, label='v3')
    # ax.title('v3')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Accleration ($\mathrm{m/s^2}$)")
    ax.legend()
    if save:
        fig.savefig(
            f"start_signal_v3_touch_hold_ch1_{time_type}time_{size_name}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"start_signal_v3_touch_hold_ch1_{time_type}time_{size_name}.svg",
            dpi=300,
            format="svg",
        )

    plt.show()
    spectromgram_touch(
        data_short3,
        f"spectrogram_start_signal_v3_touch_hold_ch1_{time_type}time_{size_name}",
        size,
        save=save,
    )


def spectromgram_touch(channel, name, size, save=False):
    sample_rate = 150e3
    nfft = 64
    xextent = (-len(channel) / sample_rate, len(channel) / sample_rate)
    size = size
    # size_name = '075'
    fig, ax = figure_size_setup_thesis(size)
    print(channel.shape)
    spec = ax.specgram(
        np.squeeze(channel),
        Fs=sample_rate,
        NFFT=nfft,
        noverlap=(nfft // 2),
        cmap="viridis",
        window=np.hanning(nfft),
        # xextent=xextent
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    # limit frequency
    ax.set_ylim(0, 50000)
    fig.colorbar(spec[3], ax=ax)
    if save:
        fig.savefig(f"{name}.png", dpi=300, format="png")
        fig.savefig(f"{name}.svg", dpi=300, format="svg")
    plt.show()


def comsol_data200():
    cmap = "plasma"
    path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\Disp_on_plate_top_case3_15kHz_pulse.txt"
    with open(path, "r") as f1:
        i = 0
        time_axis = np.linspace(0, 1000, num=501)
        x_pos = np.zeros(200)
        y_pos = np.zeros(200)
        z_pos = np.zeros(200)
        wave_data = np.zeros((200, 501))
        xcorr_scale = 0.345 / 200
        for idx, line in enumerate(f1):
            tmp = line.split()
            if tmp[0] != "%":
                x_index = int(
                    float(tmp[0]) / 0.00173
                )  # convert x coordinate to index. comes from length of plate divided by number of points
                wave_data[x_index] = tmp[3:]
                x_pos[i] = float(tmp[0])
                y_pos[i] = float(tmp[1])
                z_pos[i] = float(tmp[2])
                i += 1

    # Plot the data
    # plt.plot(wave_data[3])
    # plt.show()
    # plt.plot(wave_data[199])
    # plt.show()
    plt.imshow(wave_data, aspect="auto", cmap=cmap, extent=[0, 0.345, 0, 501])
    plt.colorbar()
    plt.title("Wave Through Plate at y=200mm")
    plt.ylabel("Time (samples)")
    plt.xlabel("Position (mm)")
    plt.show()

    # filter data
    # wave_3_filtered = filter_general(wave_data[3], 'highpass', 30000)
    # wave_100_filtered = filter_general(wave_data[100], 'highpass', 30000)
    # plt.plot(wave_3_filtered, label='Filtered data')
    # plt.plot(wave_data[3], label='Original data')
    # plt.title(label='Filtered data')
    # plt.legend()
    # plt.show()
    # plt.plot(wave_100_filtered, label='Filtered data')
    # plt.plot(wave_data[100], label='Original data')
    # plt.title(label='Filtered data')
    # plt.legend()
    # plt.show()

    # Create a 3D figure
    x_index = np.linspace(0, 0.345, num=200)

    # Create meshgrid for x and t
    x, t = np.meshgrid(x_index, np.arange(501))  # Swap x and t
    wave_data = wave_data.T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, t, wave_data, cmap=cmap)
    # Set the axis labels and title
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Time (samples)")
    ax.set_zlabel("Amplitude")
    ax.set_title("Wave Through Plate")

    # Set the x and y axis limits
    ax.set_xlim([x_index.min(), x_index.max()])
    ax.set_ylim([0, 500])

    # Set the colorbar
    fig.colorbar(surf)

    # Show the plot
    plt.show()


def comsol_wave_prop_all(save=False, size=0.75):
    cmap = "viridis"
    top_or_bottom = ["bot", "top", "top", "top", "top", "bot"]
    details = ["y_02", "y_02", "y_01", "x_075", "diagonal", "diagonal"]
    details_true = ["y=0.2", "y=0.2", "y=0.1", "x=0.75", "diagonal", "diagonal"]
    size_name = str(size).replace(".", "")
    for i in range(5, 11):
        wave_data, x_pos, y_pos, z_pos, time_axis = get_comsol_data(i)
        fig, axs = figure_size_setup_thesis(size)
        if details[i - 5] in ["x_075", "diagonal"]:
            # need to skip the 5 first sensors as they are in the source
            print(wave_data.shape)
            wave_data = wave_data[5:]
            print(wave_data.shape)
        min_val = np.amin(wave_data)
        max_val = np.amax(wave_data)
        print(f"min val is: {min_val}")
        print(f"max val is: {max_val}")
        if details[i - 5] == "x_075":
            im = axs.imshow(
                wave_data.T,
                aspect="auto",
                cmap=cmap,
                extent=[y_pos[5], y_pos[-1], 1000, 0],
                vmin=min_val,
                vmax=max_val,
            )
        else:
            im = axs.imshow(
                wave_data.T,
                aspect="auto",
                cmap=cmap,
                extent=[x_pos[0], x_pos[-1], 1000, 0],
                vmin=min_val,
                vmax=max_val,
            )
        cbar = plt.colorbar(im)
        cbar.set_label(r"Accleration ($\mathrm{m/s^2}$)")
        if not save:
            axs.set_title(
                f"Wave Through Plate at {top_or_bottom[i-5]} {details_true[i-5]}"
            )
        axs.set_ylabel(r"Time ($\mathrm{\mu s}}$)")
        axs.set_xlabel("Position (m)")
        # save figure as png and svg
        if save:
            fig.savefig(
                f"wave_prop_{top_or_bottom[i-5]}_{details[i-5]}_20mm_virdis_{size_name}_notitle.png",
                dpi=300,
                format="png",
            )
            fig.savefig(
                f"wave_prop_{top_or_bottom[i-5]}_{details[i-5]}_20mm_virdis_{size_name}_notitle.svg",
                dpi=300,
                format="svg",
            )
        plt.show()


def comsol_data50(save=False):
    path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_top_LDPE20mm_15kHz_pulse_line2.txt"
    cmap = "viridis"
    number_of_nodes = 50
    with open(path, "r") as f1:
        i = 0
        time_axis = np.linspace(0, 1000, num=501)
        x_pos = np.zeros(number_of_nodes)
        y_pos = np.zeros(number_of_nodes)
        z_pos = np.zeros(number_of_nodes)
        wave_data = np.zeros((number_of_nodes, 501))
        xcorr_scale = 0.345 / number_of_nodes
        print(f"xcorr scale is: {xcorr_scale}")
        for idx, line in enumerate(f1):
            tmp = line.split()
            if tmp[0] != "%":
                # x_index = int(float(tmp[0]) / 0.007)# convert x coordinate to index. Had to hardcode the values as the last value gives 50 as index and not 49
                wave_data[i] = tmp[3:]
                x_pos[i] = float(tmp[0])
                y_pos[i] = float(tmp[1])
                z_pos[i] = float(tmp[2])
                i += 1

    # Plot the data
    # plt.plot(wave_data[3])
    # plt.show()
    # plt.plot(wave_data[10])
    # plt.show()
    # for the x axis we need to ignore the first sensors as they are in the source

    # wave_data = wave_data[5:]
    fig, axs = figure_size_setup_thesis()
    min_val = np.amin(wave_data)
    max_val = np.amax(wave_data)
    print(f"min val is: {min_val}")
    print(f"max val is: {max_val}")

    im = axs.imshow(
        wave_data.T,
        aspect="auto",
        cmap=cmap,
        extent=[x_pos[0], x_pos[-1], 1000, 0],
        vmin=min_val,
        vmax=max_val,
    )
    cbar = plt.colorbar(im)
    cbar.set_label(r"Accleration ($\mathrm{m/s^2}$)")
    axs.set_title("Wave Through Plate at y=0.2m")
    axs.set_ylabel(r"Time ($\mathrm{\mu s}}$)")
    axs.set_xlabel("Position (m)")
    # save figure as png and svg
    if save:
        fig.savefig("wave_prop_top_y_02_20mm_virdis.png", dpi=300, format="png")
        fig.savefig("wave_prop_top_y_02_20mm_virdis.svg", dpi=300, format="svg")
    plt.show()

    # # Create a 3D figure
    # x_index = np.linspace(0, 0.345, num=number_of_nodes)

    # # Create meshgrid for x and t
    # x, t = np.meshgrid(x_index, np.arange(501))  # Swap x and t
    # wave_data = wave_data.T
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # surf = ax.plot_surface(x, t, wave_data, cmap=cmap)
    # # Set the axis labels and title
    # ax.set_xlabel("Position (mm)")
    # ax.set_ylabel("Time (samples)")
    # ax.set_zlabel("Amplitude")
    # ax.set_title("Wave Through Plate")

    # # Set the x and y axis limits
    # ax.set_xlim([x_index.min(), x_index.max()])
    # ax.set_ylim([0, 500])

    # # Set the colorbar
    # fig.colorbar(surf)

    # # Show the plot
    # plt.show()


def comsol_data200_phase_diff(idx1=1, idx2=10):
    path = r"C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\Disp_on_plate_top_case3_15kHz_pulse.txt"
    with open(path, "r") as f1:
        i = 0
        time_axis = np.linspace(0, 1000, num=501)
        x_pos = np.zeros(200)
        y_pos = np.zeros(200)
        z_pos = np.zeros(200)
        wave_data = np.zeros((200, 501))

        for idx, line in enumerate(f1):
            tmp = line.split()
            if tmp[0] != "%":
                x_index = int(float(tmp[0]) / 0.00173)  # convert x coordinate to index
                wave_data[x_index] = tmp[3:]
                x_pos[i] = float(tmp[0])
                y_pos[i] = float(tmp[1])
                z_pos[i] = float(tmp[2])
                i += 1
    phase = wp.phase_difference_div(wave_data[idx1], wave_data[idx2], pos_only=True)
    plt.plot(wave_data[idx1], label=f"x1={round(x_pos[idx1])}mm")
    plt.plot(wave_data[idx2], label=f"x2={round(x_pos[idx2])}mm")
    plt.legend()
    plt.show()
    freq = np.fft.fftfreq(len(wave_data[idx1]))
    freq = freq[freq > 0]
    plt.plot(
        freq, phase, label=f"x1={round(x_pos[idx1])}mm and x2={round(x_pos[idx2])}mm"
    )
    plt.legend()
    plt.show()
    phase_vel = wp.phase_velocity(
        freq, phase, distance=x_pos[idx2] - x_pos[idx1], plot=True
    )


def wave_type_plots():
    df = csv_to_df_thesis(
        "plate10mm\\setup2\\chirp", "chirp3_ch3top_ch2bot_ch1_sidemid_v1"
    )
    df_no_wave = df.drop(["wave_gen"], axis=1)
    time_axis = np.linspace(0, len(df) // 150e3, num=len(df))
    plt.plot(time_axis, df_no_wave)
    plt.show()

    # plotting just above and under the plate
    plt.plot(time_axis, df["channel 2"], label="channel 2")
    plt.plot(time_axis, df["channel 3"], label="channel 3")
    plt.legend()
    plt.show()

    # filter the signals
    df_filt = filter_general(
        df_no_wave,
        filtertype="highpass",
        cutoff_highpass=100,
        cutoff_lowpass=15000,
        order=4,
    )
    plt.plot(time_axis, df_filt)
    plt.show()

    # plotting just above and under the plate
    plt.plot(time_axis, df_filt["channel 2"], label="channel 2")
    plt.plot(time_axis, df_filt["channel 3"], label="channel 3")
    plt.legend()
    plt.show()

    # normalize channel 2 and 3 so that they are on the same scale
    df_filt["channel 2"] = df_filt["channel 2"] / np.max(df_filt["channel 2"])
    df_filt["channel 3"] = df_filt["channel 3"] / np.max(df_filt["channel 3"])
    # plot the difference between the two signals
    plt.plot(time_axis, df_filt["channel 2"] + df_filt["channel 3"], label="difference")
    # plt.plot(time_axis, df_filt['channel 2'], label='channel 2')
    # plt.plot(time_axis, df_filt['channel 3'], label='channel 3')
    plt.legend()
    plt.show()


def show_A0_and_S0_wave_comsol_y0_2(position):
    top_data, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(6)
    bot_data, x_pos_bot, y_pos_bot, z_pos_bot, time_axis_bot = get_comsol_data(5)
    plt.plot(time_axis_top, top_data[position], label="top raw data")
    plt.plot(time_axis_bot, bot_data[position], label="bottom raw data")
    plt.title(f"Raw data at x={x_pos_top[position]}mm")
    plt.legend()
    plt.show()
    # the way the sensor is read in comsol the S0 andd A0 is calculated opposite of what is done for real data
    S0 = (top_data[position] - bot_data[position]) / 2
    A0 = (top_data[position] + bot_data[position]) / 2
    plt.plot(time_axis_top, S0, label="S0")
    plt.plot(time_axis_top, top_data[position], label="top raw data")
    plt.title(label=f"S0 at x={x_pos_top[position]}mm")
    plt.legend()
    plt.show()
    plt.plot(time_axis_top, A0, label="A0")
    plt.plot(time_axis_top, top_data[position], label="top raw data")
    plt.title(label=f"A0 at x={x_pos_top[position]}mm")
    plt.legend()
    plt.show()
    plt.plot(time_axis_top, A0, label="A0")
    plt.plot(time_axis_top, S0, label="S0")
    plt.title(label=f"A0 and S0 at x={x_pos_top[position]}mm")
    plt.legend()
    plt.show()


def show_A0_and_S0_wave_comsol_diagonal(
    position, save=False, size=0.75, expected_arrival_time=False
):
    vline = "novline"
    size_name = str(size).replace(".", "_")
    top_data, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(9)
    bot_data, x_pos_bot, y_pos_bot, z_pos_bot, time_axis_bot = get_comsol_data(10)
    diag_dist = np.sqrt(
        (x_pos_top[position] - x_pos_top[0]) ** 2
        + (y_pos_top[position] - y_pos_top[0]) ** 2
    )
    time_axis_top = time_axis_top * 1e3
    time_axis_bot = time_axis_bot * 1e3
    fig, ax = figure_size_setup_thesis(size)
    ax.plot(time_axis_top, top_data[position], label="top raw data")
    ax.plot(time_axis_bot, bot_data[position], label="bottom raw data")
    if expected_arrival_time:
        vline = "vline"
        A0, S0 = read_DC_files()
        A0_vel_ph = A0["A0 Phase velocity (m/ms)"]
        S0_vel_ph = S0["S0 Phase velocity (m/ms)"]
        A0_vel_gr = A0["A0 Energy velocity (m/ms)"]
        S0_vel_gr = S0["S0 Energy velocity (m/ms)"]
        # find max velocity and convert from (m/ms) to (m/s)
        A0_vel_ph_max = np.max(A0_vel_ph) * 1000
        S0_vel_ph_max = np.max(S0_vel_ph) * 1000
        A0_vel_gr_max = np.max(A0_vel_gr) * 1000
        S0_vel_gr_max = np.max(S0_vel_gr) * 1000
        freq = A0["A0 f (kHz)"]
        # find expected arrival time for S0 and A0 using max group velocity
        A0_expected_arrival_time = (diag_dist / A0_vel_gr_max) * 1000
        S0_expected_arrival_time = (diag_dist / S0_vel_gr_max) * 1000

        # plot vline at this time in the plot
        ax.axvline(
            A0_expected_arrival_time,
            color="k",
            linestyle="--",
            label="A0 expected arrival time",
        )
        ax.axvline(
            S0_expected_arrival_time,
            color="k",
            linestyle=":",
            label="S0 expected arrival time",
        )

    # caluclate the diagonal distance
    print(f"distance from source to sensor: {diag_dist}")
    # ax.set_title(f"Raw data at x={round(x_pos_top[position],2)}m")
    ax.set_ylabel(r"Acceleration ($\mathrm{m/s^2}$)")
    ax.set_xlabel("Time [ms]")
    ax.legend()
    if save:
        # relace . with _ in x pos string and distance
        x_pos_string = str(round(x_pos_top[position], 2)).replace(".", "_")
        diag_dist_string = str(round(diag_dist, 2)).replace(".", "_")
        fig.savefig(
            f"top_bot_raw_data_x_{x_pos_string}_{diag_dist_string}_{size_name}_notitle_{vline}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"top_bot_raw_data_x_{x_pos_string}_{diag_dist_string}_{size_name}_notitle_{vline}.svg",
            dpi=300,
            format="svg",
        )
    plt.show()
    # the way the sensor is read in comsol the S0 andd A0 is calculated opposite of what is done for real data
    fig, ax = figure_size_setup_thesis(size)
    S0 = (top_data[position] - bot_data[position]) / 2
    A0 = (top_data[position] + bot_data[position]) / 2
    ax.plot(time_axis_top, S0, label="S0")
    ax.plot(time_axis_top, top_data[position], label="top raw data")
    ax.set_ylabel(r"Acceleration ($m/s^2$)")
    ax.set_xlabel("Time [ms]")
    if expected_arrival_time:
        ax.axvline(
            S0_expected_arrival_time,
            color="k",
            linestyle=":",
            label="S0 expected arrival time",
        )
    # ax.set_title(label=f"S0 at x={round(x_pos_top[position],2)}m")
    ax.legend()
    if save:
        fig.savefig(
            f"S0_x_{x_pos_string}_{diag_dist_string}_{size_name}_notitle_{vline}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"S0_x_{x_pos_string}_{diag_dist_string}_{size_name}_notitle_{vline}.svg",
            dpi=300,
            format="svg",
        )
    plt.show()
    fig, ax = figure_size_setup_thesis(size)
    ax.plot(time_axis_top, A0, label="A0")
    ax.plot(time_axis_top, top_data[position], label="top raw data")
    ax.set_ylabel(r"Acceleration ($m/s^2$)")
    ax.set_xlabel("Time [ms]")
    # ax.set_title(label=f"A0 at x={round(x_pos_top[position],2)}m")
    if expected_arrival_time:
        ax.axvline(
            A0_expected_arrival_time,
            color="k",
            linestyle="--",
            label="A0 expected arrival time",
        )
    ax.legend()
    if save:
        fig.savefig(
            f"A0_x_{x_pos_string}_{diag_dist_string}_{size_name}_notitle_{vline}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"A0_x_{x_pos_string}_{diag_dist_string}_{size_name}_notitle_{vline}.svg",
            dpi=300,
            format="svg",
        )
    plt.show()
    fig, ax = figure_size_setup_thesis(size)
    ax.plot(time_axis_top, A0, label="A0")
    ax.plot(time_axis_top, S0, label="S0")
    ax.set_ylabel(r"Acceleration ($m/s^2$)")
    ax.set_xlabel("Time [ms]")
    # ax.set_title(label=f"A0 and S0 at x={round(x_pos_top[position],2)}m")
    if expected_arrival_time:
        ax.axvline(
            A0_expected_arrival_time,
            color="k",
            linestyle="--",
            label="A0 expected arrival time",
        )
        ax.axvline(
            S0_expected_arrival_time,
            color="k",
            linestyle=":",
            label="S0 expected arrival time",
        )
    ax.legend()
    if save:
        fig.savefig(
            f"A0_S0_x_{x_pos_string}_{diag_dist_string}_{size_name}_notitle_{vline}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"A0_S0_x_{x_pos_string}_{diag_dist_string}_{size_name}_notitle_{vline}.svg",
            dpi=300,
            format="svg",
        )
    plt.show()
    # pad data before fft
    top_data_fft = np.fft.fft(top_data[position], n=2**18)
    bot_data_fft = np.fft.fft(bot_data[position], n=2**18)
    A0_fft = np.fft.fft(A0, n=2**18)
    S0_fft = np.fft.fft(S0, n=2**18)
    # get the frequency axis for the padded fft data
    freq_axis = np.fft.fftfreq(len(top_data_fft), d=1 / 500e3) / 1000
    # only look at the positive frequencies and fft data
    freq_axis = freq_axis[: len(freq_axis) // 2]
    top_data_fft = top_data_fft[: len(top_data_fft) // 2]
    bot_data_fft = bot_data_fft[: len(bot_data_fft) // 2]
    A0_fft = A0_fft[: len(A0_fft) // 2]
    S0_fft = S0_fft[: len(S0_fft) // 2]
    # plot the fft data in db scale
    fig, ax = figure_size_setup_thesis(size)
    ax.plot(freq_axis, 20 * np.log10(np.abs(top_data_fft)), label="top data")
    ax.plot(freq_axis, 20 * np.log10(np.abs(bot_data_fft)), label="bottom data")
    # limit the x axis to the range of interest
    ax.set_xlim([0, 40])
    ax.set_ylabel("Amplitude [dB]")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_title(label=f"FFT of raw data at x={round(x_pos_top[position],2)}m")
    ax.legend()
    if save:
        fig.savefig(
            f"fft_x_{x_pos_string}_{diag_dist_string}_{size_name}_notitle_{vline}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"fft_x_{x_pos_string}_{diag_dist_string}_{size_name}_notitle_{vline}.svg",
            dpi=300,
            format="svg",
        )
    plt.show()
    fig, ax = figure_size_setup_thesis(size)
    ax.plot(freq_axis, 20 * np.log10(np.abs(A0_fft)), label="A0")
    ax.plot(freq_axis, 20 * np.log10(np.abs(S0_fft)), label="S0")
    # limit the x axis to the range of interest
    ax.set_xlim([0, 40])
    ax.set_ylabel("Amplitude [dB]")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_title(label=f"FFT of A0 and S0 at x={round(x_pos_top[position],2)}m")
    ax.legend()
    if save:
        fig.savefig(
            f"fft_A0_S0_x_{x_pos_string}_{diag_dist_string}_{size_name}_notitle_{vline}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"fft_A0_S0_x_{x_pos_string}_{diag_dist_string}_{size_name}_notitle_{vline}.svg",
            dpi=300,
            format="svg",
        )
    plt.show()


def show_S0_wave(normalize=False, plot=True):
    df = csv_to_df_thesis(
        "plate10mm\\setup2\\chirp", "chirp3_ch3top_ch2bot_ch1_sidemid_v1"
    )
    df_no_wave = df.drop(["wave_gen"], axis=1)
    time_axis = np.linspace(0, len(df) // 150e3, num=len(df))
    if plot:
        plt.plot(time_axis, df_no_wave, labels=["channel 1", "channel 2", "channel 3"])
        plt.title("Raw data with all channels")
        plt.legend()
        plt.show()

        # plotting just above and under the plate
        plt.plot(time_axis, df["channel 2"], label="channel 2")
        plt.plot(time_axis, df["channel 3"], label="channel 3")
        plt.title("Channel 3 is the top plate and channel 2 is the bottom plate")
        plt.legend()
        plt.show()

    # filter the signals
    df_filt = filter_general(
        df_no_wave,
        filtertype="highpass",
        cutoff_highpass=100,
        cutoff_lowpass=15000,
        order=4,
    )

    if plot:
        # plotting just above and under the plate
        plt.plot(time_axis, df_filt["channel 2"], label="channel 2")
        plt.plot(time_axis, df_filt["channel 3"], label="channel 3")
        plt.title(
            label="Channel 3 is the top plate and channel 2 is the bottom plate, filtered"
        )
        plt.legend()
        plt.show()

    if normalize:
        # normalize channel 2 and 3 so that they are on the same scale
        df_filt["channel 2"] = df_filt["channel 2"] / np.max(df_filt["channel 2"])
        df_filt["channel 3"] = df_filt["channel 3"] / np.max(df_filt["channel 3"])
    S0 = df_filt["channel 2"] + df_filt["channel 3"] / 2
    if plot:
        plt.plot(time_axis, S0, label="S0")
        plt.title(label="S0")
        plt.legend()
        plt.show()
    return S0


def show_A0_wave(normalize=False, plot=True):
    df = csv_to_df_thesis(
        "plate10mm\\setup2\\chirp", "chirp3_ch3top_ch2bot_ch1_sidemid_v1"
    )
    df_no_wave = df.drop(["wave_gen"], axis=1)
    time_axis = np.linspace(0, len(df) // 150e3, num=len(df))
    if plot:
        plt.plot(time_axis, df_no_wave, labels=["channel 1", "channel 2", "channel 3"])
        plt.title("Raw data with all channels")
        plt.legend()
        plt.show()

        # plotting just above and under the plate
        plt.plot(time_axis, df["channel 2"], label="channel 2")
        plt.plot(time_axis, df["channel 3"], label="channel 3")
        plt.title("Channel 3 is the top plate and channel 2 is the bottom plate")
        plt.legend()
        plt.show()

    # filter the signals
    df_filt = filter_general(
        df_no_wave,
        filtertype="highpass",
        cutoff_highpass=100,
        cutoff_lowpass=15000,
        order=4,
    )

    if plot:
        # plotting just above and under the plate
        plt.plot(time_axis, df_filt["channel 2"], label="channel 2")
        plt.plot(time_axis, df_filt["channel 3"], label="channel 3")
        plt.title(
            label="Channel 3 is the top plate and channel 2 is the bottom plate, filtered"
        )
        plt.legend()
        plt.show()

    if normalize:
        # normalize channel 2 and 3 so that they are on the same scale
        df_filt["channel 2"] = df_filt["channel 2"] / np.max(df_filt["channel 2"])
        df_filt["channel 3"] = df_filt["channel 3"] / np.max(df_filt["channel 3"])
    A0 = df_filt["channel 3"] - df_filt["channel 2"] / 2
    if plot:
        plt.plot(time_axis, A0, label="A0")
        plt.title(label="A0")
        plt.legend()
        plt.show()
    return A0


def show_A0_and_S0_wave(normalize=False):
    S0 = show_S0_wave(normalize=normalize, plot=False)
    A0 = show_A0_wave(normalize=normalize, plot=False)
    time_axis = np.linspace(0, len(S0) // 150e3, num=len(S0))
    plt.plot(time_axis, S0, label="S0")
    plt.plot(time_axis, A0, label="A0")
    plt.title(label="S0 and A0")
    plt.legend()
    plt.show()


def velocities():
    df = csv_to_df_thesis("plate20mm\\setup1\\chirp", "chirp_100_40000_2s_v1")
    # frequency range from 100 to 40000 Hz samplerate of 150000 Hz
    freqs = np.linspace(100, 40000, 150000)

    # wp.plot_theoretical_velocities(freqs)
    phase10, freq10 = wp.phase_plotting_chirp(
        df,
        channels=["channel 2", "channel 3"],
        detrend=True,
        method="div",
        n_pi=0,
        SAMPLE_RATE=150000,
        BANDWIDTH=[100, 40000],
        save_fig=False,
        file_name="phase_difference.png",
        file_format="png",
        figsize=0.75,
        use_recorded_chirp=True,
    )
    wp.plot_velocities(
        phase10,
        freq10,
        0.10,
        savefig=False,
        filename="phase_velocity_10cm.svg",
        file_format="svg",
        material="HDPE",
    )


def pressure_wave_oscilloscope(size=0.75, save=False):
    data = csv_to_df_thesis("scope", "scope_2_1", scope=True)
    size_str = str(size).replace(".", "_")
    peaks, _ = signal.find_peaks(data["Volt"], height=0.05)
    data["second"] = data["second"] * 1e6
    time = data["second"][peaks]
    fig, ax = figure_size_setup_thesis(size)
    # convert time axis to microseconds
    time_unit = r"$\mathrm{\mu}$s"
    ax.plot(data["second"], data["Volt"])
    ax.plot(data["second"][peaks], data["Volt"][peaks], "x")
    ax.set_xlabel(r"Time ($\mathrm{\mu}$s)")
    ax.set_ylabel("Voltage (V)")
    # annotate the first peak with the time at that peak
    ax.annotate(
        f"{time.values[0]} {time_unit}", (time.values[0], data["Volt"][peaks[0]])
    )
    if save:
        fig.savefig(
            f"pressure_wave_oscilloscope_{size_str}.svg",
            format="svg",
            dpi=300,
            # bbox_inches="tight",
        )
        fig.savefig(
            f"pressure_wave_oscilloscope_{size_str}.png",
            format="png",
            dpi=300,
            # bbox_inches="tight",
        )
    plt.show()

    print(f"time is {time.values[0]} s")
    plate_wave_vel = (2 * 0.02) / (time.values[0] * 1e-6)
    print(f"plate wave velocity is {plate_wave_vel} m/s")


def comsol_pulse():
    # Define the parameters
    f0 = 15000  # Centre frequency
    t_pulse = 2.6667e-4  # Pulse length
    bandwidth = 0.7  # Bandwidth
    f_var = -((bandwidth * f0) ** 2) / (8 * np.log(0.5))
    t_var = 1 / (4 * np.pi**2 * f_var)

    # Define the pulse function
    def pulse(t):
        exp_factor = np.exp(-((t - t_pulse / 2) ** 2) / (2 * t_var))
        sin_factor = np.sin(2 * np.pi * f0 * (t - t_pulse / 2))
        pulse_value = exp_factor * sin_factor * (t <= t_pulse)
        return pulse_value

    # Create time values
    t = np.linspace(0, t_pulse, 1000)

    # Compute pulse values
    pulse_values = pulse(t)
    fig, axs = figure_size_setup()
    # Plot the pulse waveform
    axs.plot(t * 1000, pulse_values, "b-", linewidth=2)
    axs.set_xlabel("time (ms)")
    axs.set_ylabel("p(t)")
    axs.set_title("Pulse Waveform")
    axs.grid(True)
    fig.savefig("comsol_pulse_ms.png", dpi=300, format="png")
    fig.savefig("comsol_pulse_ms.svg", dpi=300, format="svg")
    plt.show()


def gen_pulse_dispersion(size=0.75, save=False, distance=0.1):
    tc = 4.460504007831342e-04  # from matlab with this: tc = gauspuls('cutoff',3e3,0.6,[],-20);
    dt = 1e-6
    fs = 1 / dt
    t = np.arange(-tc, tc + dt, dt)
    yi, yq, ye = signal.gausspulse(t, fc=3e3, bw=0.6, retquad=True, retenv=True)
    pulse = yi
    pulse_window = signal.hann(len(pulse))
    size_str = str(size).replace(".", "_")
    # fig, axs = figure_size_setup_thesis(size)
    # axs.plot(t * 1000, pulse)
    # axs.set_xlabel("Time (ms)")
    # axs.set_ylabel("Amplitude")
    # axs.set_title("Pulse")
    # if save:
    #     fig.savefig("pulse_yi.png", dpi=300, format="png")
    #     fig.savefig("pulse_yi.svg", dpi=300, format="svg")
    # plt.show()

    pulse = pulse * pulse_window

    t_axis = np.arange(min(t), 0.01, dt)  # Adjusted size to match pulse

    # Zero-padding
    num_zeros = len(t_axis) - len(pulse)
    pulse_padded = np.pad(pulse, (0, num_zeros), "constant", constant_values=(0, 0))

    # fig, axs = figure_size_setup_thesis(size)
    # axs.plot(t_axis * 1000, pulse_padded)
    # axs.set_xlabel("Time (ms)")
    # axs.set_ylabel("Amplitude")
    # axs.set_title("Pulse Padded")
    # if save:
    #     fig.savefig("pulse_padded_yi.png", dpi=300, format="png")
    #     fig.savefig("pulse_padded_yi.svg", dpi=300, format="svg")
    # plt.show()

    A, S = read_DC_files(5)
    # create a mask to limit the frequency range
    mask = (A["A0 f (kHz)"] > 0.1) & (A["A0 f (kHz)"] < 40)
    A0_t_phase = A["A0 Phase velocity (m/ms)"] * 1000
    A0_t_freq = A["A0 f (kHz)"] * 1000
    A0_group = A["A0 Energy velocity (m/ms)"] * 1000
    A0_wavenumber = A["A0 Wavenumber (rad/mm)"]
    A0_wavenumber = A0_wavenumber[mask]
    A0_t_phase = A0_t_phase[mask]
    A0_t_freq = A0_t_freq[mask]
    A0_group = A0_group[mask]
    print(f"type of A0_t_phase: {type(A0_t_phase)}")
    print(f"type of A0_t_freq: {type(A0_t_freq)}")
    print(f"type of A0_group: {type(A0_group)}")
    # save values in a dict and store as a mat file
    data = {
        "phase_velocity": A0_t_phase.to_numpy(),
        "frequency": A0_t_freq.to_numpy(),
        "group_velocity": A0_group.to_numpy(),
        "wavenumber": A0_wavenumber.to_numpy(),
    }
    scipy.io.savemat("A0.mat", data)

    v_gr = A0_group
    v_ph = A0_t_phase
    # v_ph = (1 + 0.22) * v_ph
    f = A0_t_freq.to_numpy()

    k = (
        f / v_ph
    )  # to make the same as A0_wavenumber, multiply A0_wavenumber by 1000 and multiply k by 2*pi
    k[np.isnan(k)] = 0
    # f = f * 2 * np.pi

    max_distance_phase = max(t_axis) * max(v_ph)
    max_distance_group = max(t_axis) * max(v_gr)
    print(f"max distance phase: {max_distance_phase}")
    print(f"max distance group: {max_distance_group}")
    N = len(pulse_padded)
    freq_range = np.linspace(f[0], f[-1], N // 2 + 1)
    v_ph_interp = interpolate.interp1d(f, v_ph, kind="linear")(freq_range)
    # fig, axs = figure_size_setup_thesis(size)
    # axs.plot(freq_range, v_ph_interp, "b")
    # axs.plot(f, v_ph, "r")
    # axs.set_ylabel("Phase velocity (m/s)")
    # axs.set_xlabel("Frequency (Hz)")
    # axs.set_title("Phase Velocity")
    # axs.legend(["Interpolated", "Original"])
    # if save:
    #     fig.savefig("phase_velocity.png", dpi=300, format="png")
    #     fig.savefig("phase_velocity.svg", dpi=300, format="svg")
    # plt.show()
    omega = 2 * np.pi * freq_range / v_ph_interp
    omega[np.isnan(omega)] = 0
    # max_distance / 2
    distance_str = str(distance).replace(".", "_")
    print(f"distance: {distance} m")
    exp_term = np.exp(-1j * omega * distance)
    pulse_fft = np.fft.fft(pulse_padded)
    pulse_fft = pulse_fft[
        : N // 2 + 1
    ]  # should be multiplied by 2 to compensate for removal of negative part
    shifted_pulse_fft = pulse_fft * exp_term
    shifted_pulse_fft = shifted_pulse_fft / np.sqrt(distance)
    shifted_pulse = np.fft.ifft(shifted_pulse_fft, N)
    shifted_pulse = np.real(shifted_pulse)

    fig, axs = figure_size_setup_thesis(size)
    axs.plot(t_axis * 1000, pulse_padded)
    axs.plot(t_axis * 1000, shifted_pulse)
    axs.set_ylabel("Amplitude")
    axs.set_xlabel("Time (ms)")
    # axs.set_title("Shifted Pulse")
    axs.legend(["Original", "Shifted"])
    if save:
        fig.savefig(
            f"shifted_pulse{distance_str}_yi_{size_str}.png", dpi=300, format="png"
        )
        fig.savefig(
            f"shifted_pulse{distance_str}_yi_{size_str}.svg", dpi=300, format="svg"
        )
    plt.show()
    d_step, out_signal = dispersion_compensation(
        t_axis, shifted_pulse, f, k, group_velocity=v_gr
    )
    # d_step_equallyspace = np.linspace(min(d_step), max(d_step), len(d_step))
    out_signal_real = np.real(out_signal)

    analytic_signal = signal.hilbert(out_signal_real)
    envelope = np.abs(analytic_signal)
    peak_index = np.argmax(envelope)
    peak_distance = d_step[peak_index]
    peak_value = envelope[peak_index]

    fig, axs = figure_size_setup_thesis(size)
    axs.plot(d_step, out_signal_real, label="Compensated signal")
    axs.plot(d_step, envelope, label="Envelope")
    # annotate and mark the maximum peak of the envelope with the distance with a dot and the distance with two decimals
    axs.plot(
        d_step[peak_index],
        envelope[peak_index],
        "o",
        label=f"Peak at {round(d_step[peak_index],2)} m",
    )
    # axs.plot(t_axis * 1000, pulse_padded)
    axs.set_ylabel("Amplitude")
    axs.set_xlabel("Distance (m)")
    # axs.set_title("Dispersion Compensated Pulse")
    # axs.legend(["Compensated", "Original"])
    axs.legend()
    if save:
        fig.savefig(
            f"dispersion_compensated_pulse{distance_str}_yi_{size_str}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"dispersion_compensated_pulse{distance_str}_yi_{size_str}.svg",
            dpi=300,
            format="svg",
        )
    plt.show()

    print(f"peak distance: {peak_distance}")
    print(f"true distance: {distance}")
    error_distance = peak_distance - distance
    error_distance_div = peak_distance / distance
    error_distance_perc = error_distance / distance * 100
    error_distance_div_v2 = distance / peak_distance
    print(f"error distance: {error_distance}")
    print(f"error distance div: {error_distance_div}")
    print(f"error distance div v2: {error_distance_div_v2}")
    print(f"error distance perc: {error_distance_perc}")
    correct_distance = peak_distance * 1 / 0.16
    print(f"correct distance: {correct_distance}")
    correction_value = 0.16

    print(f"type of d_step: {type(d_step)}")
    print(f"type of dispersion signal: {type(out_signal_real)}")
    # Find the index that is closest to the desired shift
    d_step_corrected = d_step / 0.16

    # Interpolate your signal to the new distance axis
    f = interpolate.interp1d(
        d_step_corrected, out_signal_real, bounds_error=False, fill_value=0
    )
    signal_corrected = f(d_step)
    analytic_signal = signal.hilbert(signal_corrected)
    envelope = np.abs(analytic_signal)
    peak_index = np.argmax(envelope)
    d_step[peak_index]
    fig, axs = figure_size_setup_thesis(size)
    axs.plot(d_step, signal_corrected, label="Corrected signal")
    # plot envelope
    axs.plot(d_step, envelope, label="Envelope")
    # annotate and mark the maximum peak of the envelope with the distance with a dot and the distance with two decimals
    axs.plot(
        d_step[peak_index],
        envelope[peak_index],
        "o",
        label=f"Peak at {round(d_step[peak_index],2)} m",
    )

    # axs.plot(t_axis * 1000, pulse_padded)
    axs.set_ylabel("Amplitude")
    axs.set_xlabel("Distance (m)")
    # axs.set_title("Dispersion Compensated Pulse")
    # axs.legend(["Compensated", "Original"])
    axs.legend()
    if save:
        fig.savefig(
            f"dispersion_compensated_pulse{distance_str}_yi_corrected_{size_str}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"dispersion_compensated_pulse{distance_str}_yi_corrected_{size_str}.svg",
            dpi=300,
            format="svg",
        )
    plt.show()


def theory_disperson_curve(save=False):
    A, S = read_DC_files(2)
    fig, axs = figure_size_setup()
    number_modes = 3
    thickness = 0.02
    for i in range(number_modes):
        Afrekvens_string = "A" + str(i) + " f (kHz)"  # A0 f (kHz)
        Sfrekvens_string = "S" + str(i) + " f (kHz)"  # S0 f (kHz)
        A_veloc_string = (
            "A" + str(i) + " Phase velocity (m/ms)"
        )  # A0 Phase velocity (m/ms)
        S_veloc_string = (
            "S" + str(i) + " Phase velocity (m/ms)"
        )  # S0 Phase velocity (m/ms)
        A_x = A[Afrekvens_string]
        S_x = S[Sfrekvens_string]
        A_y = A[A_veloc_string]
        S_y = S[S_veloc_string]
        axs.plot(A_x, A_y, "r", label="A" + str(i))
        axs.plot(S_x, S_y, "b", label="S" + str(i))
        axs.annotate(
            "A" + str(i),
            xy=(A_x.iloc[1], A_y.iloc[1]),
            xytext=(10, 10),
            textcoords="offset points",
            color="red",
        )
        axs.annotate(
            "S" + str(i),
            xy=(S_x.iloc[1], S_y.iloc[1]),
            xytext=(10, 10),
            textcoords="offset points",
            color="blue",
        )
    # axs.set_xlabel(r'Frequency (kHz)$\cdot$ thickness (mm)')
    axs.set_xlabel(r"Frequency (kHz)")
    axs.set_ylabel("Phase velocity (m/ms)")
    axs.set_title("Dispersion curve LDPE")
    # set x limit
    axs.set_xlim([0, 80])
    # axs.legend()
    if save:
        fig.savefig(
            "dispersion_curve_theory_LDPE_notthickness.png", dpi=300, format="png"
        )
        fig.savefig(
            "dispersion_curve_theory_LDPE_notthickness.svg", dpi=300, format="svg"
        )
    plt.show()


def COMSOL_velocity_curve(
    size=0.75,
    save=False,
    filenum=4,
    y_max=1.7,
    m_per_s=True,
    freq_max=35,
    name="COMSOL",
    number_modes=2,
):
    A, S = read_DC_files(filenum)
    size_name = str(size).replace(".", "_")
    fig, axs = figure_size_setup_thesis(size)
    thickness = 0.02

    if m_per_s:
        multiplier = 1000
    else:
        multiplier = 1
    y_max = y_max * multiplier
    for i in range(number_modes):
        Afrekvens_string = "A" + str(i) + " f (kHz)"  # A0 f (kHz)
        Sfrekvens_string = "S" + str(i) + " f (kHz)"  # S0 f (kHz)
        A_veloc_string = (
            "A" + str(i) + " Phase velocity (m/ms)"
        )  # A0 Phase velocity (m/ms)
        S_veloc_string = (
            "S" + str(i) + " Phase velocity (m/ms)"
        )  # S0 Phase velocity (m/ms)
        A_group_veloc_string = (
            "A" + str(i) + " Energy velocity (m/ms)"
        )  # A0 Energy velocity (m/ms)
        S_group_veloc_string = (
            "S" + str(i) + " Energy velocity (m/ms)"
        )  # S0 Energy velocity (m/ms)

        A_x = A[Afrekvens_string]
        S_x = S[Sfrekvens_string]
        A_y = A[A_veloc_string] * multiplier
        S_y = S[S_veloc_string] * multiplier
        A_group_y = A[A_group_veloc_string] * multiplier
        S_group_y = S[S_group_veloc_string] * multiplier

        axs.plot(A_x, A_y, color="C0")
        axs.plot(S_x, S_y, color="C1")
        axs.plot(A_x, A_group_y, color="C0", linestyle="--")
        axs.plot(S_x, S_group_y, color="C1", linestyle="--")
        annotation_y_A = min(
            y_max - 0.22 * multiplier, A_y.iloc[1]
        )  # limit to max of 3.5
        annotation_y_S = min(
            y_max - 0.22 * multiplier, S_y.iloc[1]
        )  # limit to max of 3.5

        axs.annotate(
            "A" + str(i),
            xy=(A_x.iloc[1], annotation_y_A),
            xytext=(10, 10),
            textcoords="offset points",
            color="C0",
        )
        axs.annotate(
            "S" + str(i),
            xy=(S_x.iloc[1], annotation_y_S),
            xytext=(10, 10),
            textcoords="offset points",
            color="C1",
        )
    # axs.set_xlabel(r'Frequency (kHz)$\cdot$ thickness (mm)')
    axs.set_xlabel(r"Frequency (kHz)")
    if m_per_s:
        axs.set_ylabel("Velocity (m/s)")
    else:
        axs.set_ylabel("Velocity (m/ms)")

    # axs.set_title("Dispersion curve LDPE")
    # set x limit
    axs.set_xlim([0, freq_max])
    axs.set_ylim([0, y_max])
    # Create custom lines for the legend
    phase_line = mlines.Line2D([], [], color="k", label="Phase velocity")
    group_line = mlines.Line2D(
        [], [], color="k", linestyle="--", label="Group velocity"
    )

    # Add the lines to the legend
    axs.legend(handles=[phase_line, group_line])

    if save:
        fig.savefig(
            f"dispersion_curve_{name}_LDPE_notthickness_{size_name}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"dispersion_curve_{name}_LDPE_notthickness_{size_name}.svg",
            dpi=300,
            format="svg",
        )
    plt.show()


def objective(x, density, vp):
    # x[0] corresponds to Poisson's ratio (v), x[1] corresponds to Young's modulus (E)

    # Calculate measured bulk modulus
    measured_M = density * vp**2

    # Calculate bulk modulus with the given values of v and E
    calculated_M = (x[1] * (1 - x[0])) / ((1 + x[0]) * (1 - 2 * x[0]))

    # Return the difference
    return abs(measured_M - calculated_M)


def find_best_values(density=935.7, vp=2527.646):
    # Initial guess (midpoint of expected range)
    v0 = (0.38 + 0.46) / 2
    E0 = (0.9e9 + 3.5e9) / 2
    x0 = np.array([v0, E0])

    # Bounds for v and E
    bounds = [(0.38, 0.46), (0.9e9, 3.5e9)]

    # Run the optimization
    result = minimize(objective, x0, args=(density, vp), bounds=bounds, tol=1e-12)
    best_v = result.x[0]
    best_E = result.x[1]
    smallest_diff = result.fun
    print(f"Best Poisson's ratio: {best_v}")
    print(f"Best Young's modulus: {best_E}")
    print(f"Smallest difference: {smallest_diff}")
    return best_v, best_E, smallest_diff


def calculate_velocity(E, rho, nu):
    return ((E * (1 - nu)) / ((1 + nu) * (1 - 2 * nu) * rho)) ** 0.5


def find_combinations_for_velocity_range(target_velocity=2527):
    def find_combinations_for_velocity(target_velocity):
        density = 936  # Range of densities in kg/m^3
        youngs_modulus_range = [
            i * 0.05 for i in range(18, 121)
        ]  # Range of Young's moduli in GPa with a step size of 0.05
        print(f"max of youngs_modulus_range: {max(youngs_modulus_range)}")
        poisson_ratio_range = [
            i * 0.05 for i in range(0, 11)
        ]  # Range of Poisson's ratios with a step size of 0.05

        closest_combination = None
        closest_velocity_diff = float("inf")
        found_combinations = []

        for youngs_modulus in youngs_modulus_range:
            for poisson_ratio in poisson_ratio_range:
                # Convert Young's modulus from GPa to Pa
                E = youngs_modulus * 1e9
                velocity = calculate_velocity(E, density, poisson_ratio)
                velocity_diff = abs(velocity - target_velocity)

                if velocity_diff < closest_velocity_diff:
                    closest_combination = (density, youngs_modulus, poisson_ratio)
                    closest_velocity_diff = velocity_diff

                if velocity == target_velocity:
                    found_combinations.append((density, youngs_modulus, poisson_ratio))

        if found_combinations:
            return found_combinations
        else:
            return closest_combination

    combinations = find_combinations_for_velocity(target_velocity)

    if isinstance(combinations, list):
        print("Combinations that give a velocity of", target_velocity, "m/s:")
        for combination in combinations:
            youngs_modulus = combination[1]
            poisson_ratio = combination[2]
            velocity = calculate_velocity(youngs_modulus * 1e9, density, poisson_ratio)
            print(
                "Density:",
                density,
                "kg/m^3, Young's Modulus:",
                youngs_modulus,
                "GPa, Poisson's Ratio:",
                poisson_ratio,
            )
            print("Velocity:", velocity, "m/s")
        return combinations
    else:
        youngs_modulus = combinations[1]
        poisson_ratio = combinations[2]
        velocity = calculate_velocity(youngs_modulus * 1e9, density, poisson_ratio)
        print("No exact match found. Closest combination:")
        print(
            "kg/m^3, Young's Modulus:",
            youngs_modulus,
            "GPa, Poisson's Ratio:",
            poisson_ratio,
        )
        print("Velocity:", velocity, "m/s")
        return combinations


def REAL_plate_velocities(theoretical=True, size=0.75, save=False):
    df = csv_to_df_thesis("plate20mm\\setup1_vegard\\chirp", "chirp_v5")
    # frequency range from 100 to 40000 Hz samplerate of 150000 Hz
    freqs = np.linspace(100, 40000, 150000)
    channels = ["channel 2", "channel 3"]
    df = wp.preprocess_df(df, detrend=True)
    SAMPLE_RATE = 150000
    BANDWIDTH = [100, 40000]
    duration_cut = 40
    size_str = str(size).replace(".", "_")

    # check if dataframe has a column with name wave_gen
    if "wave_gen" in df.columns:
        chirp = df["wave_gen"].to_numpy()

    df_sig_only = df.drop(columns=["wave_gen"])
    df_sig_only_copy = df_sig_only.copy()
    # set all values before 170000 to 0

    df_sig_only.iloc[:170000, :] = 0
    compressed_df = wp.compress_chirp(df_sig_only, chirp, use_recorded_chirp=True)
    # plt.Figure(figsize=(16, 14))
    # plt.plot(compressed_df[channels[0]], label=channels[0])
    # plt.plot(compressed_df[channels[1]], label=channels[1])
    # plt.xlabel("Samples")
    # plt.ylabel("Amplitude")
    # plt.legend()

    # plt.show()

    # exit()
    start_index_ch1 = get_first_index_above_threshold(compressed_df[channels[0]], 600)
    end_index_ch1 = start_index_ch1 + duration_cut
    start_index_ch2 = get_first_index_above_threshold(compressed_df[channels[1]], 600)
    end_index_ch2 = start_index_ch2 + duration_cut
    cut_ch1 = compressed_df[channels[0]][start_index_ch1:end_index_ch1]
    cut_ch2 = compressed_df[channels[1]][start_index_ch2:end_index_ch2]
    # add window to signal
    window = np.hamming(duration_cut)
    cut_ch1_win = cut_ch1 * window
    cut_ch2_win = cut_ch2 * window

    # plot the cut signals
    # fig2 = plt.Figure(figsize=(16, 14))
    # plt.plot(cut_ch1_win, label=channels[0])
    # plt.plot(cut_ch2_win, label=channels[1])
    # plt.legend()
    # plt.show()
    # exit()
    s1t = np.zeros(len(compressed_df[channels[0]]))
    s1t[start_index_ch1:end_index_ch1] = cut_ch1_win
    s2t = np.zeros(len(compressed_df[channels[1]]))
    s2t[start_index_ch2:end_index_ch2] = cut_ch2_win

    plt.plot(s1t)
    plt.plot(s2t)
    plt.show()
    phase, freq = wp.phase_difference_plot(
        s1t,
        s2t,
        # method=method,
        n_pi=-0.97,
        SAMPLE_RATE=SAMPLE_RATE,
        BANDWIDTH=BANDWIDTH,
        save_fig=False,
        # file_name=file_name,
        # file_format=file_format,
        # figsize=figsize,
        show=True,
    )
    phase_vel = wp.phase_velocity(phase, freq, 0.1, plot=True)
    freq = freq / 1000
    if theoretical:
        A0_t, S0_t = read_DC_files(5)
        A0_t_phase = A0_t["A0 Phase velocity (m/ms)"]
        A0_t_freq = A0_t["A0 f (kHz)"]
        S0_t_phase = S0_t["S0 Phase velocity (m/ms)"]
        S0_t_freq = S0_t["S0 f (kHz)"]
        # convert from m/ms to m/s
        A0_t_phase = A0_t_phase * 1000
        S0_t_phase = S0_t_phase * 1000
    fig, ax = figure_size_setup_thesis()
    ax.plot(freq, phase_vel, label="Measured")
    if theoretical:
        ax.plot(A0_t_freq, A0_t_phase, label="Theoretical")
        # ax.plot(S0_t_freq, S0_t_phase, label="S0")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Phase velocity (m/s)")
    ax.set_xlim([0, 40])
    ax.legend()
    if save:
        # save as png and svg
        plt.savefig(
            f"realplate_velocities_{size_str}.png",
            format="png",
            dpi=300,
            # bbox_inches="tight",
        )
        plt.savefig(
            f"realplate_velocities_{size_str}.svg",
            format="svg",
            dpi=300,
            # bbox_inches="tight",
        )
    plt.show()


def print_material_values(rho=935.7, v_p=2527.646):
    p, E, _ = find_best_values()
    print(f"E: {E}")
    nu = r"$\nu$"
    print(f"{nu}: {p}")
    G = E / (2 * (1 + p))
    print(f"G: {G}")
    V_s = np.sqrt(G / rho)
    print(f"V_s: {V_s}")
    V_p = np.sqrt(E / rho)
    print(f"V_p calculated: {V_p}")
    print(f"V_p measured: {v_p}")


def COMSOL_dispersion(
    position=20, size=0.75, save=False, theoretical=True, reflections=False
):
    A, S = read_DC_files(4)
    size_str = str(size).replace(".", "_")
    n_vlines = 1
    # Setting up velocities
    mask = (A["A0 f (kHz)"] > 0.1) & (A["A0 f (kHz)"] < 40)
    A0_t_phase = A["A0 Phase velocity (m/ms)"] * 1000
    A0_t_freq = A["A0 f (kHz)"] * 1000
    A0_group = A["A0 Energy velocity (m/ms)"] * 1000
    A0_wavenumber = A["A0 Wavenumber (rad/mm)"]
    (
        mean_velocities_A0,
        mean_velocities_S0,
        frequencies,
    ) = all_calculate_phase_velocities()
    # A0_wavenumber = A0_wavenumber[mask]
    # A0_t_phase = A0_t_phase[mask]
    # A0_t_freq = A0_t_freq[mask]
    # A0_group = A0_group[mask]

    if theoretical:
        v_gr = A0_group
        v_ph = A0_t_phase
        f = A0_t_freq.to_numpy()

    else:
        v_ph = mean_velocities_A0
        f = frequencies * 1000
        v_gr = wp.group_velocity_phase(v_ph, f, plot=True)

    # v_ph = (1 + 0.22) * v_ph

    k = (
        f / v_ph
    )  # to make the same as A0_wavenumber, multiply A0_wavenumber by 1000 and multiply k by 2*pi
    k[np.isnan(k)] = 0

    # f = f * 2 * np.pi
    # setting up comsol file
    # top_data, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(9)
    # bot_data, x_pos_bot, y_pos_bot, z_pos_bot, time_axis_bot = get_comsol_data(10)

    top_data, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(11)
    bot_data, x_pos_bot, y_pos_bot, z_pos_bot, time_axis_bot = get_comsol_data(12)
    distance = np.sqrt(
        (x_pos_top[position] - x_pos_top[0]) ** 2
        + (y_pos_top[position] - y_pos_top[0]) ** 2
    )
    distance_str = str(round(distance, 2)).replace(".", "_")
    # time_axis_ms = time_axis_top * 1e-3
    # time_axis_seconds = time_axis_top * 1e-6
    time_axis_seconds = time_axis_top
    time_axis_ms = time_axis_seconds * 1e3
    S0_pos = (top_data[position] - bot_data[position]) / 2
    A0_pos = (top_data[position] + bot_data[position]) / 2
    # A0_pos = top_data[position]
    max_distance_phase = max(time_axis_seconds) * max(v_ph)
    max_distance_group = max(time_axis_seconds) * max(v_gr)
    print(f"max distance phase: {max_distance_phase}")
    print(f"max distance group: {max_distance_group}")
    # normalize the top data[0] and the A0_pos in order to compare them
    top_data[0] = top_data[0] / max(top_data[0])
    A0_pos = A0_pos / max(A0_pos)
    fig, axs = figure_size_setup_thesis(size)
    axs.plot(time_axis_ms, top_data[0], label="Source pulse")
    axs.plot(time_axis_ms, A0_pos, label=f"Pulse at {round(distance,2)} m")
    axs.set_ylabel("Amplitude")
    axs.set_xlabel("Time (ms)")
    axs.legend()
    # axs.set_title("Shifted Pulse")

    if save:
        fig.savefig(
            f"COMSOL_top_A0_{distance_str}_{size_str}_3khz.png", dpi=300, format="png"
        )
        fig.savefig(
            f"COMSOL_top_A0_{distance_str}_{size_str}_3khz.svg", dpi=300, format="svg"
        )
    plt.show()

    if theoretical:
        d_step, out_signal = dispersion_compensation(
            time_axis_seconds, A0_pos, f, k, group_velocity=v_gr
        )
    else:
        d_step, out_signal = dispersion_compensation(time_axis_seconds, A0_pos, f, k)
    # d_step_equallyspace = np.linspace(min(d_step), max(d_step), len(d_step))
    out_signal_real = np.real(out_signal)

    analytic_signal = signal.hilbert(out_signal_real)
    envelope = np.abs(analytic_signal)
    peak_index = np.argmax(envelope)
    peak_distance = d_step[peak_index]
    peak_value = envelope[peak_index]

    # find the index where the envelope crosses a threshold value
    threshold = 0.0016
    threshold_index = (
        np.where(envelope[40 + position :] > threshold)[0][0] + 40 + position
    )
    print(f"threshold index: {threshold_index}")
    threshold_distance = d_step[threshold_index]
    print(f"threshold distance: {threshold_distance}, true distance: {distance}")
    estimated_distance_str = str(round(threshold_distance, 4)).replace(".", "_")
    if reflections:
        distances, arrival_times = draw_simulated_plate(velocity=max(v_gr))
    fig, axs = figure_size_setup_thesis(size)
    axs.plot(d_step, out_signal_real, label="Compensated signal")
    axs.plot(d_step, envelope, label="Envelope")
    # plot the vline at the true distance
    axs.axvline(x=distance, color="k", linestyle="--", label="True distance")
    axs.axvline(
        x=threshold_distance, color="r", linestyle="--", label="Estimated distance"
    )
    # annotate and mark the maximum peak of the envelope with the distance with a dot and the distance with two decimals
    # axs.plot(
    #     d_step[peak_index],

    #     envelope[peak_index],
    #     "o",
    #     label=f"Peak at {round(d_step[peak_index],2)} m",
    # )
    # axs.plot(t_axis * 1000, pulse_padded)
    axs.set_ylabel("Amplitude")
    axs.set_xlabel("Distance (m)")

    # axs.set_title("Dispersion Compensated Pulse")
    # axs.legend(["Compensated", "Original"])
    if reflections:
        n_vlines = 4
        for i in range(1, n_vlines + 1):
            axs.axvline(
                x=distances[i],
                color="k",
                linestyle="--",
                label=f"Simulated reflection {i+1}",
            )
    axs.legend()
    if save:
        fig.savefig(
            f"dispersion_compensated_pulse_3khz_COMSOL_top_A0{distance_str}_{size_str}_estimated_{estimated_distance_str}_n_vlines_{n_vlines}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"dispersion_compensated_pulse_3khz_COMSOL_top_A0{distance_str}_{size_str}_estimated_{estimated_distance_str}_n_vlines_{n_vlines}.svg",
            dpi=300,
            format="svg",
        )
    print("showing figure")
    plt.show()

    # print(f"peak distance: {peak_distance}")
    # print(f"true distance: {distance}")
    # error_distance = peak_distance - distance
    # error_distance_div = peak_distance / distance
    # error_distance_perc = error_distance / distance * 100
    # error_distance_div_v2 = distance / peak_distance
    # print(f"error distance: {error_distance}")
    # print(f"error distance div: {error_distance_div}")
    # print(f"error distance div v2: {error_distance_div_v2}")
    # print(f"error distance perc: {error_distance_perc}")
    # correct_distance = peak_distance * 1 / 0.16
    # print(f"correct distance: {correct_distance}")
    # correction_value = 0.16

    # Find the index that is closest to the desired shift
    # d_step_corrected = d_step / 0.16

    # # Interpolate your signal to the new distance axis
    # f = interpolate.interp1d(
    #     d_step_corrected, out_signal_real, bounds_error=False, fill_value=0
    # )
    # signal_corrected = f(d_step)
    # analytic_signal = signal.hilbert(signal_corrected)
    # envelope = np.abs(analytic_signal)
    # peak_index = np.argmax(envelope)
    # d_step[peak_index]
    # fig, axs = figure_size_setup_thesis(size)
    # axs.plot(d_step, signal_corrected, label="Corrected signal")
    # # plot envelope
    # axs.plot(d_step, envelope, label="Envelope")
    # # annotate and mark the maximum peak of the envelope with the distance with a dot and the distance with two decimals
    # axs.plot(
    #     d_step[peak_index],
    #     envelope[peak_index],
    #     "o",
    #     label=f"Peak at {round(d_step[peak_index],2)} m",
    # )

    # # axs.plot(t_axis * 1000, pulse_padded)
    # axs.set_ylabel("Amplitude")
    # axs.set_xlabel("Distance (m)")
    # # axs.set_title("Dispersion Compensated Pulse")
    # # axs.legend(["Compensated", "Original"])
    # axs.legend()
    # if save:
    #     fig.savefig(
    #         f"dispersion_compensated_pulse{distance_str}_yi_corrected_{size_str}.png",
    #         dpi=300,
    #         format="png",
    #     )
    #     fig.savefig(
    #         f"dispersion_compensated_pulse{distance_str}_yi_corrected_{size_str}.svg",
    #         dpi=300,
    #         format="svg",
    #     )
    # plt.show()


def REAL_dispersion(
    distance=0.2, size=0.75, save=False, theoretical=True, reflections=False
):
    """
    In setup 3 vegard does we have a sensor under the touch location and one at 10 cm distance both on the top and bottom

    """
    A, S = read_DC_files(6)
    size_str = str(size).replace(".", "_")
    n_vlines = 1
    # Setting up velocities
    mask = (A["A0 f (kHz)"] > 0.1) & (A["A0 f (kHz)"] < 40)
    A0_t_phase = A["A0 Phase velocity (m/ms)"] * 1000
    A0_t_freq = A["A0 f (kHz)"] * 1000
    A0_group = A["A0 Energy velocity (m/ms)"] * 1000
    A0_wavenumber = A["A0 Wavenumber (rad/mm)"]
    # A0_wavenumber = A0_wavenumber[mask]
    # A0_t_phase = A0_t_phase[mask]
    # A0_t_freq = A0_t_freq[mask]
    # A0_group = A0_group[mask]
    sample_rate = 150e3
    size_name = str(size).replace(".", "_")
    folder = "plate20mm\\setup4_vegard\\touch"
    # file_name = "touch_v1"
    file_name = "touch_ca2_4_v1"
    # file_name3 = "touch_v5"

    # fig1, ax1 = figure_size_setup(0.45)
    # data = csv_to_df_thesis(folder, file_name)
    data = csv_to_df_thesis(folder, file_name)
    # drop wave_gen channel
    # plt.plot(data["channel 1"], label="channel 1")
    # plt.plot(data["channel 2"], label="channel 2")
    # plt.plot(data["channel 3"], label="channel 3")
    # plt.legend()
    # plt.show()
    data = data.drop(columns=["wave_gen"])
    # data3 = csv_to_df_thesis(folder, file_name3)
    data_start = data["channel 1"]
    data_top_50 = data["channel 2"]
    # data_bot_10 = data["channel 3"]
    data_top_75 = data["channel 3"]
    time_axis = np.linspace(0, len(data) // sample_rate, num=len(data))
    print(f"max time axis: {max(time_axis)}")
    plt.plot(time_axis, data_start, label="start")
    plt.plot(time_axis, data_top_50, label="top 50 cm")
    plt.plot(time_axis, data_top_75, label="top 75 cm")
    plt.legend()
    plt.show()

    # start point is when data_start is above 0.005
    start_point = get_first_index_above_threshold(data_start, 0.007)
    seconds = 0.008
    # create new time axis which starts in 0, from start point index
    time_axis_new = (
        time_axis[start_point : start_point + int(seconds * sample_rate)]
        - time_axis[start_point]
    )
    print(f"max time axis new: {max(time_axis_new)}")
    # remove the data before the start point from all the data
    data_start = data_start[start_point : start_point + int(seconds * sample_rate)]
    data_top_50 = data_top_50[start_point : start_point + int(seconds * sample_rate)]
    data_top_75 = data_top_75[start_point : start_point + int(seconds * sample_rate)]
    # plot the data
    # plt.plot(time_axis_new, data_start)
    plt.plot(time_axis_new, data_top_50)
    plt.plot(time_axis_new, data_top_75)
    plt.show()
    exit()
    if distance == 0.5:
        A0_pos = data_top_50
    elif distance == 0.75:
        A0_pos = data_top_75
    # A0_pos = (data_top_10 - data_bot_10) / 2
    # S0_pos = (data_top_10 + data_bot_10) / 2
    # A0_pos = data_top_50
    # A_pos2 = data_top_75
    # plot top and A0
    # plt.plot(time_axis_new, data_top_50, label="top 50 cm")
    # #plt.plot(time_axis_new, A0_pos, label="A0")
    # plt.legend()
    # plt.show()
    # normalize the S0 and A0
    # S0_pos = S0_pos / max(S0_pos)
    # A0_pos = A0_pos / max(A0_pos)
    # plt.plot(time_axis_new, A0_pos, label="A0")
    # # plt.plot(time_axis_new, S0_pos, label="S0")
    # plt.legend()
    # plt.show()

    if theoretical:
        v_gr = A0_group
        v_ph = A0_t_phase
        f = A0_t_freq.to_numpy()

    else:
        exit()

    # v_ph = (1 + 0.22) * v_ph

    k = (
        f / v_ph
    )  # to make the same as A0_wavenumber, multiply A0_wavenumber by 1000 and multiply k by 2*pi
    k[np.isnan(k)] = 0

    distance_str = str(round(distance, 2)).replace(".", "_")

    max_distance_phase = max(time_axis_new) * max(v_ph)
    max_distance_group = max(time_axis_new) * max(v_gr)
    print(f"max distance phase: {max_distance_phase}")
    print(f"max distance group: {max_distance_group}")
    # normalize the top data[0] and the A0_pos in order to compare them
    fig, axs = figure_size_setup_thesis(size)
    axs.plot(time_axis_new * 1000, A0_pos, label=f"Signal at {distance}m")
    axs.set_ylabel(r"Acceleration ($\mathrm{m/s^2}$)")
    axs.set_xlabel("Time (ms)")
    axs.legend()
    # axs.set_title("Shifted Pulse")

    if save:
        fig.savefig(
            f"REAL_PLATE_A0_setup4_touch_{distance_str}_{size_str}.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"REAL_PLATE_A0_setup4_touch_{distance_str}_{size_str}.svg",
            dpi=300,
            format="svg",
        )
    plt.show()

    if theoretical:
        d_step, out_signal = dispersion_compensation(
            time_axis_new, A0_pos, f, k, group_velocity=v_gr
        )

    else:
        d_step, out_signal = dispersion_compensation(time_axis_seconds, A0_pos, f, k)
    # d_step_equallyspace = np.linspace(min(d_step), max(d_step), len(d_step))
    out_signal_real = np.real(out_signal)

    analytic_signal = signal.hilbert(out_signal_real)
    envelope = np.abs(analytic_signal)
    peak_index = np.argmax(envelope)
    peak_distance = d_step[peak_index]
    peak_value = envelope[peak_index]

    # find the index where the envelope crosses a threshold value
    threshold = 0.04
    ignore_index = np.where(d_step > 0.4)[0][0]
    threshold_index = np.where(envelope[ignore_index:] > threshold)[0][0] + ignore_index
    # ignore distance values below 0.4m

    print(f"threshold index: {threshold_index}")
    threshold_distance = d_step[threshold_index]
    print(f"threshold distance: {threshold_distance}, true distance: {distance}")
    estimated_distance_str = str(round(threshold_distance, 4)).replace(".", "_")
    if reflections:
        distances, arrival_times = draw_simulated_plate(velocity=max(v_gr))
    # find index of d_step which is above 2m
    # index_above_25m = np.where(d_step > 25)[0][0]
    index_above_25m = -1
    fig, axs = figure_size_setup_thesis(size)
    axs.plot(
        d_step[:index_above_25m],
        out_signal_real[:index_above_25m],
        label="Compensated signal",
    )
    axs.plot(d_step[:index_above_25m], envelope[:index_above_25m], label="Envelope")
    # plot the vline at the true distance
    axs.axvline(x=distance, color="k", linestyle="--", label="True distance")
    # axs.axvline(
    #     x=threshold_distance, color="r", linestyle="--", label="Estimated distance"
    # )
    # annotate and mark the maximum peak of the envelope with the distance with a dot and the distance with two decimals
    # axs.plot(
    #     d_step[peak_index],

    #     envelope[peak_index],
    #     "o",
    #     label=f"Peak at {round(d_step[peak_index],2)} m",
    # )
    # axs.plot(t_axis * 1000, pulse_padded)
    axs.set_ylabel("Amplitude")
    axs.set_xlabel("Distance (m)")

    # axs.set_title("Dispersion Compensated Pulse")
    # axs.legend(["Compensated", "Original"])
    if reflections:
        n_vlines = 4
        for i in range(1, n_vlines + 1):
            axs.axvline(
                x=distances[i],
                color="k",
                linestyle="--",
                label=f"Simulated reflection {i+1}",
            )
    axs.legend()
    if save:
        fig.savefig(
            f"dispersion_compensated_pulse_realplate_top_A0_setup4_touch{distance_str}_{size_str}_estimated_{estimated_distance_str}_short.png",
            dpi=300,
            format="png",
        )
        fig.savefig(
            f"dispersion_compensated_pulse_realplate_top_A0_setup4_touch{distance_str}_{size_str}_estimated_{estimated_distance_str}_short.svg",
            dpi=300,
            format="svg",
        )

    plt.show()


if __name__ == "__main__":
    CROSS_CORR_PATH1 = "\\Measurements\\setup2_korrelasjon\\"
    CROSS_CORR_PATH2 = "\\first_test_touch_passive_setup2\\"
    custom_chirp = csv_to_df(
        file_folder="Measurements\\div_files",
        file_name="chirp_custom_fs_150000_tmax_2_100-40000_method_linear",
        channel_names=CHIRP_CHANNEL_NAMES,
    )
    # set base new base after new measurements:
    # execution_time = timeit.timeit(ccp.SetBase, number=1)
    # print(f'Execution time: {execution_time}')
    # ccp.SetBase(CROSS_CORR_PATH2)
    # ccp.run_test(tolatex=True)
    # ccp.run_test(tolatex=True,data_folder='\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\first_test_touch_passive_setup2\\',filename='results_correlation_old_samples.csv')
    # find position
    # ccp.FindTouchPosition(f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\Measurements\\setup2_korrelasjon\\A2_v3.csv')
    df1 = csv_to_df("Measurements\\setup3_0\\", "prop_speed_chirp3_setup3_0_v1")
    df = csv_to_df(
        "\\Measurements\\setup10_propagation_speed_15cm\\chirp\\", "chirp_v1"
    )
    sensor_test_df = csv_to_df("\\Measurements\\sensortest\\rot_clock_123", "chirp_v1")
    df_touch_10cm = csv_to_df(
        "Measurements\\setup9_propagation_speed_short\\touch\\", "touch_v1"
    )
    time = np.linspace(0, len(df1) / SAMPLE_RATE, num=len(df1))
