import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, Button
from pathlib import Path
from objects import Table, Actuator, Sensor
from setups import Setup2, Setup3, Setup3_2, Setup3_4, Setup6, Setup9
from constants import SAMPLE_RATE, CHANNEL_NAMES, CHIRP_CHANNEL_NAMES
from data_processing import cross_correlation_position as ccp
from csv_to_df import csv_to_df, csv_to_df_thesis
from data_viz_files.visualise_data import compare_signals, plot_vphs, plot_fft, plot_plate_speed_sliders_book, plot_estimated_reflections_with_sliders, compare_signals_v2, plot_compare_signals_v2
from data_processing.preprocessing import get_first_index_above_threshold, interpolate_waveform, crop_data, filter_general, compress_chirp, get_phase_and_vph_of_compressed_signal,cut_out_signal, manual_cut_signal, compress_df_touch, cut_out_pulse_wave, shift_signal
from data_processing.detect_echoes import find_first_peak, get_hilbert_envelope, get_travel_times
from data_processing.find_propagation_speed import find_propagation_speed_with_delay
from data_viz_files.drawing import plot_legend_without_duplicates
import timeit
import data_processing.wave_properties as wp
import data_processing.sensor_testing as st
from data_viz_files.visualise_data import inspect_touch, figure_size_setup, to_dB
import data_processing.wave_properties as wp
import data_processing.sensor_testing as st
import data_processing.preprocessing as pp
from matplotlib import style
import data_viz_files.visualise_data as vd
import os
#import mph

def results_setup1():
    ## Results for phase velocity test in the beginning of the thesis.
    custom_chirp = csv_to_df(file_folder='Measurements\\div_files', file_name='chirp_custom_fs_150000_tmax_2_100-40000_method_linear', channel_names=CHIRP_CHANNEL_NAMES)
    # df1 = csv_to_df('Measurements\\setup3_0\\', 'chirp_100_40000_2s_v1')
    # phase10, freq10  = wp.phase_plotting(df1, chirp=custom_chirp, use_recorded_chirp=True,start_stops=[(241000,508570),(241000,508570)], BANDWIDTH=[100,40000], save_fig=False, file_name='phase_plot_10cm0_45.svg', file_format='svg',figsize=0.45, n_pi=1)
    # wp.plot_velocities(phase10, freq10, 0.10, savefig=False, filename='phase_velocity_10cm.svg', file_format='svg')
    df_PE = csv_to_df_thesis('plate20mm\\setup1\\chirp', 'chirp_100_40000_2s_v1')
    df_teflon = csv_to_df_thesis('plate10mm\\setup1\\chirp', 'chirp_100_40000_2s_v1')
    #filter signal
    df_PE_filt = filter_general(df_PE, filtertype='bandpass', cutoff_highpass=10000, cutoff_lowpass=15000, order=4)
    #df_teflon_filt = filter_general(df_teflon, filtertype='bandpass', cutoff_highpass=100, cutoff_lowpass=40000, order=4)
    phase_PE, freq_PE  = wp.phase_plotting_chirp(df_PE, BANDWIDTH=[100,40000], save_fig=False, file_name='phase_plot_PE_45.svg', file_format='svg',figsize=0.45, n_pi=1)
    #phase_teflon, freq_teflon  = wp.phase_plotting_chirp(df_teflon, BANDWIDTH=[5000,40000], save_fig=False, file_name='phase_plot_teflon_45.svg', file_format='svg',figsize=0.45, n_pi=1)
    #wp.plot_velocities(phase_PE, freq_PE, 0.10, savefig=False, filename='phase_velocity_PE.svg', file_format='svg')
    #wp.plot_velocities(phase_teflon, freq_teflon, 0.10, savefig=False, filename='phase_velocity_teflon.svg', file_format='svg')
    #wp.plot_velocities(phase_PE, freq_PE, 0.10, material='HDPE', savefig=False, filename='phase_velocity_teflon.svg', file_format='svg')
    #wp.plot_velocities(phase_PE, freq_PE, 0.10, material='LDPE', savefig=False, filename='phase_velocity_teflon.svg', file_format='svg')
    print(wp.max_peak_velocity(df_PE, material='HDPE'))
    #print(wp.max_peak_velocity(df_teflon))
    #print(wp.max_peak_velocity(df_PE))

def data_viz(viz_type, folder, filename, semester='thesis', channel='wave_gen'):
    if semester == 'thesis':
        df = csv_to_df_thesis(folder, filename)
    else:
        df = csv_to_df(folder, filename)
    if viz_type == 'scaleogram':
        
        vd.plot_scaleogram(df, channels=['wave_gen'])
    elif viz_type == 'wvd':
        vd.wigner_ville_dist(df, channel)
    elif viz_type == 'custom_wvd':
        vd.custom_wigner_ville_batch(df, channel)
    elif viz_type == 'ssq':
        vd.ssqueeze_spectrum(df, channel)


def load_simulated_data1():
    signals = ['sg8.txt','sg28.txt', 'sg108.txt']
    freq_signals = ['wt8.txt','wt28.txt', 'wt108.txt']
    channels = []
    for sig in signals:
        path = os.path.join(r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm', sig)
        data = np.loadtxt(path)
        channels.append(data)
    return channels

def simulated_data_vel():
    distances = [0.173, 0.41, 1.359]
    channels = load_simulated_data1()
    for idx, ch in enumerate(channels):
        plt.plot(ch, label=f'channel {idx+1}')
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
    freq = np.fft.fftfreq(data8.size, d=1/1000)

    # Select positive-frequency values
    # pos_freq = freq > 0
    # freq = freq[pos_freq]
    # fft8 = np.abs(fft8[pos_freq]) / data8.size
    # fft28 = np.abs(fft28[pos_freq]) / data28.size
    # fft108 = np.abs(fft108[pos_freq]) / data108.size

    # Convert to dB scale
    fft8_db = 20*np.log10(fft8)
    fft28_db = 20*np.log10(fft28)
    fft108_db = 20*np.log10(fft108)

    # Plot FFTs of the three data sets in dB scale
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    axs[0].plot(freq, fft8_db)
    axs[1].plot(freq, fft28_db)
    axs[2].plot(freq, fft108_db)

    # Set title and axis labels
    axs[0].set_title('FFT for punkt 8')
    axs[1].set_title('FFT for punkt 28')
    axs[2].set_title('FFT for punkt 108')
    for ax in axs:
        ax.set_xlabel('Frekvens (kHz)')
        ax.set_ylabel('Amplitude (dB)')
    plt.tight_layout()
    plt.show()
    phase = wp.phase_difference_div(channels[0], channels[1])
    phase1 = wp.phase_difference_div(channels[0], channels[2])
    # Compute complex transfer functions
    tf28_8 = fft28 / fft8
    tf108_8 = fft108 / fft8
    print(f'the length of fft28 is {len(fft28)}')
    # Compute phase differences between transfer functions
    phase_diff_28_8 = np.unwrap(np.angle(tf28_8))
    phase_diff_108_8 = np.unwrap(np.angle(tf108_8))

    # Create frequency axis
    #freq = np.fft.fftfreq(data8.size, d=1/1000)

    # Select positive-frequency values
    # pos_freq = freq >= 0
    # freq = freq[pos_freq]
    # phase_diff_28_8 = phase_diff_28_8[pos_freq]
    # phase_diff_108_8 = phase_diff_108_8[pos_freq]

    # Plot phase differences
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(freq, phase)
    axs[0].set_title('Phase difference tf28/tf8')
    axs[1].plot(freq, phase1)
    axs[1].set_title('Phase difference tf108/tf8')
    for ax in axs:
        ax.set_xlabel('Frekvens (kHz)')
        ax.set_ylabel('Faseforskyvning (rad)')
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
    shift = nfft/16
    K = 6
    tm = 6.0
    flock = 0.01
    tlock = 5
    S = np.abs(libtfr.tfr_spec(s=x, N=nfft, step=shift, Np=Np, K=K, tm=tm, flock=flock, tlock=tlock))
    print(np.shape(S), np.max(S), np.min(S))
    S = librosa.amplitude_to_db(S, ref=np.max, top_db =100)
    print(np.shape(S), np.max(S), np.min(S))
    fig, ax = plt.subplots(figsize=(20,5))
    display.specshow(D, y_axis='log', cmap='viridis')

def wave_number_graph(number=1):
    # Given data points
    # Given data points
    
    fs = 501000
    data, x_pos, y_pos, z_pos, time_axis = get_comsol_data(number)
    phase = wp.phase_difference_div(data[10], data[20], fs, pos_only=True)
    freq = np.fft.fftfreq(data[0].size, 1/fs)
    freq = freq[freq>0]
    fig, ax = plt.subplots()
    ax.plot(freq, phase)
    ax.set_xlabel('Frekvens (Hz)')
    ax.set_ylabel('Faseforskyvning (rad)')
    ax.set_title('Faseforskyvning mellom punkt 10 og 20')
    plt.show()
    distance = x_pos[20]-x_pos[10]
    #phase_vel = wp.phase_velocity(phase, freq, distance)
    wp.plot_velocities(phase, freq, distance, material='LDPE_tonni7mm')


def dispersion_compensation_Wilcox(file_n=2, postion=5, fs=501000, dx=0.001):
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
    upper_freq = 40000
    lower_freq = 0
    new_fs = 80000
    
    #frequency axis from 0 to 40kHz with 80000 samples
    #freq = np.linspace(lower_freq, upper_freq, new_fs)
    

    wave_data_top, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(9)
    wave_data_bottom, x_pos_bottom, y_pos_bottom, z_pos_bottom, time_axis_bottom = get_comsol_data(10)
    plt.title(f'Raw data for top and bottom at postion: {postion}')
    plt.plot(wave_data_top[postion], label='top')
    plt.plot(wave_data_bottom[postion], label='bottom')
    plt.legend()
    plt.show()
    signal = (wave_data_top[postion]+wave_data_bottom[postion])/2 #A0 mode
    plt.plot(signal, label='A0 mode')
    plt.title('A0 mode')
    plt.legend()
    plt.show()
    n_times = 8 #number of points that is an integral power of two and at least eight times as many as in the original signal.
    
    m = len(signal)
    n_fft = 2 ** int(np.ceil(np.log2(n_times * m)))
    print(f'length of signal before padding: {m}, length of signal after padding {n_fft}')
    signal_padded = np.pad(signal, (0, n_fft - m), mode='constant')
    plt.plot(signal_padded)
    plt.title('Padded signal')
    plt.show()
    #desired_sampling_rate = 40000
    #downsample_factor = int(round(m / (fs / desired_sampling_rate)))
    
    G_w = np.fft.fft(signal_padded)
    print(f'shape of G_w: {G_w.shape}')
    dt = 1/fs
    print(f'dt: {dt}')
    freq_vel = np.fft.fftfreq(G_w.size, dt)
    plt.plot(freq_vel, np.abs(G_w))
    plt.title('Fourier transform of padded signal')
    plt.show()
    print(f'shape of freq_vel: {freq_vel.shape}')
    #freq_range = (freq_vel>lower_freq) & (freq_vel<upper_freq)
    #freq_vel = freq_vel[freq_range]
    #f_nyq = freq_vel[-1]/2
    #G_w = G_w[freq_range]
    G_w = G_w[freq_vel>0]
    freq_vel = freq_vel[freq_vel>0]
    f_nyq = fs/2
    print(f'f_nyq: {f_nyq}')
    print(f'last element in freq_vel: {freq_vel[-1]}')
    plt.plot(freq_vel, np.abs(G_w))
    plt.title('Fourier transform of padded signal after frequency range')
    plt.show()
    print(f'freq_vel: {freq_vel.shape}')
    #dt = 1/upper_freq
    v_gr, v_ph = wp.theoretical_group_phase_vel(freq_vel, material='LDPE_tonni20mm', plot=True)
    print(f'v_gr: {v_gr.shape}')
    print(f'v_ph: {v_ph.shape}')
    k = (2*np.pi*freq_vel)/v_ph
    print(f'k: {k.shape}')
    v_nyq = get_velocity_at_freq(f_nyq)['A0']['phase_velocity']
    #print(f'k_max = {k[-1]}') 
    v_max = get_velocity_at_freq(upper_freq)['A0']['phase_velocity']
    k_nyq = (2*np.pi*f_nyq)/v_nyq
    k_max = k[-1] #doesnt matter if i use this or this 2*np.pi*upper_freq/v_max since both are equal or 2 times k_nyq
    print(f'k_nyq: {k_nyq} ')
    w = 2*np.pi*freq_vel
    print(f'shape of w: {w.shape}')
    n = len(k)
    print(f'length of k: {n}')
    #print(f'altnerative length of k: {int(np.ceil(2 * f_nyq / (1 / (dx * m))))}')
    plt.plot(k, freq_vel)
    plt.xlabel('Wavenumber')
    plt.ylabel('Frequency')
    plt.title('Wavenumber vs frequency')
    plt.show()
    # Perform FFT on the padded signal

    #print(len(k))
    
    # Calculate wavenumber step and number of points in the wavenumber domain
    print(f'Checking if n*delta_x is larger than m*delta_t*v_max. n*delta_x is {n_fft*dx}, m*delta_t*v_max is {m*dt*np.max(v_gr)}')
    #k_nyq = #k[round(1/(2*dt))]
    print(f'Checking if Delta x is less or equal to 1/(2k_nyq). Delta x is {dx}, 1/(2k_nyq) is {1/(2*k_nyq)}')
    dk = 1 / (n_fft * dx)

    print(f'n should be larger than 2 * k_nyq / dk, n is {n_fft}, 2 * k_nyq / dk is {2 * k_nyq / dk}')
    #print(f'this number of points in the wavenumber domain is {n}')
    
    # Interpolate the FFT to equally spaced k values
    k_new = np.arange(0, k_max + dk, dk)
    print(f'shape of k new: {k.shape}')
    # Interpolate G(w) to find G(k)
    G_interp = interpolate.interp1d(k, G_w, kind='linear', bounds_error=False, fill_value=0)(k_new)
    plt.plot(k_new, G_interp.real, label='interpolated G(k)')
    plt.plot(k, G_w.real, label='G(k)')
    plt.xlabel('Wavenumber')
    plt.ylabel('Amplitude')
    plt.title('Interpolated G(k)')
    plt.show()
    print('G_interp created')
    # Calculate the group velocity of the guided wave mode at the wavenumber points
    v_gr_interp = interpolate.interp1d(k, v_gr, kind='linear', bounds_error=False, fill_value=0)(k_new)

    print(f'shape of G_interp: {G_interp.shape}')
    print(f'shape of v_gr_interp: {v_gr_interp.shape}')
  

    # Compute H(k) = G(k) * vgr(k)
    H_k = G_interp * v_gr_interp

    # Apply inverse FFT to H(k) to obtain the dispersion compensated distance-trace
    h_x = np.fft.ifft(H_k)

    # Remove zero-padding from the compensated signal
    h_x = h_x[:m]
    #normalize h_x and signal
    h_x = h_x/np.max(h_x)
    signal = signal/np.max(signal)
    print(f'shape of h_x: {h_x.shape}')
    print(f'shape of signal: {signal.shape}')
    #Create a subplot with the dispersion compensated signal and the original signal
    plt.subplot(2, 1, 1)
    plt.plot(h_x.real, label='Dispersion compensated signal')
    plt.xlabel('sample')
    plt.ylabel('Amplitude')
    plt.title('Dispersion compensated signal')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time_axis_top,signal, label='Original signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Original signal')
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
    path_A = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\LDPE20_A_Lamb.xlsx' 
    path_S = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\LDPE20_S_Lamb.xlsx'
    dc_A0 = pd.read_excel(path_A)
    dc_S0 = pd.read_excel(path_S)
    return dc_A0, dc_S0

def get_wavelength_DC(plot=True):
    A0, S0 = read_DC_files()
    wavelength_A0 = A0['A0 Wavelength (mm)']
    wavelength_S0 = S0['S0 Wavelength (mm)']
    freq = A0['A0 f (kHz)']
    if plot:
        plt.plot(freq, wavelength_A0, label='A0')
        plt.plot(freq, wavelength_S0, label='S0')
        plt.xlabel('Frequency [kHz]')
        plt.ylabel('Wavelength [mm]')
        plt.title('Wavelength of A0 and S0 mode')
        plt.legend()
        plt.show()
    return wavelength_A0, wavelength_S0
            
def get_velocity_at_freq(freq, meter_per_second=True):
    ''' 
    This returns the group and phase velocity of the A0 and S0 mode at a given frequency.
    The velocities are in m/ms. 

    '''
    A0, S0 = read_DC_files()
    idx = (A0['A0 f (kHz)'] - freq).abs().idxmin()
    group_velocity_A0 = A0.loc[idx, 'A0 Energy velocity (m/ms)']
    phase_velocity_A0 = A0.loc[idx, 'A0 Phase velocity (m/ms)']
    group_velocity_S0 = S0.loc[idx, 'S0 Energy velocity (m/ms)']
    phase_velocity_S0 = S0.loc[idx, 'S0 Phase velocity (m/ms)']
    if meter_per_second:
        group_velocity_A0 = group_velocity_A0 * 1000
        phase_velocity_A0 = phase_velocity_A0 * 1000
        group_velocity_S0 = group_velocity_S0 * 1000
        phase_velocity_S0 = phase_velocity_S0 * 1000
    velocities = {'A0': {'group_velocity': group_velocity_A0, 'phase_velocity': phase_velocity_A0},
                  'S0': {'group_velocity': group_velocity_S0, 'phase_velocity': phase_velocity_S0}}
    #print(velocities)
    return velocities

def velocites_modes():
    plate_width = 0.35  # meters
    plate_height = 0.25  # meters
    plate_thickness = 0.007  # meters
    samplerate = 501000  # Hz
    num_samples = 501
    delta_i = plate_height / (num_samples - 1)
    freq = 30000  # Hz

    # Generate example signal data
    wave_top, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(number=9)
    wave_bottom, x_pos_bottom, y_pos_bottom, z_pos_bottom, time_axis_bottom = get_comsol_data(number=10)
    wavelength  = get_velocity_at_freq(freq=freq)['A0']['phase_velocity']/freq
    print(wavelength)   
    distances = np.sqrt((x_pos_bottom[19] - x_pos_bottom[10])**2 + (y_pos_bottom[19] - y_pos_bottom[10])**2)
    print(distances)
    S0 = (wave_top - wave_bottom) / 2
    A0 = (wave_top + wave_bottom) / 2
    plt.plot(S0[19], label='S0_19')
    plt.plot(S0[10], label='S0_10')
    plt.legend()
    plt.show()
    plt.plot(A0[19], label='A0_19')
    plt.plot(A0[10], label='A0_10')
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
    #plot the cut
    plt.plot(S0_cut_19, label='S0_19')
    plt.plot(S0_cut_10, label='S0_10')
    plt.legend()
    plt.show()
    plt.plot(A0_cut_19, label='A0_19')
    plt.plot(A0_cut_10, label='A0_10')
    plt.legend()
    plt.show()
    phase_AO, freq_AO = wp.phase_difference_plot(A0_cut_19, A0_cut_10, SAMPLE_RATE=501000, BANDWIDTH=[0,30000], title=f'A0 Phase difference, distance: {distances} ')
    phase_SO, freq_SO = wp.phase_difference_plot(S0_cut_19, S0_cut_10, SAMPLE_RATE=501000, BANDWIDTH=[0,30000], title=f'S0 Phase difference, distance: {distances} ')
    phase_velocity_A0 = wp.phase_velocity(phase_AO, freq_AO, distance=distances, plot=True, title=f'A0 Phase velocity, distance: {distances} ')
    phase_velocity_S0 = wp.phase_velocity(phase_SO, freq_SO, distance=distances, plot=True, title=f'S0 Phase velocity, distance: {distances} ')
    


    
    


def wave_number_to_omega():
    plate_width = 0.35  # meters
    plate_height = 0.25  # meters
    plate_thickness = 0.007  # meters
    samplerate = 501000  # Hz
    num_samples = 501
    delta_i = plate_height / (num_samples - 1)

    # Generate example signal data
    wave_top, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(number=9)
    wave_bottom, x_pos_bottom, y_pos_bottom, z_pos_bottom, time_axis_bottom = get_comsol_data(number=10)
    
    # Calculate S0 and A0 modes
    S0 = (wave_top - wave_bottom) / 2
    A0 = (wave_top + wave_bottom) / 2
    
    # Perform 2D FFT
    S0_fft = np.fft.fft2(S0)
    A0_fft = np.fft.fft2(A0)

    # Obtain frequency and wavenumber information
    sampling_period = 1 / samplerate
    omega = (2 * np.pi / sampling_period) * np.arange(num_samples)
    k_expected = omega / v_s  # Expected wavenumber using phase velocity (v_s) of S0 mode

    # Calculate wavenumbers from 2D FFT
    k_fft = np.fft.fftshift(np.fft.fftfreq(num_samples, delta_i)) * 2 * np.pi

    # Plot Omega to K (Wavenumber) relationship for S0 mode
    plt.plot(omega, k_expected, label='Expected')
    plt.plot(omega, k_fft, label='FFT')
    plt.xlabel('Angular Frequency (omega)')
    plt.ylabel('Wavenumber (k)')
    plt.title('Omega to K (Wavenumber) Plot - S0 Mode')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def warping_map():
    # Given data points
    fs = 501000
    min_freq = 0
    max_freq = 30000
    x_line_data = [2, 8]
    positions = [12,30]
    file_number = 2
    data, x_pos, y_pos, z_pos, time_axis = get_comsol_data(file_number)
   
    phase, freq = wp.phase_difference_div_improved(data[positions[0]], data[positions[1]], fs, pos_only=True, n_pi=-1)
    #phase = wp.phase_difference_div(data[positions[0]], data[positions[1]], pos_only=True)
    #freq = np.fft.fftfreq(data[positions[0]].size, 1/fs)
    slices = (freq>min_freq) & (freq<max_freq)
    #freq = freq[slices]
    #phase = phase[slices]
    plt.plot(freq, phase)
    plt.xlabel('Frekvens (Hz)')
    plt.ylabel('Faseforskyvning (rad)')
    plt.show()
    if file_number in x_line_data:
        distance = y_pos[positions[1]]-y_pos[positions[0]]
        print('x_line_data')
        print(f'distance is {distance}')
    else:
        distance = x_pos[positions[1]]-x_pos[positions[0]]
        print('y_line_data')
        print(f'distance is {distance}')
    phase_vel = wp.phase_velocity(phase, freq, distance, plot=True)
    phase_velocities_flexural, corrected_phase_velocities, phase_velocity_shear, material = wp.theoretical_velocities(freq, material='LDPE_tonni7mm')
    vg_theoretical = wp.group_velocity_theoretical(freq, material=material)
    vg = wp.group_velocity_phase(phase_vel, freq, distance)
    wavenumber_measured = 2 * np.pi * freq / phase_vel
    wavenumber_theoretical = 2 * np.pi * freq / corrected_phase_velocities
    # Define the normalization parameter
    K_measured = 0.5 * (1 / (np.max(freq) * vg))
    K_theoretical = 0.5 * (1 / (np.max(freq) * vg_theoretical))
    w_inv_measured = freq / phase_vel
    w_inv_theoretical = freq / corrected_phase_velocities
    plt.plot(freq, phase_vel, label='Measured')
    plt.plot(freq, corrected_phase_velocities, label='Theoretical')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase Velocity (m/s)')
    plt.title('Phase Velocity (Unknown Wavenumbers)')
    plt.grid(True)
    plt.legend()
    plt.show()
    warping_map_measured = K_measured * w_inv_measured
    warping_map_theoretical = K_theoretical * w_inv_theoretical
    plt.plot(warping_map_measured, freq, label='Measured')
    plt.plot(warping_map_theoretical, freq, label='Theoretical')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Warping Map')
    plt.title('Warping Map (Unknown Wavenumbers)')
    plt.grid(True)
    plt.legend()
    plt.show()

def get_comsol_data(number=2):
    if number == 1:
        path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\Disp_on_plate_top_case3_15kHz_pulse.txt' #y=0.2 this not acceleration data
        data_nodes = 200
    elif number == 2:
        path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\az_on_plate_top_case3_15kHz_pulse_line3.txt' #x=0.075, starting in the middle of the source
        data_nodes = 50
    elif number == 3:
        path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\az_on_plate_top_case3_15kHz_pulse_line2.txt' #y=0.1
        data_nodes = 50
    elif number == 4:
        path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\az_on_plate_top_case3_15kHz_pulse_line1.txt'#y=0.2
        data_nodes = 50
    elif number == 5:
        path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_bottom_LDPE20mm_15kHz_pulse_line1.txt'#y=0.2
        data_nodes = 50
    elif number == 6:
        path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_top_LDPE20mm_15kHz_pulse_line1.txt' #y=0.2
        data_nodes = 50
    elif number == 7:
        path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_top_LDPE20mm_15kHz_pulse_line2.txt' #y=0.1
        data_nodes = 50
    elif number == 8:
        path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_top_LDPE20mm_15kHz_pulse_line3.txt' #x=0.075
        data_nodes = 50
    elif number == 9:
        path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_top_LDPE20mm_15kHz_pulse_diagonal.txt' #diagonal
        data_nodes = 100
    elif number == 10:
        path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_20mm\az_on_plate_bottom_LDPE20mm_15kHz_pulse_diagonal.txt' #diagonal
        data_nodes = 100

    with open(path, 'r') as f1:
        i = 0
        time_axis = np.linspace(0, 1000, num=501)
        x_pos = np.zeros(data_nodes)
        y_pos = np.zeros(data_nodes)
        z_pos = np.zeros(data_nodes)
        wave_data = np.zeros((data_nodes, 501))
            
        for idx, line in enumerate(f1):
            tmp = line.split()
            if tmp[0] != '%':

                wave_data[i] = tmp[3:]
                x_pos[i] = float(tmp[0])
                y_pos[i] = float(tmp[1])
                z_pos[i] = float(tmp[2])
                i += 1
    return wave_data, x_pos, y_pos, z_pos, time_axis
def comsol_data200():
    path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\Disp_on_plate_top_case3_15kHz_pulse.txt'
    with open(path, 'r') as f1:
        i = 0
        time_axis = np.linspace(0, 1000, num=501)
        x_pos = np.zeros(200)
        y_pos = np.zeros(200)
        z_pos = np.zeros(200)
        wave_data = np.zeros((200, 501))
        xcorr_scale = 0.345/200
        for idx, line in enumerate(f1):
            tmp = line.split()
            if tmp[0] != '%':

                x_index = int(float(tmp[0]) / 0.00173)# convert x coordinate to index. comes from length of plate divided by number of points
                wave_data[x_index] = tmp[3:]
                x_pos[i] = float(tmp[0])
                y_pos[i] = float(tmp[1])
                z_pos[i] = float(tmp[2])
                i += 1
        
    # Plot the data
    plt.plot(wave_data[3])
    plt.show()
    plt.plot(wave_data[199])
    plt.show()
    plt.imshow(wave_data, aspect='auto', cmap='jet', extent=[0, 501, 0, 0.345])
    plt.colorbar()
    plt.title('Wave Through Plate at y=200mm')
    plt.xlabel('Time (samples)')
    plt.ylabel('Position (mm)')
    plt.show()

    #filter data
    wave_3_filtered = filter_general(wave_data[3], 'highpass', 30000)
    wave_100_filtered = filter_general(wave_data[100], 'highpass', 30000)
    plt.plot(wave_3_filtered, label='Filtered data')
    plt.plot(wave_data[3], label='Original data')
    plt.title(label='Filtered data')
    plt.legend()
    plt.show()
    plt.plot(wave_100_filtered, label='Filtered data')
    plt.plot(wave_data[100], label='Original data')
    plt.title(label='Filtered data')
    plt.legend()
    plt.show()

    # Create a 3D figure
    x_index = np.linspace(0, 0.345, num=200)

    # Create meshgrid for x and t
    x, t = np.meshgrid(x_index, np.arange(501)) # Swap x and t
    wave_data = wave_data.T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, t, wave_data, cmap='jet')
    # Set the axis labels and title
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Time (samples)')
    ax.set_zlabel('Amplitude')
    ax.set_title('Wave Through Plate')

    # Set the x and y axis limits
    ax.set_xlim([x_index.min(), x_index.max()])
    ax.set_ylim([0, 500])

    # Set the colorbar
    fig.colorbar(surf)

    # Show the plot
    plt.show()

def comsol_data50():
    path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\az_on_plate_top_case3_15kHz_pulse_line1.txt'
    with open(path, 'r') as f1:
        i = 0
        time_axis = np.linspace(0, 1000, num=501)
        x_pos = np.zeros(50)
        y_pos = np.zeros(50)
        z_pos = np.zeros(50)
        wave_data = np.zeros((50, 501))
        xcorr_scale = 0.345/50
        print(f'xcorr scale is: {xcorr_scale}')
        for idx, line in enumerate(f1):
            tmp = line.split()
            if tmp[0] != '%':

                #x_index = int(float(tmp[0]) / 0.007)# convert x coordinate to index. Had to hardcode the values as the last value gives 50 as index and not 49
                wave_data[i] = tmp[3:]
                x_pos[i] = float(tmp[0])
                y_pos[i] = float(tmp[1])
                z_pos[i] = float(tmp[2])
                i += 1
   
    # Plot the data
    plt.plot(wave_data[3])
    plt.show()
    plt.plot(wave_data[10])
    plt.show()
    plt.imshow(wave_data, aspect='auto', cmap='jet', extent=[0, 501, 0, 0.345])
    plt.colorbar()
    plt.title('Wave Through Plate at y=200mm')
    plt.xlabel('Time (samples)')
    plt.ylabel('Position (mm)')
    plt.show()

    # Create a 3D figure
    x_index = np.linspace(0, 0.345, num=50)

    # Create meshgrid for x and t
    x, t = np.meshgrid(x_index, np.arange(501)) # Swap x and t
    wave_data = wave_data.T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, t, wave_data, cmap='jet')
    # Set the axis labels and title
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Time (samples)')
    ax.set_zlabel('Amplitude')
    ax.set_title('Wave Through Plate')

    # Set the x and y axis limits
    ax.set_xlim([x_index.min(), x_index.max()])
    ax.set_ylim([0, 500])

    # Set the colorbar
    fig.colorbar(surf)

    # Show the plot
    plt.show()

def comsol_data200_phase_diff(idx1=1, idx2=10):
    path = r'C:\Users\vegar\OneDrive - NTNU\NTNU\Masteroppgave\spring2023\tonnidata\LDPE_7mm\Disp_on_plate_top_case3_15kHz_pulse.txt'
    with open(path, 'r') as f1:
        i = 0
        time_axis = np.linspace(0, 1000, num=501)
        x_pos = np.zeros(200)
        y_pos = np.zeros(200)
        z_pos = np.zeros(200)
        wave_data = np.zeros((200, 501))
        
        for idx, line in enumerate(f1):
            tmp = line.split()
            if tmp[0] != '%':

                x_index = int(float(tmp[0]) / 0.00173)# convert x coordinate to index
                wave_data[x_index] = tmp[3:]
                x_pos[i] = float(tmp[0])
                y_pos[i] = float(tmp[1])
                z_pos[i] = float(tmp[2])
                i += 1
    phase = wp.phase_difference_div(wave_data[idx1], wave_data[idx2], pos_only=True)
    plt.plot(wave_data[idx1], label=f'x1={round(x_pos[idx1])}mm')
    plt.plot(wave_data[idx2], label=f'x2={round(x_pos[idx2])}mm')
    plt.legend()
    plt.show()
    freq = np.fft.fftfreq(len(wave_data[idx1]))
    freq = freq[freq > 0]
    plt.plot(freq, phase, label=f'x1={round(x_pos[idx1])}mm and x2={round(x_pos[idx2])}mm')
    plt.legend()
    plt.show()
    phase_vel = wp.phase_velocity(freq, phase, distance=x_pos[idx2] - x_pos[idx1], plot=True)
    

def wave_type_plots():
    df = csv_to_df_thesis('plate10mm\\setup2\\chirp', 'chirp3_ch3top_ch2bot_ch1_sidemid_v1')
    df_no_wave = df.drop(['wave_gen'], axis=1)
    time_axis = np.linspace(0, len(df) // 150e3, num=len(df))
    plt.plot(time_axis, df_no_wave)
    plt.show()

    #plotting just above and under the plate
    plt.plot(time_axis, df['channel 2'], label='channel 2')
    plt.plot(time_axis, df['channel 3'], label='channel 3')
    plt.legend()
    plt.show()

    #filter the signals
    df_filt = filter_general(df_no_wave, filtertype='highpass', cutoff_highpass=100, cutoff_lowpass=15000, order=4)
    plt.plot(time_axis, df_filt)
    plt.show()

    #plotting just above and under the plate
    plt.plot(time_axis, df_filt['channel 2'], label='channel 2')
    plt.plot(time_axis, df_filt['channel 3'], label='channel 3')
    plt.legend()
    plt.show()

    #normalize channel 2 and 3 so that they are on the same scale
    df_filt['channel 2'] = df_filt['channel 2'] / np.max(df_filt['channel 2'])
    df_filt['channel 3'] = df_filt['channel 3'] / np.max(df_filt['channel 3'])
    #plot the difference between the two signals
    plt.plot(time_axis, df_filt['channel 2'] + df_filt['channel 3'], label='difference')
    #plt.plot(time_axis, df_filt['channel 2'], label='channel 2')
    #plt.plot(time_axis, df_filt['channel 3'], label='channel 3')
    plt.legend()
    plt.show()

def show_A0_and_S0_wave_comsol_y0_2(position):
    top_data, x_pos_top, y_pos_top, z_pos_top, time_axis_top = get_comsol_data(6)
    bot_data, x_pos_bot, y_pos_bot, z_pos_bot, time_axis_bot = get_comsol_data(5)
    plt.plot(time_axis_top, top_data[position], label='top raw data')
    plt.plot(time_axis_bot, bot_data[position], label='bottom raw data')
    plt.title(f'Raw data at x={x_pos_top[position]}mm')
    plt.legend()
    plt.show()
    S0 = (top_data[position] + bot_data[position])/2
    A0 = (top_data[position] - bot_data[position])/2
    plt.plot(time_axis_top, S0, label='S0')
    plt.plot(time_axis_top, top_data[position], label='top raw data')
    plt.title(label=f'S0 at x={x_pos_top[position]}mm')
    plt.legend()
    plt.show()
    plt.plot(time_axis_top, A0, label='A0')
    plt.plot(time_axis_top, top_data[position], label='top raw data')
    plt.title(label=f'A0 at x={x_pos_top[position]}mm')
    plt.legend()
    plt.show()
    plt.plot(time_axis_top, A0, label='A0')
    plt.plot(time_axis_top, S0, label='S0')
    plt.title(label=f'A0 and S0 at x={x_pos_top[position]}mm')
    plt.legend()
    plt.show()



def show_S0_wave(normalize=False, plot=True):
    df = csv_to_df_thesis('plate10mm\\setup2\\chirp', 'chirp3_ch3top_ch2bot_ch1_sidemid_v1')
    df_no_wave = df.drop(['wave_gen'], axis=1)
    time_axis = np.linspace(0, len(df) // 150e3, num=len(df))
    if plot:
        plt.plot(time_axis, df_no_wave, labels=['channel 1', 'channel 2', 'channel 3'])
        plt.title('Raw data with all channels')
        plt.legend()
        plt.show()  

        #plotting just above and under the plate
        plt.plot(time_axis, df['channel 2'], label='channel 2')
        plt.plot(time_axis, df['channel 3'], label='channel 3')
        plt.title('Channel 3 is the top plate and channel 2 is the bottom plate')
        plt.legend()
        plt.show()

    #filter the signals
    df_filt = filter_general(df_no_wave, filtertype='highpass', cutoff_highpass=100, cutoff_lowpass=15000, order=4)
    
    if plot:
        #plotting just above and under the plate
        plt.plot(time_axis, df_filt['channel 2'], label='channel 2')
        plt.plot(time_axis, df_filt['channel 3'], label='channel 3')
        plt.title(label='Channel 3 is the top plate and channel 2 is the bottom plate, filtered')
        plt.legend()
        plt.show()

    if normalize:
        #normalize channel 2 and 3 so that they are on the same scale
        df_filt['channel 2'] = df_filt['channel 2'] / np.max(df_filt['channel 2'])
        df_filt['channel 3'] = df_filt['channel 3'] / np.max(df_filt['channel 3'])
    S0 = df_filt['channel 2'] + df_filt['channel 3'] / 2
    if plot:
        plt.plot(time_axis, S0, label='S0')
        plt.title(label='S0')
        plt.legend()
        plt.show()
    return S0


def show_A0_wave(normalize=False, plot=True):
    df = csv_to_df_thesis('plate10mm\\setup2\\chirp', 'chirp3_ch3top_ch2bot_ch1_sidemid_v1')
    df_no_wave = df.drop(['wave_gen'], axis=1)
    time_axis = np.linspace(0, len(df) // 150e3, num=len(df))
    if plot:
        plt.plot(time_axis, df_no_wave, labels=['channel 1', 'channel 2', 'channel 3'])
        plt.title('Raw data with all channels')
        plt.legend()
        plt.show()  

        #plotting just above and under the plate
        plt.plot(time_axis, df['channel 2'], label='channel 2')
        plt.plot(time_axis, df['channel 3'], label='channel 3')
        plt.title('Channel 3 is the top plate and channel 2 is the bottom plate')
        plt.legend()
        plt.show()

    #filter the signals
    df_filt = filter_general(df_no_wave, filtertype='highpass', cutoff_highpass=100, cutoff_lowpass=15000, order=4)
    
    if plot:
        #plotting just above and under the plate
        plt.plot(time_axis, df_filt['channel 2'], label='channel 2')
        plt.plot(time_axis, df_filt['channel 3'], label='channel 3')
        plt.title(label='Channel 3 is the top plate and channel 2 is the bottom plate, filtered')
        plt.legend()
        plt.show()

    if normalize:
        #normalize channel 2 and 3 so that they are on the same scale
        df_filt['channel 2'] = df_filt['channel 2'] / np.max(df_filt['channel 2'])
        df_filt['channel 3'] = df_filt['channel 3'] / np.max(df_filt['channel 3'])
    A0 = df_filt['channel 3'] - df_filt['channel 2'] / 2
    if plot:
        plt.plot(time_axis, A0, label='A0')
        plt.title(label='A0')
        plt.legend()
        plt.show()
    return A0

def show_A0_and_S0_wave(normalize=False):
    S0 = show_S0_wave(normalize=normalize, plot=False)
    A0 = show_A0_wave(normalize=normalize, plot=False)
    time_axis = np.linspace(0, len(S0) // 150e3, num=len(S0))
    plt.plot(time_axis, S0, label='S0')
    plt.plot(time_axis, A0, label='A0')
    plt.title(label='S0 and A0')
    plt.legend()
    plt.show()

def velocities():
    df = csv_to_df_thesis('plate20mm\\setup1\\chirp', 'chirp_100_40000_2s_v1')
    #frequency range from 100 to 40000 Hz samplerate of 150000 Hz
    freqs = np.linspace(100, 40000, 150000)
    
    #wp.plot_theoretical_velocities(freqs)
    phase10, freq10  = wp.phase_plotting_chirp(df,
                channels=['channel 2', 'channel 3'], 
                detrend=True,
                method='div', 
                n_pi=0, 
                SAMPLE_RATE=150000, 
                BANDWIDTH=[100,40000], 
                save_fig=False,
                file_name='phase_difference.png',
                file_format='png',
                figsize=0.75,
                use_recorded_chirp=True)
    wp.plot_velocities(phase10, freq10, 0.10, savefig=False, filename='phase_velocity_10cm.svg', file_format='svg', material='HDPE')

if __name__ == '__main__':
    CROSS_CORR_PATH1 = '\\Measurements\\setup2_korrelasjon\\'
    CROSS_CORR_PATH2 = '\\first_test_touch_passive_setup2\\'
    custom_chirp = csv_to_df(file_folder='Measurements\\div_files', file_name='chirp_custom_fs_150000_tmax_2_100-40000_method_linear', channel_names=CHIRP_CHANNEL_NAMES)
    #set base new base after new measurements:
    #execution_time = timeit.timeit(ccp.SetBase, number=1)
    #print(f'Execution time: {execution_time}')
    #ccp.SetBase(CROSS_CORR_PATH2)
    #ccp.run_test(tolatex=True)
    #ccp.run_test(tolatex=True,data_folder='\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\first_test_touch_passive_setup2\\',filename='results_correlation_old_samples.csv')
    #find position
    #ccp.FindTouchPosition(f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\Measurements\\setup2_korrelasjon\\A2_v3.csv')
    df1 = csv_to_df('Measurements\\setup3_0\\', 'prop_speed_chirp3_setup3_0_v1')
    df = csv_to_df('\\Measurements\\setup10_propagation_speed_15cm\\chirp\\', 'chirp_v1')
    sensor_test_df = csv_to_df('\\Measurements\\sensortest\\rot_clock_123', 'chirp_v1')
    df_touch_10cm = csv_to_df('Measurements\\setup9_propagation_speed_short\\touch\\', 'touch_v1')
    time = np.linspace(0, len(df1) / SAMPLE_RATE, num=len(df1))
    