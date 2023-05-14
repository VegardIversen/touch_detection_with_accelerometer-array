import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import pandas as pd
from pathlib import Path
import seaborn as sb
from data_processing.preprocessing import get_first_index_above_threshold, compress_single_touch, compress_chirp, manual_cut_signal
import data_processing.preprocessing as pp
#sb.set_theme(style="darkgrid")
#sb.set(font_scale=12/10)
from data_viz_files.visualise_data import figure_size_setup

def get_estimated_gamma_correction(nu):
    nu_true = np.array([0.2, 0.3, 0.4, 0.5])
    gamma_true = np.array([0.689, 0.841, 0.919, 0.955])

    # Interpolate to estimate gamma for nu = 0.45
    
    gamma_interp = np.interp(nu, nu_true, gamma_true)

    # Print the estimated value of gamma at nu = 0.45
    print(f"Estimated value of gamma at nu = {nu}: {gamma_interp}")
    return gamma_interp


def phase_difference_sub(sig1, sig2):
    """
    Calculates the phase difference between two signals using the FFT method.
    :param sig1: First signal
    :param sig2: Second signal
    :return: Phase difference between the two signals
    """
    sig1_fft = np.fft.fft(sig1)
    sig2_fft = np.fft.fft(sig2)
    sig1_fft = np.fft.fftshift(sig1_fft)
    sig2_fft = np.fft.fftshift(sig2_fft)
    sig1_fft = np.abs(sig1_fft)
    sig2_fft = np.abs(sig2_fft)
    sig1_fft = sig1_fft/np.max(sig1_fft)
    sig2_fft = sig2_fft/np.max(sig2_fft)
    sig1_fft = np.log(sig1_fft)
    sig2_fft = np.log(sig2_fft)
    sig1_fft = np.unwrap(np.angle(sig1_fft))
    sig2_fft = np.unwrap(np.angle(sig2_fft))
    phase_diff = sig1_fft - sig2_fft
    return phase_diff

def phase_difference_div_improved(sig1, sig2, fs, n_pi=0, improved_fft=True, pos_only=False, min_freq=0, max_freq=30000):

    S1f, freq, _ = pp.improved_fft(sig1, fs=fs, interpolation_factor=8, methods=['zero_padding', 'windowing', 'interpolation'])
    S2f, freq, _  = pp.improved_fft(sig2, fs=fs, interpolation_factor=8, methods=['zero_padding', 'windowing', 'interpolation'])
    
    if pos_only:
        #freq = np.fft.fftfreq(len(sig1), 1/fs)
        slices = (freq > min_freq) & (freq < max_freq)
        S1f = S1f[slices]
        S2f = S2f[slices]
        freq = freq[slices]
        plt.plot(freq, np.abs(S1f), label='S1f')
        plt.plot(freq, np.abs(S2f), label='S2f')
        plt.xlabel(xlabel='Frequency (Hz)')
        plt.ylabel(ylabel='Amplitude')
        plt.legend()
        plt.show()

    phase = np.unwrap(np.angle(S2f/S1f)) +n_pi*np.pi
    return phase, freq

def phase_difference_div(sig1, sig2,fs=501000, n_pi=0, pos_only=False, freq_min=0, freq_max=30000):
   
    S1f = np.fft.fft(sig1)
    S2f = np.fft.fft(sig2)
    if pos_only:
        freq = np.fft.fftfreq(len(sig1), 1/fs)
        slices = (freq > freq_min) & (freq < freq_max) 
        S1f = S1f[slices]
        S2f = S2f[slices]
        freq = freq[slices]
        plt.plot(freq, np.abs(S1f), label='S1f')
        plt.plot(freq, np.abs(S2f), label='S2f')
        plt.legend()
        plt.show()

    phase = np.unwrap(np.angle(S2f/S1f)) +n_pi*np.pi
    return phase

def phase_difference_gpt(sig1, sig2):
    fft_sig1 = np.fft.fft(sig1)
    fft_sig2 = np.fft.fft(sig2)

    # Find the phase angles of the FFTs of the two pulse compressed signals
    phase_angles_sig1 = np.angle(fft_sig1)
    phase_angles_sig2 = np.angle(fft_sig2)

    # Use the unwrap function to remove any jumps in phase between consecutive elements
    unwrapped_phase_angles_sig1 = np.unwrap(phase_angles_sig1)
    unwrapped_phase_angles_sig2 = np.unwrap(phase_angles_sig2)

    # Calculate the phase difference between the two pulse compressed signals
    phi = unwrapped_phase_angles_sig2 - unwrapped_phase_angles_sig1

    return phi

def phase_difference(sig1, sig2, method='div', n_pi=0):
    if method == 'sub':
        return phase_difference_sub(sig1, sig2)
    elif method == 'div':
        return phase_difference_div(sig1, sig2, n_pi=n_pi)
    elif method == 'gpt':
        return phase_difference_gpt(sig1, sig2)
    else:
        raise ValueError('Invalid method')

def phase_difference_plot(
                        sig1, 
                        sig2, 
                        method='div', 
                        n_pi=0, 
                        SAMPLE_RATE=150000, 
                        BANDWIDTH=None, 
                        save_fig=False,
                        file_name='phase_difference.png',
                        file_format='png',
                        figsize=0.75,
                        title=None):
    phase_diff = phase_difference(sig1, sig2, method=method, n_pi=n_pi)
    
    freq = np.fft.fftfreq(len(sig1), 1/SAMPLE_RATE)
    #freq = np.fft.fftshift(freq)
    fig, axs = figure_size_setup(figsize)
    if BANDWIDTH is not None:
        slices = (freq>BANDWIDTH[0]) & (freq<BANDWIDTH[1])
        phase_diff = phase_diff[slices]
        freq = freq[slices]
    axs.plot(freq, phase_diff)
    if title is not None:
        axs.set_title(title)
    axs.set_xlabel('Frequency [Hz]')
    axs.set_ylabel('Phase difference [rad]')
    if save_fig:
        fig.savefig(file_name, format=file_format)
    plt.show()
    plt.clf()
    
    return phase_diff, freq
    
def preprocess_df(df, detrend=True):
    if detrend:
        for col in df.columns:
            df[col] = signal.detrend(df[col])
    return df

def compress_and_cut_df_touch(df, channels=['channel 1', 'channel 3'], direct_samples=55):
    compressed_df = df.copy()
    for ch in channels:
        compressed_df[ch] = compress_single_touch(df[ch], set_threshold_man=True, n_sampl=direct_samples)
    return compressed_df

def plot_results(df, channels=['channel 1', 'channel 3'], direct_samples=55, detrend=True, method='gpt', n_pi=0, SAMPLE_RATE=150000, BANDWIDTH=None, chirp=None):
    df = preprocess_df(df, detrend=detrend)
    if chirp is not None:
        df = compress_chirp(df, chirp)
    df = compress_and_cut_df_touch(df, channels=channels, direct_samples=direct_samples)
    phase_difference_plot(df[channels[0]], df[channels[1]], method=method, n_pi=n_pi, SAMPLE_RATE=SAMPLE_RATE, BANDWIDTH=BANDWIDTH)

def phase_plotting(
                df,
                channels=['channel 1', 'channel 3'], 
                detrend=True, chirp=None, method='div', 
                n_pi=0, 
                SAMPLE_RATE=150000, 
                BANDWIDTH=None, 
                use_recorded_chirp=True, 
                start_stops=None, 
                duration_cut=55,
                save_fig=False,
                file_name='phase_difference.png',
                file_format='png',
                figsize=0.75,
                threshold1=400,
                threshold2=400,):
    df = preprocess_df(df, detrend=detrend)
    # check if dataframe has a column with name wave_gen
    if 'wave_gen' in df.columns:
        chirp = df['wave_gen'].to_numpy()
    # if save_fig:
    #     #plot channel 1
    #     fig1=plt.Figure(figsize=(16, 14))
    #     plt.plot(df[channels[0]])
    #     plt.xlabel('Samples')
    #     plt.ylabel('Amplitude')
    #     plt.savefig('sig'+file_name, format=file_format,dpi=300)
    #     plt.show()
        
    if start_stops is not None:
        start1, end1 = start_stops[0]
        start2, end2 = start_stops[1]
    else:
        start1,end1 = manual_cut_signal(signal=df[channels[0]])
        start2,end2 = manual_cut_signal(signal=df[channels[1]])
    #empty array to fill
    temp_arr = np.zeros((len(df[channels[0]]),len(channels)))
    #filling the array
    temp_arr[start1:end1,0] = df[channels[0]].iloc[start1:end1]
    temp_arr[start2:end2,1] = df[channels[1]].iloc[start2:end2]
    df_sig_only = pd.DataFrame(temp_arr, columns=channels)
    compressed_df = compress_chirp(df_sig_only, chirp, use_recorded_chirp=use_recorded_chirp)
    plt.Figure(figsize=(16, 14))
    plt.plot(compressed_df[channels[0]], label=channels[0])
    plt.plot(compressed_df[channels[1]], label=channels[1])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    #plt.savefig('sig'+file_name, format=file_format,dpi=300)
    plt.show()
    
    
    start_index_ch1 = get_first_index_above_threshold(compressed_df[channels[0]], threshold1)
    end_index_ch1 = start_index_ch1 + duration_cut
    start_index_ch2 = get_first_index_above_threshold(compressed_df[channels[1]], threshold2)
    end_index_ch2 = start_index_ch2 + duration_cut
    cut_ch1 = compressed_df[channels[0]][start_index_ch1:end_index_ch1]
    cut_ch2 = compressed_df[channels[1]][start_index_ch2:end_index_ch2]
    #add window to signal
    window = np.hamming(duration_cut)
    cut_ch1_win = cut_ch1 * window
    cut_ch2_win = cut_ch2 * window

    #plot the cut signals
    fig2=plt.Figure(figsize=(16, 14))
    plt.plot(cut_ch1_win, label=channels[0])
    plt.plot(cut_ch2_win, label=channels[1])
    # if save_fig:
    #     plt.savefig('cut'+file_name, format=file_format, dpi=300)
    plt.legend()
    plt.show()
    
    s1t = np.zeros(len(compressed_df[channels[0]]))
    s1t[start_index_ch1:end_index_ch1] = cut_ch1_win
    s2t = np.zeros(len(compressed_df[channels[1]]))
    s2t[start_index_ch2:end_index_ch2] = cut_ch2_win
    
    
    phase, freq = phase_difference_plot(
                                    s1t, 
                                    s2t, 
                                    method=method, 
                                    n_pi=n_pi, 
                                    SAMPLE_RATE=SAMPLE_RATE, 
                                    BANDWIDTH=BANDWIDTH, 
                                    save_fig=save_fig,
                                    file_name=file_name,
                                    file_format=file_format,
                                    figsize=figsize)
    return phase, freq


def phase_plotting_chirp(
                df,
                channels=['channel 1', 'channel 3'], 
                detrend=True,
                method='div', 
                n_pi=0, 
                SAMPLE_RATE=150000, 
                BANDWIDTH=[100,40000], 
                save_fig=False,
                file_name='phase_difference.png',
                file_format='png',
                figsize=0.75,
                use_recorded_chirp=True):
    df = preprocess_df(df, detrend=detrend)
    # check if dataframe has a column with name wave_gen
    if 'wave_gen' in df.columns:
        chirp = df['wave_gen'].to_numpy()
 
        
    df_sig_only = df.drop(columns=['wave_gen'])
    compressed_df = compress_chirp(df_sig_only, chirp, use_recorded_chirp=use_recorded_chirp)
    plt.Figure(figsize=(16, 14))
    plt.plot(compressed_df[channels[0]], label=channels[0])
    plt.plot(compressed_df[channels[1]], label=channels[1])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    #plt.savefig('sig'+file_name, format=file_format,dpi=300)
    plt.show()
    
    
    
    s1t = compressed_df[channels[0]]

    s2t = compressed_df[channels[1]]
    
    phase, freq = phase_difference_plot(
                                    s1t, 
                                    s2t, 
                                    method=method, 
                                    n_pi=n_pi, 
                                    SAMPLE_RATE=SAMPLE_RATE, 
                                    BANDWIDTH=BANDWIDTH, 
                                    save_fig=save_fig,
                                    file_name=file_name,
                                    file_format=file_format,
                                    figsize=figsize)
    return phase, freq

def theoretical_group_phase_vel(freqs, material='teflon', plot=False, kind='cubic'):
    omega  = 2*np.pi*freqs
    phase_velocities_flexural, corrected_phase_velocities, phase_velocity_shear, material = theoretical_velocities(freqs, material=material)
    #univ_s = interpolate.InterpolatedUnivariateSpline(freqs, phase_velocities_flexural)
    #vp_prime = univ_s.derivative()     
    #vg = np.square(phase_velocities_flexural) * (1/(freqs - vp_prime(freqs)*freqs))
    #interp_vg = interpolate.interp1d(freqs, vg, kind=kind)


    d_v_ph = np.gradient(corrected_phase_velocities, omega)
    v_g_numpy = corrected_phase_velocities / (1 - (d_v_ph * omega/corrected_phase_velocities))
    v_g_numpy[np.isnan(v_g_numpy)] = 0

    if plot:
        plt.plot(freqs, v_g_numpy, label='Group velocity')
        #plt.plot(freqs, interp_vg, label='Interpolated')
        plt.plot(freqs, corrected_phase_velocities, label='Phase velocity' )
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('velocity [m/s]')
        plt.legend()
        plt.show()

    return v_g_numpy, corrected_phase_velocities

def group_velocity_theoretical(freqs, material='teflon', plot=False, kind='cubic'):

    
    omega  = 2*np.pi*freqs
    phase_velocities_flexural, corrected_phase_velocities, phase_velocity_shear, material = theoretical_velocities(freqs, material=material)

    univ_s = interpolate.InterpolatedUnivariateSpline(freqs, phase_velocities_flexural)
    vp_prime = univ_s.derivative()     
    vg = np.square(phase_velocities_flexural) * (1/(freqs - vp_prime(freqs)*freqs))
    interp_vg = interpolate.interp1d(freqs, vg, kind=kind)


    d_v_ph = np.gradient(corrected_phase_velocities, omega)
    v_g_numpy = corrected_phase_velocities / (1 - (d_v_ph * omega/corrected_phase_velocities))
    if plot:
        plt.plot(freqs, v_g_numpy, label='Numpy')
        plt.plot(freqs, vg, label='Interpolated')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Group velocity [m/s]')
        plt.legend()
        plt.show()
    return v_g_numpy


def group_velocity_phase(vph, freqs, method='t', distance=0.1, material='teflon', plot=False):
    '''
    Function for calulating the group velocity of a signal.
    phase is the phase of the signal. freqs is the frequency range you are interested in.
    method is either 't' for theoretical or 'e' experimental. 
    Material is important to define if you want to use the theoretical method.

    '''
    omega  = 2*np.pi*freqs
    d_v_ph = np.gradient(vph, omega)
    v_g_numpy = vph / (1 - (d_v_ph * omega/vph))
    v_g_numpy[np.isnan(v_g_numpy)] = 0

    if plot:
        plt.plot(freqs, v_g_numpy)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Group velocity [m/s]')
        plt.show()
    return v_g_numpy





def phase_velocity(phase, freq, distance, plot=False, title=None):
    phase_vel = 2*np.pi*freq*distance/np.abs(phase)
    if plot:
        fig3=plt.Figure(figsize=(16, 14))
        plt.plot(freq, phase_vel)
        if title is not None:
            plt.title(title)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase velocity [m/s]')
        plt.show()
        
    return phase_vel

def theoretical_velocities(freq, material='teflon'):
    #values dispersion calculator
    if material == 'teflon':
        poisson_ratio = 0.46
        youngs_modulus = 0.475e9
        plate_thickness = 0.01
        density = 2170
        correction_factor = get_estimated_gamma_correction(poisson_ratio)
    elif material == 'HDPE':
        poisson_ratio = 0.361337
        youngs_modulus = 3.49897e9
        plate_thickness = 0.02
        density = 940
        correction_factor = get_estimated_gamma_correction(poisson_ratio)
    elif material == 'LDPE':
        poisson_ratio = 0.348275
        youngs_modulus = 3.82535e9
        plate_thickness = 0.02
        density = 910
        correction_factor = get_estimated_gamma_correction(poisson_ratio)
        print('LDPE')
    elif material == 'LDPE_tonni7mm':
        density = 920 
        poisson_ratio = 0.45
        youngs_modulus = 1e9
        plate_thickness = 0.007
        correction_factor = get_estimated_gamma_correction(poisson_ratio)
    elif material == 'LDPE_tonni20mm':
        density = 920 
        poisson_ratio = 0.45
        youngs_modulus = 1e9
        plate_thickness = 0.02
        correction_factor = get_estimated_gamma_correction(poisson_ratio)

    # plate_thickness = 0.02  # m
    # youngs_modulus = 3.8 * 10 ** 9  # Pa
    # density = (650 + 800) / 2  # kg/m^3
    # poisson_ratio = 0.2  # -
    #plate_thickness = 0.01  # m
    #youngs_modulus = 5.52e8  # Pa
    #youngs_modulus = 1e9  # Pa
    #density = 2.2e3  # kg/m^3
    #density = 950  # kg/m^3
    #poisson_ratio = 0.42  # -
    phase_velocity_longitudinal = np.sqrt(youngs_modulus /
                                          (density * (1 - poisson_ratio ** 2)))
    phase_velocity_shear = (phase_velocity_longitudinal *
                            np.sqrt((1 - poisson_ratio) / 2))
    phase_velocities_flexural = np.sqrt(1.8 * phase_velocity_longitudinal *
                                        plate_thickness * freq)


    # group_velocity_flexural = (2 * np.sqrt(2 * np.pi * freq) *
    #                            (youngs_modulus) ** (1 / 4))
    """As the plate is not considered 'thin', we need to correct the
    calculated velocities with correction factor depending on the
    Poisson ratio:
    poisson_ratio = 0.2: correction_factor = 0.689
    poisson_ratio = 0.3: correction_factor = 0.841
    (according to Vigran's 'Building acoustics').
    """
    #correction_factor = 0.689
    c_G = phase_velocity_shear  # mysterious factor that the source doesnt explain
    corrected_phase_velocities = (1 /
                                  ((1 / (phase_velocities_flexural ** 3)) +
                                   (1 / ((correction_factor ** 3) *
                                    (c_G ** 3))))) ** (1 / 3)
    corrected_phase_velocities[np.isnan(corrected_phase_velocities)] = 0

    return phase_velocities_flexural, corrected_phase_velocities, phase_velocity_shear, material

def plot_theoretical_velocities(freq, material='teflon'):
    phase_velocities_flexural, corrected_phase_velocities, phase_velocity_shear, material = theoretical_velocities(freq,material)
    group_vel = group_velocity_theoretical(freq)
    fig, axs = figure_size_setup()
    axs.plot(freq, phase_velocities_flexural, label='Simulated velocity')
    axs.plot(freq, corrected_phase_velocities, label='Simulated corrected velocity')
    #axs.plot(freq, phase_velocity_shear, label='Simulated shear velocity')
    axs.plot(freq, group_vel, label='Simulated group velocity')
    axs.set_xlabel('Frequency [Hz]')
    axs.set_ylabel('Velocity [m/s]')
    axs.set_title('Theoretical velocities for ' + material)
    axs.legend()
    plt.show()

    return axs

def plot_velocities(phase, freq, distance, savefig=False, filename=None, file_format='png', material='teflon'):
    phase_vel = phase_velocity(phase, freq, distance)
    phase_velocities_flexural, corrected_phase_velocities, phase_velocity_shear, material = theoretical_velocities(freq,material)
    vg_theoretical = group_velocity_theoretical(freq, material=material)
    vg = group_velocity_phase(phase_vel, freq, distance)
    wavenumbers_measured = 2*np.pi*freq/phase_vel
    wavenumbers_theoretical = 2*np.pi*freq/corrected_phase_velocities
    freq = freq/1000
    fig, axs = figure_size_setup()
    #axs.plot(freq, phase_vel, label='Measured velocity')
    axs.plot(freq, phase_velocities_flexural, label='Simulated velocity', linestyle='--')
    axs.plot(freq, corrected_phase_velocities, label='Simulated corrected velocity', linestyle='--')
    #axs.plot(freq, vg, label='Measured group velocity')
    axs.plot(freq, vg_theoretical, label='Simulated group velocity', linestyle='--')
    axs.set_xlabel('Frequency [kHz]')
    axs.set_ylabel('Phase velocity [m/s]')
    axs.set_title(f'Phase velocity for {material} plate')
    axs.legend()
    if savefig:
        fig.savefig(filename, format=file_format, dpi=300)
    plt.show()
    fig, axs = figure_size_setup()
    #axs.plot(freq, phase_vel, label='Measured velocity')
    axs.plot(freq, wavenumbers_measured, label='Simulated wavenumber', linestyle='--')
    axs.plot(freq, wavenumbers_theoretical, label='theoretical wavenumber', linestyle='--')
    axs.set_xlabel('Frequency [kHz]')
    axs.set_ylabel('wavenumber')
    axs.set_title(f'Wavenumbers for {material} plate')
    axs.legend()
    plt.show()
    if savefig:
        filename = filename[:-4] + '_wavenumber' + filename[-4:]
        fig.savefig(filename, format=file_format, dpi=300)
    


def max_peak_velocity(df, distance=0.1, sampling_rate=150000, material='teflon'):
    #Test indicates that the pressure wave travels with 1000 m/s for teflon and 1500 m/s for PE
    # Extract the signal data from the DataFrame
    signal1 = df.iloc[:, 2].values
    signal2 = df.iloc[:, 1].values
    if material == 'teflon':

        # Find peaks in signal 1
        peaks1, _ = signal.find_peaks(signal1, threshold=0.007, prominence=0.005)
        # Find peaks in signal 2
        peaks2, _ = signal.find_peaks(signal2, threshold=0.007, prominence=0.005)
    elif material == 'HDPE' or material == 'LDPE':
        # Find peaks in signal 1
        peaks1, _ = signal.find_peaks(signal1, threshold=0.0015)
        # Find peaks in signal 2
        peaks2, _ = signal.find_peaks(signal2, threshold=0.0015)

    # Plot the signals with the detected peaks
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.suptitle('Detected Peaks')
    ax[0].plot(signal1)
    ax[0].plot(peaks1, signal1[peaks1], "x")
    ax[0].set_ylabel('Signal 1')
    ax[1].plot(signal2)
    ax[1].plot(peaks2, signal2[peaks2], "x")
    ax[1].set_ylabel('Signal 2')
    plt.show()
    
    # Find the closest pair of peaks
    
    t1 = peaks1[0] / sampling_rate  # Convert sample index to time in seconds
    t2 = peaks2[0] / sampling_rate
    
    # Calculate the time difference between the two peaks
    dt = t2 - t1
    
    # Calculate the velocity
    velocity = distance / dt

    return velocity

def plot_velocities_2distance(phase1, freq1, distance1, phase2, freq2, distance2, savefig=False, filename=None, file_format='png'):
    phase_vel1 = phase_velocity(phase1, freq1, distance1)
    phase_vel2 = phase_velocity(phase2, freq2, distance2)
    phase_velocities_flexural, corrected_phase_velocities, phase_velocity_shear = theoretical_velocities(freq1)
    
    freq1 = freq1/1000
    fig, axs = figure_size_setup()
    axs.plot(freq1, phase_vel1, label=f'Measured velocity d={distance1}m')
    axs.plot(freq1, phase_vel2, label=f'Measured velocity d={distance2}m')
    axs.plot(freq1, phase_velocities_flexural, label='Simulated velocity', linestyle='--')
    axs.plot(freq1, corrected_phase_velocities, label='Simulated corrected velocity', linestyle='--')
    axs.set_xlabel('Frequency [kHz]')
    axs.set_ylabel('Phase velocity [m/s]')
    axs.legend()
    if savefig:
        fig.savefig(filename, format=file_format, dpi=300)
    plt.show()