import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, Button

from objects import Table, Actuator, Sensor
from setups import Setup2, Setup3, Setup3_2, Setup3_4, Setup6
from constants import SAMPLE_RATE, CHANNEL_NAMES, CHIRP_CHANNEL_NAMES

from csv_to_df import csv_to_df
from data_viz_files.visualise_data import compare_signals, plot_vphs, plot_fft, plot_plate_speed_sliders_book, plot_estimated_reflections_with_sliders
from data_processing.preprocessing import crop_data, filter_general, compress_chirp, get_phase_and_vph_of_compressed_signal,cut_out_signal, manual_cut_signal, compress_df_touch
from data_processing.detect_echoes import find_first_peak, get_hilbert_envelope, get_travel_times
from data_processing.find_propagation_speed import find_propagation_speed_with_delay
from data_viz_files.drawing import plot_legend_without_duplicates


def main():
    """CONFIG"""
    CROP = False
    TIME_START = 0.75724  # s
    TIME_END = TIME_START + 0.010  # s
    FILTER = True
    BANDWIDTH = np.array([100, 40000]) #this area the phase velocity is more or less constant, but still differs bwtween the channels
    #SETUP = Setup2()
    #SETUP.draw(savefig=True, filename='setup2_new', fileformat='png')
    # Generate the chirp signal
    # Generate the chirp signal
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=12/10)
    t = np.linspace(0, 1, 1000)
    f0 = 3
    f1 = 30
    k = (f1 - f0) / 1
    signal = np.sin(2 * np.pi * (f0 * t + k * t * t / 2))

    # Compress the signal using matched filtering
    signal_compressed = np.convolve(signal, signal[::-1], mode='same')

    # Plot the original and compressed signals in separate subplots
    plt.figure(figsize=(16, 14))
    
    plt.plot(t, signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.savefig('compression_signal.png', dpi=300, format='png')
    #clear figure
    plt.show()
    plt.clf()
    plt.figure(figsize=(16, 14))
    #ax1.set_title('Original Signal')
    plt.plot(t, signal_compressed)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    #ax2.set_title('Compressed Signal')
    plt.savefig('compression_result.png', dpi=300, format='png')
    #plt.tight_layout()
    plt.show()
    plt.clf()
    # plot the fft of the respective signals with only positive frequencies
    plt.figure(figsize=(16, 14))
    plt.plot(np.fft.rfftfreq(len(signal), 1 / 1000), np.abs(np.fft.rfft(signal)))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.savefig('compression_fft_signal.png', dpi=300, format='png')
    #save as svg
    plt.savefig('compression_fft_signal.svg', dpi=300, format='svg')
    plt.show()
    plt.clf()
    plt.figure(figsize=(16, 14))
    plt.plot(np.fft.rfftfreq(len(signal_compressed), 1 / 1000), np.abs(np.fft.rfft(signal_compressed)))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.savefig('compression_fft_result.png', dpi=300, format='png')
    plt.savefig('compression_fft_result.svg', dpi=300, format='svg')
    plt.show()
    plt.clf()
    #plot the ffts in the same plot with labels "signal" and "compressed signal" and limit the maximum frequency to 100 Hz and normalize the amplitude so they are around the same amplitude
    plt.figure(figsize=(16, 14))
    #plot normalized ffts
    plt.plot(np.fft.rfftfreq(len(signal), 1 / 1000), np.abs(np.fft.rfft(signal))/np.max(np.abs(np.fft.rfft(signal))), label='signal')
    plt.plot(np.fft.rfftfreq(len(signal_compressed), 1 / 1000), np.abs(np.fft.rfft(signal_compressed))/np.max(np.abs(np.fft.rfft(signal_compressed))), label='compressed signal')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.xlim(0, 100)
    plt.legend()
    plt.savefig('compression_fft_both_norm.png', dpi=300, format='png')
    plt.savefig('compression_fft_both_norm.svg', dpi=300, format='svg')
    plt.show()
    exit()

    
    """Open file"""
    # measurements = csv_to_df(file_folder='prop_speed_files/setup3_2',
    #                          file_name='prop_speed_chirp3_setup3_2_v8',
    #                          channel_names=CHIRP_CHANNEL_NAMES)
    measurements = csv_to_df(file_folder='div_files',
                              file_name='prop_speed_fingertouch_hard_setup3_v1',
                              channel_names=['channel 1', 'channel 2', 'channel 3'])
    """Preprocessing"""
    #df1 = measurements.drop(columns=['wave_gen'], axis=1)
    df1 = measurements
    df1.plot()
    plt.show()
    #df1 = measurements
    #create time axis
    #compare_signals(measurements['channel 1'],measurements['channel 2'],measurements['channel 3'])
    
    time = np.linspace(0, len(df1) / SAMPLE_RATE, num=len(df1))
    #plot_fft(df1)
    
    # plt.plot(time, df1)
    # plt.ylabel(ylabel='Amplitude')
    # plt.xlabel(xlabel='Time [s]')
    # plt.legend(df1.columns)
    # plt.grid()
    # plt.show()

    if CROP:
        measurements = crop_data(measurements, TIME_START, TIME_END)
    if FILTER:
        measurements_filt = filter_general(measurements,
                                           filtertype='highpass',
                                           cutoff_highpass=BANDWIDTH[0],
                                           # cutoff_lowpass=BANDWIDTH[1],
                                           order=4)
    else:
        measurements_filt = measurements

    """Compress chirp signals"""
    #print(np.linalg.norm(SETUP.sensor_1.coordinates-SETUP.sensor_3.coordinates))
    #measurements_comp = compress_chirp(measurements_filt, custom_chirp=None)

    measurements_comp = compress_df_touch(measurements_filt, set_threshold_man=True, n_sampl=50)
    # plt.plot(measurements_comp['channel 1'], label='ch1')
    # plt.plot(measurements_comp['channel 2'], label='ch2')
    # plt.legend()
    # plt.show()
    # phase1, vph1, freq1 = get_phase_and_vph_of_compressed_signal(measurements_comp,threshold1=300, threshold2=50,ch1='channel 1', ch2='channel 2', distance=np.linalg.norm(SETUP.sensor_1.coordinates-SETUP.sensor_2.coordinates), bandwidth=BANDWIDTH)
    # phase2, vph2, freq2 = get_phase_and_vph_of_compressed_signal(measurements_comp,threshold1=300, threshold2=150, ch2='channel 3', distance=np.linalg.norm(SETUP.sensor_1.coordinates-SETUP.sensor_3.coordinates), bandwidth=BANDWIDTH)
    # phase3, vph3, freq3 = get_phase_and_vph_of_compressed_signal(measurements_comp,threshold1=50, threshold2=150, ch1='channel 2', ch2='channel 3', distance=np.linalg.norm(SETUP.sensor_2.coordinates-SETUP.sensor_3.coordinates), bandwidth=BANDWIDTH)
    # avg_vph = (np.mean(vph1)+np.mean(vph2)+np.mean(vph3))/3
    # print('Average phase velocity: ', avg_vph)
    # ax = plt.subplot(211)
    # plt.title('phase velocity')
    # plt.plot(freq1, vph1, label=f'channel 1 - channel 2, distance (m): {round(np.linalg.norm(SETUP.sensor_1.coordinates-SETUP.sensor_2.coordinates), 4)}', color='tab:blue')
    # plt.plot(freq2, vph2, label=f'channel 1 - channel 3, distance (m): {round(np.linalg.norm(SETUP.sensor_1.coordinates-SETUP.sensor_3.coordinates), 4)}', color='tab:orange')
    # plt.plot(freq3, vph3, label=f'channel 2 - channel 3, distance (m): {round(np.linalg.norm(SETUP.sensor_2.coordinates-SETUP.sensor_3.coordinates), 4)}', color='tab:green')
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('phase velocity [m/s]')
    # plt.legend()
    # plt.subplot(212, sharex=ax)
    # plt.title('phase')
    # plt.plot(freq1, phase1, label=f'channel 1 - channel 2, distance (m): {round(np.linalg.norm(SETUP.sensor_1.coordinates-SETUP.sensor_2.coordinates),4)}', color='tab:blue')
    # plt.plot(freq2, phase2, label=f'channel 1 - channel 3, distance (m): {round(np.linalg.norm(SETUP.sensor_1.coordinates-SETUP.sensor_3.coordinates), 4)}', color='tab:orange')
    # plt.plot(freq3,phase3, label=f'channel 2 - channel 3, distance (m): {round(np.linalg.norm(SETUP.sensor_2.coordinates-SETUP.sensor_3.coordinates), 4)}', color='tab:green')
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('phase [rad]')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #plot_plate_speed_sliders_book()
    #exit()
    # plt.savefig('phase_and_vph_prop_speed_chirp3_setup3_2_v1.png', dpi=200)
    # #save in vector format
    # plt.savefig('phase_and_vph_prop_speed_chirp3_setup3_2_v1.svg', dpi=200)
    # #save in eps format
    # plt.savefig('phase_and_vph_prop_speed_chirp3_setup3_2_v1.eps', dpi=200)
    # plt.show()


    #get_phase_and_vph_of_compressed_signal(measurements_comp,threshold=1, ch1='channel 2', ch2='channel 3', distance=np.linalg.norm(SETUP.sensor_2.coordinates-SETUP.sensor_3.coordinates), set_thresh_man=True)
    #compare_signals(measurements_comp['channel 1'], measurements_comp['channel 2'], measurements_comp['channel 3'])
    #get_phase_and_vph_of_compressed_signal(measurements_comp,threshold=13,ch2='channel 3', distance=np.linalg.norm(SETUP.sensor_1.coordinates-SETUP.sensor_3.coordinates), set_thresh_man=True, plot=True)
    """Generate Hilbert transforms"""
    #plt.plot(measurements_comp['channel 3'])
    #plt.show()
    #all_ch, dfm = plot_vphs('prop_speed_files\\setup3_0', Setup3(), bandwidth=BANDWIDTH, threshold1=400, threshold2=400, multichannel=True)
    # dfm12 = plot_vphs('prop_speed_files\\setup3_0', Setup3(), bandwidth=BANDWIDTH, threshold1=400, threshold2=400, ch2='channel 2')
    # dfm13 = plot_vphs('prop_speed_files\\setup3_0', Setup3(), bandwidth=BANDWIDTH, threshold1=400, threshold2=400, ch2='channel 3')
    # dfm23 = plot_vphs('prop_speed_files\\setup3_0', Setup3(), bandwidth=BANDWIDTH, threshold1=400, threshold2=400, ch1='channel 2', ch2='channel 3')
    # dfm12['distance'] = np.linalg.norm(SETUP.sensor_1.coordinates-SETUP.sensor_2.coordinates)
    # dfm13['distance'] = np.linalg.norm(SETUP.sensor_1.coordinates-SETUP.sensor_3.coordinates)
    # dfm23['distance'] = np.linalg.norm(SETUP.sensor_2.coordinates-SETUP.sensor_3.coordinates)
    # print('done')
    # df = pd.concat([dfm12, dfm13, dfm23])
    # sns.lineplot(data=df, x='freq', y='vph', hue='distance')
    # plt.show()
    #plt.show()
    measurements_hilb = get_hilbert_envelope(measurements_filt)

    measurements_comp_hilb = get_hilbert_envelope(measurements_comp)
    #plt.plot(measurements_comp_hilb['channel 3'])
    #plt.show()
    # start, end = manual_cut_signal(measurements_comp_hilb['channel 1'])
    # plt.plot(measurements_comp_hilb['channel 1'][start:end])
    # plt.show()
    # #zero pad the signal
    # measurements_comp_hilb_zero_pad = np.zeros(len(measurements_comp_hilb['channel 1']))
    # measurements_comp_hilb_zero_pad[start:end] = measurements_comp_hilb['channel 1'][start:end]
    # #find the mass centrum of measurements_comp_hilb whichs is the expected value of measurements_comp_hilb
    # expected_value = (1/len(measurements_comp_hilb_zero_pad) )*np.sum(measurements_comp_hilb_zero_pad*np.arange(0,len(measurements_comp_hilb_zero_pad)))
    # print(expected_value)
    # #plot expected value on measurements_comp_hilb['channel 1']
    # plt.plot(measurements_comp_hilb['channel 1'])
    # plt.plot(int(expected_value),measurements_comp_hilb['channel 1'][int(expected_value)],'ro')
    # plt.show()
    """Place setup objects"""
    #plot_estimated_reflections_with_sliders(SETUP,measurements_comp)
    #exit()
    SETUP.draw()
    actuator, sensors = SETUP.get_objects()

    # """Calculate wave propagation speed"""
    # prop_speed = SETUP.get_propagation_speed(measurements_comp['channel 1'],
    #                                          measurements_comp['channel 2'])
    # prop_speed *= 1.3
    #print(f'Prop speed: {prop_speed}')
    #SETUP.set_propagation_speed(avg_vph)
    
    #prop_speed = SETUP.propagation_speed
    prop_speed = 603.1585605364801
    print(f'Prop speed: {prop_speed}')
    """Calculate wave arrival times"""
    arrival_times = np.array([])
    for sensor in sensors:
        time, _ = get_travel_times(actuator[0],
                                   sensor,
                                   prop_speed,
                                   ms=False,
                                   print_info=True,
                                   relative_first_reflection=False)
        time = time + 2.5
        arrival_times = np.append(arrival_times, time)
    """Reshape arrival_times to a 2D array with len(sensor) rows"""
    arrival_times = np.reshape(arrival_times, (len(sensors), len(arrival_times) // len(sensors)))

    """Plot the measurements"""
    compare_signals(measurements_comp['channel 1'],
                    measurements_comp['channel 2'],
                    measurements_comp['channel 3'],
                    freq_max=BANDWIDTH[1],
                    nfft=16,
                    sync_time=True)

    """Plot the spectrograms along with lines for expected reflections"""
    dynamic_range_db = 60
    vmin = 10 * np.log10(np.max(measurements_comp['channel 1'])) - dynamic_range_db
    for i, sensor in enumerate(sensors):
        plt.subplot(311 + i, sharex=plt.gca())
        plt.title('Correlation between chirp and channel ' + str(i + 1))
        plt.specgram(measurements_comp['channel ' + str(i + 1)], Fs=SAMPLE_RATE, NFFT=16, noverlap=(16 // 2), vmin=vmin)
        plt.axis(ymax=BANDWIDTH[1])
        plt.title('Spectrogram')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.colorbar()
        #plt.axvline(arrival_times[i][0], linestyle='--', color='r', label='Direct wave')
        #x = [plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])]
        #y = [plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times[i][5:])]
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [V]')
        plot_legend_without_duplicates()
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    """Plot the correlation between the chirp signal and the measured signal"""
    # time_axis_corr = np.linspace(-1000 * len(measurements_comp) / SAMPLE_RATE,
    time_axis_corr = np.linspace(0,
                                 1000 * len(measurements_comp) / SAMPLE_RATE,
                                 (len(measurements_comp)))

    arrival_times *= 1000   # Convert to ms
    for i, sensor in enumerate(sensors):
        plt.subplot(311 + i, sharex=plt.gca())
        plt.title('Correlation between chirp and channel ' + str(i + 1))
        plt.plot(time_axis_corr, measurements_comp['channel ' + str(i + 1)], label='Correlation')
        plt.plot(time_axis_corr, measurements_comp_hilb['channel ' + str(i + 1)], label='Hilbert envelope')
        #plt.axvline(arrival_times[i][0], linestyle='--', color='r', label='Direct wave')
        #[plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])]
        #[plt.axvline(line, linestyle='--', color='purple', label='2nd reflections') for line in (arrival_times[i][5:])]
        #print([plt.axvline(line, linestyle='--', color='g', label='1st reflections') for line in (arrival_times[i][1:5])])
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [V]')
        plot_legend_without_duplicates()
        plt.grid()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    arrival_times /= 1000   # Convert back to s

if __name__ == '__main__':
    main()
