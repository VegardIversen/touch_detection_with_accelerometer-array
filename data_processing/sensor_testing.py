import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from pathlib import Path
import seaborn as sb
#sb.set_theme(style="darkgrid")
sb.set(font_scale=12/10)
from data_viz_files.visualise_data import figure_size_setup

def time_domain_analysis(df, ch, Fs=150000):
    # Compute the autocorrelation of the sensor data and chirp signal
    sensor_data = df[ch]
    chirp_signal = df['wave_gen']
    acorr_sensor = signal.correlate(sensor_data, sensor_data, mode='same')
    acorr_chirp = signal.correlate(chirp_signal, chirp_signal, mode='same')
    dt = 1/Fs
    # Compute the cross-correlation of the sensor data and chirp signal
    xcorr = signal.correlate(sensor_data, chirp_signal, mode='same')

    # Compute the time values corresponding to the correlation results
    time = np.arange(-len(sensor_data)/2, len(sensor_data)/2) * dt

    plt.figure()
    plt.plot(time, acorr_sensor)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Autocorrelation of sensor data')

    plt.figure()
    plt.plot(time, acorr_chirp)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Autocorrelation of chirp signal')

    plt.figure()
    plt.plot(time, xcorr)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Cross-correlation of sensor data and chirp signal')

    plt.show()

def fft_analysis_plot(df, ch, Fs=150000):
    sensor_data = df[ch]
    chirp_signal = df['wave_gen']
    #detrend df to remove linear trend
    sensor_data = signal.detrend(sensor_data)
    chirp_signal = signal.detrend(chirp_signal)
    fft_sensor = np.fft.fft(sensor_data)
    fft_chirp = np.fft.fft(chirp_signal)

    fft_sensor_centered = np.fft.fftshift(fft_sensor)
    fft_chirp_centered = np.fft.fftshift(fft_chirp)

    # Compute the frequency values corresponding to the FFT results
    freq_sensor = np.fft.fftfreq(len(sensor_data), d=1.0/Fs)
    freq_chirp = np.fft.fftfreq(len(chirp_signal), d=1.0/Fs)

    # Shift the frequency values to match the zero-centered FFT results
    freq_sensor_centered = np.fft.fftshift(freq_sensor)
    freq_chirp_centered = np.fft.fftshift(freq_chirp)

    # Plot the FFT results in dB scale
    plt.figure()
    plt.plot(freq_sensor_centered, 20*np.log10(np.abs(fft_sensor_centered)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('FFT of sensor data')

    plt.figure()
    plt.plot(freq_chirp_centered, 20*np.log10(np.abs(fft_chirp_centered)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('FFT of chirp signal')

    plt.show()

def fft_analysis_all_sensors(folder, n_runs=1, position=-3, Fs=150000, plot_chirp_fft=False, savefig=False, file_format='png', channel_names=['channel 1', 'channel 2', 'channel 3','wave_gen'], bandwidth=None):
    '''
    We have a sensor test where 3 sensors are placed in a equilateral triangle with a distance of x between them.
    split into 3 folders, one for each rotation type. A rotation type is where we placed the sensors in a certain way.
    for example the first rotation type "123" means that the sensors are placed in the order 1,2,3.
    the second rotation type "231" means that now sensor 2 is placed where sensor 1 was, sensor 3 is placed where sensor 2 was and sensor 1 is placed where sensor 3 was. and so on. 

    The reason for this switch is to remove dependency in the plate. 

    position -3 is the left bottom sensor, position -2 is the middletop sensor and position -1 is the rightbottom sensor.

    '''
    if position not in [-3, -2, -1]:
        raise ValueError('position must be -3, -2 or -1')

    ROOT_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
    folder_path = f'{ROOT_FOLDER}\\{folder}\\'
    # Get a list of the directories in the directory
    dir_path = Path(folder_path)
    rotation_type_dirs = list(dir_path.glob('*'))
    fig, axs = figure_size_setup()
    if bandwidth is not None:
        min_freq = bandwidth[0]
        max_freq = bandwidth[1]
    else:
        min_freq = 0
        max_freq = 50000
    # Iterate over the rotation type directories
    sensor_position = {-3: '1', -2: '2', -1: '3', 0: 'no_chirp'}
    for idx, rotation_type_dir in enumerate(rotation_type_dirs):
        # Get a list of the files in the rotation type directory
        file_list = list(rotation_type_dir.glob('*.csv'))
        file_name = rotation_type_dir.name
        print(rotation_type_dir.name)
        channel_to_use = int(file_name[position])-1
        ch = channel_names[channel_to_use]
        
        #Iterate over the first n files in the list
        for jdx, file in enumerate(file_list[:n_runs]):
            # Read the data from the file
            df = pd.read_csv(filepath_or_buffer=file, names=channel_names)
            #take the fft of the entire dataframe
            df[ch] = signal.detrend(df[ch])
            fft_sensor = np.fft.fft(df[ch])
            # Compute the frequency values corresponding to the FFT results
            freq_axis = np.fft.fftfreq(len(df), d=1.0/Fs)
            # Shift the frequency values to match the zero-centered FFT results
            fft_sensor_centered = np.fft.fftshift(fft_sensor)
            freq_axis_centered = np.fft.fftshift(freq_axis)
            #ignore negative frequencies
            freq_axis_plot = freq_axis_centered[(freq_axis_centered>min_freq) & (freq_axis_centered<max_freq)]
            fft_sensor_plot = fft_sensor_centered[(freq_axis_centered>min_freq) & (freq_axis_centered<max_freq)]
            # plot the fft of the data from the 1st sensor + the nuber of different rotations types in the same plot
            axs.plot(freq_axis_plot/1000, 20*np.log10(np.abs(fft_sensor_plot)), label=f'{ch} pos_({sensor_position[position]})')
    if plot_chirp_fft:
        #take the fft of the chirp signal
        sensor_position[0] = 'chirp'
        df['wave_gen'] = signal.detrend(df['wave_gen'])
        fft_chirp = np.fft.fft(df['wave_gen'])

        # Compute the frequency values corresponding to the FFT results
        freq_axis = np.fft.fftfreq(len(df), d=1.0/Fs)
        # Shift the frequency values to match the zero-centered FFT results
        fft_chirp_centered = np.fft.fftshift(fft_chirp)
        freq_axis_centered = np.fft.fftshift(freq_axis)
        #ignore negative frequencies
        freq_axis_plot = freq_axis_centered[(freq_axis_centered>min_freq) & (freq_axis_centered<max_freq)]
        fft_chirp_plot = fft_chirp_centered[(freq_axis_centered>min_freq )& (freq_axis_centered<max_freq)]
        #plot
        axs.plot(freq_axis_plot/1000, 20*np.log10(np.abs(fft_chirp_plot)), label='chirp')

    axs.set_xlabel('Frequency (kHz)')
    axs.set_ylabel('Amplitude (dB)')
    #plt.title(f'Sensor test')
    axs.legend(loc='upper right')
    if savefig:
        fig.savefig(f'sensortest_nfiles_{n_runs}_{sensor_position[position]}_{sensor_position[0]}.{file_format}', format=file_format, dpi=300)
    plt.show()

def fft_analysis_all_sensors_compare(folder, n_runs=1, positions=[-3, -2, -1], Fs=150000, plot_chirp_fft=False, savefig=False, file_format='png', channel_names=['channel 1', 'channel 2', 'channel 3','wave_gen'], bandwidth=None):
    '''
    We have a sensor test where 3 sensors are placed in a equilateral triangle with a distance of x between them.
    split into 3 folders, one for each rotation type. A rotation type is where we placed the sensors in a certain way.
    for example the first rotation type "123" means that the sensors are placed in the order 1,2,3.
    the second rotation type "231" means that now sensor 2 is placed where sensor 1 was, sensor 3 is placed where sensor 2 was and sensor 1 is placed where sensor 3 was. and so on. 

    The reason for this switch is to remove dependency in the plate. 

    position -3 is the left bottom sensor, position -2 is the middletop sensor and position -1 is the rightbottom sensor.

    '''
    if not isinstance(positions, list) or not all(x in [-3, -2, -1] for x in positions):
        raise ValueError('positions must be a list containing only the values -3, -2, or -1')

    ROOT_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
    folder_path = f'{ROOT_FOLDER}\\{folder}\\'
    # Get a list of the directories in the directory
    dir_path = Path(folder_path)
    rotation_type_dirs = list(dir_path.glob('*'))
    fig, axs = figure_size_setup()
    if bandwidth is not None:
        min_freq = bandwidth[0]
        max_freq = bandwidth[1]
    else:
        min_freq = 0
        max_freq = 50000
    # Iterate over the rotation type directories
    for position in positions:
        # Get the channel indices corresponding to the position
        channel_indices = [int(rotation_type_dir.name[position])-1 for rotation_type_dir in rotation_type_dirs]
        # Get the channel names corresponding to the channel indices
        channel_names_to_use = [channel_names[idx] for idx in channel_indices]

        # Iterate over the rotation type directories
        sensor_position = {-3: 'left_bottom', -2: 'middle_top', -1: 'right_bottom', 0: 'no_chirp'}
        for idx, rotation_type_dir in enumerate(rotation_type_dirs):
            # Get a list of the files in the rotation type directory
            file_list = list(rotation_type_dir.glob('*.csv'))
            file_name = rotation_type_dir.name
            print(rotation_type_dir.name)
            # Use the channel names corresponding to the current position
            ch = channel_names_to_use[idx]
            
            #Iterate over the first n files in the list
            for jdx, file in enumerate(file_list[:n_runs]):
                # Read the data from the file
                df = pd.read_csv(filepath_or_buffer=file, names=channel_names)
                #take the fft of the entire dataframe
                df[ch] = signal.detrend(df[ch])
                fft_sensor = np.fft.fft(df[ch])
                # Compute the frequency values corresponding to the FFT results
                freq_axis = np.fft.fftfreq(len(df), d=1.0/Fs)
                # Shift the frequency values to match the zero-centered FFT results
                fft_sensor_centered = np.fft.fftshift(fft_sensor)
                freq_axis_centered = np.fft.fftshift(freq_axis)
                #ignore negative frequencies
                freq_axis_plot = freq_axis_centered[(freq_axis_centered>min_freq) & (freq_axis_centered<max_freq)]
                fft_sensor_plot = fft_sensor_centered[(freq_axis_centered>min_freq) & (freq_axis_centered<max_freq)]
                # plot the fft of the data from the 1st sensor + the nuber of different rotations types in the same plot
                axs.plot(freq_axis_plot/1000, 20*np.log10(np.abs(fft_sensor_plot)), label = f'{ch} pos_({sensor_position[position]})')
    if plot_chirp_fft:
        #take the fft of the chirp signal
        sensor_position[0] = 'chirp'
        df['wave_gen'] = signal.detrend(df['wave_gen'])
        fft_chirp = np.fft.fft(df['wave_gen'])

        # Compute the frequency values corresponding to the FFT results
        freq_axis = np.fft.fftfreq(len(df), d=1.0/Fs)
        # Shift the frequency values to match the zero-centered FFT results
        fft_chirp_centered = np.fft.fftshift(fft_chirp)
        freq_axis_centered = np.fft.fftshift(freq_axis)
        #ignore negative frequencies
        freq_axis_plot = freq_axis_centered[(freq_axis_centered>min_freq) & (freq_axis_centered<max_freq)]
        fft_chirp_plot = fft_chirp_centered[(freq_axis_centered>min_freq )& (freq_axis_centered<max_freq)]
        #plot
        axs.plot(freq_axis_plot/1000, 20*np.log10(np.abs(fft_chirp_plot)), label='chirp')

    axs.set_xlabel('Frequency (kHz)')
    axs.set_ylabel('Amplitude (dB)')
    #plt.title(f'Sensor test')
    axs.legend(loc='upper right')
    if savefig:
        fig.savefig(f'sensortest_nfiles_{n_runs}_{sensor_position[position]}_all_compared.{file_format}', format=file_format, dpi=300)
    plt.show()

def fft_analysis_all_positions_compare(folder, file_num=0, Fs=150000, plot_chirp_fft=False, savefig=False,filename='rot_123', file_format='png', channel_names=['channel 1', 'channel 2', 'channel 3','wave_gen'], bandwidth=None):
    '''
    We have a sensor test where 3 sensors are placed in a equilateral triangle with a distance of x between them.
    split into 3 folders, one for each rotation type. A rotation type is where we placed the sensors in a certain way.
    for example the first rotation type "123" means that the sensors are placed in the order 1,2,3.
    the second rotation type "231" means that now sensor 2 is placed where sensor 1 was, sensor 3 is placed where sensor 2 was and sensor 1 is placed where sensor 3 was. and so on. 

    The reason for this switch is to remove dependency in the plate. 

    position -3 is the left bottom sensor, position -2 is the middletop sensor and position -1 is the rightbottom sensor.

    '''
    

    ROOT_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
    folder_path = f'{ROOT_FOLDER}\\{folder}\\'
    # Get a list of the directories in the directory
    dir_path = Path(folder_path)
    
    fig, axs = figure_size_setup()
    if bandwidth is not None:
        min_freq = bandwidth[0]
        max_freq = bandwidth[1]
    else:
        min_freq = 0
        max_freq = 50000
    # Iterate over the rotation type directories
    file_list = list(dir_path.glob('*.csv'))
    #check if the file_num exists in the folder
    if file_num > len(file_list):
        print(f'file_num {file_num} does not exist in folder {dir_path}')
        return

    df = pd.read_csv(filepath_or_buffer=file_list[file_num], names=channel_names)
    df_fft = df.copy()
    df_detrended = df.copy()
    freq_axis = np.fft.fftfreq(len(df), d=1.0/Fs)
    freq_axis_shifted = np.fft.fftshift(freq_axis)
    for ch in channel_names:
        df_detrended[ch] = signal.detrend(df[ch])
        df_fft[ch] = np.fft.fftshift(np.fft.fft(df_detrended[ch]))
        freq_axis_shifted_lim = freq_axis_shifted[(freq_axis_shifted>min_freq) & (freq_axis_shifted<max_freq)]
        df_fft_lim = df_fft[ch][(freq_axis_shifted>min_freq) & (freq_axis_shifted<max_freq)]
        if ch == 'wave_gen':
            ch = 'chirp'
        axs.plot(freq_axis_shifted_lim/1000, 20*np.log10(np.abs(df_fft_lim)), label=ch)
   


    
    

    axs.set_xlabel('Frequency (kHz)')
    axs.set_ylabel('Amplitude (dB)')
    #plt.title(f'Sensor test')
    axs.legend(loc='upper right')
    if savefig:
        fig.savefig(f'sensortest_position_file_n{file_num}_folder_{filename}.{file_format}', format=file_format, dpi=300)
    plt.show()


def fft_analysis_one_sens_all_pos(folder, n_runs=1,ch='channel 1',start_positions=1, Fs=150000, plot_chirp_fft=False, savefig=False, file_format='png', channel_names=['channel 1', 'channel 2', 'channel 3','wave_gen'], bandwidth=None):
    '''
    We have a sensor test where 3 sensors are placed in a equilateral triangle with a distance of x between them.
    split into 3 folders, one for each rotation type. A rotation type is where we placed the sensors in a certain way.
    for example the first rotation type "123" means that the sensors are placed in the order 1,2,3.
    the second rotation type "231" means that now sensor 2 is placed where sensor 1 was, sensor 3 is placed where sensor 2 was and sensor 1 is placed where sensor 3 was. and so on. 

    The reason for this switch is to remove dependency in the plate. 

    position -3 is the left bottom sensor, position -2 is the middletop sensor and position -1 is the rightbottom sensor.

    '''
    

    ROOT_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
    folder_path = f'{ROOT_FOLDER}\\{folder}\\'
    # Get a list of the directories in the directory
    dir_path = Path(folder_path)
    rotation_type_dirs = list(dir_path.glob('*'))
    fig, axs = figure_size_setup()
    if bandwidth is not None:
        min_freq = bandwidth[0]
        max_freq = bandwidth[1]
    else:
        min_freq = 0
        max_freq = 50000
    # Iterate over the rotation type directories
    
    for idx, rotation_type_dir in enumerate(rotation_type_dirs):
        
        # Get a list of the files in the rotation type directory
        file_list = list(rotation_type_dir.glob('*.csv'))
        file_name = rotation_type_dir.name
        # Use the channel names corresponding to the current position
        
        #Iterate over the first n files in the list
        for jdx, file in enumerate(file_list[:n_runs]):
            # Read the data from the file
            df = pd.read_csv(filepath_or_buffer=file, names=channel_names)
            #take the fft of the entire dataframe
            df[ch] = signal.detrend(df[ch])
            fft_sensor = np.fft.fft(df[ch])
            # Compute the frequency values corresponding to the FFT results
            freq_axis = np.fft.fftfreq(len(df), d=1.0/Fs)
            # Shift the frequency values to match the zero-centered FFT results
            fft_sensor_centered = np.fft.fftshift(fft_sensor)
            freq_axis_centered = np.fft.fftshift(freq_axis)
            #ignore negative frequencies
            freq_axis_plot = freq_axis_centered[(freq_axis_centered>min_freq) & (freq_axis_centered<max_freq)]
            fft_sensor_plot = fft_sensor_centered[(freq_axis_centered>min_freq) & (freq_axis_centered<max_freq)]
            # plot the fft of the data from the 1st sensor + the nuber of different rotations types in the same plot
            axs.plot(freq_axis_plot/1000, 20*np.log10(np.abs(fft_sensor_plot)), label = f'{ch} pos_({start_positions})')
        start_positions += 1
    if plot_chirp_fft:
        #take the fft of the chirp signal
        
        df['wave_gen'] = signal.detrend(df['wave_gen'])
        fft_chirp = np.fft.fft(df['wave_gen'])

        # Compute the frequency values corresponding to the FFT results
        freq_axis = np.fft.fftfreq(len(df), d=1.0/Fs)
        # Shift the frequency values to match the zero-centered FFT results
        fft_chirp_centered = np.fft.fftshift(fft_chirp)
        freq_axis_centered = np.fft.fftshift(freq_axis)
        #ignore negative frequencies
        freq_axis_plot = freq_axis_centered[(freq_axis_centered>min_freq) & (freq_axis_centered<max_freq)]
        fft_chirp_plot = fft_chirp_centered[(freq_axis_centered>min_freq )& (freq_axis_centered<max_freq)]
        #plot
        axs.plot(freq_axis_plot/1000, 20*np.log10(np.abs(fft_chirp_plot)), label='chirp')

    axs.set_xlabel('Frequency (kHz)')
    axs.set_ylabel('Amplitude (dB)')
    #plt.title(f'Sensor test')
    axs.legend(loc='upper right')
    if savefig:
        plt.savefig(f'sensortest_nfiles_{n_runs}_ch{ch}.{file_format}', format=file_format, dpi=300)
    plt.show()

def transfer_function_plate(folder, n_files=1, Fs=150000,channels=['channel 1'], savefig=True, file_format='png',plot_response=False, channel_names=['channel 1', 'channel 2', 'channel 3','wave_gen'], bandwidth=None):
    '''
    We have a sensor test where 3 sensors are placed in a equilateral triangle with a distance of x between them.
    split into 3 folders, one for each rotation type. A rotation type is where we placed the sensors in a certain way.
    for example the first rotation type "123" means that the sensors are placed in the order 1,2,3.
    the second rotation type "231" means that now sensor 2 is placed where sensor 1 was, sensor 3 is placed where sensor 2 was and sensor 1 is placed where sensor 3 was. and so on. 

    The reason for this switch is to remove dependency in the plate. 

    position -3 is the left bottom sensor, position -2 is the middletop sensor and position -1 is the rightbottom sensor.

    '''
    ROOT_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
    folder_path = f'{ROOT_FOLDER}\\{folder}\\'
    # Get a list of the directories in the directory
    dir_path = Path(folder_path)
    if bandwidth is not None:
        min_freq = bandwidth[0]
        max_freq = bandwidth[1]
    else:
        min_freq = 0
        max_freq = 50000
    file_list = list(dir_path.glob('*.csv'))
    plot_sig_fft = 'no_sig_fft'
    #Iterate over the first n files in the list
    fig, axs = figure_size_setup()
    for jdx, file in enumerate(file_list[:n_files]):
        # Read the data from the file
        df = pd.read_csv(filepath_or_buffer=file, names=channel_names)
        freq_axis = np.fft.fftfreq(len(df), d=1.0/Fs)
        chirp = df['wave_gen']
        chirp_fft = np.fft.fft(chirp)
        chirp_fft_centered = np.fft.fftshift(chirp_fft)
        freq_axis_centered = np.fft.fftshift(freq_axis)

        for ch in channels:
            #take the fft of the entire dataframe
            fft_sensor = np.fft.fft(df[ch])
            # Compute the frequency values corresponding to the FFT results
            # Shift the frequency values to match the zero-centered FFT results
            fft_sensor_centered = np.fft.fftshift(fft_sensor)
            #ignore negative frequencies
            freq_axis_plot = freq_axis_centered[(freq_axis_centered>min_freq) & (freq_axis_centered<max_freq)]
            fft_sensor_plot = fft_sensor_centered[(freq_axis_centered>min_freq) & (freq_axis_centered<max_freq)]
            chirp_fft_centered_plot = chirp_fft_centered[(freq_axis_centered>min_freq) & (freq_axis_centered<max_freq)]
            transfer_func = fft_sensor_plot/chirp_fft_centered_plot
            # plot the fft of the data from the 1st sensor + the nuber of different rotations types in the same plot
            axs.plot(freq_axis_plot/1000, 20*np.log10(np.abs(transfer_func)), label = f'transfer function with {ch}')
            if plot_response:
                plot_sig_fft = 'with_sig_fft'
                axs.plot(freq_axis_plot/1000, 20*np.log10(np.abs(fft_sensor_plot)), label = f'{ch} response')
    axs.set_label('Frequency (kHz)')
    axs.set_label('Amplitude (dB)')
    #plt.title(f'Sensor test')
    axs.legend(loc='lower right')
    if savefig:
        fig.savefig(f'sensortest__transfer_nfiles_{n_files}_nch{len(channels)}_sens{plot_sig_fft}.{file_format}', format=file_format, dpi=300)
    plt.show()