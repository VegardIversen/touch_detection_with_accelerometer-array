import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
from scipy import signal




def crop_data(data, crop_mode):
    """CROP_MODE:
    Manual,
    Auto
    """
    if crop_mode == "Auto":
        # Removes zero sections of the data
        data_cropped = data.loc[(df!=0).any(1)]
    elif crop_mode == "Manual":
        data_cropped = data.truncate(before=CROP_BEFORE, after=CROP_AFTER)

    return data_cropped

def plot_fft(df, Fs=150000, window=False):
    
    if window:
        hamming_window = scipy.signal.hamming(len(df))
        data_fft = scipy.fft.fft(df.values * hamming_window)
    else:
        data_fft = scipy.fft.fft(df.values, axis=0)
    
    #print(df.values)
    
    fftfreq = scipy.fft.fftfreq(len(data_fft),1/Fs)
    N = int(len(data_fft)/2)
    #fft_x_axis = np.linspace(0,(Fs/ 2),N)
    plt.grid()
    plt.title('fft of signal')
    plt.xlabel("Frequency [hz]")
    plt.ylabel("Amplitude")
    plt.plot(fftfreq, 20 * np.log10(np.abs(data_fft)))
    # Only plot positive frequencies
    ax = plt.subplot(1, 1, 1)
    ax.set_xlim(0)
    plt.show()


def plot_fft_with_hamming(df, Fs=80000):    
    plot_fft(df, window=True)

def plot_data(df, crop=True):
    if crop:
        df = crop_data(df, CROP_MODE)

    df.plot()
    plt.legend(df.columns)
    plt.grid()
    plt.show()


def plot_spectogram(df, include_signal=True, sample_rate=150000, channel='channel 1', freq_max=None):
    
    if include_signal:
        time_axis = np.linspace(0, len(df)//sample_rate, num=len(df))
        ax1 = plt.subplot(211)
        plt.plot(time_axis, df[channel])
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        ax2 = plt.subplot(212, sharex=ax1)
        plt.specgram(df[channel],Fs=sample_rate)
        plt.axis(ymax=freq_max)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
    else:
        plt.specgram(df[channel],Fs=sample_rate)
        plt.axis(ymax=freq_max)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
    plt.show()

if __name__=='__main__':

    # Config 
    SAMPLE_RATE = 150000     # Hz

    CROP_MODE = "Auto"      # Auto or Manual
    CROP_BEFORE = 80000     # samples
    CROP_AFTER = 120000     # samples

    DATA_DELIMITER = ","

    data_folder = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'

    test_file = data_folder + '\\fingernail_test_passive_setup2\\touch_test_fingernail_passive_setup2_place_A1_center_v2.csv'
    print(test_file)
    df = pd.read_csv(test_file, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'] )
    #print(df.head())
    print(len(df['channel 1'].values))
    print(int(50e-3*SAMPLE_RATE))
    df_crop = crop_data(df, CROP_MODE)
    plot_spectogram(df)
