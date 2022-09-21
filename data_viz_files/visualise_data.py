import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy



# Config 
SAMPLE_RATE = 80000     # Hz

CROP_MODE = "Auto"      # Auto or Manual
CROP_BEFORE = 80000     # samples
CROP_AFTER = 120000     # samples

DATA_DELIMITER = ","

data_folder = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
test_file = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_A1_center_v1.csv'
print(test_file)
df = pd.read_csv(test_file, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'] )
print(df.head())


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

def plot_fft(df, Fs=80000, window=False):
    
    if window:
        hamming_window = scipy.signal.hamming(len(df))
        data_fft = scipy.fft.fft(df.values * hamming_window)
    else:
        data_fft = scipy.fft.fft(df.values)
    
    fftfreq = scipy.fft.fftfreq(len(data_fft),1/Fs)
    N = int(len(data_fft)/2)
    #fft_x_axis = np.linspace(0,(Fs/ 2),N)

    plt.title('fft of signal')
    plt.xlabel("Frequency [hz]")
    plt.ylabel("Amplitude")
    plt.plot(fftfreq[fftfreq > 0], 20*np.log10(np.abs(data_fft[fftfreq > 0])))
    plt.show()


def plot_fft_with_hamming(df, Fs=80000):    
    plot_fft(df, window=True)




if __name__=='__main__':
    plot_fft(df['channel 3'])
    plot_fft_with_hamming(df['channel 3'])