import numpy as np
import pandas as pd
import pathlib as pl


d = 0.08


# if speed in material is 300, then the maximum time delay between two sensor
# are t_max=0.08s/300m/s = 0.00026666666 s, with sampling freq on 150k Hz
# this will approximate 40 samples
def correlate(x, y, upSamplingFactor=1, maxdelay=40, mod='valid'):  # maxdelay calculated from formula, bc of distance between mics
    corr = np.abs(np.correlate(x, y.iloc[(maxdelay * upSamplingFactor):-(maxdelay * upSamplingFactor)], mode=mod))
    return corr


# calculate degree from lags array, from the lag of the different mics
def degree(lags, positive_angl=True):
    theta = np.arctan2(np.sqrt(3) * (lags[0] + lags[1]), (lags[0] - lags[1] - 2 * lags[2])) #formula for theta
    if theta < 0 and positive_angl:
        return 360 + np.degrees(theta)  # not to get negative degrees. 
    return np.degrees(theta)


def lag(corr, upSamplingFactor=1, maxdelay=40): # max delay on our circuit is 5, tor is 9.
    lag = np.argmax(corr) - (maxdelay * upSamplingFactor) # calculate lag, 
    return lag


def degree_calc(df, upsampl=1):
    
    crosscorr2_1 = correlate(df['channel 1'], df['channel 2'], upSamplingFactor=upsampl) # mic 2 and mic 1
    crosscorr3_1 = correlate(df['channel 3'], df['channel 2'], upSamplingFactor=upsampl) # mic 3 and 1
    crosscorr3_2 = correlate(df['channel 3'], df['channel 1'], upSamplingFactor=upsampl) # mic 3 and 2
    autocorr1_1 = correlate(df['channel 2'], df['channel 2'], upSamplingFactor=upsampl) # mic 1 mic 1, autocorr
    lags = np.array([lag(crosscorr2_1, upSamplingFactor=upsampl), lag(crosscorr3_1, upSamplingFactor=upsampl), lag(crosscorr3_2, upSamplingFactor=upsampl)])
    degrees = degree(lags)
    print(f'\n{degrees} degrees calculated')
    
    return degrees

if __name__ == '__main__':
    # Config 
    SAMPLE_RATE = 150000     # Hz

    CROP_MODE = "Auto"      # Auto or Manual
    CROP_BEFORE = 80000     # samples
    CROP_AFTER = 120000     # samples

    DATA_DELIMITER = ","

    data_folder = f'{pl.Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
    test_file = data_folder + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_C1_center_v2.csv'
    print(test_file)
    df = pd.read_csv(test_file, delimiter=DATA_DELIMITER, names=['channel 1', 'channel 2', 'channel 3'] )
    # print(df.head())
    degree_calc(df)