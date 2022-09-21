import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# Config 
SAMPLE_RATE = 80000     # Hz

CROP_MODE = "Auto"      # Auto or Manual
CROP_BEFORE = 80000     # samples
CROP_AFTER = 120000     # samples

DATA_DELIMITER = ","

df = pd.read_csv('..\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_C1_center_v1.csv', delimiter=DATA_DELIMITER)


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


df = crop_data(df, CROP_MODE)
df.plot()

plt.legend(["Channel 1", "Channel 2", "Channel 3"])
plt.grid()
plt.show()


# Number of sample points
N = len(df)
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
yf = scipy.fftpack.fft(df.iloc[:,0])
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.grid()
plt.show()