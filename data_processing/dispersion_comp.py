import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt


def disp_comp(
    time_step,
    in_signal,
    frequency,
    wavenumber,
    truncate=True,
    oversampling_factor=16,
    interpolation_method="linear",
    group_velocity=None,
):
    # basic disp comp starting from time signal, with time origin at first point

    # calc maximum group velocity
    # print(wavenumber)
    if group_velocity is None:
        temp_k = np.linspace(min(wavenumber), max(wavenumber), 100)
        temp_f = interp1d(wavenumber, frequency, kind="cubic", fill_value=0)(temp_k)
        group_vel = np.gradient(temp_f) / (temp_k[1] - temp_k[0])
        print("max group velocity from interpolation:", np.max(np.abs(group_vel)))

    else:
        group_vel = group_velocity
    max_group_velocity = np.max(np.abs(group_vel))
    print("max group velocity from file:", max_group_velocity)

    # get zero padded time signals into frequency domain
    in_signal = np.squeeze(in_signal)
    time_pts = in_signal.shape[0]
    fft_pts = 2 ** int(np.ceil(np.log2(time_pts * oversampling_factor)))
    freq_step = 1 / (time_step * fft_pts)
    in_spec = np.fft.fft(in_signal, fft_pts)
    in_spec = in_spec[: fft_pts // 2 + 1]
    freq = np.arange(fft_pts) * freq_step
    freq = freq[: fft_pts // 2 + 1]
    freq = np.round(freq, decimals=9)
    frequency = np.round(frequency, decimals=9)
    wavenumber = np.round(wavenumber, decimals=9)
    # set up wavenumber vector for output
    out_kstep = 1 / (time_pts * time_step * max_group_velocity)
    f_nyq = 1 / (2 * time_step)

    k_nyq = interp1d(
        frequency, wavenumber, kind=interpolation_method, fill_value="extrapolate"
    )(f_nyq)

    out_kpts = 2 ** int(np.ceil(np.log2(2 * k_nyq / out_kstep)))
    out_k = np.arange(out_kpts) * out_kstep
    out_k = out_k[: out_kpts // 2 + 1]

    # interpolate to get k at each frequency

    k_at_freq = interp1d(
        frequency,
        wavenumber,
        kind=interpolation_method,
        fill_value=np.nan,
        bounds_error=False,
    )(freq)

    valid_range = np.where(~np.isnan(k_at_freq))[0]

    out_kspec = interp1d(
        k_at_freq[valid_range],
        in_spec[valid_range],
        kind=interpolation_method,
        fill_value=0,
        bounds_error=False,
    )(out_k)

    out_signal = np.fft.ifft(out_kspec, out_kpts)
    d_step = 1 / (out_kpts * out_kstep)

    if truncate:
        max_dist = max_group_velocity * time_step * in_signal.shape[0]
        last_index = int(np.ceil(max_dist / d_step))
        if last_index <= out_signal.shape[0]:
            out_signal = out_signal[:last_index]

    return d_step, out_signal


def dispersion_compensation(
    time,
    in_signal,
    frequency,
    wavenumber,
    truncate=True,
    oversampling_factor=16,
    interpolation_method="linear",
    group_velocity=None,
):
    # Applies dispersion compensation to a time domain signal to convert it to the distance domain

    time_step = abs(time[0] - time[1])

    # do positive time half of signal
    pos_in_sig = in_signal[time > 0]
    if time[0] != 0:
        pos_in_sig = np.concatenate((in_signal[time == 0], pos_in_sig))

    if pos_in_sig.size != 0:
        d_step, pos_out_sig = disp_comp(
            time_step,
            pos_in_sig,
            frequency,
            wavenumber,
            truncate,
            oversampling_factor,
            interpolation_method,
            group_velocity,
        )
        pos_dist = np.arange(1, pos_out_sig.shape[0]) * d_step
        zero_pos_out_sig = pos_out_sig[0]
        pos_out_sig = pos_out_sig[1:]
    else:
        pos_dist = np.array([])
        pos_out_sig = np.array([])
        zero_pos_out_sig = np.zeros(in_signal.shape[0])

    # do negative time half of signal
    neg_in_sig = np.flipud(in_signal[time < 0])
    if neg_in_sig.size != 0:
        neg_in_sig = np.concatenate((in_signal[time == 0], neg_in_sig))
        d_step, neg_out_sig = disp_comp(
            time_step,
            neg_in_sig,
            frequency,
            wavenumber,
            truncate,
            oversampling_factor,
            interpolation_method,
            group_velocity,
        )
        neg_dist = np.arange(1, neg_out_sig.shape[0]) * d_step
        neg_dist = -np.flipud(neg_dist)
        zero_neg_out_sig = neg_out_sig[0]
        neg_out_sig = neg_out_sig[1:]
        neg_out_sig = np.flipud(neg_out_sig)
    else:
        neg_dist = np.array([])
        neg_out_sig = np.array([])
        zero_neg_out_sig = 0  # np.zeros(len(in_signal))

    dist = np.concatenate((neg_dist, [0], pos_dist))
    out_signal = np.concatenate(
        (
            neg_out_sig,
            np.atleast_1d(0.5 * (zero_neg_out_sig + zero_pos_out_sig)),
            pos_out_sig,
        )
    )

    return dist, out_signal
