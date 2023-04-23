# Changelog

## 6.1.1
Export analytic signals for use in DOA estimation functions. Currently cropped to what I believe to be the direct signal. The Matlab functions for DOA estimation result in around -17 degrees broadside angle, which is only a few percent off what it is expected to be (if the understanding of the broadside figure in the script is correct). Tried with cropping to two sources, not great results. Probably too few sensors to get a good estimate.

## 6.1.0
Export signals from setup 4 (three sensors 5 cm from edge, 1 cm apart), currently filtered at 30 kHz and cropped to (most likely) only include the direct signal (I think. If the wave travels at around 1000 m/s and it has to travel approximately 10 cm longer for the second reflection, it should appear 0.1 ms after).

## 5.1.0
Looking at touch release and swiping. Really interesting to see the relative power in the touch release, and the energy in the swiping. Even at 30 kHz to 35 kHz bandpass the SNR is rather good, and promising for phase based DoA estimation.

# 4.1.0
Inspecting the immediate touch signal, before propagation using setup 3. Exporting the cropped touch to use for simulations.

## 3.2.0
Same experiment as 3.1.0, but with results from applying the estimation to a simulation. Also great results.

## 3.1.0
Estimate touch location using ToA and two sensors at 6010 Hz. Good results.

## 2.1.0 - 2023-03-03
Use a sensor array consisting of four sensors (or choose another amount) placed in a line, and simulate non-dispersive signals from a randomly placed actuator/touch on a 100x70 cm plate.

## 1.1.0 - 2023-01-18
Trying to make use of the correlation technique described by Ziola and Gorman in "Source location in thin plates using cross-correlation". The results did not yield any great results. Somewhat similar to using regular bandpassing, but I suspect it is limited by the triangular shape of pulse compression. Could however keep testing different frequencies and standard deviations for the gaussian pulse.

## 1.0.0 - 2023-01-09
Final changes before exporting to a different repository for delivering the project.
