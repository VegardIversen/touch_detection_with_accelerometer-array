# Changelog


### 10.1.3
Same as 10.1.2, but bandpassed at 35 kHz. Looks a bit wonky, but the implementation makes sense at least.

### 10.1.2
Adjustments to v10.1.1 again, using the full bandwidth of the measurements. Looks good. Filter ideal signal as well with the same filter as the measured signal.

### 10.1.1
Minor improvements to v10.1.0, shift and scale ideal signals for comparison using Sensor 1 as reference instead of doing it individually.

## 10.1.0
EDIT: Highpassed at 35 kHz, not bandpassed.
Just saving a nice comparison of the ULA measurements and the expected arrival times.
Quite easy to see what is going on, at a group velocity of 1000 m/s at 35 kHz, and around 23 dB/m attenuation.
Similar gaussian pulse as the one transmitted.


### 10.0.6
Spot on result for Root-MUSIC, good results for Root-WSF and ESPRIT.
MUSIC is way off and phi_4 is wrong, which is ok
Using 35kHz and phase velocity 825 m/s, which makes sense as this gives a wavelength of 2.357 cm.
Does not seem to be very dependent on the velocity from first tests, as in withing a cm or two for each 5 m/s.
Not great, not terrible.

This fits with the hypothesis in many ways:
- Higher frequency yields better results, especially as the wavelength approaches 2 cm (for this setup).
- Even though the phase velocity is not as calculated beforehand, it makes somewhat sense given the measured group velocity. Possible to estimate phase velocity from a measured group velocity? If it is easier.
- Angles are quite accurate, and although the other methods are worse, they are still relatively close.

### 10.0.5
Good guess for both Root-WSF and Root-MUSIC, with angle errors less than 3.5 degrees off.
Phi_4 is way off still.
Estimating four angles at 725 m/s for 30 kHz.

Method: Root-WSF \
Real phi_1: 30.964, Estimated phi_1: 27.615 \
Real phi_2: -43.698, Estimated phi_2: -41.216 \
Real phi_3: 26.147, Estimated phi_3: 27.615 \
Real phi_4: -38.019, Estimated phi_4: -12.521 \
Error in phi_1: 3.349 degrees. \
Error in phi_2: 2.482 degrees. \
Error in phi_3: 1.468 degrees. \
Error in phi_4: 25.498 degrees. \
r_sa: [0.453, 0.290] \
Estimated location error: 0.020 m

Method: Root-MUSIC \
Real phi_1: 30.964, Estimated phi_1: 31.059 \
Real phi_2: -43.698, Estimated phi_2: -43.371 \
Real phi_3: 26.147, Estimated phi_3: 29.236 \
Real phi_4: -38.019, Estimated phi_4: -13.168 \
Error in phi_1: 0.096 degrees. \
Error in phi_2: 0.327 degrees. \
Error in phi_3: 3.089 degrees. \
Error in phi_4: 24.851 degrees. \
r_sa: [0.467, 0.317] \
Estimated location error: 0.051 m

### 10.0.4
Quite close guess for ESPRIT, despite angles being quite off.
It finds three angles when asked to find 4, and all three angles are more than 3 degrees off.
Certain that the cropping is correct, see v10.0.3.
Results are from using 15kHz, 7 sensors, 2nd order filter, q=0.1, phase velocity 600 m/s.

It may seem that an error in the guessed phase velocity is wrong, it may still be able to correct itself as long as the estimated angles are systematically wrong.
This open up the ability for a potential user to manually tune the velocity to be correct.

Method: ESPRIT \
Real phi_1: 30.964, Estimated phi_1: 38.968 \
Real phi_2: -43.698, Estimated phi_2: -49.475 \
Real phi_3: 26.147, Estimated phi_3: 29.480 \
Real phi_4: -38.019, Estimated phi_4: 0.000 \
Error in phi_1: 8.004 degrees. \
Error in phi_2: 5.777 degrees. \
Error in phi_3: 3.333 degrees. \
Error in phi_4: 38.019 degrees. \
r_sa: [0.443, 0.307] \
Estimated location error: 0.038 m

### 10.0.3
Nothing exciting really, plotting spectrograms and FFTs of measured signals from the 15 kHz pulse.
The spectrogram relatively clearly shows the expected arrival times of the first four signals, considering that the last sensor has a higher concentration of the first arriving signals (before 1 ms).
Confused about the group velocity, since it seems really high at around 1108 m/s.

### 10.0.2
3.33 cm off with ESPRIT on full 22 kHz signal, PROPAGATION_SPEED = 443 for 22 kHz, 20.12 mm wavelength.
Sharp tukey window (alpha=0.05), corrected array start position.

Method: Root-WSF \
Real phi_1: 27.072, Estimated phi_1: 26.221 \
Real phi_2: -46.245, Estimated phi_2: -46.695 \
Real phi_3: 22.694, Estimated phi_3: 19.778 \
Real phi_4: -40.515, Estimated phi_4: -40.311 \
Error in phi_1: 0.851 degrees. \
Error in phi_2: 0.450 degrees. \
Error in phi_3: 2.916 degrees. \
Error in phi_4: 0.205 degrees. \
r_sa: [0.422, 0.188] \
Estimated location error: 0.051 m

Method: Root-MUSIC  \
Real phi_1: 27.072, Estimated phi_1: 26.654 \
Real phi_2: -46.245, Estimated phi_2: -45.652 \
Real phi_3: 22.694, Estimated phi_3: 18.785 \
Real phi_4: -40.515, Estimated phi_4: -36.903 \
Error in phi_1: 0.418 degrees. \
Error in phi_2: 0.593 degrees. \
Error in phi_3: 3.909 degrees. \
Error in phi_4: 3.612 degrees. \
r_sa: [0.461, 0.191] \
Estimated location error: 0.041 m

Method: MUSIC \
Real phi_1: 27.072, Estimated phi_1: 39.000 \
Real phi_2: -46.245, Estimated phi_2: -45.000 \
Real phi_3: 22.694, Estimated phi_3: 27.000 \
Real phi_4: -40.515, Estimated phi_4: 19.000 \
Error in phi_1: 11.928 degrees. \
Error in phi_2: 1.245 degrees. \
Error in phi_3: 4.306 degrees. \
Error in phi_4: 59.515 degrees. \
r_sa: [1.262, 0.694] \
Estimated location error: 0.935 m

Method: ESPRIT \
Real phi_1: 27.072, Estimated phi_1: 25.418 \
Real phi_2: -46.245, Estimated phi_2: -46.197 \
Real phi_3: 22.694, Estimated phi_3: 22.061 \
Real phi_4: -40.515, Estimated phi_4: -35.408 \
Error in phi_1: 1.654 degrees. \
Error in phi_2: 0.049 degrees. \
Error in phi_3: 0.633 degrees. \
Error in phi_4: 5.108 degrees. \
r_sa: [0.423, 0.212] \
Estimated location error: 0.033 m

### 10.0.1
EDIT: The array was starting at y=0.06 m instead of 0.05 as it should.
0.9 cm off the correct touch location using Root-WSF set to detect 3 sources.
Phase velocity still set to 954 m/s as from the calculation.
Applying a Hamming window to the cropped signal, suspecting that this suppresses phi_4 signal quite a bit.
Or perhaps it already is suppressed in the Comsol simulation.
Still having the issue of wrong sign for phi_2 / phi_4, that is supposed to be negative.

Info from run:

Real phi_1: 29.539, Estimated phi_1: 25.478 \
Real phi_2: -44.680, Estimated phi_2: -42.183 \
Real phi_3: 24.874, Estimated phi_3: 25.478 \
Error in phi_1: 4.061 degrees. \
Error in phi_2: 2.497 degrees. \
Error in phi_3: 0.604 degrees. \
r_sa: [0.442, 0.258] \
Estimated location error: 0.009 m

## 10.0.0
Trying to estimate touch location with S0 waves from the Comsol simulation.
Testing at 25 kHz. Getting quite close, 6.4 cm off, with angles from Root-WSF.
The issue is phi_4 seems to consistently be estimated to arrive from above instead of below.
Not sure what is going on there.
Root-MUSIC was way off on this one, only being able to detect three angles.

### 9.0.1
Some edits to v8.0.0.
Plot with the parameters Tonni sent in February, way higher velocities there. Fix ylabel to not say "Phase velocity".

## 9.0.0
Not much, but some ok results with my simulations. Changed to using the MEMS accelerometers, which allowed for smaller sensors and can be placed at the distance required by the simulated phase velocity of 250 m/s at 25 kHz. Look more at this later. Should probably move to a model that simulates both group and phase velocity, maybe directly in sum_signals()?

## 8.0.0
See v9.0.1 for the current version of this.
Plot the phase velocities, energy (group) velocities and the wavelengths of A0 and S0 waves.
Concluding that the best frequency band should be around 25 kHz as this allows for the A0 waves to arrive ahead of the S0 waves with the best margin (the group velocity is lower for S0 than A0), while still having a relatively short wavelength of ~1 cm.

### 7.1.0
Use estimated angles from the Matlab Root-WSF to estimate the touch location based on the r_sa vector. Results vary with the chosen center frequency and t_var, but overall seems promising. Also comparing the estimated angles to the real anlges, calulcated given the location of the sensors and the actuator.

### 7.0.1
Measure speed of the 25 kHz wave.
This is approximately four times higher than the expected value.
Not sure if it is S0 or A0, as it is both the first and the strongest wave to arrive.
A similar result is found by ''eye'', i.e. the time between the first peaks. (991,85 m/s).

## 7.0.0
Make setup 5: an 8-element ULA spaced by 1 cm starting at [0.05, 0.65]. Adding an option to generate_ideal_signal() for passing model type as a parameter, allowing for choosing between using dirac pulses and a measurement sample to delay.

#### 6.0.1
Export analytic signals for use in DOA estimation functions. Currently cropped to what I believe to be the direct signal. The Matlab functions for DOA estimation result in around -17 degrees broadside angle, which is only a few percent off what it is expected to be (if the understanding of the broadside figure in the script is correct). Tried with cropping to two sources, not great results. Probably too few sensors to get a good estimate.

## 6.0.0
Export signals from setup 4 (three sensors 5 cm from edge, 1 cm apart), currently filtered at 30 kHz and cropped to (most likely) only include the direct signal (I think. If the wave travels at around 1000 m/s and it has to travel approximately 10 cm longer for the second reflection, it should appear 0.1 ms after).

## 5.0.0
Looking at touch release and swiping. Really interesting to see the relative power in the touch release, and the energy in the swiping. Even at 30 kHz to 35 kHz bandpass the SNR is rather good, and promising for phase based DoA estimation.

## 4.0.0
Inspecting the immediate touch signal, before propagation using setup 3. Exporting the cropped touch to use for simulations.

### 3.1.0
Same experiment as 3.1.0, but with results from applying the estimation to a simulation. Also great results.

## 3.0.0
Estimate touch location using ToA and two sensors at 6010 Hz. Good results.

## 2.0.0 - 2023-03-03
Use a sensor array consisting of four sensors (or choose another amount) placed in a line, and simulate non-dispersive signals from a randomly placed actuator/touch on a 100x70 cm plate.

### 1.1.0 - 2023-01-18
Trying to make use of the correlation technique described by Ziola and Gorman in "Source location in thin plates using cross-correlation". The results did not yield any great results. Somewhat similar to using regular bandpassing, but I suspect it is limited by the triangular shape of pulse compression. Could however keep testing different frequencies and standard deviations for the gaussian pulse.

## 1.0.0 - 2023-01-09
Final changes before exporting to a different repository for delivering the project.
