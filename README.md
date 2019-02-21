# Filtering Tests

These are some python scripts that use numpy/scipy and matplotlib to test out some simple DSP concepts. 
They were written just out of personal interest.

## `fft.py`

Calculates and plots the FFT of a simple sinusoid.

## `square_wave_dt.py`

Takes a discrete-time square wave and applies a low-pass Butterworth IIR filter to produce a discrete-time sinsoid.

Plots the frequency response of the original and the filtered signal, the frequency response of the IIR filter, 
and the original and filtered signal in the time domain.

## `square_wave_ct.py`

Takes a continuous-time square wave (simulated as a DT wave sampled at a really high frequency) and applies a low-pass 
RC filter.

Plots the frequency response of the original and the filtered signal, the frequency response of the RC filter, 
and the original and filtered signal in the time domain.

## `pwm.py`

Takes a PWM wave whose duty cycle varies according to a sinusoid and passes it through an RC filter. This is meant to
test out a way of generating a sinusoid using PWM. 

Plots the frequency response of the original and the filtered signal, the frequency response of the RC filter, 
and the original and filtered signal in the time domain.
