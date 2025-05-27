import numpy as np

def build_sine_wave(X, frequency, amplitude):
    return amplitude * np.sin(X/frequency)


def build_modified_sine_wave(X, frequency, amplitude, loc, length):
    return modify_sine(build_sine_wave(X,frequency, amplitude), loc, length)


def modify_sine(sine_wave, loc, length):
    x_value = sine_wave[loc]
    sine_wave[loc:loc+length] = x_value
    return x_value

