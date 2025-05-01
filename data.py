import numpy as np

def build_sine_wave(X, frequency, amplitude):
    return amplitude * np.sin(X/frequency)


def build_modified_sine_wave(X, frequency, amplitude):
    return modify_sine(build_sine_wave(X,frequency, amplitude))


def modify_sine(sine_wave):
    pass