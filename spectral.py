import numpy
from scipy import signal

def welch(x, fs: float=1., frange=None, *args, **kwargs):
    '''Estimate power spectral density using Welch's method.'''
    if frange is None:
        freqs, Pxx =  signal.welch(x, fs, *args, **kwargs)
    else:
        fstart, fstop, fstep = frange
        freqs, Pxx = signal.welch(x, fs, *args, nperseg=round(fs / fstep), **kwargs)
        indices = numpy.fromiter((i for i, f in enumerate(freqs) if fstart <= f < fstop), int)
        freqs = freqs[     indices]
        Pxx   = Pxx  [..., indices]
    return freqs, Pxx
