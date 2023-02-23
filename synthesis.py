import numpy
import system
import fitting
import control

def Hsyn(F: control.StateSpace, G: control.StateSpace):
    Ginv = 1. / control.xferfcn._convert_to_transfer_function(G)
    control.namedio.common_timebase(Ginv, F)
    return control.statesp._convert_to_statespace(Ginv * F)

def Ksyn(F: control.StateSpace, G: control.StateSpace):
    '''Build a closed-loop controller based on a filter F and a plant input response system G.'''
    H = Hsyn(F, G)
    I = control.StateSpace((), (), (), numpy.identity(F.noutputs))
    return system.inv(I + F) * H

def Ssyn(G: control.lti.LTI, H: control.StateSpace):
    I = control.StateSpace((), (), (), numpy.identity(G.noutputs))
    return I + G * H

def Stsyn(G: control.lti.LTI, K: control.StateSpace):
    return system.feedback(1., G * K, 1)
        
def closed_loop(P: control.lti.LTI | control.InputOutputSystem, K: control.StateSpace):
    '''Build a closed-loop circuit based on a plant P and a controller K.'''
    Duu = numpy.identity(K.noutputs)
    Dun = numpy.zeros((P.ninputs - K.noutputs, K.noutputs))
    comp = control.StateSpace((), (), (), numpy.vstack((Duu, Dun)))
    return system.feedback(P, comp * K, 1)

def psdfit(freqs, psd, poles=None, niter=8):
    sdata = 2j * numpy.pi * numpy.asarray(freqs)
    return fitting.magvectfit(sdata, psd, poles, niter=niter)

def predictorfit(freqs: numpy.ndarray, delay: numpy.ndarray, poles=None, dt=None, niter=8):
    delay = system._check_delay(delay)
    sdata = 2j * numpy.pi * numpy.atleast_1d(freqs)
    ddata = numpy.exp((dt if dt else 1.) * numpy.outer(delay, sdata))
    return fitting.vectfit(sdata, ddata, poles, strictly_proper=False, dt=dt, niter=niter)

def white_noise(size: int, length: int, dt: float, seed=None):
    rs = numpy.random.RandomState(seed)
    return rs.normal(0., dt**-.5, (size, length))

def phase_noise(size: int, length: int, dt: float, seed=None):
    sigma = (length / dt)**.5
    rs = numpy.random.RandomState(seed)
    mnoise = numpy.full((size, 1), sigma)
    pnoise = sigma * numpy.exp(1j * rs.uniform(0., 2. * numpy.pi, (size, (length - 1) // 2)))
    nnoise = numpy.flip(pnoise.conj(), 1)
    pnoise = pnoise if length % 2 else numpy.hstack((pnoise, mnoise))
    noise = numpy.hstack((mnoise, pnoise, nnoise))
    return numpy.fft.ifft(noise).real

def violet_noise(size: int, length: int, dt: float, seed=None):
    return numpy.diff(white_noise(size, length + 1, dt, seed))
