import numpy
import model
import simul
import system
import control
import fitting
from matplotlib.axes._axes import Axes

def crop_time_series(tdata: numpy.ndarray, series: tuple[numpy.ndarray], duration=None):

    tmask = tdata >= tdata.max() - (duration if duration else 500. * tdata[1])
    return tdata[tmask], tuple(ydata[..., tmask] for ydata in series)

def delay(axes: tuple[Axes, Axes, Axes]):
    Ts = 1e-3
    H = model.alpha_gamma_filter()
    I = numpy.identity(H.noutputs)
    S = I + H
    S11 = system.feedback(I, system.StateSpace(H.feedback(I), delay=3e-3, delay_type='output'), 1)
    S21 = system.feedback(I, system.StateSpace(H.feedback(I), delay=5e-3, delay_type='output'), 1)
    S31 = system.feedback(I, system.StateSpace(H.feedback(I), delay=1e-2, delay_type='output'), 1)
    Phi1 = model.discrete_predictor(10.,  3)
    Phi2 = model.discrete_predictor(10.,  5)
    Phi3 = model.discrete_predictor(10., 10)
    z1 = numpy.exp(2j * numpy.pi * 10. * Ts)
    z2 = numpy.exp(2j * numpy.pi * 40. * Ts)
    H1 = model.alpha_gamma_filter(c1=1./numpy.abs(Phi1(z1)), c2=-.5/numpy.abs(Phi1(z2)))
    H2 = model.alpha_gamma_filter(c1=1./numpy.abs(Phi2(z1)), c2=-.5/numpy.abs(Phi2(z2)))
    H3 = model.alpha_gamma_filter(c1=1./numpy.abs(Phi3(z1)), c2=-.5/numpy.abs(Phi3(z2)))
    S12 = system.feedback(I, system.StateSpace(Phi1 * system.div(H1.sample(Ts), S.sample(Ts)), delay= 3, delay_type='output'), 1)
    S22 = system.feedback(I, system.StateSpace(Phi2 * system.div(H2.sample(Ts), S.sample(Ts)), delay= 5, delay_type='output'), 1)
    S32 = system.feedback(I, system.StateSpace(Phi3 * system.div(H3.sample(Ts), S.sample(Ts)), delay=10, delay_type='output'), 1)
    # sample
    S   = S.sample(Ts)
    S11 = S11.sample(Ts).sample_delays()
    S21 = S21.sample(Ts).sample_delays()
    S31 = S31.sample(Ts).sample_delays()
    S12 = S12.sample_delays()
    S22 = S22.sample_delays()
    S32 = S32.sample_delays()
    # stability check
    if not system.is_stable(S12):
        raise ValueError('S12 is unstable')
    if not system.is_stable(S22):
        raise ValueError('S22 is unstable')
    if not system.is_stable(S32):
        raise ValueError('S32 is unstable')
    # freqresp
    freqs = numpy.linspace(0., 80., 1000)
    Smagn = simul.freqresp(S  , freqs).magnitude.squeeze()
    S11mg = simul.freqresp(S11, freqs).magnitude.squeeze()
    S21mg = simul.freqresp(S21, freqs).magnitude.squeeze()
    S31mg = simul.freqresp(S31, freqs).magnitude.squeeze()
    S12mg = simul.freqresp(S12, freqs).magnitude.squeeze()
    S22mg = simul.freqresp(S22, freqs).magnitude.squeeze()
    S32mg = simul.freqresp(S32, freqs).magnitude.squeeze()
    # log conversion
    logSmagn = 20. * numpy.log10(Smagn)
    logS11mg = 20. * numpy.log10(S11mg)
    logS21mg = 20. * numpy.log10(S21mg)
    logS31mg = 20. * numpy.log10(S31mg)
    logS12mg = 20. * numpy.log10(S12mg)
    logS22mg = 20. * numpy.log10(S22mg)
    logS32mg = 20. * numpy.log10(S32mg)
    # plot
    for ax in axes:
        ax.plot (freqs, logSmagn, 'k')
    axes[0].plot(freqs, logS11mg, 'C0--')
    axes[0].plot(freqs, logS12mg, 'C3--')
    axes[1].plot(freqs, logS21mg, 'C0--')
    axes[1].plot(freqs, logS22mg, 'C3--')
    axes[2].plot(freqs, logS31mg, 'C0--')
    axes[2].plot(freqs, logS32mg, 'C3--')

def activity(
    P: control.StateSpace
      | system.StateSpace
      | system.NonlinearStateSystem,
    axes: tuple[Axes, Axes],
    seed: int | None=None
):
    H = model.alpha_gamma_filter()
    circuit = simul.Circuit(P, H, seed=seed)
    y0dat, udata, ydata = circuit.stimulate(uamp=0.)
    # spectral density
    fpsd, y0sd = circuit.welch(y0dat)
    # log conversion
    logy0sd = 10. * numpy.log10(y0sd)
    # crop time series
    tdata, (y0dat, udata, ydata) = crop_time_series(circuit.time(), (y0dat, udata, ydata))
    # plot
    axes[1].plot(fpsd, logy0sd, 'C0')
    axes[0].plot(tdata, y0dat, 'C0')

def psdfit(
    P: control.StateSpace
      | system.StateSpace
      | system.NonlinearStateSystem,
    axes: tuple[Axes, Axes, Axes, Axes],
    seed: int | None=None
):
    H = model.alpha_gamma_filter()
    circuit = simul.Circuit(P, H, seed=seed)
    y0dat, udata, ydata = circuit.stimulate(uamp=5e-3)
    # spectral density
    fpsd, y0sd = circuit.welch(y0dat)
    fpsd, upsd = circuit.welch(udata)
    fpsd, ypsd = circuit.welch(ydata)
    # tf magnitude
    Gpsd = (ypsd - y0sd) / upsd
    Gneg = Gpsd < 0.
    if Gneg.any():
        Gpsd = Gpsd * numpy.logical_not(Gneg)
        control.statesp.warn('negative magnitude encountered')
    # fitting
    sdata = 2j * numpy.pi * fpsd
    poles = circuit.G.poles() if not system.is_delayed(circuit.G) else (circuit.G.nstates // 2) * 2
    G = fitting.magvectfit(sdata, Gpsd, poles, niter=8)
    # frequency response
    freqs = circuit.frequency()
    Gresp = simul.freqresp(circuit.G, freqs)
    Gfitr = simul.freqresp(G, freqs)
    Gmag = Gresp.magnitude.squeeze()
    Gphase = Gresp.phase.squeeze()
    Gfitm = Gfitr.magnitude.squeeze()
    Gfitp = Gfitr.phase.squeeze()
    # log conversion
    Gphased = numpy.rad2deg(numpy.unwrap(Gphase))
    Gfitphd = numpy.rad2deg(numpy.unwrap(Gfitp))
    logGfit = 20. * numpy.log10(Gfitm)
    logGmag = 20. * numpy.log10(Gmag)
    logGpsd = 10. * numpy.log10(Gpsd)
    logy0sd = 10. * numpy.log10(y0sd)
    logypsd = 10. * numpy.log10(ypsd)
    logupsd = 10. * numpy.log10(upsd)
    # crop time series
    tdata, (y0dat, udata, ydata) = crop_time_series(circuit.time(), (y0dat, udata, ydata))
    #tdata /= 1e3
    #freqs *= 1e3
    #fpsd  *= 1e3
    # plot
    axes[0].plot(tdata, udata  , 'C2', label='$u$'  )
    axes[0].plot(tdata, ydata  , 'C3', label='$y$'  )
    axes[0].plot(tdata, y0dat  , 'C0', label='$y_0$')
    axes[1].plot(fpsd , logupsd, 'C2', label='$u$'  )
    axes[1].plot(fpsd , logypsd, 'C3', label='$y$'  )
    axes[1].plot(fpsd , logy0sd, 'C0', label='$y_0$')
    axes[2].plot(freqs, logGmag, 'k'   )
    axes[2].plot(fpsd , logGpsd, 'C3'  )
    axes[2].plot(freqs, logGfit, 'C9--')
    axes[3].plot(freqs, Gphased, 'k'   )
    axes[3].plot(freqs, Gfitphd, 'C9--')

def noise_level(
    P: control.StateSpace
      | system.StateSpace
      | system.NonlinearStateSystem,
    maxes: tuple[Axes, Axes, Axes],
    paxes: tuple[Axes, Axes, Axes],
    seed: int | None=None
):
    H = model.alpha_gamma_filter()
    circuit = simul.Circuit(P, H, seed=seed)
    y0dat1, udata, ydata1 = circuit.stimulate(uamp=5e-3, namp=.1)
    circuit.seed = None if circuit.seed is None else circuit.seed + 999
    y0dat2, udata, ydata2 = circuit.stimulate(uamp=5e-3, namp=1.)
    circuit.seed = None if circuit.seed is None else circuit.seed + 999
    y0dat3, udata, ydata3 = circuit.stimulate(uamp=5e-3, namp=5.)
    # spectral density
    fpsd, upsd = circuit.welch(udata)
    fpsd, y0sd1 = circuit.welch(y0dat1)
    fpsd, ypsd1 = circuit.welch(ydata1)
    fpsd, y0sd2 = circuit.welch(y0dat2)
    fpsd, ypsd2 = circuit.welch(ydata2)
    fpsd, y0sd3 = circuit.welch(y0dat3)
    fpsd, ypsd3 = circuit.welch(ydata3)
    # tf magnitude
    Gpsd1 = (ypsd1 - y0sd1) / upsd
    Gneg1 = Gpsd1 < 0.
    if Gneg1.any():
        Gpsd1 = Gpsd1 * numpy.logical_not(Gneg1)
        control.statesp.warn('negative magnitude encountered for noise 1')
    Gpsd2 = (ypsd2 - y0sd2) / upsd
    Gneg2 = Gpsd2 < 0.
    if Gneg2.any():
        Gpsd2 = Gpsd2 * numpy.logical_not(Gneg2)
        control.statesp.warn('negative magnitude encountered for noise 2')
    Gpsd3 = (ypsd3 - y0sd3) / upsd
    Gneg3 = Gpsd3 < 0.
    if Gneg3.any():
        Gpsd3 = Gpsd3 * numpy.logical_not(Gneg3)
        control.statesp.warn('negative magnitude encountered for noise 3')
    # fitting
    sdata = 2j * numpy.pi * fpsd
    poles = circuit.G.poles() if not system.is_delayed(circuit.G) else (circuit.G.nstates // 2) * 2
    G1 = fitting.magvectfit(sdata, Gpsd1, poles, niter=8)
    G2 = fitting.magvectfit(sdata, Gpsd2, poles, niter=8)
    G3 = fitting.magvectfit(sdata, Gpsd3, poles, niter=8)
    # frequency response
    freqs = circuit.frequency()
    Gresp = simul.freqresp(circuit.G, freqs)
    Gmag = Gresp.magnitude.squeeze()
    Gphase = Gresp.phase.squeeze()
    Gfitr1 = simul.freqresp(G1, freqs)
    Gfitm1 = Gfitr1.magnitude.squeeze()
    Gfitp1 = Gfitr1.phase.squeeze()
    Gfitr2 = simul.freqresp(G2, freqs)
    Gfitm2 = Gfitr2.magnitude.squeeze()
    Gfitp2 = Gfitr2.phase.squeeze()
    Gfitr3 = simul.freqresp(G3, freqs)
    Gfitm3 = Gfitr3.magnitude.squeeze()
    Gfitp3 = Gfitr3.phase.squeeze()
    # log conversion
    Gfitphd1 = numpy.rad2deg(numpy.unwrap(Gfitp1))
    logGfit1 = 20. * numpy.log10(Gfitm1)
    Gfitphd2 = numpy.rad2deg(numpy.unwrap(Gfitp2))
    logGfit2 = 20. * numpy.log10(Gfitm2)
    Gfitphd3 = numpy.rad2deg(numpy.unwrap(Gfitp3))
    logGfit3 = 20. * numpy.log10(Gfitm3)
    Gphased = numpy.rad2deg(numpy.unwrap(Gphase))
    logGmag = 20. * numpy.log10(Gmag)
    logGpsd1 = 10. * numpy.log10(Gpsd1)
    logGpsd2 = 10. * numpy.log10(Gpsd2)
    logGpsd3 = 10. * numpy.log10(Gpsd3)
    # plot
    for ax in maxes:
        ax.plot(freqs, logGmag, 'k')
    maxes[0].plot(freqs, logGfit1, 'C9--')
    maxes[1].plot(freqs, logGfit2, 'C9--')
    maxes[2].plot(freqs, logGfit3, 'C9--')
    maxes[1].scatter(fpsd , logGpsd2, 5., 'C3' )
    maxes[2].scatter(fpsd , logGpsd3, 5., 'C3' )
    maxes[0].scatter(fpsd , logGpsd1, 5., 'C3' )
    for ax in paxes:
        ax.plot(freqs, Gphased, 'k')
    paxes[0].plot(freqs, Gfitphd1, 'C9--')
    paxes[1].plot(freqs, Gfitphd2, 'C9--')
    paxes[2].plot(freqs, Gfitphd3, 'C9--')

def open_loop_control(
    P: control.StateSpace
      | system.StateSpace
      | system.NonlinearStateSystem,
    axes: tuple[Axes, Axes, Axes],
    seed: int | None=None
):
    H = model.alpha_gamma_filter()
    circuit = simul.Circuit(P, H, seed=seed)
    y0dat, udata, ydata, Cmag2, umag1 = circuit.open_loop_control(uamp=5e-3)
    # spectral density
    fpsd, y0sd = circuit.welch(y0dat)
    fpsd, upsd = circuit.welch(udata)
    fpsd, ypsd = circuit.welch(ydata)
    # log conversion
    logCmag = 10. * numpy.log10(Cmag2)
    logy0sd = 10. * numpy.log10(y0sd)
    logypsd = 10. * numpy.log10(ypsd)
    logupsd = 10. * numpy.log10(upsd)
    logCpsd = logypsd - logy0sd
    # crop time series
    tdata, (y0dat, udata, ydata) = crop_time_series(circuit.time(), (y0dat, udata, ydata))
    # plot
    freqs = circuit.frequency()
    axes[0].plot(tdata, udata  , 'C0', label='$u$'  )
    axes[0].plot(tdata, ydata  , 'C3', label='$y$'  )
    axes[0].plot(tdata, y0dat  , 'k' , label='$y_0$')
    axes[1].plot(fpsd , logupsd, 'C0', label='$u$'  )
    axes[1].plot(fpsd , logypsd, 'C3', label='$y$'  )
    axes[1].plot(fpsd , logy0sd, 'k' , label='$y_0$')
    axes[2].plot(freqs, logCmag, 'C3--')
    axes[2].plot(fpsd , logCpsd, 'C3'  )

def reference_based_control(
    P: control.StateSpace
      | system.StateSpace
      | system.NonlinearStateSystem,
    axes: tuple[Axes, Axes, Axes],
    seed: int | None=None,
    delay: numpy.ndarray=()
):
    H = model.alpha_gamma_filter()
    circuit = simul.Circuit(P, H, seed=seed)
    K = model.PID_controller(25., 1000.)
    I = system.identity(H.noutputs)
    freqs = circuit.frequency()
    Smag = simul.freqresp(I + H, freqs).magnitude.squeeze()
    delay = system._check_delay(delay)
    if delay.any():
        circuit.P = circuit.P.sample(circuit.dt, 'zoh')
        circuit.G = circuit.G.sample(circuit.dt, 'zoh')
        circuit.H = circuit.H.sample(circuit.dt, 'zoh')
        K = K.sample(circuit.dt, 'zoh')
        Theta = system.pure_delay(delay).sample(circuit.dt)
        K = (Theta * model.smith_predictor(circuit.G, K, Theta.delay)).sample_delays()
    y0dat, rdata, udata, ydata, Cmag2, Dmag2 = circuit.reference_based_control(K)
    # spectral density
    fpsd, y0sd = circuit.welch(y0dat)
    fpsd, upsd = circuit.welch(udata)
    fpsd, ypsd = circuit.welch(ydata)
    # log conversion
    logSmag = 20. * numpy.log10(Smag)
    logCmag = 10. * numpy.log10(Cmag2)
    logDmag = 10. * numpy.log10(Dmag2)
    logy0sd = 10. * numpy.log10(y0sd)
    logypsd = 10. * numpy.log10(ypsd)
    logupsd = 10. * numpy.log10(upsd)
    logCpsd = logypsd - logy0sd
    logDpsd = logupsd - logy0sd
    # crop time series
    tdata, (y0dat, udata, ydata) = crop_time_series(circuit.time(), (y0dat, udata, ydata))
    # plot
    axes[0].plot(tdata, udata  , 'C2', label='$u$'  )
    axes[0].plot(tdata, ydata  , 'C3', label='$y$'  )
    axes[0].plot(tdata, y0dat  , 'C0', label='$y_0$')
    axes[1].plot(fpsd , logupsd, 'C2', label='$u$'  )
    axes[1].plot(fpsd , logypsd, 'C3', label='$y$'  )
    axes[1].plot(fpsd , logy0sd, 'C0', label='$y_0$')
    #axes[2].plot(freqs, logDmag, 'C0--')
    #axes[2].plot(fpsd , logDpsd, 'C0'  )
    axes[2].plot(freqs, logSmag, 'k'   )
    axes[2].plot(freqs, logCmag, 'C3--')
    #axes[2].plot(fpsd , logCpsd, 'C3' )

def model_based_control(
    P: control.StateSpace
      | system.StateSpace
      | system.NonlinearStateSystem,
    axes: tuple[Axes, Axes, Axes],
    seed: int | None=None,
    delay: numpy.ndarray=()
):
    H = model.alpha_gamma_filter()
    circuit = simul.Circuit(P, H, seed=seed)
    I = system.identity(H.noutputs)
    freqs = circuit.frequency()
    delay = system._check_delay(delay)
    if delay.any():
        circuit.P = circuit.P.sample(circuit.dt, 'zoh')
        circuit.G = circuit.G.sample(circuit.dt, 'zoh')
        circuit.H = circuit.H.sample(circuit.dt, 'zoh')
        Theta = system.pure_delay(delay).sample(circuit.dt)
        Phi = model.discrete_predictor(10., int(Theta.delay))
        H = model.alpha_gamma_filter(c1=1.  / numpy.abs(Phi(numpy.exp(2j * numpy.pi * 10. * circuit.dt))),
                                     c2=-.5 / numpy.abs(Phi(numpy.exp(2j * numpy.pi * 40. * circuit.dt))))
        H = system.StateSpace(Phi * H.sample(Theta.dt, 'zoh'), delay=Theta.delay, delay_type='output').sample_delays()
        y0dat, udata, ydata, Cmag2, Dmag2 = circuit.model_based_control(H)

    else:
        y0dat, udata, ydata, Cmag2, Dmag2 = circuit.model_based_control()
    Smag = simul.freqresp(I + circuit.H, freqs).magnitude.squeeze()
    # spectral density
    fpsd, y0sd = circuit.welch(y0dat)
    fpsd, upsd = circuit.welch(udata)
    fpsd, ypsd = circuit.welch(ydata)
    # log conversion
    logSmag = 20. * numpy.log10(Smag)
    logCmag = 10. * numpy.log10(Cmag2)
    logDmag = 10. * numpy.log10(Dmag2)
    logy0sd = 10. * numpy.log10(y0sd)
    logypsd = 10. * numpy.log10(ypsd)
    logupsd = 10. * numpy.log10(upsd)
    logCpsd = logypsd - logy0sd
    logDpsd = logupsd - logy0sd
    # crop time series
    tdata, (y0dat, udata, ydata) = crop_time_series(circuit.time(), (y0dat, udata, ydata))
    # plot
    axes[0].plot(tdata, udata  , 'C2', label='$u$'  )
    axes[0].plot(tdata, ydata  , 'C3', label='$y$'  )
    axes[0].plot(tdata, y0dat  , 'C0', label='$y_0$')
    axes[1].plot(fpsd , logupsd, 'C2', label='$u$'  )
    axes[1].plot(fpsd , logypsd, 'C3', label='$y$'  )
    axes[1].plot(fpsd , logy0sd, 'C0', label='$y_0$')
    #axes[2].plot(freqs, logDmag, 'C0--')
    #axes[2].plot(fpsd , logDpsd, 'C0'  )
    axes[2].plot(freqs, logSmag, 'k'   )
    axes[2].plot(freqs, logCmag, 'C3--')
    #axes[2].plot(fpsd , logCpsd, 'C3' )

def fit_model_based_control(
    P: control.StateSpace
      | system.StateSpace
      | system.NonlinearStateSystem,
    axes: tuple[Axes, Axes, Axes, Axes],
    seed: int | None=None,
    delay: numpy.ndarray=()
):
    H = model.alpha_gamma_filter()
    circuit = simul.Circuit(P, H, seed=seed)
    # fitting
    y0dat, udata, ydata = circuit.stimulate(uamp=5e-3)
    # spectral density
    fpsd, y0sd = circuit.welch(y0dat)
    fpsd, upsd = circuit.welch(udata)
    fpsd, ypsd = circuit.welch(ydata)
    # tf magnitude
    Gpsd = (ypsd - y0sd) / upsd
    Gneg = Gpsd < 0.
    if Gneg.any():
        Gpsd = Gpsd * numpy.logical_not(Gneg)
        control.statesp.warn('negative magnitude encountered')
    # fitting
    sdata = 2j * numpy.pi * fpsd
    poles = circuit.G.poles() if not system.is_delayed(circuit.G) else (circuit.G.nstates // 2) * 2
    G = fitting.magvectfit(sdata, Gpsd, poles, niter=8)
    # frequency response
    freqs = circuit.frequency()
    Gresp = simul.freqresp(circuit.G, freqs)
    Gfitr = simul.freqresp(G, freqs)
    Gmag = Gresp.magnitude.squeeze()
    Gphase = Gresp.phase.squeeze()
    Gfitm = Gfitr.magnitude.squeeze()
    Gfitp = Gfitr.phase.squeeze()
    # closed-loop
    I = system.identity(H.noutputs)
    freqs = circuit.frequency()
    delay = system._check_delay(delay)
    if delay.any():
        circuit.P = circuit.P.sample(circuit.dt, 'zoh')
        circuit.G = circuit.G.sample(circuit.dt, 'zoh')
        Theta = system.pure_delay(delay).sample(circuit.dt)
        Phi = model.discrete_predictor(10., int(Theta.delay))
        Phi.dt = circuit.dt
        H = model.alpha_gamma_filter(c1=1.  / numpy.abs(Phi(numpy.exp(2j * numpy.pi * 10. * circuit.dt))),
                                     c2=-.5 / numpy.abs(Phi(numpy.exp(2j * numpy.pi * 40. * circuit.dt))))
        K11 = system.div(system.div(H, I + circuit.H), G)
        K11 = Phi * K11.sample(circuit.dt, 'zoh')
        #K11 = system.div(model.alpha_gamma_filter(f1=11, c2=-.55, f2=43.).feedback(I), G)
        circuit.H = circuit.H.sample(circuit.dt, 'zoh')
        K = system.pure_delay(5, circuit.dt) * K11
        H = (Theta * Phi * H.sample(Theta.dt, 'zoh')).sample_delays()
        G = G.sample(circuit.dt)
        y0dat, udata, ydata, Cmag2, Dmag2 = circuit.model_based_control(H, G, K)
    else:
        y0dat, udata, ydata, Cmag2, Dmag2 = circuit.model_based_control(None, G)
    Smag = simul.freqresp(I + circuit.H, freqs).magnitude.squeeze()
    # spectral density
    fpsd, y0sd = circuit.welch(y0dat)
    fpsd, upsd = circuit.welch(udata)
    fpsd, ypsd = circuit.welch(ydata)
    # log conversion
    Gphased = numpy.rad2deg(numpy.unwrap(Gphase))
    Gfitphd = numpy.rad2deg(numpy.unwrap(Gfitp))
    logGfit = 20. * numpy.log10(Gfitm)
    logGmag = 20. * numpy.log10(Gmag)
    logSmag = 20. * numpy.log10(Smag)
    logCmag = 10. * numpy.log10(Cmag2)
    logy0sd = 10. * numpy.log10(y0sd)
    logypsd = 10. * numpy.log10(ypsd)
    logupsd = 10. * numpy.log10(upsd)
    # crop time series
    tdata, (y0dat, udata, ydata) = crop_time_series(circuit.time(), (y0dat, udata, ydata))
    #tdata /= 1e3
    #freqs *= 1e3
    #fpsd  *= 1e3
    # plot
    axes[2].plot(fpsd , logupsd, 'C2', label='$u$'  )
    axes[2].plot(fpsd , logypsd, 'C3', label='$y$'  )
    axes[2].plot(fpsd , logy0sd, 'C0', label='$y_0$')
    axes[3].plot(freqs, logSmag, 'k'   )
    axes[3].plot(freqs, logCmag, 'C3--')
    axes[0].plot(freqs, logGmag, 'k'   )
    axes[0].plot(freqs, logGfit, 'C9--')
    axes[1].plot(freqs, Gphased, 'k'   )
    axes[1].plot(freqs, Gfitphd, 'C9--')

def predictor_stability(ax: Axes):
    Ts = 1e-3
    H = model.alpha_gamma_filter()
    I = system.identity(H.noutputs)
    adata = numpy.linspace(-.25, 1., 200)
    mdata = numpy.empty((3, len(adata)))
    for i, delay in enumerate((3, 5, 10)):
        for j, a in enumerate(adata):
            phi = model.discrete_predictor_from_pole(a, delay)
            Hp = model.alpha_gamma_filter(
                c1=1.  / numpy.abs(phi(numpy.exp(2j * numpy.pi * 10. * Ts))),
                c2=-.5 / numpy.abs(phi(numpy.exp(2j * numpy.pi * 40. * Ts)))
            )
            GK = system.div(Hp, I + H).sample(Ts)
            S = I.feedback(system.StateSpace(phi * GK, delay=delay, delay_type='output').sample_delays(), 1)
            mdata[i, j] = numpy.abs(S.poles()).max()
    # plot
    ax.plot(adata, 20. * numpy.log10(mdata[0]), 'k'  , label='$\\tau = 3$ ms')
    ax.plot(adata, 20. * numpy.log10(mdata[1]), 'k--', label='$\\tau = 5$ ms')
    ax.plot(adata, 20. * numpy.log10(mdata[2]), 'k:' , label='$\\tau = 10$ ms')
    ax.fill_between(adata, 0., mdata.max(), alpha=.4, color='k', label='unstable region')
    ax.annotate('unstable region', (adata.max() / 2.,  .125), (adata.max() / 2.,  .125), color='k')
    ax.annotate(  'stable region', (adata.max() / 2., -.2  ), (adata.max() / 2., -.2  ), color='k')
    '''
    import csv
    with open('predictor_stability_data', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(('a', 'pole magnitude (tau=3ms)', 'pole magnitude (tau=5ms)', 'pole magnitude (tau=10ms)'))
        writer.writerows(zip(adata, mdata[0], mdata[1], mdata[2]))
    '''
