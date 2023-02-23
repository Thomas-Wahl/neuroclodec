import numpy
import system
import control
import spectral
import synthesis


def freqresp(sys, freqs, squeeze=None) -> control.FrequencyResponseData:
    return control.frequency_response(sys, 2. * numpy.pi * freqs, squeeze)


def input_output_response(
    sys,
    T=None,
    U=0.,
    transpose=False,
    interpolate=False,
    return_x=None,
    squeeze=None,
) -> control.TimeResponseData:
    X0 = 0.
    if system.is_delayed(sys):
        if sys.isctime():
            if T is None:
                raise ValueError('``T`` must be specified for delayed systems')
            else:
                Ts = T[1] - T[0]
            sys = sys.sample(Ts)
            if system.is_delayed(sys):
                sys = sys.sample_delays()
    if isinstance(sys, control.lti.LTI):
        return control.forced_response(sys, T, U, X0, transpose, interpolate, return_x, squeeze)
    else:
        try:
            X0 = system.find_eqpt(sys)
            if X0 is None or None in X0:
                raise ValueError
        except (NotImplementedError, ValueError):
            X0 = 0.
            Tmask = T < 1.
            resp = control.input_output_response(sys, T[Tmask], U[..., Tmask], X0, return_x=True)
            X0 = resp.x[..., -1]
        return control.input_output_response(sys, T, U, X0, {}, transpose, return_x, squeeze)


def get_seeds(seed: int | None =None, size=1, offset=0):
    if seed is None:
        return numpy.full(size, None)
    else:
        rs = numpy.random.RandomState(seed + offset)
        return rs.randint(0, 0x80000000, size)


def signal_labels(label: str, n=1):
    return (label,) if n == 1 else tuple(label + str(i + 1) for i in range(n))


class Circuit:
    '''Closed-loop circuit based on a plant P and a filter H.'''
    def __init__(self,
        P: control.StateSpace | system.NonlinearStateSystem,
        H: control.StateSpace,
        ninputs=1,
        tmax=30.,
        dt=1e-3,
        fmax=80.,
        fres=1000,
        seed=None
    ):
        self.P = P
        self.H = H
        self.tmax = tmax
        self.dt = dt
        self.fmax = fmax
        self.fres = fres
        self.seed = seed
        G = P[:, :ninputs]
        self.G = G.linearize() if isinstance(G, system.NonlinearStateSystem) else G
    
    def time(self):
        return numpy.arange(0., self.tmax, self.dt)
    
    def frequency(self):
        return numpy.linspace(0., self.fmax, self.fres)
    
    def welch(self, x, fstep=1.):
        return spectral.welch(x, self.fs, (0., self.fmax, fstep))
    
    def handle_delay(self, K: control.StateSpace | system.StateSpace):
        P = self.P.copy()
        if system.is_delayed(P) or system.is_delayed(K):
            if P.isctime():
                P = P.sample(self.dt, 'foh')
            if system.is_delayed(P):
                P = P.sample_delays()
            if K.isctime():
                K = K.sample(self.dt, 'foh')
            if system.is_delayed(K):
                K = K.sample_delays()
        return P, K
    
    def stimulate(self, uamp=1., namp=1.):
        seeds = get_seeds(self.seed, 3, 3)
        # simulation
        tdata = self.time()
        ndat1 = synthesis.white_noise(self.nnoises, len(tdata), self.dt, seeds[0]) * namp
        ndat2 = synthesis.white_noise(self.nnoises, len(tdata), self.dt, seeds[1]) * namp
        udata = synthesis.phase_noise(self.ninputs, len(tdata), self.dt, seeds[2]) * uamp
        y0rsp = input_output_response(self.P, tdata, numpy.pad(ndat1, ((self.ninputs, 0), (0, 0))))
        yresp = input_output_response(self.P, tdata, numpy.vstack((udata, ndat2)))
        # squeeze
        y0dat = y0rsp.y.squeeze()
        ydata = yresp.y.squeeze()
        udata = udata.squeeze()
        return y0dat, udata, ydata
    
    def open_loop_control(self, uamp=1.):
        seeds = get_seeds(self.seed, 3)
        I = numpy.identity(self.noutputs)
        G0 = self.P[:, self.ninputs:]
        G0 = G0.linearize() if isinstance(G0, system.NonlinearStateSystem) else G0
        # frequency response
        freqs = self.frequency()
        GHmg2 = freqresp(self.G * self.H, freqs).magnitude**2
        G0mg2 = freqresp(     G0        , freqs).magnitude**2
        y0mg2 = G0mg2.sum(1)
        Cmag2 = I + uamp**2 * GHmg2 / y0mg2
        ufrsp = freqresp(uamp * self.H, freqs)
        # simulation
        tdata = self.time()
        ndat1 = synthesis.white_noise(self.nnoises , len(tdata), self.dt, seeds[0])
        ndat2 = synthesis.white_noise(self.nnoises , len(tdata), self.dt, seeds[1])
        rdata = synthesis.phase_noise(self.noutputs, len(tdata), self.dt, seeds[2])
        uresp = input_output_response(uamp * self.H, tdata, rdata)
        y0rsp = input_output_response(self.P, tdata, numpy.pad(ndat1, ((self.ninputs, 0), (0, 0))))
        yresp = input_output_response(self.P, tdata, numpy.vstack((uresp.y, ndat2)))
        # squeeze
        Cmag2 = Cmag2.squeeze()
        umag1 = ufrsp.magnitude.squeeze()
        y0dat = y0rsp.y.squeeze()
        udata = uresp.y.squeeze()
        ydata = yresp.y.squeeze()
        return y0dat, udata, ydata, Cmag2, umag1
    
    def reference_based_control(self, K: control.StateSpace | system.StateSpace):
        seeds = get_seeds(self.seed, 2, 1)
        # transfer function
        I = system.identity(self.noutputs)
        F = I + self.H
        T = (self.G * K).feedback(I)
        S = I - T
        KS = system.feedback(K, self.G)
        # frequency response
        I = numpy.identity(self.noutputs)
        freqs = self.frequency()
        Smag2 = freqresp( S    , freqs).magnitude**2
        TFmg2 = freqresp( T * F, freqs).magnitude**2
        KSmg2 = freqresp(KS    , freqs).magnitude**2
        KSFm2 = freqresp(KS * F, freqs).magnitude**2
        Cmag2 = Smag2 + TFmg2
        Dmag2 = KSmg2 + KSFm2
        # closed-loop
        P, K = self.handle_delay(K)
        sub = system.summing_junction(self.noutputs, -1)
        input  = signal_labels('u', self.ninputs )
        noise  = signal_labels('ξ', self.nnoises )
        error  = signal_labels('e', self.noutputs)
        refer  = signal_labels('r', self.noutputs)
        output = signal_labels('y', self.noutputs)
        if isinstance(P, control.InputOutputSystem):
            P.input_index  = {label: index for label, index in zip(input + noise, P.input_index.values() )}
            P.output_index = {label: index for label, index in zip(output       , P.output_index.values())}
            P.name = 'P'
        else:
            P = control.LinearIOSystem(P  , inputs=input+noise , output=output, name='P')
        K     = control.LinearIOSystem(K  , input =error       , output=input , name='K')
        sub   = control.LinearIOSystem(sub, inputs=refer+output, output=error           )
        syslist = [P, K, sub]
        inplist = list(refer  + noise)
        outlist = list(output + input)
        C = control.interconnect(syslist, None, inplist, outlist)
        # check_stability
        if isinstance(C, control.lti.LTI):
            try:
                stable = system.is_stable(C)
            except ValueError:
                stable = True
            if not stable:
                raise ValueError('closed-loop is unstable.')
        # simulation
        tdata = self.time()
        ndat1 = synthesis.white_noise(self.nnoises, len(tdata), self.dt, seeds[0])
        ndat2 = synthesis.white_noise(self.nnoises, len(tdata), self.dt, seeds[1])
        y0rsp = input_output_response(self.P, tdata, numpy.pad(ndat1, ((self.ninputs, 0), (0, 0))))
        rresp = input_output_response(     F, tdata, y0rsp.y)
        yresp = input_output_response(     C, tdata, numpy.vstack((rresp.y, ndat2)))
        # squeeze
        Cmag2 = Cmag2.squeeze()
        Dmag2 = Dmag2.squeeze()
        y0dat = y0rsp.y.squeeze()
        rdata = rresp.y.squeeze()
        udata = yresp.y[self.noutputs:].squeeze()
        ydata = yresp.y[:self.noutputs].squeeze()
        return y0dat, rdata, udata, ydata, Cmag2, Dmag2
    
    def model_based_control(self, H: system.StateSpace | None=None, G: control.StateSpace | None=None, K: control.StateSpace | None=None):
        seeds = get_seeds(self.seed, 2, 2)
        # transfer function
        I = system.identity(self.noutputs)
        if G is None:
            if H is None:
                S = I + self.H
                D = system.div(self.H, self.G)
                K = system.div(system.feedback(self.H, I), self.G) if K is None else K
            else:
                S = system.feedback(I, system.div(H, I + self.H), 1)
                D = system.div(system.div(H, self.G), (I + self.H - H))
                K = system.div(system.div(H, self.G), I + self.H) if K is None else K
        else:
            if K is None:
                if H is None:
                    K = system.div(system.feedback(self.H, I), G)
                else:
                    K = system.div(system.div(H, G), I + self.H)
            S = system.feedback(I, self.G * K, 1)
            D = system.div(self.H, G)######## INCORRECT TODO: change
        # frequency response
        I = numpy.identity(self.noutputs)
        freqs = self.frequency()
        Cmag2 = freqresp(S, freqs).magnitude**2
        Dmag2 = freqresp(D, freqs).magnitude**2
        # closed-loop
        P, K = self.handle_delay(K)
        input  = signal_labels('u', self.ninputs )
        noise  = signal_labels('ξ', self.nnoises )
        output = signal_labels('y', self.noutputs)
        if isinstance(P, control.InputOutputSystem):
            P.input_index  = {label: index for label, index in zip(input + noise, P.input_index.values() )}
            P.output_index = {label: index for label, index in zip(output       , P.output_index.values())}
            P.name = 'P'
        else:
            P = control.LinearIOSystem(P  , inputs=input+noise , output=output, name='P')
        K     = control.LinearIOSystem(K  , input =output      , output=input , name='K')
        syslist = [P, K]
        inplist = list(         noise)
        outlist = list(output + input)
        C = control.interconnect(syslist, None, inplist, outlist)
        # check_stability
        if isinstance(C, control.lti.LTI):
            try:
                stable = system.is_stable(C)
            except ValueError:
                stable = True
            if not stable:
                raise ValueError('closed-loop is unstable.')
        # simulation
        P = self.P.sample_delays() if isinstance(self.P, system.StateSpace) else self.P
        tdata = self.time()
        ndat1 = synthesis.white_noise(self.nnoises, len(tdata), self.dt, seeds[0])
        ndat2 = synthesis.white_noise(self.nnoises, len(tdata), self.dt, seeds[1])
        y0rsp = input_output_response(P, tdata, numpy.pad(ndat1, ((self.ninputs, 0), (0, 0))))
        yresp = input_output_response(C, tdata,           ndat2)
        # squeeze
        Cmag2 = Cmag2.squeeze()
        Dmag2 = Dmag2.squeeze()
        y0dat = y0rsp.y.squeeze()
        udata = yresp.y[self.noutputs:].squeeze()
        ydata = yresp.y[:self.noutputs].squeeze()
        return y0dat, udata, ydata, Cmag2, Dmag2
    
    @property
    def ninputs(self):
        return self.G.ninputs
    
    @property
    def nnoises(self):
        return self.P.ninputs - self.ninputs
    
    @property
    def noutputs(self):
        return self.P.noutputs

    @property
    def fs(self):
        return 1. / self.dt
    
def test(K: control.StateSpace | system.StateSpace):
    from matplotlib import pyplot
    pyplot.style.use('dark_background')
    pyplot.close(pyplot.gcf())
    freqs = numpy.linspace(0., 80., 1000)
    pyplot.plot(freqs, freqresp(K, freqs).fresp.real.squeeze(), 'w')
    pyplot.margins(x=0.)
    pyplot.show()
