import numpy
import control
import warnings
from scipy import linalg
from scipy import signal
from scipy import optimize
from scipy import integrate


def _check_delay(delay: numpy.ndarray):
    delay = numpy.atleast_1d(numpy.squeeze(delay))
    if delay.ndim > 1:
        raise ValueError(f'delay cannot be more than 1-dimensional: {delay.ndim = }')
    return delay


def _check_io_delay(delay: numpy.ndarray):
    delay = numpy.atleast_2d(delay)
    if delay.ndim > 2:
        raise ValueError(f'IO delay cannot be more than 2-dimensional: {delay.ndim = }')
    return delay


def is_delayed(sys: control.InputOutputSystem):
    return hasattr(sys, 'is_delayed') and sys.is_delayed()


def is_stable(sys: control.lti.LTI):
    if sys.isctime():
        return (sys.poles() <= 0.).all()
    else:
        return (numpy.abs(sys.poles()) <= 1.).all()


def is_proper(sys: control.lti.LTI):
    if isinstance(sys, control.TransferFunction):
        return not any([[len(num) for num in col] for col in sys.num] >
                       [[len(den) for den in col] for col in sys.den])
    else:
        return True


def is_strictly_proper(sys: control.lti.LTI):
    if isinstance(sys, control.TransferFunction):
        return not any([[len(num) for num in col] for col in sys.num] >=
                       [[len(den) for den in col] for col in sys.den])
    elif isinstance(sys, control.StateSpace):
        return not sys.D.any()
    else:
        raise TypeError(f'sys must be {control.StateSpace.__name__} or {control.TransferFunction.__name__}: '
                        f'type(sys) = {type(sys).__name__}')


def inv(sys: control.lti.LTI):
    if isinstance(sys, control.StateSpace):
        D = numpy.linalg.inv(sys.D)
        A = sys.A - sys.B @ D @ sys.C
        B =         sys.B @ D
        C =                -D @ sys.C
        return control.StateSpace(A, B, C, D, sys.dt)
    else:
        return 1. / sys


def div(sys: control.lti.LTI, other: control.lti.LTI):
    if isinstance(sys, control.StateSpace):
        if is_strictly_proper(other):
            sys = control.xferfcn._convert_to_transfer_function(sys)
            return control.statesp._convert_to_statespace(sys / other)
        else:
            return sys * inv(other)
    else:
        return sys / other


def feedback(sys: control.lti.LTI, other: control.lti.LTI, sign=-1):
    if not isinstance(sys, StateSpace):
        if isinstance(other, control.TransferFunction):
            sys = control.xferfcn._convert_to_transfer_function(sys)
        else:
            sys = control.statesp._convert_to_statespace(sys)
            if isinstance(other, StateSpace):
                sys = StateSpace(sys, reduced=False)
    return sys.feedback(other, sign)


def symmetric(sys: control.TransferFunction):
    warnings.simplefilter('ignore', signal.BadCoefficients)
    if is_delayed(sys):
        if sys.purely_io_delay():
            sys = sys.H12 * sys.H21
        else:
            raise ValueError('cannot compute symmetric of system with non purely IO delay')
    syslist = sys.returnScipySignalLTI()
    num = tuple(tuple([] for _ in range(sys.ninputs)) for _ in range(sys.noutputs))
    den = tuple(tuple([] for _ in range(sys.ninputs)) for _ in range(sys.noutputs))
    for i in range(sys.noutputs):
        for j in range(sys.ninputs):
            zpk = signal.ZerosPolesGain(syslist[i][j])
            if sys.isctime():
                symz = numpy.hstack((zpk.zeros, -zpk.zeros))
                symp = numpy.hstack((zpk.poles, -zpk.poles))
            else:
                symz = numpy.hstack((zpk.zeros, 1. / zpk.zeros))
                symp = numpy.hstack((zpk.poles, 1. / zpk.poles))
            symk = zpk.gain**2
            b, a = signal.zpk2tf(symz, symp, symk)
            num[i][j].extend(b)
            den[i][j].extend(a)
    warnings.simplefilter('default', signal.BadCoefficients)
    return control.TransferFunction(num, den, sys.dt)


def derivative(sys: control.TransferFunction):
    if is_delayed(sys):
        if sys.has_delay_feedback():
            raise ValueError('cannot compute derivative of system with delay feedback')
        else:
            H11 = sys.H11
            H12 = sys.H12
            H21 = sys.H21
            Thetasys = control.StateSpace((), (), (), numpy.identity(sys.ndelays), sys.dt)
            Theta = StateSpace(Thetasys, delay=sys.delay, delay_type='input')
            Delay = control.StateSpace((), (), (), numpy.diag(-sys.delay), sys.dt)
            return derivative(H11) + ((derivative(H12) + H12 * Delay) * H21 + H12 * derivative(H21)) * Theta
    sys = control.xferfcn._convert_to_transfer_function(sys)
    num = tuple(tuple([] for _ in range(sys.ninputs)) for _ in range(sys.noutputs))
    den = tuple(tuple([] for _ in range(sys.ninputs)) for _ in range(sys.noutputs))
    for i in range(sys.noutputs):
        for j in range(sys.ninputs):
            u = numpy.poly1d(sys.num[i][j])
            v = numpy.poly1d(sys.den[i][j])
            num[i][j].extend((u.deriv() * v - u * v.deriv()).c)
            den[i][j].extend((v**2).c)
    return control.TransferFunction(num, den, sys.dt)


def extremums(sys: control.lti.LTI, tol=1e-6):
    sym = symmetric(sys)
    symderiv1 = derivative(sym)
    symderiv2 = derivative(symderiv1)
    peaks   = numpy.full((sys.noutputs, sys.ninputs), None)
    valleys = numpy.full((sys.noutputs, sys.ninputs), None)
    filter_zeros = (lambda z: 1j * z[numpy.abs(z.real) < tol].imag) \
                   if sys.isctime() else \
                   (lambda z: numpy.abs(z[numpy.abs(z) - 1. < tol]))
    for i in range(sys.noutputs):
        for j in range(sys.ninputs):
            zeros = filter_zeros(symderiv1[i, j].zeros())
            curve = symderiv2[i, j](zeros).real
            peaks  [i, j] = zeros[curve < 0.]
            valleys[i, j] = zeros[curve > 0.]
    return peaks, valleys


def find_eqpt(sys: control.NonlinearIOSystem, x0=0., ueq=0., **kwargs):
    if isinstance(sys, (NonlinearStateSystem, NonlinearDynamicsSystem)):
        return sys.find_eqpt(x0, ueq, **kwargs)
    else:
        return control.find_eqpt(sys, x0, ueq, **kwargs)


def identity(n: int, dt=None):
    return control.StateSpace((), (), (), numpy.identity(n), dt)


def summing_junction(n: int, sign=1, dt=None):
    I = numpy.identity(n)
    return control.StateSpace((), (), (), numpy.hstack((I, sign * I)), dt)


def pure_delay(delay: numpy.ndarray, dt=None):
    delay = _check_delay(delay)
    if delay.any():
        if dt:
            delay = delay.astype(int)
            Ad = linalg.block_diag(*(numpy.eye(d,    k= -1) for d in delay))
            Bd = linalg.block_diag(*(numpy.eye(d, 1       ) for d in delay))
            Cd = linalg.block_diag(*(numpy.eye(1, d, k=d-1) for d in delay))
            return control.StateSpace(Ad, Bd, Cd, 0., dt)
        else:
            D = numpy.identity(len(delay))
            return StateSpace(control.StateSpace((), (), (), D, dt), delay=delay, delay_type='input')
    else:
        return control.StateSpace((), (), (), numpy.identity(max(1, len(delay))), dt)


class StateSpace(control.lti.LTI):
    # Allow ndarray * StateSpace to give StateSpace._rmul_() priority
    __array_priority__ = 12     # override ndarray and matrix types

    def __new__(cls,
        H11: control.StateSpace,
        B2: numpy.ndarray=(),
        C2: numpy.ndarray=(),
        D12: numpy.ndarray=(),
        D21: numpy.ndarray=(),
        D22: numpy.ndarray=(),
        delay: numpy.ndarray=(),
        delay_type='',
        reduced=True
    ):
        if reduced:
            if delay_type.strip():
                return super().__new__(cls) if numpy.any(delay) else control.StateSpace(H11)
            else:
                # if H12 and H21
                if ((numpy.any(D12) or (numpy.any(B2) and numpy.any(H11.C)))
                and (numpy.any(D21) or (numpy.any(H11.B) and numpy.any(C2)))):
                    return super().__new__(cls)
                else:
                    return control.StateSpace(H11)
        else:
            return super().__new__(cls)

    def __init__(self,
        H11: control.StateSpace,
        B2: numpy.ndarray=(),
        C2: numpy.ndarray=(),
        D12: numpy.ndarray=(),
        D21: numpy.ndarray=(),
        D22: numpy.ndarray=(),
        delay: numpy.ndarray=(),
        delay_type='',
        reduced=True
    ):
        H11 = control.statesp._convert_to_statespace(H11)
        raw_delay_type = delay_type
        delay_type = delay_type.strip().lower()
        if delay_type:
            if any((B2, C2, D12, D21, D22)):
                raise ValueError('cannot specify both delay type and delay matrices')
            # H11 = 0(ny nu)
            D11 = numpy.zeros_like(H11.D)
            if delay_type == 'io':
                delay = _check_io_delay(delay)
                # H22 = 0(nd nd)
                D22 = numpy.zeros((delay.size, delay.size))
                raise NotImplementedError('IO delay specification not implemented yet')
            elif delay_type == 'output':
                delay = _check_delay(delay)
                # H22 = 0(nd nd)
                D22 = numpy.zeros((len(delay), len(delay)))

                B1  = H11.B
                C1  = numpy.zeros_like(H11.C)
                B2  = numpy.zeros((H11.nstates, len(delay)))
                if len(delay) == H11.noutputs:
                    # H12 = I(nd)
                    # H21 = H11
                    C2  = H11.C
                    D21 = H11.D
                    D12 = numpy.identity(len(delay))
                elif len(delay) == 1:
                    # H12 = 1(ny)
                    # H21 = 1(ny) @ H11
                    yones = numpy.ones(H11.noutputs)
                    C2  = yones @ H11.C
                    D21 = yones @ H11.D
                    D12 = yones[numpy.newaxis].T
                else:
                    raise ValueError('delay must have a length of noutputs: '
                                    f'noutputs = {H11.noutputs}, {len(delay) = }')
            elif delay_type == 'input':
                delay = _check_delay(delay)
                # H22 = 0(nd nd)
                D22 = numpy.zeros((len(delay), len(delay)))

                C1  = H11.C
                B1  = numpy.zeros_like(H11.B)
                C2  = numpy.zeros((len(delay), H11.nstates))
                if len(delay) == H11.noutputs:
                    # H12 = H11
                    # H21 = I(nd)
                    B2  = H11.B
                    D12 = H11.D
                    D21 = numpy.identity(len(delay))
                elif len(delay) == 1:
                    # H12 = H11 @ 1(nu)
                    # H21 =       1(nu)
                    uones = numpy.ones(H11.ninputs)
                    B2  = H11.B @ uones
                    D12 = H11.D @ uones
                    D21 = uones[numpy.newaxis]
                else:
                    raise ValueError('delay must have a length of ninputs: '
                                    f'ninputs = {H11.ninputs}, {len(delay) = }')
            else:
                raise ValueError(f"unknown delay type: '{raw_delay_type}'")
            
            H11 = control.StateSpace(H11.A, B1, C1, D11, H11.dt)
        else:
            delay = _check_delay(delay)
            for name, M in ('B2', B2), ('D12', D12), ('D22', D22):
                M = numpy.atleast_2d(M)
                if M.shape[1] != len(delay):
                    raise ValueError(f'{name} must have ndelays columns: {len(delay) = }, {name}.shape[1] = {M.shape[1]}')
            for name, M in ('C2', B2), ('D21', D12), ('D22', D22):
                M = numpy.atleast_2d(M)
                if M.shape[1] != len(delay):
                    raise ValueError(f'{name} must have ndelays rows: {len(delay) = }, {name}.shape[0] = {M.shape[0]}') 
            B2  = numpy.reshape(B2,  (H11.nstates , len(delay) ))
            C2  = numpy.reshape(C2,  (len(delay)  , H11.nstates))
            D12 = numpy.reshape(D12, (H11.noutputs, len(delay) ))
            D21 = numpy.reshape(D21, (len(delay)  , H11.ninputs))
            D22 = numpy.reshape(D22, (len(delay)  , len(delay) ))
        
        super().__init__(H11.ninputs, H11.noutputs, H11.nstates, dt=H11.dt)
        self.H11 = H11
        self.B2  = B2
        self.C2  = C2
        self.D12 = D12
        self.D21 = D21
        self.D22 = D22
        self.delay = delay

        self.__remove_zero_delays()
        #self.__remove_unreachable_delays() #this method erroneously removes reachable delays
        #self.__merge_duplicate_delays() #this method produces incorrect results
    
    def __mask_delay(self, delay_mask: numpy.ndarray):
        if not delay_mask.all():
            anti_mask = numpy.logical_not(delay_mask)
            Ik = numpy.diag(anti_mask)
            I  = numpy.identity(self.ndelays)

            D22_Ik = self.D22 @ Ik
            if D22_Ik.any():
                F = I - D22_Ik
                # Precompute F\C2, F\D21 and F\other.D22 (E = inv(F))
                # We can solve two linear systems in one pass, since the
                # coefficients matrix F is the same. Thus, we perform the LU
                # decomposition (cubic runtime complexity) of F only once!
                # The remaining back substitutions are only quadratic in runtime.
                E_CDD = numpy.linalg.solve(F, numpy.hstack((self.C2, self.D21, self.D22)))
                E_C2  = E_CDD[:, :self.nstates]
                E_D21 = E_CDD[:, self.nstates: self.nstates + self.ninputs]
                E_D22 = E_CDD[:,              self.nstates + self.ninputs:]
            else:
                E_C2  = self.C2
                E_D21 = self.D21
                E_D22 = self.D22
            Ik_E_C2  = Ik @ E_C2
            Ik_E_D21 = Ik @ E_D21
            Ik_E_D22 = Ik @ E_D22
            
            T = Ik_E_D22 + I
            
            A   = self.A   + self.B2  @ Ik_E_C2
            B1  = self.B1  + self.B2  @ Ik_E_D21
            C1  = self.C1  + self.D12 @ Ik_E_C2
            D11 = self.D11 + self.D12 @ Ik_E_D21
            H11 = control.StateSpace(A, B1, C1, D11, self.dt)
            B2  = (self.B2  @ T)[:, delay_mask]
            D12 = (self.D12 @ T)[:, delay_mask]
            D22 = (self.D22 @ T)[:, delay_mask]   [delay_mask]
            C2  = (self.C2  + self.D22 @ Ik_E_C2 )[delay_mask]
            D21 = (self.D21 + self.D22 @ Ik_E_D21)[delay_mask]
            delay = self.delay[delay_mask]

            self.H11 = H11
            self.B2  = B2
            self.C2  = C2
            self.D12 = D12
            self.D21 = D21
            self.D22 = D22
            self.delay = delay

    def __remove_zero_delays(self):
        self.__mask_delay(self.delay.astype(bool))
    
    # TODO: fix __remove_unreachable_delays removing reachable delays
    def __remove_unreachable_delays(self):
        if not self.D22.any():
            B1 = self.B1.any()
            C1 = self.C1.any()
            reachable = numpy.empty(self.ndelays, bool)
            for i in range(self.ndelays):
                # H12 and H21
                reachable[i] = (self.D12[:, i].any() and self.D21[i].any()) \
                            or (self.B2 [:, i].any() and self.C2 [i].any() and B1 and C1) \
                            or (self.D12[:, i].any() and self.C2 [i].any() and B1) \
                            or (self.B2 [:, i].any() and self.D21[i].any() and C1)
            if not reachable.all():
                hdata1 = self(10j)
                self.B2    = self.B2   [:, reachable]
                self.D12   = self.D12  [:, reachable]
                self.C2    = self.C2   [reachable]
                self.D21   = self.D21  [reachable]
                self.D22   = self.D22  [reachable][:, reachable]
                self.delay = self.delay[reachable]
                # check if accidentally removed reachable delays
                hdata2 = self(10j)
                err = numpy.abs(hdata2 - hdata1)
                if numpy.mean(err) > 1e-12:
                    raise ArithmeticError('removed reachable delays: '
                                         f'{numpy.arange(len(reachable))[numpy.logical_not(reachable)]}\n'
                                         f'{err = }')
    
    # TODO: implement _StateSpace__merge_duplicate_delays
    def __merge_duplicate_delays(self):
        delay = tuple(self.delay)
        delay_index = numpy.fromiter((delay.index(d) for d in delay), int)
        K = numpy.zeros((self.ndelays, self.ndelays), bool)
        for j in range(self.ndelays):
            i = delay.index(delay[j])
            K[i, j] = True
        print(K.shape)
        self.delay = self.delay[numpy.diag(K)]
        K = K[numpy.any(K, 1)]
        print(K.shape)
        #BDD = numpy.linalg.lstsq(K.T, numpy.vstack((self.B2, self.D12, K @ self.D22)).T, rcond=None)[0].T

        self.C2  = K @ self.C2
        self.D21 = K @ self.D21
        self.B2 = self.B2 @ K.T
        self.D12 = self.D12 @ K.T
        self.D22 = K @ self.D22 @ K.T
        #self.B2  = BDD[:self.nstates]
        #self.D12 = BDD[self.nstates: self.nstates + self.noutputs]
        #self.D22 = BDD[self.nstates + self.noutputs:]
        #self.delay = self.delay[delay_index == numpy.arange(self.ndelays)]
    
    def __str__(self):
        Mvars = 'A', 'B1', 'C1', 'D11', 'B2', 'C2', 'D12', 'D21', 'D22'
        values = self.A, self.B1, self.C1, self.D11, self.B2, self.C2, self.D12, self.D21, self.D22
        MvarMs = ((Mvar, str(value).replace('\n', '\n' + ' ' * (len(Mvar) + 3))) for Mvar, value in zip(Mvars, values))
        prefix = '\n\n'.join(f'{Mvar} = {M}' for Mvar, M in MvarMs)
        suffix = f'\n\ndt = {self.dt}' if self.isdtime() else ''
        return f'{prefix}\n\ndelay = {self.delay}{suffix}'
    
    def __repr__(self):
        matrices = self.A, self.B1, self.C1, self.D11, self.B2, self.C2, self.D12, self.D21, self.D22
        prefix = ', '.join(repr(numpy.asarray(M)) for M in matrices)
        suffix = f', {self.dt}' if self.isdtime() else ''
        return f'({prefix}, {self.delay}{suffix})'
    
    def __neg__(self):
        H11 = -self.H11
        D12 = -self.D12
        return type(self)(H11, self.B2, self.C2, D12, self.D21, self.D22, self.delay)
    
    def __add__(self, other):
        if isinstance(other, control.lti.LTI):
            if isinstance(other, type(self)):
                H11 = self.H11 + other.H11
                D12 = numpy.hstack((self.D12, other.D12))
                D21 = numpy.vstack((self.D21, other.D21))
                B2  = linalg.block_diag(self.B2 , other.B2 )
                C2  = linalg.block_diag(self.C2 , other.C2 )
                D22 = linalg.block_diag(self.D22, other.D22)
                delay = numpy.hstack((self.delay, other.delay))
                return type(self)(H11, B2, C2, D12, D21, D22, delay)
            else:
                H11 = self.H11 + other
                B2 = numpy.vstack((self.B2, numpy.zeros((other.nstates, self.ndelays))))
                C2 = numpy.hstack((self.C2, numpy.zeros((self.ndelays, other.nstates))))
                return type(self)(H11, B2, C2, self.D12, self.D21, self.D22, self.delay)
        else:
            H11 = self.H11 + other
            return type(self)(H11, self.B2, self.C2, self.D12, self.D21, self.D22, self.delay)
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other):
        if isinstance(other, control.lti.LTI):
            # Check to see if the right operator has priority
            if getattr(other, '__array_priority__', None) and \
               getattr(self , '__array_priority__', None) and \
               other.__array_priority__ > self.__array_priority__:
                return other.__rmul__(self)
            if isinstance(other, type(self)):
                H11 = self.H11 * other.H11
                B2  = linalg.block_diag(other.B2, self.B2)
                C2  = linalg.block_diag(other.C2, self.C2)
                B2[other.nstates:, :other.ndelays] = self.B1 @ other.D12
                C2[other.ndelays:, :other.nstates] = self.D21 @ other.C1
                D12 = numpy.hstack((self.D11 @ other.D12, self.D12))
                D21 = numpy.vstack((other.D21, self.D21 @ other.D11))
                D22 = linalg.block_diag(other.D22, self.D22)
                D22[other.ndelays:, :other.ndelays] = self.D21 @ other.D12
                delay = numpy.hstack((other.delay, self.delay))
                return type(self)(H11, B2, C2, D12, D21, D22, delay)
            else:
                H11 = self.H11 * other
                other = control.statesp._convert_to_statespace(other)
                B2  = numpy.vstack((numpy.zeros((other.nstates, self.ndelays)), self.B2))
                C2  = numpy.hstack((self.D21 @ other.C, self.C2))
                D21 = self.D21 @ other.D
                return type(self)(H11, B2, C2, self.D12, D21, self.D22, self.delay)
        else:
            H11 = other * self.H11 # hotfix control.StateSpace reversed scalar multiplication
            D21 = self.D21 * other
            return type(self)(H11, self.B2, self.C2, self.D12, D21, self.D22, self.delay)
    
    def __rmul__(self, other):
        if isinstance(other, control.lti.LTI):
            return type(self)(other, reduced=False) * self
        else:
            H11 = self.H11 * other # hotfix control.StateSpace reversed scalar multiplication
            D12 = other * self.D12
            return type(self)(H11, self.B2, self.C2, D12, self.D21, self.D22, self.delay)

    def __call__(self, x, squeeze=None, warn_infinite=True):
        out = self.horner(x, warn_infinite)
        return control.statesp._process_frequency_response(self, x, out, squeeze=squeeze)
    
    def horner(self, x, warn_infinite=True):
        x_arr = numpy.atleast_1d(numpy.squeeze(x))
        delay = -self.delay if self.has_delay_feedback() else self.delay
        theta = (lambda s: numpy.exp(-delay * s)) if self.isctime() else (lambda z: z**-delay)
        hout = self.H.horner(x, warn_infinite)
        hout11 = hout[:self.noutputs, :self.ninputs]
        hout12 = hout[:self.noutputs, self.ninputs:]
        hout21 = hout[self.noutputs:, :self.ninputs]
        out = numpy.empty_like(hout11)
        if self.has_delay_feedback():
            hout22 = hout[self.noutputs:, self.ninputs:]
            for i, x in enumerate(x_arr):
                invden = numpy.linalg.inv(numpy.diag(theta(x)) - hout22[:, :, i])
                out[:, :, i] = hout11[:, :, i] + hout12[:, :, i] @               invden @ hout21[:, :, i]
        else:
            for i, x in enumerate(x_arr):
                out[:, :, i] = hout11[:, :, i] + hout12[:, :, i] @ numpy.diag(theta(x)) @ hout21[:, :, i]
        return out
    
    def poles(self):
        if self.has_delay_feedback():
            raise ValueError('cannot compute the poles of a system with internal delay')
        else:
            return self.H11.poles()
    
    def zeros(self):
        if not self.is_delayed():
            return self.H11.zeros()
        elif self.purely_io_delay():
            return self.H.zeros() # TODO: fix slycot error in some cases
        else:
            raise ValueError('cannot compute the zeros of a non-purely io delay system')
    
    def feedback(self, other=1, sign=-1):
        if isinstance(other, control.lti.LTI):
            if isinstance(other, type(self)):
                H11 = self.H11.feedback(other.H11, sign)
                if not (sign or other.D11.any() or self.D11.any()):
                    E_D11 = other.D11
                    E_C1  = other.C1
                    E_D12 = other.D12
                else:
                    F = numpy.identity(self.noutputs) - sign * other.D11 @ self.D11
                    # Precompute F\other.D11, F\other.C1 and F\other.D12 (E = inv(F))
                    # We can solve two linear systems in one pass, since the
                    # coefficients matrix F is the same. Thus, we perform the LU
                    # decomposition (cubic runtime complexity) of F only once!
                    # The remaining back substitutions are only quadratic in runtime.
                    E_DCD = numpy.linalg.solve(F, numpy.hstack((other.D11, other.C1, other.D12)))
                    E_D11 = E_DCD[:, :other.ninputs]
                    E_C1  = E_DCD[:, other.ninputs: other.ninputs + other.nstates]
                    E_D12 = E_DCD[:,               other.ninputs + other.nstates:]

                T1 = numpy.identity(self.noutputs) + sign * self.D11 @ E_D11
                T2 = numpy.identity(self.ninputs ) + sign * E_D11 @ self.D11

                D12 = numpy.hstack((T1 @ self.D12,       sign * self.D11 @ E_D12))
                D21 = numpy.vstack((     self.D21 , other.D21 @ self.D11        ))  @ T2
                B2  = numpy.vstack((numpy.hstack(( self.B2  + sign * self.B1  @ E_D11 @ self.D12,             sign *  self.B1             @ E_D12)),
                                    numpy.hstack((other.B1                    @ T1    @ self.D12, other.B2  + sign * other.B1  @ self.D11 @ E_D12))))
                C2  = numpy.vstack((numpy.hstack(( self.C2  + sign * self.D21 @ E_D11 @ self.C1 ,             sign *  self.D21            @ E_C1 )),
                                    numpy.hstack((other.D21                   @ T1    @ self.C1 , other.C2  + sign * other.D21 @ self.D11 @ E_C1 ))))
                D22 = numpy.vstack((numpy.hstack(( self.D22 + sign * self.D21 @ E_D11 @ self.D12,             sign *  self.D21            @ E_D12)),
                                    numpy.hstack((other.D21                   @ T1    @ self.D12, other.D22 + sign * other.D21 @ self.D11 @ E_D12))))
                delay = numpy.hstack((self.delay, other.delay))
                return type(self)(H11, B2, C2, D12, D21, D22, delay)
            else:
                H11 = self.H11.feedback(other, sign)
                other = control.statesp._convert_to_statespace(other)
                if not (sign or other.D.any() or self.D11.any()):
                    E_D = other.D
                    E_C = other.C
                else:
                    F = numpy.identity(self.noutputs) - sign * other.D @ self.D11
                    E_DC = numpy.linalg.solve(F, numpy.hstack((other.D, other.C)))
                    E_C = E_DC[:, other.ninputs:]
                    E_D = E_DC[:, :other.ninputs]
                
                T1 = numpy.identity(self.noutputs) + sign * self.D11 @ E_D
                T2 = numpy.identity(self.ninputs ) + sign * E_D @ self.D11

                D12 = T1 @ self.D12
                D21 = self.D21 @ T2
                D22 = self.D22 + sign * self.D21 @ E_D @ self.D12
                B2  = numpy.vstack((self.B2 + sign * self.B1  @ E_D @ self.D12,       other.B   @ T1 @ self.D12))
                C2  = numpy.hstack((self.C2 + sign * self.D21 @ E_D @ self.C1 , sign * self.D21 @ E_C          ))
                return type(self)(H11, B2, C2, D12, D21, D22, self.delay)
        else:
            H11 = self.H11.feedback(other, sign)
            other = control.statesp._ssmatrix(other)
            if not (sign or other.any() or self.D11.any()):
                E_D = other
            else:
                F = numpy.identity(self.noutputs) - sign * other @ self.D11
                E_D = numpy.linalg.solve(F, other)
            
            T1 = numpy.identity(self.noutputs) + sign * self.D11 @ E_D
            T2 = numpy.identity(self.ninputs ) + sign * E_D @ self.D11

            D12 = T1 @ self.D12
            D21 = self.D21 @ T2
            B2  = self.B2  + sign * self.B1  @ E_D @ self.D12
            C2  = self.C2  + sign * self.D21 @ E_D @ self.C1
            D22 = self.D22 + sign * self.D21 @ E_D @ self.D12
            return type(self)(H11, B2, C2, D12, D21, D22, self.delay)
    
    def append(self, other):
        if not isinstance(other, control.lti.LTI):
            other = control.statesp._convert_to_statespace(other)
        if isinstance(other, control.StateSpace):
            H11 = self.H11.append(other)
            return type(self)(H11, self.B2, self.C2, self.D12, self.D21, self.D22, self.delay)
        elif isinstance(other, type(self)):
            H11 = self.H11.append(other.H11)
            B2  = linalg.block_diag(self.B2 , other.B2 )
            C2  = linalg.block_diag(self.C2 , other.C2 )
            D12 = linalg.block_diag(self.D12, other.D12)
            D21 = linalg.block_diag(self.D21, other.D21)
            D22 = linalg.block_diag(self.D22, other.D22)
            delay = numpy.hstack((self.delay, other.delay))
            return type(self)(H11, B2, C2, D12, D21, D22, delay)
    
    def __getitem__(self, indices):
        H11 = self.H11[indices]
        i = indices[0]
        j = indices[1]
        D12 = self.D12[i]
        D21 = self.D21[:, j]
        return type(self)(H11, self.B2, self.C2, D12, D21, self.D22, self.delay)
    
    def sample(self, Ts, method='zoh', alpha=None, prewarp_frequency=None):
        H = self.H.sample(Ts, method, alpha, prewarp_frequency)
        A   = H.A
        B1  = H.B[:             , :self.ninputs]
        B2  = H.B[:             , self.ninputs:]
        C1  = H.C[:self.noutputs]
        C2  = H.C[self.noutputs:]
        D11 = H.D[:self.noutputs, :self.ninputs]
        D12 = H.D[:self.noutputs, self.ninputs:]
        D21 = H.D[self.noutputs:, :self.ninputs]
        D22 = H.D[self.noutputs:, self.ninputs:]
        delay = self.delay / Ts
        H11 = control.StateSpace(A, B1, C1, D11, H.dt)
        return type(self)(H11, B2, C2, D12, D21, D22, delay)
    
    def sample_delays(self, tol=1e-12):
        if self.isdtime():
            # convert delay
            integer_delay = self.delay.round()
            if numpy.max(numpy.abs((self.delay - integer_delay) / integer_delay)) > tol:
                raise ValueError('cannot sample non-integer delays')
            delay = integer_delay.astype(int)
            # build system
            A = linalg.block_diag(*(numpy.eye(d,    k= -1) for d in delay))
            B = linalg.block_diag(*(numpy.eye(d, 1       ) for d in delay))
            C = linalg.block_diag(*(numpy.eye(1, d, k=d-1) for d in delay))
            Theta = control.StateSpace(A, B, C, 0., self.dt)
            return self.H.lft(Theta, self.ndelays, self.ndelays)
        else:
            raise ValueError('cannot sample delays for continuous systems')
    
    def dcgain(self, warn_infinite=False):
        return self._dcgain(warn_infinite)

    def dynamics(self, t, x, u=None, w=None):
        h11dyn = self.H11.dynamics(t, x, u)
        if u is None:
            u = numpy.zeros(self.ninputs)
        if w is None:
            w = numpy.zeros(self.ndelays)
        # received t, x, u and w ignore t
        x = numpy.reshape(x, (-1, 1))  # force to a column in case matrix
        u = numpy.reshape(u, (-1, 1))  # force to a column in case matrix
        w = numpy.reshape(w, (-1, 1))  # force to a column in case matrix
        if w.size != self.ndelays:
            raise ValueError('len(w) must be equal to number of delays: '
                            f'ndelays = {self.ndelays}, len(w) = {w.size}')
        return h11dyn + (self.B2 @ w).reshape(-1)

    def output(self, t, x, u=None, w=None):
        h11out = self.H11.output(t, x, u)
        if u is None:
            u = numpy.zeros(self.ninputs)
        if w is None:
            w = numpy.zeros(self.ndelays)
        # received t, x, u and w, ignore t
        w = numpy.reshape(w, (-1, 1))  # force to a column in case matrix
        if w.size != self.ndelays:
            raise ValueError('len(w) must be equal to number of delays: '
                            f'ndelays = {self.ndelays}, len(w) = {w.size}')
        return h11out + (self.D12 @ w).reshape(-1)
    
    def lags(self, t, x, u=None, w=None):
        if u is None:
            u = numpy.zeros(self.ninputs)
        if w is None:
            w = numpy.zeros(self.ndelays)
        # received t, x, u and w ignore t
        x = numpy.reshape(x, (-1, 1))  # force to a column in case matrix
        u = numpy.reshape(u, (-1, 1))  # force to a column in case matrix
        w = numpy.reshape(w, (-1, 1))  # force to a column in case matrix
        if x.size != self.nstates:
            raise ValueError('len(x) must be equal to number of states: '
                            f'nstates = {self.nstates}, len(x) = {x.size}')
        if u.size != self.ninputs:
            raise ValueError('len(u) must be equal to number of inputs: '
                            f'ninputs = {self.ninputs}, len(u) = {u.size}')
        if w.size != self.ndelays:
            raise ValueError('len(w) must be equal to number of delays: '
                            f'ndelays = {self.ndelays}, len(w) = {w.size}')
        return (self.C2 @ x + self.D21 @ u + self.D22 @ w).reshape(-1)
    
    def is_delayed(self):
        # H12 and H21
        return self.D12.any() or (self.B2.any() and self.C1.any()) \
           and self.D21.any() or (self.B1.any() and self.C2.any())
     
    def has_delay_feedback(self):
        # if delayed and H22
        return self.is_delayed() and (self.D22.any() or (self.B2.any() and self.C2.any()))
    
    def purely_io_delay(self):
        # not (H11 or H22)
        return not (self.D11.any() or (self.B1.any() and self.C1.any())
                 or self.D22.any() or (self.B2.any() and self.C2.any()))
    
    def copy(self):
        return type(self)(
            self.H11.copy(),
            self.B2.copy(),
            self.C2.copy(),
            self.D12.copy(),
            self.D21.copy(),
            self.D22.copy(),
            self.delay.copy()
        )
    
    @property
    def A(self):
        return self.H11.A
    
    @property
    def B1(self):
        return self.H11.B
    
    @property
    def C1(self):
        return self.H11.C
    
    @property
    def D11(self):
        return self.H11.D
    
    @property
    def B(self):
        return numpy.hstack((self.B1, self.B2))
    
    @property
    def C(self):
        return numpy.vstack((self.C1, self.C2))
    
    @property
    def D(self):
        return numpy.vstack((numpy.hstack((self.D11, self.D12)),
                             numpy.hstack((self.D21, self.D22))))
    
    @property
    def H12(self):
        return control.StateSpace(self.A, self.B2, self.C1, self.D12, self.dt)
    
    @property
    def H21(self):
        return control.StateSpace(self.A, self.B1, self.C2, self.D21, self.dt)
    
    @property
    def H22(self):
        return control.StateSpace(self.A, self.B2, self.C2, self.D22, self.dt)
    
    @property
    def H(self):
        return control.StateSpace(self.A, self.B, self.C, self.D, self.dt)
    
    @property
    def ndelays(self):
        return len(self.delay)


class NonlinearDynamicsSystem(control.NonlinearIOSystem):
    def __init__(self, F, J, Jb, C, D, dt=None):
        ninputs = numpy.atleast_2d(D).shape[1]
        nstates = numpy.atleast_2d(C).shape[1]
        sys = control.StateSpace(numpy.identity(nstates), numpy.ones((nstates, ninputs)), C, D)
        C, D = sys.C, sys.D

        def updfcn(t, x, u=0., params={}):
            u += numpy.zeros(self.ninputs)
            return self.F(x, u)
        
        def outfcn(t, x, u=0., params={}):
            x += numpy.zeros(self.nstates)
            u += numpy.zeros(self.ninputs)
            return self.C @ x + self.D @ u
        
        super().__init__(updfcn, outfcn, inputs=sys.ninputs, outputs=sys.noutputs, states=sys.nstates, dt=dt)

        self.F  = F
        self.J  = J
        self.Jb = Jb
        self.C  = C
        self.D  = D
        
    def __getitem__(self, i):
        if isinstance(i, tuple):
            raise ValueError(f'cannot index {type(self)} inputs')
        return type(self)(self.F, self.J, self.Jb, self.C[i], self.D[i], self.dt)
    
    def sample(self, Ts: float, method='zoh'):
        def F(x, u):
            fun = lambda t, x: self.F(x, u)
            jac = lambda t, x: self.J(x)
            sol = integrate.solve_ivp(fun, (0., Ts), x)
            return sol.y[..., -1]
        
        def J(x):
            y = numpy.empty((self.nstates, self.nstates))
            for i in range(self.nstates):
                fun = lambda t, x: self.J(x)[i]
                sol = integrate.solve_ivp(fun, (0., Ts), x)
                y[i] = sol.y[..., -1]
            return y
        
        def Jb(u):
            y = numpy.empty((self.nstates, self.ninputs))
            for i in range(self.ninputs):
                fun = lambda t, u: self.Jb(u)[i]
                sol = integrate.solve_ivp(fun, (0., Ts), u)
                y[i] = sol.y[..., -1]
            return y
        
        return type(self)(F, J, Jb, self.C, self.D, Ts)

    def find_eqpt(self, x0=0., ueq=0., **kwargs):
        x0  += numpy.zeros(self.nstates)
        ueq += numpy.zeros(self.ninputs)
        if self.isctime():
            fun = lambda x: self.F(x, ueq)
            jac =           self.J
        else:
            I = numpy.identity(self.nstates)
            fun = lambda x: self.F(x, ueq) - x
            #jac = lambda x: self.J(x)      - I
            jac=None
        root = optimize.root(fun, x0, jac=jac, **kwargs)
        return root.x if root.success else None

    def linearize(self, xeq=None, ueq=0.):
        if xeq is None:
            xeq = self.find_eqpt(ueq=ueq)
        return control.StateSpace(self.J(xeq), self.Jb(ueq), self.C, self.D, self.dt)


class NonlinearStateSystem(control.NonlinearIOSystem):
    def __init__(self, F, J, B, C, D, dt=None):
        nstates = numpy.atleast_1d(B).shape[0]
        sys = control.StateSpace(numpy.identity(nstates), B, C, D)
        B, C, D = sys.B, sys.C, sys.D

        def updfcn(t, x, u=0., params={}):
            u += numpy.zeros(self.ninputs)
            return self.F(x) + self.B @ u
    
        def outfcn(t, x, u=0., params={}):
            x += numpy.zeros(self.nstates)
            u += numpy.zeros(self.ninputs)
            return self.C @ x + self.D @ u
        
        super().__init__(updfcn, outfcn, inputs=sys.ninputs, outputs=sys.noutputs, states=sys.nstates, dt=dt)
        
        self.F = F
        self.J = J
        self.B = B
        self.C = C
        self.D = D
    
    def __getitem__(self, indices):
        i = indices[0]
        j = indices[1]
        B = self.B[:, j]
        C = self.C[i   ]
        D = self.D[i, j]
        return type(self)(self.F, self.J, B, C, D, self.dt)
    
    def sample(self, Ts: float, method='zoh'):
        def F(x, u):
            fun = lambda t, x: self.F(x, u)
            jac = lambda t, x: self.J(x)
            sol = integrate.solve_ivp(fun, (0., Ts), x)
            return sol.y[..., -1]
        
        def J(x):
            y = numpy.empty((self.nstates, self.nstates))
            for i in range(self.nstates):
                fun = lambda t, x: self.J(x)[i]
                sol = integrate.solve_ivp(fun, (0., Ts), x)
                y[i] = sol.y[..., -1]
            return y
        
        Jb = lambda u: self.B
        
        return NonlinearDynamicsSystem(F, J, Jb, self.C, self.D, Ts)
    
    def find_eqpt(self, x0=0., ueq=0., **kwargs):
        x0  += numpy.zeros(self.nstates)
        ueq += numpy.zeros(self.ninputs)
        if self.isctime():
            fun = lambda x: self.F(x) + self.B @ ueq
            jac =           self.J
        else:
            I = numpy.identity(self.nstates)
            fun = lambda x: self.F(x) + self.B @ ueq - x
            jac = lambda x: self.J(x)                - I
        root = optimize.root(fun, x0, jac=jac, **kwargs)
        # TODO: replace hotfix for neural mass model
        if not root.success:
            x0 = (
                 .04076362464992866  ,
                -.09402108567794648  ,
                 .04076362464992866  ,
                 .04076362464992866  ,
                 .01066677768228507  ,
                 .0013229895781895903,
                 .01066677768228507  ,
                 .01066677768228507
            )
            x0 = numpy.asarray(x0)
            count=0
            while not root.success:
                count+=1
                print(count)
                root = optimize.root(fun, x0 + (numpy.random.rand(*x0.shape) - .5) * 1e-1, jac=jac, **kwargs)
            #print('(')
            #print(*root.x, sep=',\n')
            #print(')')
        return root.x if root.success else None
    
    def linearize(self, xeq=None, ueq=0.):
        if xeq is None:
            xeq = self.find_eqpt(ueq=ueq)
        return control.StateSpace(self.J(xeq), self.B, self.C, self.D, self.dt)


class NonlinearDelayedStateSystem(NonlinearStateSystem):
    def __init__(self, F, J, B, C, D, F2, J2, C2, delay, dt=None):
        super().__init__(F, J, B, C, D, dt)

        C2 = numpy.atleast_2d(C2)
        delay = _check_delay(delay)
        if C2.shape[0] != len(delay):
            raise ValueError(f'C2 must have ndelays rows: {len(delay) = }, {C2.shape[0] = }')
        if C2.shape[1] != self.nstates:
            raise ValueError(f'C2 must have nstates columns: nstates = {self.nstates}, {C2.shape[1] = }') 

        self.F2    = F2
        self.J2    = J2
        self.C2    = C2
        self.delay = delay
    
    def __getitem__(self, indices):
        i = indices[0]
        j = indices[1]
        B = self.B[:, j]
        C = self.C[i   ]
        D = self.D[i, j]
        return type(self)(self.F, self.J, B, C, D, self.F2, self.J2, self.C2, self.delay, self.dt)
    
    def sample(self, Ts: float, method='zoh', tol=1e-12):
        # convert delay
        delay = self.delay / Ts
        integer_delay = delay.round()
        if numpy.max(numpy.abs((delay - integer_delay) / integer_delay)) > tol:
            raise ValueError('Ts must divide all delays')
        delay = integer_delay.astype(int)
        # sample system
        tot_delay = delay.sum()
        Ad = linalg.block_diag(*(numpy.eye(d,    k= -1) for d in delay))
        Bd = linalg.block_diag(*(numpy.eye(d, 1       ) for d in delay))
        Cd = linalg.block_diag(*(numpy.eye(1, d, k=d-1) for d in delay))
        B = numpy.vstack((self.B, numpy.zeros((tot_delay,  self.ninputs))))
        C = numpy.hstack((self.C, numpy.zeros((self.noutputs, tot_delay))))

        def F(x, u):
            x1 = x[:self.nstates]
            xd = x[self.nstates:]
            fun = lambda t, x: self.F(x) + self.B @ u + self.F2(Cd @ xd)
            jac = lambda t, x: self.J(x)
            sol = integrate.solve_ivp(fun, (0., Ts), x1)
            x1next = sol.y[..., -1]
            xdnext = Bd @ self.C2 @ x1 + Ad @ xd
            return numpy.hstack((x1next, xdnext))
        
        def J(x):
            x1 = x[:self.nstates]
            xd = x[self.nstates:]
            y = numpy.empty((self.nstates, self.nstates + tot_delay))
            for i in range(self.nstates):
                fun = lambda t, x: self.J(x)[i]
                sol = integrate.solve_ivp(fun, (0., Ts), x1)
                y[i] = sol.y[..., -1]
            for i in range(tot_delay):
                fun = lambda t, x: self.J2(x)[i]
                sol = integrate.solve_ivp(fun, (0., Ts), x1)
                y[self.nstates + i] = sol.y[..., -1]
            return numpy.vstack((y, numpy.hstack(Cd, Ad)))

        Jb = lambda u: B

        return NonlinearDynamicsSystem(F, J, Jb, C, self.D, Ts)
    
    def find_eqpt(self, x0=0., ueq=0., **kwargs):
        x0  += numpy.zeros(self.nstates)
        ueq += numpy.zeros(self.ninputs)
        if self.isctime():
            fun = lambda x: self.F(x) + self.F2(self.C2 @ x) + self.B @ ueq
            jac = lambda x: self.J(x) + self.J2(self.C2 @ x) @ self.C2
        else:
            I = numpy.identity(self.nstates)
            fun = lambda x: self.F(x) + self.F2(self.C2 @ x) + self.B @ ueq - x
            jac = lambda x: self.J(x) + self.J2(self.C2 @ x) @ self.C2      - I
        root = optimize.root(fun, x0, jac=jac, **kwargs)
        return root.x if root.success else None
    
    def linearize(self, xeq=None, ueq=0.):
        if xeq is None:
            xeq = self.find_eqpt(ueq=ueq)
        A   = self.J (          xeq)
        B2  = self.J2(self.C2 @ xeq)
        D12 = numpy.zeros((self.noutputs, self.ndelays))
        D21 = numpy.zeros((self.ndelays , self.ninputs))
        D22 = numpy.zeros((self.ndelays , self.ndelays))
        H11 = control.StateSpace(A, self.B, self.C, self.D, self.dt)
        return StateSpace(H11, B2, self.C2, D12, D21, D22, self.delay)
    
    def dynamics(self, t, x, u=0., w=0.):
        w += numpy.zeros(self.ndelays)
        return self.updfcn(t, x, u) + self.F2(w)
    
    def output(self, t, x, u=0., w=0.):
        return self.outfcn(t, x, u)
    
    def lags(self, t, x, u=0., w=0.):
        x += numpy.zeros(self.nstates)
        return self.C2 @ x
    
    def is_delayed(self):
        return self.C2.any()
    
    @property
    def ndelays(self):
        return len(self.delay)
