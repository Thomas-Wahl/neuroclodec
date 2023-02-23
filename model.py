import numpy
import system
import control
from scipy import linalg
from scipy import special

def phasefit(f0: float, delay=0., tol=1e6):
    if delay:
        ω0 = 2. * numpy.pi * f0
        k = -numpy.sign(numpy.sin(ω0 * delay))
        a = -numpy.tan(((ω0 * delay % numpy.pi) - numpy.pi)/ 2.)
        c = k * (control.TransferFunction((-a, ω0), (a, ω0)) if a < tol else -1.)
        return control.statesp._convert_to_statespace(c)
    else:
        return 1.

def phasevfit(s, p, tol=1e-6):
    if numpy.any(p):
        s = numpy.atleast_1d(s)
        p = numpy.atleast_1d(p)
        eN = numpy.arange(1, len(s) + 1)
        sN =   s ** eN
        mN = (-1)**(eN - 1)
        pn = numpy.exp(1j * ((-p) % numpy.pi))
        A = (mN + pn[numpy.newaxis].T) @ numpy.diag(sN)
        b = 1. - pn
        A = numpy.vstack((A.real, A.imag))
        b = numpy.hstack((b.real, b.imag))
        x, residuals, rank, _ = numpy.linalg.lstsq(A, b)
        den = numpy.append(numpy.flip(      x), 1.)
        num = numpy.append(numpy.flip(-mN * x), 1.)
        sys = control.TransferFunction(num, den).minreal(tol)
        return control.statesp._convert_to_statespace(sys)
    else:
        return 1.

def bandpass_filter(f0: float, B: float):
    ω0 = 2. * numpy.pi * f0
    Bp = 2. * numpy.pi * B
    A = numpy.array(((-Bp, -ω0**2), (1., 0.)))
    B = numpy.array((Bp, 0.))
    C = numpy.array((1., 0.))
    return control.StateSpace(A, B, C, 0.)

def alpha_gamma_filter(*,
    f1=10.,
    f2=40.,
    B1= 4.,
    B2=30.,
    c1=1.,
    c2=-.5
):
    F1 = bandpass_filter(f1, B1)
    F2 = bandpass_filter(f2, B2)
    return c1 * F1 + c2 * F2

def phasefitted_alpha_gamma_filter(*,
    f1=10.,
    f2=45.,
    B1= 4.,
    B2=30.,
    c1=1.,
    c2=-.5,
    delay=0.
):
    p1 = phasefit(f1, delay)
    p2 = phasefit(f2, delay)
    F1 = bandpass_filter(f1, B1)
    F2 = bandpass_filter(f2, B2)
    return p1 * c1 * F1 + p2 * c2 * F2

def PID_controller(Kp=1., Ki=0., Kd=0., fc=100.):
    ωc = 2. * numpy.pi * fc
    A = numpy.array(((0., 0.), (0., -ωc)))
    B = numpy.ones((2, 1))
    C = numpy.array((Ki, Kd))
    D = Kp
    return control.StateSpace(A, B, C, D)#.minreal()

def smith_predictor(G: control.StateSpace, K: control.StateSpace, delay: numpy.ndarray):
    dt = control.namedio.common_timebase(G.dt, K.dt)
    delay = system._check_delay(delay)
    Theta = system.pure_delay(delay, dt)
    I = control.StateSpace((), (), (), numpy.identity(Theta.ninputs), dt)
    return system.feedback(K, G * (I - Theta))

def continuous_predictor(delay: numpy.ndarray, fc=5e2):
    ωc = 2. * numpy.pi * fc
    delay = system._check_delay(delay)
    A = ωc
    B = 1.
    C = -delay * ωc**2
    D = 1. + delay * ωc
    return control.StateSpace(A, B, C, D, 0)

def discrete_predictor(maxgain: float, step=1):
    if step:
        gain = maxgain**(1./step)
        A = (3. - gain) / (1. + gain)
        z = control.TransferFunction.z
        return control.statesp._convert_to_statespace(((2. - A) * z - 1.)**step / (z - A)**step)
    else:
        return control.StateSpace((), (), (), 1., True)
    '''
    B = 1.
    C = A  - 1.
    D = 2. - A
    sys = control.StateSpace(A, B, C, D, True)
    sysn = sys
    for _ in range(step - 1):
        sysn *= sys
    return sysn
    '''

def discrete_predictor_from_pole(a: float, step=1):
    if step:
        z = control.TransferFunction.z
        return control.statesp._convert_to_statespace(((2. - a) * z - 1.)**step / (z - a)**step)
    else:
        return control.StateSpace((), (), (), 1., True)

def neural_oscillator(*,
    # coefficients
    taue1= 5e-3,#s
    taui1=20e-3,#s
    taue2= 5e-3,#s
    taui2=20e-3,#s
    N11=1.15,
    N21= .63,
    N12=2.52,
    N22=6.6 ,
    # input coupling strengths
    b1=.18,
    b2=.18,
    b3=.14,
    b4=.14,
    # observation
    ce=1.,
    ci=0.,
    # noise
    Q1=1e-4,
    Q2=1e-4,
    # neurons
    N=1000
):
    # matrices
    Tauinv = numpy.diag(numpy.array((taue1, taui1, taue2, taui2))**-1)
    M1 = numpy.array(((-1. + N11,     - N11),
                      (      N21, -1. - N21)))
    M2 = numpy.array(((-1. + N12,     - N12),
                      (      N22, -1. - N22)))
    Bu = numpy.array((b1, b2, b3, b4))[numpy.newaxis].T
    Bn = (numpy.array(((Q1, 0.), (0., 0.), (0., Q2), (0., 0.))) / N)**.5
    # variables
    A = Tauinv @ linalg.block_diag(M1, M2)
    B = Tauinv @ numpy.hstack((Bu, Bn))
    C = numpy.array((ce, ci, ce, ci))
    return control.StateSpace(A, B, C, 0.)

def cortico_thalamic(*,
    # synaptic scales
    taue  =10e-3,#s
    taui  =50e-3,#s
    tauthe= 5e-3,#s
    tauthi=30e-3,#s
    tauret= 8e-3,#s
    tauce = 5e-3,#s
    tauci =20e-3,#s
    # cortico-subcortical delay
    tau   =40e-3,#s
    # coupling strengths
    Fe  =1.  ,
    Fi  =2.  ,
    Fccx= .05,
    Fct =1.2 ,
    Ftc =1.  ,
    Ftr =1.  ,
    Frt = .3 ,
    Frc = .6 ,
    # intra-populations coupling strengths
    Fcx=2.18,
    Mcx=3.88,
    # coupling strength
    Fcxth=.1,
    # coupling strength ratio
    S1=1.7,
    # mean input noise
    mue  = .1 ,
    mui  =0.  ,
    muthe=1.3 ,
    muthi=1.  ,
    muret=0.  ,
    muce = .05,
    muci = .05,
    # constant input
    Ie =2.7,
    Ii =1.7,
    Ice=1.1,
    Ici= .4,
    # input coupling strengths
    b1=1.,
    b2=1.,
    b3=1.,
    b4=1.,
    bret=0.,
    # input noise variances
    Qe_taue    =.3  ,
    Qi_taui    =.2  ,
    Qthe_tauthe=.05 ,
    Qthi_tauthi=.014,
    Qret_tauret=.17 ,
    Qce        =.005,
    Qci        =.004,
    # observation
    w = .3,
    # neurons
    N = 1000
):
    Qe   = Qe_taue     * taue
    Qi   = Qi_taui     * taui
    Qthe = Qthe_tauthe * tauthe
    Qthi = Qthi_tauthi * tauthi
    Qret = Qret_tauret * tauret
    Tauinv = numpy.diag(numpy.array((taue, taui, tauthe, tauthi, tauret, tauce, tauci))**-1)
    Bn = (numpy.diag((Qe, Qi, Qthe, Qthi, Qret, Qce, Qci)) / N)**.5
    Bu = numpy.array((b1, b2,  0.,  0.,  bret, b3, b4))[numpy.newaxis].T
    # finite size fluctuations
    # input noise variances
    # mean noise input
    s2c   = Qe_taue + Qi_taui
    s2ret = Qret_tauret
    s2th  = Qthe_tauthe + Qthi_tauthi
    s2ce  = Qce / tauce
    s2ci  = Qci / tauci
    # transfer functions
    def erf_transfer(x, s2): return (1. - special.erf(-x / (2. * s2)**.5)) / 2.
    
    def Tc  (x): return      erf_transfer(x, s2c  )  
    def Tret(x): return      erf_transfer(x, s2ret)
    def Tth (x): return      erf_transfer(x, s2th )
    def Se  (x): return S1 * erf_transfer(x, s2ce )
    def Si  (x): return      erf_transfer(x, s2ci )
    
    def F(V):
        '''Delay free part of the transfer function'''
        # vector unpacking
        Ve, Vi, Vthe, Vthi, Vret, u, v = V
        # transfer functions computations
        TcV   = Tc  (Ve - Vi    )
        TretV = Tret(Vret       )
        TthV  = Tth (Vthe - Vthi)
        SeV   = Se  (u          )
        SiV   = Si  (v          )
        # dynamics
        FVe   = -Ve   + Fe  * TcV  + Fccx * SeV + mue  + Ie
        FVi   = -Vi   + Fi  * TcV               + mui  + Ii
        FVthe = -Vthe                           + muthe
        FVthi = -Vthi + Ftr * TretV             + muthi
        FVret = -Vret + Frt * TthV              + muret
        Fu    = -u    + Fcx * SeV  - Mcx  * SiV + muce + Ice
        Fv    = -v    - Fcx * SiV  + Mcx  * SeV + muci + Ici
        # vector
        return Tauinv @ numpy.array((FVe, FVi, FVthe, FVthi, FVret, Fu, Fv))
    
    def F2(W):
        '''Delayed part of the transfer function'''
        # vector unpacking
        Wc, Wth = W
        # transfer function computation
        TcW  = Tc (Wc )
        TthW = Tth(Wth)
        # dynamics
        F2Ve   = Fct   * TthW
        F2Vthe = Ftc   * TcW
        F2Vret = Frc   * TcW
        F2u    = Fcxth * TthW
        # vector
        return Tauinv @ numpy.array((F2Ve, 0., F2Vthe, 0., F2Vret, F2u, 0.))
    
    # transfer functions derivatives
    def erf_transfer_prime(x, s2): return numpy.exp(-x**2 / (2. * s2)) / (2. * numpy.pi * s2)**.5
    
    def Tcp  (x): return      erf_transfer_prime(x, s2c  )  
    def Tretp(x): return      erf_transfer_prime(x, s2ret)
    def Tthp (x): return      erf_transfer_prime(x, s2th )
    def Sep  (x): return S1 * erf_transfer_prime(x, s2ce )
    def Sip  (x): return      erf_transfer_prime(x, s2ci )

    def J(V):
        '''Jacobian matrix of the delay free part of the transfer function'''
        # vector unpacking
        Ve, Vi, Vthe, Vthi, Vret, u, v = V
        # transfer functions computations
        TcpVe     = Tcp  (Ve   - Vi  )
        TcpVi     = -TcpVe
        TretpVret = Tretp(Vret       )
        TthpVthe  = Tthp (Vthe - Vthi)
        TthpVthi  = -TthpVthe
        Sepu      = Sep  (u)
        Sipv      = Sip  (v)
        # dynamics
        dFVe_dVe     = -1. + Fe   * TcpVe
        dFVi_dVe     =       Fi   * TcpVe
        dFVe_dVi     =       Fe   * TcpVi
        dFVi_dVi     = -1. + Fi   * TcpVi
        dFVthe_dVthe = -1.
        dFVret_dVthe =       Frt  * TthpVthe
        dFVthi_dVthi = -1.
        dFVret_dVthi =       Frt  * TthpVthi
        dFVthi_dVret =       Ftr  * TretpVret
        dFVret_dVret = -1.
        dFVe_du      =       Fccx * Sepu
        dFu_du       = -1. + Fcx  * Sepu
        dFv_du       =       Mcx  * Sepu
        dFu_dv       =     - Mcx  * Sipv
        dFv_dv       = -1. - Fcx  * Sipv
        # vector
        return Tauinv @ numpy.array(((dFVe_dVe  , dFVe_dVi  , 0.          , 0.          , 0.          , dFVe_du, 0.    ),
                                     (dFVi_dVe  , dFVi_dVi  , 0.          , 0.          , 0.          , 0.     , 0.    ),
                                     (0.        , 0.        , dFVthe_dVthe, 0.          , 0.          , 0.     , 0.    ),
                                     (0.        , 0.        , 0.          , dFVthi_dVthi, dFVthi_dVret, 0.     , 0.    ),
                                     (0.        , 0.        , dFVret_dVthe, dFVret_dVthi, dFVret_dVret, 0.     , 0.    ),
                                     (0.        , 0.        , 0.          , 0.          , 0.          , dFu_du , dFu_dv),
                                     (0.        , 0.        , 0.          , 0.          , 0.          , dFv_du , dFv_dv)))
    
    def J2(W):
        '''Jacobian matrix of the delayed part of the transfer function'''
        # vector unpacking
        Wc, Wth = W
        # transfer function computation
        TcpWc   = Tcp (Wc )
        TthpWth = Tthp(Wth)
        # dynamics
        dFVthe_dWc = Ftc   * TcpWc
        dFVret_dWc = Frc   * TcpWc
        dFVe_dWth  = Fct   * TthpWth
        dFu_dWth   = Fcxth * TthpWth
        # matrix
        return Tauinv @ numpy.array(((        0., dFVe_dWth),
                                     (        0.,        0.),
                                     (dFVthe_dWc,        0.),
                                     (        0.,        0.),
                                     (dFVret_dWc,        0.),
                                     (        0.,  dFu_dWth),
                                     (        0.,        0.)))

    B = Tauinv @ numpy.hstack((Bu, Bn))
    # constant input
    C = numpy.array((1.,  0.,  0.,  0.,  0.,   w,  0.))
    # state delay
    C2 = numpy.array((( 1., -1.,  0.,  0., 0., 0., 0.),
                      ( 0.,  0.,  1., -1., 0., 0., 0.)))
    return system.NonlinearDelayedStateSystem(F, J, B, C, 0., F2, J2, C2, (tau, tau))
