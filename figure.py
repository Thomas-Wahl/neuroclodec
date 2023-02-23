import system
import science
import control
from matplotlib import pyplot

def activity(P: control.StateSpace | system.StateSpace | system.NonlinearStateSystem):
    '''Plot the rest state activity of a plant.'''
    fig, axes = pyplot.subplots(1, 2, constrained_layout=True)
    science.activity(P, axes, seed=100)
    fig.set_figwidth (fig.get_figwidth () * 1.1)
    fig.set_figheight(fig.get_figheight() * 1.1 / 3.)
    axes[1].set_ylabel('spectral density [dB]')
    axes[0].set_ylabel('signal')
    axes[1].set_xlabel('frequency [Hz]')
    axes[0].set_xlabel('time [s]')
    for ax in axes:
        ax.set_xmargin(0.)
        # Hide the right and top spines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.tick_params(direction='in')
        for line in ax.get_lines():
            line.set_zorder(1.5)
    pyplot.show()

def compare(P: control.StateSpace | system.StateSpace | system.NonlinearStateSystem):
    '''Compare model-based neurostimulation with delay compensation to reference based control with a Smith predictor.'''
    fig, axes = pyplot.subplots(3, 2, constrained_layout=True)
    science.reference_based_control(P, axes[:, 0], delay=5e-3, seed=150)
    science.model_based_control    (P, axes[:, 1], delay=5e-3, seed=160)
    fig.set_figwidth (fig.get_figwidth () * 1.1)
    fig.set_figheight(fig.get_figheight() * 1.1)
    for column in axes.T:
        column[0].set_ylabel('signal')
        column[1].set_ylabel('spectral density [dB]')
        column[2].set_ylabel('gain [dB]')
        column[0].set_xlabel('time [s]')
        column[1].set_xlabel('frequency [Hz]')
        column[2].set_xlabel('frequency [Hz]')
    for row in axes:
        for ax in row:
            ax.set_xmargin(0.)
            # Hide the right and top spines
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.tick_params(direction='in')
            for line in ax.get_lines():
                line.set_zorder(1.5)
    axes[0, 0].set_title('A)', loc='left')
    axes[0, 1].set_title('B)', loc='left')
    pyplot.show()

def psdfit(P: control.StateSpace | system.StateSpace | system.NonlinearStateSystem):
    '''Plot the different steps of plant model estimation.'''
    fig, axes = pyplot.subplots(2, 2, constrained_layout=True)
    science.psdfit(P, axes.flatten(), seed=220)
    fig.set_figwidth (fig.get_figwidth () * 1.1)
    fig.set_figheight(fig.get_figheight() * 1.1 / 1.5)
    axes[0, 1].set_ylabel('spectral density [dB]')
    axes[0, 0].set_ylabel('signal')
    axes[1, 0].set_ylabel('magnitude [dB]')
    axes[1, 1].set_ylabel('phase [°]')
    axes[0, 0].set_xlabel('time [s]')
    axes[0, 1].set_xlabel('frequency [Hz]')
    axes[1, 0].set_xlabel('frequency [Hz]')
    axes[1, 1].set_xlabel('frequency [Hz]')
    for row in axes:
        for ax in row:
            ax.set_xmargin(0.)
            # Hide the right and top spines
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.tick_params(direction='in')
            for line in ax.get_lines():
                line.set_zorder(1.5)
    axes[0, 0].set_title('A)', loc='left')
    axes[0, 1].set_title('B)', loc='left')
    axes[1, 0].set_title('C)', loc='left')
    axes[1, 1].set_title('D)', loc='left')
    pyplot.show()

def noise_level(P: control.StateSpace | system.StateSpace | system.NonlinearStateSystem):
    '''Compare efficiency of magnitude vector fitting between 3 different noise levels.'''
    fig, axes = pyplot.subplots(3, 2, constrained_layout=True)
    science.noise_level(P, axes[:, 0], axes[:, 1], seed=300)
    fig.set_figwidth (fig.get_figwidth () * 1.1)
    fig.set_figheight(fig.get_figheight() * 1.1)
    for ax in axes[:, 0]:
        ax.set_ylabel('magnitude [dB]')
        ax.set_xlabel('frequency [Hz]')
    for ax in axes[:, 1]:
        ax.set_ylabel('phase [°]')
        ax.set_xlabel('frequency [Hz]')
    for row in axes:
        for ax in row:
            ax.set_xmargin(0.)
            # Hide the right and top spines
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.tick_params(direction='in')
            for line in ax.get_lines():
                line.set_zorder(1.5)
    axes[0, 0].set_title('noise standard deviation = $3\\times 10^{-5}$', loc='left')
    axes[1, 0].set_title('noise standard deviation = $3\\times 10^{-4}$', loc='left')
    axes[2, 0].set_title('noise standard deviation = $1.6\\times 10^{-3}$', loc='left')
    pyplot.show()

def delay():
    '''Compare closed-loop gain errors for between three different delays.'''
    fig, axes = pyplot.subplots(3, constrained_layout=True)
    science.delay(axes)
    fig.set_figwidth (fig.get_figwidth () * 1.1 / 2.)
    fig.set_figheight(fig.get_figheight() * 1.1)
    for ax in axes:
        ax.set_ylabel('gain [dB]')
        ax.set_xlabel('frequency [Hz]')
        ax.set_xmargin(0.)
        # Hide the right and top spines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.tick_params(direction='in')
        for line in ax.get_lines():
            line.set_zorder(1.5)
    axes[0].set_title('$\\tau = 3$ms', loc='left')
    axes[1].set_title('$\\tau = 5$ms', loc='left')
    axes[2].set_title('$\\tau = 10$ms', loc='left')
    pyplot.show()

def fit_control(P: control.StateSpace | system.StateSpace | system.NonlinearStateSystem):
    '''Plot the result of closed-loop neurostimulation based on a fitted plant model.'''
    fig, axes = pyplot.subplots(2, 2, constrained_layout=True)
    science.fit_model_based_control(P, axes.flatten(), delay=5e-3, seed=513)
    fig.set_figwidth (fig.get_figwidth () * 1.1)
    fig.set_figheight(fig.get_figheight() * 1.1 / 1.5)
    axes[0, 0].set_ylabel('magnitude [dB]')
    axes[0, 1].set_ylabel('phase [°]')
    axes[1, 0].set_ylabel('spectral density [dB]')
    axes[1, 1].set_ylabel('gain [dB]')
    axes[0, 0].set_xlabel('frequency [Hz]')
    axes[0, 1].set_xlabel('frequency [Hz]')
    axes[1, 0].set_xlabel('frequency [Hz]')
    axes[1, 1].set_xlabel('frequency [Hz]')
    for row in axes:
        for ax in row:
            ax.set_xmargin(0.)
            # Hide the right and top spines
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.tick_params(direction='in')
            for line in ax.get_lines():
                line.set_zorder(1.5)
    axes[0, 0].set_title('A)', loc='left')
    axes[0, 1].set_title('B)', loc='left')
    axes[1, 0].set_title('C)', loc='left')
    axes[1, 1].set_title('D)', loc='left')
    pyplot.show()

def predictor_stability():
    '''Plot the stability region of the predictor.'''
    fig, ax = pyplot.subplots(constrained_layout=True)
    science.predictor_stability(ax)
    fig.set_figwidth (fig.get_figwidth () * 1.1 / 2.)
    fig.set_figheight(fig.get_figheight() * 1.1 / 3.)
    ax.set_ylabel('pole magnitude [dB]')
    ax.set_xlabel('$a$ [$s^{-1}$]')
    ax.set_ylim(-.25, .25)
    ax.set_xmargin(0.)
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(direction='in')
    for line in ax.get_lines():
        line.set_zorder(1.5)
    #pyplot.legend()
    pyplot.show()