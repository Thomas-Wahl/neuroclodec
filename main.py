import model
import figure

def main():
    #P = model.neural_oscillator()
    P = model.cortico_thalamic(
        #bret=5.
    )
    #figure.activity(P)
    #figure.compare(P)
    #figure.psdfit(P)
    #figure.noise_level(P)
    figure.fit_control(P)
    #figure.delay()
    #figure.predictor_stability()

if __name__ == '__main__':
    main()
