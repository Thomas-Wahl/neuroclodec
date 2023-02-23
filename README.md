# neuroclodec
Closed-Loop Neurostimulation with Delay Compensation

## Prerequisites
Install required python libraries with `pip install -r requirements.txt`.

## Usage
Create a `StateSpace` system with `model.py`:
* `neural_oscillator`: simple linear brain model.
* `cortico_thalamic`: non-linear brain model with cortico-thalamic loop.
* `alpha_gamma_filter`: weighted double bandpass filter with a positive weight in the alpha-band and a negative weight in the gamma-band.
**NB**: The first input of the brain systems is the stimulation input while all the other inputs are the driving noise with standard normal distribution.

Build the closed-loop circuit using the `Circuit` class in `simul.py`.

Run numerical simulations using the `input_output_response` function in `simul.py`.

## Examples
In `main.py`:
* Choose the brain model by commenting/uncommenting `P = ...` statements.
* Uncomment a given function `figure.<x>()` and run the file.
