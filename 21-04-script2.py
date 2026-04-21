import numpy as np
import sys
sys.path.append("/Users/ana/Desktop/takarada/")
from pathlib import Path

DATA_DIR = Path('/Users/ana/Desktop/takarada/')

#DATA_DIR.mkdir(parents=True, exist_ok=True)

from helpers_takarada import * 
from module_takarada import *

''' numerical parameters for the self-consistent calculation '''
parameters1 = {'n_pass' : 1e-4,
'epsilon_threshold' : 1e-7,
'N_epsilon' : 5,
'maxiter' : 500,
'eps_last' : 1e-7,
'dmu' : 1.5,
'mix2' : 0.001,
'mix3' : 1.5,
'faktor1' : 0.1,
'max_trials' : 15,
'eps0' : 0.03,}

parameters2 = {'n_pass' : 1e-4,
'epsilon_threshold' : 1e-6,
'N_epsilon' : 5,
'maxiter' : 100,
'eps_last' : 1e-6,
'dmu' : 1.5,
'mix2' : 0.001,
'mix3' : 1.5,
'faktor1' : 0.1,
'max_trials' : 15,
'eps0' : 0.03,
}


a = 1.

delta = 0.0

t = 1.0
t_ = 3.0

mu0 = 0.
include_hartree = True

Nk = 2000

epsilon = 2.0
epsilon_ = 2.0

n_target = 1.00

Vb = 3.0
Vc = 1.0

maxbrentq = 30

Nomega, eps, Gammas = 2000, 1e-5, [0.006, 0.008, 0.01]

epsilon_max = 1.0
epsilons = np.linspace(-epsilon_max,epsilon_max,5000)

omegas = np.logspace(-7,-2,50)
params = {'Nomega' : Nomega, 'eps' : 1e-4, 'deg' : 8000, 'omega0' : omegas, 'eps2' : 1e-6}

alphas = np.linspace(0,2,100)
alphas2 = np.linspace(0,0.5,100)
b = 0.0
Deltas = np.zeros((len(alphas), len(alphas2)))
for i, alpha in enumerate(alphas):
    for j, alpha2 in enumerate(alphas2):
        print(i,j)
        Vb = 3 * alpha
        Vc = alpha
        t12 = alpha2

        print(f'-' * 100, flush=True)
        print(f'evaluating alpha={alpha}, alpha2={alpha2}', flush=True)
        phys_parameters = [b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta]
        m = model(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree, n_target=n_target)
        m.GS()

        Deltas[i,j] = np.sqrt(np.abs(m.delta_b)**2 + np.abs(m.delta_c)**2)

data = {'Deltas' : Deltas, 'alphas' : alphas, 't12s' : alphas2}

np.savez('data.npz', **data)