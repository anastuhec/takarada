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


alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
b = 0.5
alphas2 = [0.0, 0.5, 1.0]

for i, alpha in enumerate(alphas):
    for j, alpha2 in enumerate(alphas2):
        Vb = 3 * alpha
        Vc = alpha
        t12 = b*(1-b)*alpha2

        print(f'-' * 100, flush=True)
        print(f'evaluating alpha={alpha}, alpha2={alpha2}', flush=True)
        phys_parameters = [b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta]
        m = model(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree, n_target=n_target)
        m.GS()
        beta0 = 1/0.015
        scale = 1.008
        betas = beta0/scale**np.arange(1,351)
        stops = [int(np.emath.logn(scale, beta0/beta)) for beta in betas][::5]
        Ts = 1/betas
        m.run_Tdependence(betas, stops, Gammas, params, maxbrentq, evaluate_transport_DC=True, evaluate_vertex_DC=False)
        data = m.collect_data()

        np.savez(DATA_DIR / f"10-04-lega_b{b}_alpha{alpha}_alpha2_{alpha2}.npz", **data)