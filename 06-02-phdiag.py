from pathlib import Path
import numpy as np

''' this is where I keep my helpers '''
import sys
sys.path.append("/scratch/stuhecana/takarada/")
from helpers_takarada import *
from module_takarada import *

DATA_DIR = Path('/project/stuhecana/takarada/data/06-02-phdiag')
DATA_DIR.mkdir(parents=True, exist_ok=True)

''' numerical parameters for the self-consistent calculation '''
parameters1 = {'n_pass' : 1e-4,
'epsilon_threshold' : 1e-8,
'N_epsilon' : 5,
'maxiter' : 500,
'eps_last' : 1e-8,
'dmu' : 5.,
'mix2' : 0.001,
'mix3' : 1.5,
'faktor1' : 0.1,
'max_trials' : 15,
'eps0' : 0.03,
}

parameters2 = {'n_pass' : 1e-4,
'epsilon_threshold' : 1e-8,
'N_epsilon' : 5,
'maxiter' : 500,
'eps_last' : 1e-8,
'dmu' : 5.,
'mix2' : 0.001,
'mix3' : 1.5,
'faktor1' : 0.1,
'max_trials' : 15,
'eps0' : 0.03,
}

mu0 = 0.

a = 1.
b = 0.5

t12 = 0.
delta = 0.

t = 1.
t_ = 0.5
Vc = 0.
Vb = 1.

mu0 = 0.
include_hartree = True

Nk = 3000

gap = -1.

epsilon = 0.5*(gap + 2*t_ + 2*t)
epsilon_ = epsilon

epsilon_ = epsilon

a = 1.
b = 0.

t12 = 0.
delta = 0.
Vc = 0.

t = 1.
t_ = 0.5

include_hartree = True

phys_parameters = [b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta]


print(f'started')
m = mt.model(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree)
m.GS()
print(f'found ground state', flush=True)

scale = 1.008
T_gap = m.gap
T_steps = np.array([0.01 * i for i in range(2,9)])
betas_stops = 1/T_steps
beta0 = betas_stops[0] * scale**10
stops = [int(np.emath.logn(scale, beta0/beta)) for beta in betas_stops]
print(f'set temperature stops', flush=True)

Gammas = [0.0005, 0.001]
N = 15
factor = 5
omegas = np.logspace(-4,1,200)
maxbrentq = 100

# redundant:
Nomega, eps = 3, 1e-6

for i in range(len(T_steps) - 1):
    print(f'temp stop {i}', flush=True)
    N = int(np.emath.logn(scale, betas_stops[i] / betas_stops[i+1]))
    betas = betas_stops[i]/scale**np.arange(1,N)
    stops = np.arange(1,N-1)
    m.run_Tdependence(betas, stops, Gammas, Nomega, eps, maxbrentq, N, factor, omegas, evaluate_transport=False)

    T = m.Ts[-1]

    m.next_T(T, 1)
    mu = m.mu

    for g, Gamma in enumerate(Gammas):
        print(f'temp stop {i}, Gamma {g}', flush=True)
        results = m.chijrho(T, omegas, Gamma, N, factor, mu)
        results["T"] = T
        results["delta"] = np.abs(m.delta_bs[-1])

        np.savez(DATA_DIR / f"dataopt_06-02-Nk{Nk}-step{i}-gamma{g+4}.npz", **results)