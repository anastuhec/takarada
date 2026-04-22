from pathlib import Path
import numpy as np
import sys
from itertools import product

''' this is where I keep my helpers '''
import sys
sys.path.append("/scratch/stuhecana/takarada/")
from helpers_takarada_lips import *
from module_takarada_lips import *

DATA_DIR = Path('/project/stuhecana/takarada/data/22-04')
DATA_DIR.mkdir(parents=True, exist_ok=True)

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
'eps0' : 0.03,
}
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

t12s = [-0.3, -0.1, 0.0, 0.1, 0.3]

''' physical parameters '''
a = 1.

delta = 0.0

t = 1.0
t_ = 3.0

mu0 = 0.
include_hartree = True

Nk = 2000

epsilon = 2.0
epsilon_ = 2.0

''' line of V_total = Vb+Vc=4 '''
V_total = 4
Vcs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

params = list(product(t12s, Vcs))
task_id = int(sys.argv[1])

(t12, Vc) = params[task_id]
    
Vb = V_total - Vc
b = Vc / (Vb + Vc)

phys_parameters = [b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta]

''' generate model and find ground state '''
m = model(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree)
m.GS()

''' temperature dependence '''
beta0 = 70
scale = 1.01
betas = beta0/scale**np.arange(1,401)
stops = [int(np.emath.logn(scale, beta0/beta)) for beta in betas][::3]
Ts = 1/betas
maxbrentq = 30

Nomega, Gammas = 3000, [0.005, 0.008, 0.01]

omegas = np.logspace(-7,-2,50)
params = {'Nomega' : Nomega, 'eps' : 1e-4, 'deg' : 8000, 'omega0' : omegas, 'eps2' : 1e-6}
m.run_Tdependence(betas, stops, Gammas, params, maxbrentq, evaluate_transport_DC=True, evaluate_vertex_DC=False)
data = m.collect_data()

np.savez(DATA_DIR / f"22-04_Vc_{Vc}_t12_{t12}.npz", **data)
