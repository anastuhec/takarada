from pathlib import Path
import json, sys, argparse
import numpy as np

''' this is where I keep my helpers '''
sys.path.append("/scratch/stuhecana/takarada/")
from helpers_takarada import *
from module_takarada import *

''' I have to get parameters from json file: '''
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
with open(args.config) as f:
    params = json.load(f)

''' this is where I will save data. if directory does not exist, I make it '''
DATA_DIR = Path('/project/stuhecana/takarada/data/30-01-phdiag')
DATA_DIR.mkdir(parents=True, exist_ok=True)

parameters1 = {'n_pass' : 1e-4,
'epsilon_threshold' : 1e-10,
'N_epsilon' : 5,
'maxiter' : 600,
'eps_last' : 1e-10,
'dmu' : 5.,
'mix2' : 0.001,
'mix3' : 1.5,
'faktor1' : 0.1,
'max_trials' : 15,
'eps0' : 0.03,
}

parameters2 = {'n_pass' : 1e-4,
'epsilon_threshold' : 1e-10,
'N_epsilon' : 5,
'maxiter' : 600,
'eps_last' : 1e-10,
'dmu' : 5.,
'mix2' : 0.001,
'mix3' : 1.5,
'faktor1' : 0.1,
'max_trials' : 15,
'eps0' : 0.03,
}
mu0 = 0.

epsilon = params["epsilon"]
Vb = params["Vb"]
gap = params["gap"]

beta0 = params["beta0"]
scale = params["scale"]
Nk = params["Nk"]
Nbeta = params["Nbeta"]

Gammas = params["Gammas"]
Nomega = params["Nomega"]
eps = params["eps"]
maxbrentq = params["maxbrentq"]
N = params["N"]
factor = params["factor"]
oms = params["oms"]

epsilon_ = epsilon

a = 1.
b = 0.

t12 = 0.
delta = 0.
Vc = 0.

t = 1.
t_ = 0.5


phys_parameters = [b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta]

include_hartree = True


print(f't={t}, t_={t_}, Vb={Vb}, Vc={Vc}, epsilon={epsilon}, epsilon_={epsilon_}, gap={gap}', flush=True)
m = model(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree)
m.GS()


betas = beta0/scale**np.arange(1,Nbeta)
stops = [int(np.emath.logn(scale, beta0/beta)) for beta in betas]
Ts = 1/betas

m.run_Tdependence(betas, stops, Gammas, Nomega, eps, maxbrentq, N, factor, oms)

data = m.collect_data()
np.savez(DATA_DIR / f"data_Vb{Vb}_gap{gap}.npz", **data)

