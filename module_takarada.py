import numpy as np
import scipy.linalg as LA
from numba import njit, prange
from scipy.optimize import brentq
from helpers_takarada import *

''' create TNS class '''
class model:
    def __init__(self, Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree):
        self.K = 2*np.pi * np.arange(-Nk/2, Nk/2) / Nk
        self.Nk = Nk
        self.parameters1 = parameters1
        self.parameters2 = parameters2
        self.phys_parameters = phys_parameters
        self.hk0 = h_k0(self.K, self.phys_parameters)

        self.mu = mu0

        self.include_hartree = include_hartree

        self.delta_bs = []
        self.delta_cs = []
        self.gaps = []
        self.mus = []
        self.errors = []
        self.occupations = []
        self.times_rho = []
        self.times_boltzmann = []
        self.times_kubo = []
        self.Ts = []
        self.betas = []

        self.K1 = []
        self.K1q = []
        self.K0 = []
        
        self.K1_b = []
        self.K0_b  = []

    def GS(self):
        _, _, _, _, _, _, Vb, Vc, _ = self.phys_parameters
        rho, err, energije, vecs, fs, n = Rho_next(self.hk0, rho0(self.Nk), self.K, 0, self.mu, self.phys_parameters, self.parameters1['eps0'],
                                                  self.parameters1['epsilon_threshold'], self.parameters1['N_epsilon'], self.parameters1['maxiter'], self.include_hartree, mix=0.5)
        self.rho = rho
        self.energije = energije
        self.vecs = vecs
        self.fs = fs
        self.err = err
        self.n = n
        self.delta_b, self.delta_c = Delta(self.K, self.rho, Vb, Vc)
        self.gap = np.min(self.energije[1]) - np.max(self.energije[0])
        #print(f'found ground state, err={err}, n_err={np.abs(n-1)}, phi={self.phi}')

    def next_T(self, T, i, show_print=None) -> None:
        _, _, _, _, _, _, Vb, Vc, _ = self.phys_parameters

        if i == 1: parameters = self.parameters1
        elif i == 2: parameters = self.parameters2

        eps0 = parameters['eps0']
        dmu = parameters['dmu']
        epsilon_threshold = parameters['epsilon_threshold']
        N_epsilon = parameters['N_epsilon']
        maxiter = parameters['maxiter']
        
        mu = NewMu2(self.mu - dmu, self.mu + dmu, self.rho, self.K, T, self.phys_parameters, eps0, epsilon_threshold, N_epsilon, maxiter, mix=0.5, xtol=1e-6, rtol=1e-6, maxiterbrentq=100)
        rho, err, energije, vecs, fs, n = Rho_next(self.hk0, self.rho, self.K, T, mu, self.phys_parameters, eps0, epsilon_threshold, N_epsilon, maxiter, mix=0.5)
        
        self.rho = rho
        self.energije = energije
        self.fs = fs
        self.vecs = vecs
        self.err = err
        self.n = n
        self.mu = mu
        self.delta_b, self.delta_c = Delta(self.K, self.rho, Vb, Vc)
        if show_print == None: print(1/T, err, n, self.delta_b.real, self.delta_c.real)

    def run(self, Ts, show_print=None):
        for _, T in enumerate(Ts):
            if T == Ts[-1]:
                self.next_T(T, 1, show_print)
                self.Ts.append(T)
                self.betas.append(1/T)
                self.delta_bs.append(self.delta_b)
                self.delta_cs.append(self.delta_c)
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)

                fs = np.zeros((2,self.Nk//2+1))
                for i in range(self.Nk//2 + 1):
                   fs[:, i] = 1/(1 + np.exp((self.energije[:,i] - self.mu)/T))
            else:
                self.next_T(T, 2, show_print)

    def run2(self, betas, stops, Gamma, Nomega, eps):
        for i, beta in enumerate(betas):
            T = 1/beta
            if i not in stops:
                if (i+1) in stops:
                    rho_save = self.rho
                    energije_save = self.energije
                    fs_save = self.fs
                    vecs_save = self.vecs
                    mu_save = self.mu
                    err_save = self.err
                    n_save = self.n
                self.next_T(T, 2)
            else:
                print(f'started evaluating at beta={beta}')
                self.next_T(T, 1)
                self.Ts.append(T)
                self.betas.append(1/T)
                self.delta_bs.append(self.delta_b)
                self.delta_cs.append(self.delta_c)
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)
                self.gap = np.min(self.energije[1] - self.energije[0])
                self.gaps.append(self.gap)

                omega_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T)
                omegas = np.linspace(-omega_max, omega_max, Nomega)

                spektralka = Spektralka(omegas, self.mu, self.energije, Gamma, )
                tok = j_tok(self.K, self.phys_parameters, self.mu)
                tok_tilde = np.einsum('ijx,jlx,mlx -> imx', self.vecs, tok, self.vecs.conj())
                phi = phi_Kubo(self.K, tok_tilde, tok_tilde, spektralka, omegas)

                m1 = mf_matrix1(self.K, self.rho, self.phys_parameters, self.mu)
                m1_tilde = np.einsum('ijx,jlx,mlx -> imx', self.vecs, m1, self.vecs.conj())
                m2 = mf_matrix2(self.K, self.rho, self.phys_parameters, self.mu)
                m2_tilde = np.einsum('ijx,jlx,mlx -> imx', self.vecs, m2, self.vecs.conj())
                m3 = mf_matrix3(self.K, self.rho, self.phys_parameters, self.mu)
                m3_tilde = np.einsum('ijx,jlx,mlx -> imx', self.vecs, m3, self.vecs.conj())
                m4 = mf_matrix4(self.K, self.rho, self.phys_parameters, self.mu)
                m4_tilde = np.einsum('ijx,jlx,mlx -> imx', self.vecs, m4, self.vecs.conj())
                phiQ = phi_Kubo(self.K, m1_tilde + m2_tilde + m3_tilde + m4_tilde, tok_tilde, spektralka, omegas)

                K0 = np.pi * np.sum(phi.real * (-fd_1(omegas, T))) * (omegas[1] - omegas[0])
                K1 = np.pi * np.sum(omegas * phi.real * (-fd_1(omegas, T))) * (omegas[1] - omegas[0])
                K1b = np.pi * np.sum(phiQ.real * (-fd_1(omegas, T))) * (omegas[1] - omegas[0])

                self.K1.append(K1)
                self.K1q.append(K1b)
                self.K0.append(K0)
                
                K0b, K1b = Kn_boltz(self.K, self.energije, self.mu, T)
                self.K1_b.append(K1b / (2 * Gamma))
                self.K0_b.append(K0b / (2 * Gamma))

                if i > 0:
                    self.rho = rho_save
                    self.energije = energije_save
                    self.fs = fs_save
                    self.vecs = vecs_save
                    self.mu = mu_save
                    self.err = err_save
                    self.n = n_save

    def reset(self, mu0):
        self.GS()
        self.mu = mu0

    def temperature_propagate(self, mu0, beta0, betas, scale=1.05):
        ends = [int(np.emath.logn(scale, beta0/beta)) for beta in betas]
        Ts = 1/np.array(betas)
        for i, _ in enumerate(Ts):
            self.reset(mu0)
            set_betas = beta0/scale**np.arange(ends[i])
            set_Ts = 1/set_betas
            self.run(set_Ts)

def find_GS_mu(Nk, mu0, phys_parameters, parameters1, parameters2,beta0=40, beta1=30, scale=1.05):
    m = model(Nk, mu0, phys_parameters, parameters1, parameters2)
    m.GS()
    betas = [beta1]
    ends = [int(np.emath.logn(scale, beta0/beta)) for beta in betas]
    Ts = 1/np.array(betas)
    for i, T in enumerate(Ts):
        m.reset(mu0)
        set_betas = beta0/scale**np.arange(ends[i])
        set_Ts = 1/set_betas
        m.run(set_Ts, show_print=False)
    mu0=0.5*(np.min(m.energije[1]) + np.max(m.energije[0]))
    m = model(Nk, mu0, phys_parameters, parameters1, parameters2)
    ends = [int(np.emath.logn(scale, beta0/beta)) for beta in betas]
    Ts = 1/np.array(betas)
    m.GS()
    for i, T in enumerate(Ts):
        m.reset(mu0)
        set_betas = beta0/scale**np.arange(ends[i])
        set_Ts = 1/set_betas
        m.run(set_Ts, show_print=False)
    print('found mu in the Ground state')
    return float(m.mu)
