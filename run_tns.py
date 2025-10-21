import numpy as np
import os
import time

''' import functions which calculate stuff '''
os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5")
import tokovi_drugic
import helpers

''' read physical and numerical parameters from the file '''
os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5\\parameters")
parameters = []
with open('parametri-input.txt', 'r') as f:
    for line in f:
        spl = line.split()
        if spl != []:
            if 'parameters' in line:
                pass
            else:
                if len(spl) == 3 and spl[2] == 'int': parameters.append(int(spl[1]))
                else: parameters.append(float(spl[1]))
a, b, b2, U, V, mu0, Ny, Nx = parameters[:8]

''' physical parameters '''
a, b, b2, U, V, mu0, Ny, Nx = parameters[:8] 

''' numerical parameters '''
parameters1 = parameters[8:15]
parameters2 = parameters[15:]

''' set up temperatures '''
scale, beta0, betas = np.load('betas.npy')
ends = [int(np.emath.logn(scale, beta0/beta)) for beta in betas]
Ts = 1/betas

''' create TNS class '''
class TNS:
    def __init__(self, a, b, b2, Ny, Nx, U, V, mu, parameters1, parameters2):
        self.parameters1 = parameters1
        self.parameters2 = parameters2
        self.Nx, self.Ny = Nx, Ny
        self.Nk = Ny * Nx
        Ky = 2*np.pi/b * np.arange(-Ny/2, Ny/2) / Ny
        Kx = 2*np.pi/a * np.arange(-Nx/2, Nx/2) / Nx
        Kxmesh, Kymesh = np.meshgrid(Kx, Ky)
        self.kxmesh = Kxmesh
        self.kymesh = Kymesh
        self.hop = helpers.H_hopping(self.kymesh, self.kxmesh, a, b)
        self.perturb = helpers.H_perturb(self.kymesh, self.kxmesh, a, b)
        self.rho = helpers.Rho0(self.Ny, self.Nx)
        self.mu = mu

        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, a, V)
        self.hartree = helpers.H_hartree(self.rho, self.Nk, U, V)
        j = tokovi_drugic.j_tok(self.kymesh, self.kxmesh, a, b, b2)
        j1 = tokovi_drugic.j_1(self.kymesh, self.kxmesh, a, b, b2)
        j2 = tokovi_drugic.j_2(self.kymesh, self.kxmesh, V, a, b, b2)

        j2_matrix_ = np.empty((2,6,6,6,Ny,Nx,Ny,Nx), dtype='complex')
        j2_matrix_[0] = j2[0]
        j2_matrix_[1] = j2[1]
        j_matrix_ = np.empty((2,6,6,Ny,Nx), dtype='complex')
        j_matrix_[0] = j[0]
        j_matrix_[1] = j[1]
        j1_matrix_ = np.empty((2,6,6,Ny,Nx), dtype='complex')
        j1_matrix_[0] = j1[0]
        j1_matrix_[1] = j1[1]
        self.j_matrix = j_matrix_
        self.j1_matrix = j1_matrix_
        self.j2_matrix = j2_matrix_

        self.rho, self.energije, self.fs, self.vecs, self.err, self.n, self.fock, self.hartree = helpers.GS(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, self.mu, 0.1, a, U, V, 1e-10, maxiter=1000, N_epsilon=5)
        self.rho0 = self.rho
        self.fock0 = self.fock
        self.hartree0 = self.hartree
        self.phi = helpers.Phi(self.kxmesh, self.Nk, self.rho, a)[0].real

        self.phis = []
        self.mus = []
        self.errors = []
        self.occupations = []
        self.times_rho = []
        self.times_boltzmann = []
        self.times_kubo = []
        self.Ts = []
        self.betas = []
        
        self.boltzmann_L1_xx, self.boltzmann_L0_xx = [], []
        self.boltzmann_L1_yy, self.boltzmann_L0_yy = [], []
        self.boltzmann_L1_xy, self.boltzmann_L0_xy = [], []

        self.kubo_L1_xx, self.kubo_L0_xx = [], []
        self.kubo_L1_yy, self.kubo_L0_yy = [], []
        self.kubo_L1_xy, self.kubo_L0_xy = [], []
        self.kubo_L1_yx, self.kubo_L0_yx = [], []

    def next_T(self, T, i) -> None:
        start = time.time()
        if i == 1: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters1
        elif i ==2: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters2
        rho, energije, fs, vecs, fock, hartree, err, mu, n = helpers.NewMu(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock,
                                                                a, U, V, T, self.mu,
                                                                dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials)
        self.rho = rho
        self.energije = energije
        self.fs = fs
        self.vecs = vecs
        self.fock = fock
        self.hartree = hartree
        self.mu = mu
        self.err = err
        self.n = n
        self.times_rho.append(time.time() - start)
        print(1/T, err, n, helpers.Phi(self.kxmesh, self.Nk, rho, a)[0].real)

    def boltzmann_koef(self, T) -> None:
        start = time.time()
        L1_xx, L0_xx, L1_yy, L0_yy, L1_xy, L0_xy = tokovi_drugic.Ln_Boltzmann(self.kymesh, self.kxmesh, self.energije, T, self.mu)
        self.boltzmann_L1_xx.append(L1_xx)
        self.boltzmann_L0_xx.append(L0_xx)
        self.boltzmann_L1_yy.append(L1_yy)
        self.boltzmann_L0_yy.append(L0_yy)
        self.boltzmann_L1_xy.append(L1_xy)
        self.boltzmann_L0_xy.append(L0_xy)  
        self.times_boltzmann.append(time.time() - start) 

    def kubo_koef(self, T) -> None:
        start = time.time()
        L1_xx, L1_yy, L1_xy, L1_yx = tokovi_drugic.L_K(self.kymesh, self.vecs, self.energije, self.j1_matrix, self.j_matrix, T, self.mu) + \
                                    tokovi_drugic.L_I_1(self.kymesh, self.vecs, self.energije, self.fs, self.j2_matrix, self.j_matrix, T, self.mu)
        L0_xx, L0_yy, L0_xy, L0_yx = tokovi_drugic.L_11(self.kymesh, self.vecs, self.energije, self.j_matrix, T, self.mu)
        self.kubo_L1_xx.append(L1_xx)
        self.kubo_L0_xx.append(L0_xx)
        self.kubo_L1_yy.append(L1_yy)
        self.kubo_L0_yy.append(L0_yy)
        self.kubo_L1_xy.append(L1_xy)
        self.kubo_L0_xy.append(L0_xy)
        self.kubo_L1_yx.append(L1_yx)
        self.kubo_L0_yx.append(L0_yx)
        self.times_kubo.append(time.time() - start)

    def run(self, Ts):
        for _, T in enumerate(Ts):
            if T == Ts[-1]:
                self.next_T(T, 1)
                self.boltzmann_koef(T)
                self.kubo_koef(T)
                self.Ts.append(T)
                self.betas.append(1/T)
                self.phis.append(helpers.Phi(self.kxmesh, self.Nk, self.rho, a)[0].real)
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)

            else:
                self.next_T(T, 2)

    def S_kubo(self):
        return - np.array(self.kubo_L1_xx) / np.array(self.kubo_L0_xx),\
                - np.array(self.kubo_L1_yy) / np.array(self.kubo_L0_yy),\
                - np.array(self.kubo_L1_xy) / np.array(self.kubo_L0_xy),\
                - np.array(self.kubo_L1_yx) / np.array(self.kubo_L0_yx)
                
    def S_boltzmann(self):
        return - np.array(self.boltzmann_L1_xx) / np.array(self.boltzmann_L0_xx),\
                - np.array(self.boltzmann_L1_yy) / np.array(self.boltzmann_L0_yy),\
                - np.array(self.boltzmann_L1_xy) / np.array(self.boltzmann_L0_xy)

    def reset(self):
        self.rho = self.rho0
        self.hartree = self.hartree0
        self.fock = self.fock0


s = TNS(a, b, b2, Ny, Nx, U, V, mu0, parameters1, parameters2)

for i, T in enumerate(Ts):
    if i != 0: s.reset()
    set_betas = beta0/scale**np.arange(ends[i])
    set_Ts = 1/set_betas
    s.run(set_Ts)
    if i > 0:
        print(f'------ beta={1/T}, phi={s.phis[-1]} ------')



###


os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5\\parameters")

''' create TNS class '''
class TNS:
    def __init__(self, a, b, b2, Ny, Nx, U, V, mu0, parameters1, parameters2, eps0):
        self.parameters1 = parameters1
        self.parameters2 = parameters2
        self.Nx, self.Ny = Nx, Ny
        self.Nk = Ny * Nx
        Ky = 2*np.pi/b * np.arange(-Ny/2, Ny/2) / Ny
        Kx = 2*np.pi/a * np.arange(-Nx/2, Nx/2) / Nx
        Kxmesh, Kymesh = np.meshgrid(Kx, Ky)
        self.kxmesh = Kxmesh
        self.kymesh = Kymesh
        self.hop = helpers.H_hopping(self.kymesh, self.kxmesh, a, b)
        self.perturb = helpers.H_perturb(self.kymesh, self.kxmesh, a, b)
        self.rho = helpers.Rho0(self.Ny, self.Nx)
        self.mu = mu0

        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, a, V)
        self.hartree = helpers.H_hartree(self.rho, self.Nk, U, V)

        self.j_matrix = tokovi_drugic.j_tok(self.kymesh, self.kxmesh, a, b, b2)
        self.j1_matrix = tokovi_drugic.j_1(self.kymesh, self.kxmesh, a, b, b2)
        self.j2_matrix = tokovi_drugic.j_MF(self.kymesh, self.kxmesh, self.rho, V, a, b, b2)

        jx, jy = tokovi_drugic.j_2(self.kymesh, self.kxmesh, V, a, b, b2)
        self.j2_matrix_full = np.zeros((2,6,6,6,Ny,Nx,Ny,Nx), dtype='complex')
        self.j2_matrix_full[0], self.j2_matrix_full[1] = jx, jy

        self.rho, self.energije, self.fs, self.vecs, self.err, self.n, self.fock, self.hartree = helpers.GS(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, self.mu, eps0, a, U, V, 1e-10, maxiter=1000, N_epsilon=5)
        self.rho0 = self.rho
        self.fock0 = self.fock
        self.hartree0 = self.hartree
        self.phi = helpers.Phi(self.kxmesh, self.Nk, self.rho, a)[0].real

        self.phis = []
        self.mus = []
        self.errors = []
        self.occupations = []
        self.times_rho = []
        self.times_boltzmann = []
        self.times_kubo = []
        self.Ts = []
        self.betas = []

        self.transportne_x = []
        self.transportne_y = []

        self.L11_xx, self.L11_yy = [], []
        self.L12_k_xx, self.L12_k_yy = [], []
        self.L12_i_xx, self.L12_i_yy = [], []
        self.L12_i_neint_xx, self.L12_i_neint_yy = [], []
        self.L12_i_full_xx, self.L12_i_full_yy = [], []

    def next_T(self, T, i) -> None:
        start = time.time()
        if i == 1: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters1
        elif i ==2: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = self.parameters2
        rho, energije, fs, vecs, fock, hartree, err, n, mu = helpers.NewMu(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock,
                                                                a, U, V, T, self.mu,
                                                                dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials)
        self.rho = rho
        self.energije = energije
        self.fs = fs
        self.vecs = vecs
        self.fock = fock
        self.hartree = hartree
        self.mu = mu
        self.err = err
        self.n = n
        self.times_rho.append(time.time() - start)
        print(1/T, err, n, helpers.Phi(self.kxmesh, self.Nk, rho, a)[0].real)

    def run(self, Ts, Gamma, domega, eps=1e-5, transport=None):
        for _, T in enumerate(Ts):
            if T == Ts[-1]:
                self.next_T(T, 1)
                self.Ts.append(T)
                self.betas.append(1/T)
                self.phis.append(helpers.Phi(self.kxmesh, self.Nk, self.rho, a)[0].real)
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)
                
                if transport == 'evaluate':
                    omega_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T)))) * 2 * T
                    omegas, trans = tokovi_drugic.transportna_phi(self.kymesh, self.vecs, self.energije, self.j_matrix, self.mu, omega_max=omega_max, domega=domega, Gamma=Gamma)
                    self.j2_matrix = tokovi_drugic.j_MF(self.kymesh, self.kxmesh, self.rho, V, a, b, b2)
                    _, transK, transI = tokovi_drugic.transportna_phiE(self.kymesh, self.vecs, self.energije, self.j_matrix, self.j1_matrix, self.j2_matrix, self.mu, Gamma=Gamma, omega_max=omega_max, domega=domega)
                    _, transI_full = tokovi_drugic.transportna_I_full(self.kymesh, self.vecs, self.energije, self.fs, self.j2_matrix_full, self.j_matrix, self.mu, omega_max, domega, Gamma=Gamma)
                    for i in range(2):
                        l11 = tokovi_drugic.L11(trans[i], omegas, T)
                        l12_k, l12_i, l12_i_full, l12_i_neint = tokovi_drugic.L12(transK[i], transI[i], transI_full[i], trans[i], omegas, T)
                        if i == 0:
                            self.L11_xx.append(l11)
                            self.L12_k_xx.append(l12_k)
                            self.L12_i_xx.append(l12_i)
                            self.L12_i_full_xx.append(l12_i_full)
                            self.L12_i_neint_xx.append(l12_i_neint)
                        else:
                            self.L11_yy.append(l11)
                            self.L12_k_yy.append(l12_k)
                            self.L12_i_yy.append(l12_i)
                            self.L12_i_full_yy.append(l12_i_full)
                            self.L12_i_neint_yy.append(l12_i_neint)
            else:
                self.next_T(T, 2)

    def reset(self, mu0):
        self.rho = self.rho0
        self.hartree = self.hartree0
        self.fock = self.fock0
        self.mu = mu0

    def reset_infty(self):
        self.rho = helpers.Rhoinfty(self.Ny, self.Nx)
        self.hartree = helpers.H_hartree(self.rho, self.Nk, U, V)
        self.fock = helpers.H_fock(self.kxmesh, self.Nk, self.rho, a, V)
        
        _, energije, fs, vecs, _, _, _, _ = helpers.Rho_next(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, a, U, V, 0, self.mu, 50, 0.5, 1e-10, eps0=0.0, N_epsilon=5)
        self.energije = energije
        self.fs = fs
        self.vecs = vecs

    def collect(self):
        self.phis = np.array(self.phis)
        self.mus = np.array(self.mus)
        self.errors = np.array(self.errors)
        self.occupations = np.array(self.occupations)
        self.times_rho = np.array(self.times_rho)
        self.times_boltzmann = np.array(self.times_boltzmann)
        self.times_kubo = np.array(self.times_kubo)
        self.Ts = np.array(self.Ts)
        self.betas = np.array(self.betas)