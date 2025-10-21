

import os
import numpy as np
import time

os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5")

import tokovi_drugic
import helpers


os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5\\parameters")

U = 2.5 # eV
V = 0.785 # eV
a = 3.51 # A
b = 15.79 # A
b2 = 1.927 # A
mu0 = 2.84
dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = 0.001, 100, 2000, 1e-9, 0.5, 0.001, 1.5, 1e-3, 30
parameters1 = [dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials]

dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials = 0.001, 10, 10, 1e-9, 0.5, 0.001, 1.5, 1e-3, 30
parameters2 = [dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials]


''' create TNS class '''
class TNS:
    def __init__(self, Ny, Nx, a=a, b=b, b2=b2, U=U, V=V, mu0=mu0, parameters1=parameters1, parameters2=parameters2, eps0=0.1, eps_gs=1e-10):
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
        self.j_matrix_new = np.copy(self.j_matrix)
        self.j1_matrix = tokovi_drugic.j_1(self.kymesh, self.kxmesh, a, b, b2,  self.rho, self.Nk, U, V, mu=self.mu)
        self.j2_matrix = tokovi_drugic.j_MF(self.kymesh, self.kxmesh, self.rho, V, a, b, b2)

        self.j2_matrix_full = np.zeros((2,6,6,6,Ny,Nx), dtype='complex')
        self.j2_matrix_full[0], self.j2_matrix_full[1] = tokovi_drugic.j_2_kq(self.kymesh, self.kxmesh, 0, 0, V, a, b, b2)
        
        self.j2_MF = tokovi_drugic.j_MF(self.kymesh, self.kxmesh, self.rho, V, a, b, b2)
        self.rho, self.energije, self.fs, self.vecs, self.err, self.n, self.fock, self.hartree = helpers.GS(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock, self.mu, eps0, a, U, V, eps_gs, maxiter=1000, N_epsilon=5)
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
        self.Omegas = []

        self.L11_xx, self.L11_yy = [], []
        self.L12_xx, self.L12_yy = [], []

        self.L11_xx_boltz, self.L11_yy_boltz = [], []
        self.L12_xx_boltz, self.L12_yy_boltz = [], []

        self.Lk_x = []
        self.Lk_y = []
        self.Li_x = []
        self.Li_y = []

    def next_T(self, T, i) -> None:
        start = time.time()
        if i == 1: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials, faktor1 = self.parameters1
        elif i ==2: dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials, faktor1 = self.parameters2
        rho, energije, fs, vecs, fock, hartree, err, n, mu = helpers.NewMu(self.kxmesh, self.rho, self.hop, self.perturb, self.hartree, self.fock,
                                                                a, U, V, T, self.mu,
                                                                dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials, faktor1=faktor1)
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
        print(i, 1/T, err, n, helpers.Phi(self.kxmesh, self.Nk, rho, a)[0].real)

    def run2(self, betas, stops, Gamma, omega_max, domega, eps=1e-5, meja=1e-3, step=20, transport=None, transportB=None, transportE=None):
        for i, beta in enumerate(betas):
            T = 1/beta
            if i not in stops:
                if (i+1) in stops:
                    rho_save = self.rho
                    energije_save = self.energije
                    fs_save = self.fs
                    vecs_save = self.vecs
                    fock_save = self.fock
                    hartree_save = self.hartree
                    mu_save = self.mu
                    err_save = self.err
                    n_save = self.n
                self.next_T(T, 2)
            else:
                self.next_T(T, 1)
                self.Ts.append(T)
                self.betas.append(1/T)
                self.phis.append(helpers.Phi(self.kxmesh, self.Nk, self.rho, a)[0].real)
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)
                if transportB == 'evaluate':
                    _, _, L11, L12 = tokovi_drugic.L_Boltzmann(self.kymesh, self.kxmesh, self.energije, self.mu, T)
                    self.L11_xx_boltz.append(L11[0])
                    self.L11_yy_boltz.append(L11[1])
                    self.L12_xx_boltz.append(L12[0])
                    self.L12_yy_boltz.append(L12[1])

                if transportE == 'evaluate':
                    l12k_x, l12k_y = [], []
                    l12i_x, l12i_y = [], []
                    omega_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T)
                    n = 50
                    self.j2_MF = tokovi_drugic.j_MF(self.kymesh, self.kxmesh, self.rho, V, a, b, b2)
                    pogoj = False
                    
                    while pogoj == False:
                        omegas = np.linspace(-omega_max, omega_max, n)
                        transportna_K, transportna_I = tokovi_drugic.transportna_phiE(self.kymesh, self.vecs, self.energije, self.j_matrix, self.j1_matrix, self.j2_MF, self.mu, omegas, Gamma=Gamma)
                        for i in range(2):
                            l12_k = np.sum(transportna_K[i] * (-tokovi_drugic.fd_1(omegas, T)) ).real * domega
                            l12_i = np.sum(transportna_I[i] * (-tokovi_drugic.fd_1(omegas, T)) ).real * domega
                            if i == 0:
                                l12k_x.append(l12_k)
                                l12i_x.append(l12_i)
                            else:
                                l12k_y.append(l12_k)
                                l12i_y.append(l12_i)
                        if len(l12k_x) > 2:
                            pogoj = (np.abs((l12k_x[-1] - l12k_x[-2])/l12k_x[-1]) < meja) * (np.abs((l12k_y[-1] - l12k_y[-2])/l12k_y[-1]) < meja) * (np.abs((l12i_x[-1] - l12i_x[-2])/l12i_x[-1]) < meja) * (np.abs((l12i_y[-1] - l12i_y[-2])/l12i_y[-1]) < meja) * (domega < 1e-2)
                            n += step
                    
                    self.Lk_x.append(l12k_x[-1])
                    self.Lk_y.append(l12k_x[-1])
                    self.Li_x.append(l12i_x[-1])
                    self.Li_y.append(l12i_y[-1])
                    print(f'{(n-50)//step} steps towards convergence')
                    

                if transport == 'evaluate':
                    omega_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T)

                    l11s_x, l11s_y = [], []
                    l12s_y, l12s_x = [], []
                    n = 50
                    pogoj = False

                    while pogoj == False:
                        omegas = np.linspace(-omega_max, omega_max, n)
                        domega = omegas[1] - omegas[0]
                        trans, jnew = tokovi_drugic.transportna_phi(self.kymesh, self.vecs, self.energije, self.j_matrix, self.mu, omegas, Gamma=Gamma)
                        for i in range(2):
                            l11 = np.sum(trans[i] * (-tokovi_drugic.fd_1(omegas, T)) ).real * domega
                            l12 = np.sum(trans[i] * omegas * (-tokovi_drugic.fd_1(omegas, T)) ).real * domega
                            if i == 0:
                                l11s_x.append(l11)
                                l12s_x.append(l12)
                            else:
                                l11s_y.append(l11)
                                l12s_y.append(l12)
                        if len(l11s_x) > 2:
                            pogoj = (np.abs((l11s_x[-1] - l11s_x[-2])/l11s_x[-1]) < meja) * (np.abs((l11s_y[-1] - l11s_y[-2])/l11s_y[-1]) < meja) * (np.abs((l12s_x[-1] - l12s_x[-2])/l12s_x[-1]) < meja) * (np.abs((l12s_y[-1] - l12s_y[-2])/l12s_y[-1]) < meja) * (domega < 1e-2)
                            n += step
                    self.j_matrix_new = jnew
                    self.transportne_x.append(trans[0])
                    self.transportne_y.append(trans[1])
                    self.Omegas.append(omegas)
                    self.L11_xx.append(l11s_x[-1])
                    self.L11_yy.append(l11s_y[-1])
                    self.L12_xx.append(l12s_x[-1])
                    self.L12_yy.append(l12s_y[-1])
                    print(f'{(n-50)//step} steps towards convergence')

                    #omegas = np.arange(-omega_max, omega_max, domega)
                    #trans, j_matrix_new = tokovi_drugic.transportna_phi(self.kymesh, self.vecs, self.energije, self.j_matrix, self.mu, omegas, Gamma=Gamma)
                    '''self.j_matrix_new = jnew
                    self.transportne_x.append(trans[0])
                    self.transportne_y.append(trans[1])
                    self.Omegas.append(omegas)
                    for i in range(2):
                        l11 = np.sum(trans[i] * (-tokovi_drugic.fd_1(omegas, T)) ) * domega
                        l12 = np.sum(trans[i] * omegas * (-tokovi_drugic.fd_1(omegas, T)) ) * domega
                        if i == 0:
                            self.L11_xx.append(l11)
                            self.L12_xx.append(l12)
                        else:
                            self.L11_yy.append(l11)
                            self.L12_yy.append(l12)'''

                if i > 0:
                    self.rho = rho_save
                    self.energije = energije_save
                    self.fs = fs_save
                    self.vecs = vecs_save
                    self.fock = fock_save
                    self.hartree = hartree_save
                    self.mu = mu_save
                    self.err = err_save
                    self.n = n_save

    def run(self, Ts, Gamma, omega_max, domega, eps=1e-5, transport=None, transportB=None, num=1000):
        for _, T in enumerate(Ts):
            if T == Ts[-1]:
                self.next_T(T, 1)
                self.Ts.append(T)
                self.betas.append(1/T)
                self.phis.append(helpers.Phi(self.kxmesh, self.Nk, self.rho, a)[0].real)
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)
                if transportB == 'evaluate':
                    _, _, L11, L12 = tokovi_drugic.L_Boltzmann(self.kymesh, self.kxmesh, self.energije, self.mu, T)
                    self.L11_xx_boltz.append(L11[0])
                    self.L11_yy_boltz.append(L11[1])
                    self.L12_xx_boltz.append(L12[0])
                    self.L12_yy_boltz.append(L12[1])
                if transport == 'evaluate':
                    omega_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T)
                    omegas = np.arange(-omega_max, omega_max, domega)
                    domega = omegas[1] - omegas[0]
                    trans, j_matrix_new = tokovi_drugic.transportna_phi(self.kymesh, self.vecs, self.energije, self.j_matrix, self.mu, omegas, Gamma=Gamma)
                    self.j_matrix_new = j_matrix_new
                    self.transportne_x.append(trans[0])
                    self.transportne_y.append(trans[1])
                    self.Omegas.append(omegas)
                    for i in range(2):
                        l11 = np.sum(trans[i].real * (-tokovi_drugic.fd_1(omegas, T)) ) * domega
                        l12 = np.sum(trans[i].real * omegas * (-tokovi_drugic.fd_1(omegas, T)) ) * domega
                        if i == 0:
                            self.L11_xx.append(l11)
                            self.L12_xx.append(l12)
                        else:
                            self.L11_yy.append(l11)
                            self.L12_yy.append(l12)
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