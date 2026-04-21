import numpy as np
import sys
from scipy.special import roots_legendre
sys.path.append("/Users/ana/Desktop/takarada/")
import helpers_takarada as ht

''' Takarada model '''
class model:
    def __init__(self, Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree, stopinfty=False, n_target=1.0):
        self.K = 2*np.pi * np.arange(-Nk/2, Nk/2) / Nk
        self.Nk = Nk
        self.parameters1 = parameters1
        self.parameters2 = parameters2
        self.n_target = n_target
        self.phys_parameters = phys_parameters
        _, _, _, _, _, _, Vb, Vc, _ = phys_parameters
        self.Vb, self.Vc = Vb, Vc
        self.mu = mu0
        self.include_hartree = include_hartree
    
        self.hk0 = ht.h_k0(self.K, self.phys_parameters)
        if not stopinfty:
            self.energy_infty, self.mu_infty, self.gap_infty = ht.Gap_infty(self.Nk, self.phys_parameters, self.parameters1, self.include_hartree)
        self.tok = ht.j_tok(self.K, self.phys_parameters)
        self.hop= ht.j_E_hop(self.K, self.phys_parameters, self.mu)
        self.rhos, self.thetas = ht.rho_operators(self.K, self.phys_parameters, self.include_hartree)
        self.geom, self.phases = ht.input_data(self.K, self.phys_parameters)
        self.g_ffts = ht.G_ffts(self.phases, self.Nk)

        self.delta_bs = []
        self.delta_cs = []
        self.gaps = []
        self.mus = []
        self.errors = []
        self.occupations = []
        self.Ts = []
        self.mean_energies = []
        self.ns0 = []
        self.ns1 = []

        self.L11 = []
        self.L12 = []
        self.L22 = []
        self.L12q = []
        self.L22q = []

        self.L11_boltz = []
        self.L12_boltz  = []
        self.L22_boltz = []

        self.L11_0 = []
        self.L11_corr = []
        self.L12_0 = []
        self.L12_corr = []
        self.L12q_0 = []
        self.L12q_corr = []


    def GS(self):
        _, _, _, _, _, _, Vb, Vc, _ = self.phys_parameters
        rho, err, energije, vecs, fs, n = ht.Rho_next(self.hk0, ht.rho0(self.Nk), self.K, 0, self.mu, self.phys_parameters, self.parameters1['eps0'],
                                                  self.parameters1['epsilon_threshold'], self.parameters1['N_epsilon'], self.parameters1['maxiter'], self.include_hartree, mix=0.5)
        self.rho = rho
        self.energije = energije
        self.vecs = vecs
        self.fs = fs
        self.err = err
        self.n = n
        self.delta_b, self.delta_c = ht.Delta(self.K, self.rho, Vb, Vc)
        self.gap = np.min(self.energije[1]) - np.max(self.energije[0])
        self.mu = 0.5 * (np.min(self.energije[1]) + np.max(self.energije[0]))
        self.hop = ht.j_E_hop(self.K, self.phys_parameters, self.mu)

    def next_T(self, T, i, show_print=None, maxbrentq=50) -> None:
        _, _, _, _, _, _, Vb, Vc, _ = self.phys_parameters

        if i == 1: parameters = self.parameters1
        elif i == 2: parameters = self.parameters2

        eps0 = parameters['eps0']
        dmu = parameters['dmu']
        epsilon_threshold = parameters['epsilon_threshold']
        N_epsilon = parameters['N_epsilon']
        maxiter = parameters['maxiter']
        n_pass = parameters['n_pass']
        
        mu, rho, err, energije, vecs, fs, n = ht.NewMu2(self.mu - dmu, self.mu + dmu, self.hk0, self.rho, self.K, T, self.phys_parameters, eps0, epsilon_threshold, N_epsilon, maxiter, self.include_hartree, mix=0.5, xtol=n_pass, rtol=n_pass, maxiterbrentq=maxbrentq, n_target=self.n_target)
    
        self.rho = rho
        self.energije = energije
        self.fs = fs
        self.vecs = vecs
        self.err = err
        self.n = n
        self.mu = mu
        self.delta_b, self.delta_c = ht.Delta(self.K, self.rho, Vb, Vc)
        if show_print == None: print(1/T, err, n, self.delta_b.real, self.delta_c.real, flush=True)

    def run_Tdependence(self, betas, stops, Gammas, params, maxbrentq, eps_ns=1e-6, evaluate_transport_DC=True, evaluate_vertex_DC=True,
                        phonon_parameters=None, phonon=False, n_workers=None):
        Nomega = params['Nomega']
        eps = params['eps']
        eps2 = params['eps2']
        deg = params['deg']
        omega0 = params['omega0']

        if phonon:
            gbx = phonon_parameters['gbx']
            gby = phonon_parameters['gby']
            gcx = phonon_parameters['gcx']
            gcy = phonon_parameters['gcy']
            omega_bx = phonon_parameters['omega_bx']
            omega_by = phonon_parameters['omega_by']
            omega_cx = phonon_parameters['omega_cx']
            omega_cy = phonon_parameters['omega_cy']
            Gamma_ph = phonon_parameters['Gamma_ph']

        if evaluate_vertex_DC:
            nodes, weights = roots_legendre(deg)
            
        self.Gammas = Gammas
        print('started', flush=True)
        for i, beta in enumerate(betas):
            print(f'{i/len(betas)}', flush=True)
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
                self.next_T(T, 2, maxbrentq=maxbrentq)
            else:
                self.next_T(T, 1, maxbrentq=maxbrentq)
                self.Ts.append(T)
                self.delta_bs.append(self.delta_b)
                self.delta_cs.append(self.delta_c)
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)

                n0, n1 = ht.Ns(self.rho, self.energy_infty, self.delta_b, self.delta_c, self.Vb, self.Vc, eps_ns, self.Ts[-1], self.mus[-1])
                self.ns0.append(n0)
                self.ns1.append(n1)

                self.gap = ht.Gap(self.energije, self.delta_bs[-1], self.delta_cs[-1], self.Vb, self.Vc, eps_ns, self.gap_infty)
                self.gaps.append(self.gap)
                self.mean_energies.append(ht.energy_average(self.K, self.rho, self.phys_parameters, self.energije, self.mu, self.Ts[-1]))
                
                self.tok_tilde = ht.operator_tilde(self.tok, self.vecs)
                self.hop = ht.j_E_hop(self.K, self.phys_parameters, self.mu)
                self.hop_tilde = ht.operator_tilde(self.hop, self.vecs)
                m3, m6, m4a, m4b = ht.compute_all_mf_matrices(self.K, self.rho, self.geom, self.phases, self.g_ffts)
                mat = m3 + m6 + m4a + m4b
                self.mat = mat
                self.mat_tilde = ht.operator_tilde(mat, self.vecs)
                self.rhos_tilde = ht.operator_tilde(self.rhos, self.vecs)

                if evaluate_transport_DC:
                    ''' DC coefficients: Boltzmann's and Kubo's, evaluating the bubble diagram (coefficients as integrals of transport functions) '''
                    self.DC_coefficients(eps, Nomega, Gammas)
                
                if evaluate_vertex_DC:
                    ''' Kubo's DC coefficients, bubble and corrections '''
                    if not phonon:
                        self.DC_bubble_corr(nodes, weights, Gammas, omega0, eps2, n_workers=n_workers)
                    else:
                        self.DC_bubble_corr(nodes, weights, Gammas, omega0, eps2,
                                            gbx=gbx, omega_bx=omega_bx,
                                            gby=gby, omega_by=omega_by,
                                            gcx=gcx, omega_cx=omega_cx,
                                            gcy=gcy, omega_cy=omega_cy,
                                            Gamma_ph=Gamma_ph, phonon=phonon, n_workers=n_workers)
                        
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
    
    def ls_kubo(self, epsilons, Gamma, mfd1):
        spektralka = ht.Spektralka(epsilons, self.mu, self.energije, Gamma)
        phi = ht.phi_Kubo(self.K, self.tok_tilde, self.tok_tilde, spektralka, epsilons)
        phiQ = ht.phi_Kubo(self.K, self.mat_tilde, self.tok_tilde, spektralka, epsilons)
        phiQ2 = ht.phi_Kubo(self.K, self.mat_tilde, self.mat_tilde, spektralka, epsilons)

        l11 = np.pi * ht.integral_omega(phi * mfd1, epsilons)
        l12 = np.pi * ht.integral_omega(epsilons * phi * mfd1, epsilons)
        l22 = np.pi * ht.integral_omega(epsilons**2 * phi * mfd1, epsilons)
        l12q = np.pi * ht.integral_omega(phiQ * mfd1, epsilons)
        l22q = np.pi * ht.integral_omega(phiQ2 * mfd1, epsilons) + 2 * np.pi * ht.integral_omega(phiQ * epsilons * mfd1, epsilons)
        return l11, l12, l22, l12q, l22q

    def DC_coefficients(self, eps, Nomega, Gammas):
        T = self.Ts[-1]
        epsilon_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T)
        epsilons = np.linspace(-epsilon_max, epsilon_max, Nomega, dtype=np.float64)

        K0b, K1b = ht.Kn_boltz(self.K, self.energije, self.mu, T)
        mfd1 = -ht.fd_1(epsilons, T)

        Ngamma = len(Gammas)

        l11 = np.zeros(Ngamma)
        l12 = np.zeros(Ngamma)
        l22 = np.zeros(Ngamma)
        l12q = np.zeros(Ngamma)
        l22q = np.zeros(Ngamma)

        l11_boltz = np.zeros(Ngamma)
        l22_boltz = np.zeros(Ngamma)
        l12_boltz = np.zeros(Ngamma)

        for g, Gamma in enumerate(Gammas):
            l11_, l12_, l22_, l12q_, l22q_ = self.ls_kubo(epsilons, Gamma, mfd1)
            l11[g] = l11_.real
            l22[g] = l22_.real
            l12[g] = l12_.real
            l12q[g] = l12q_.real
            l22q[g] = l22q_.real

            l11_boltz[g] = K0b / (2 * Gamma)
            l22_boltz[g] = K0b / (2 * Gamma)
            l12_boltz[g] = K1b / (2 * Gamma)

        self.L11.append(ht.to_scalar_if_single(l11))
        self.L12.append(ht.to_scalar_if_single(l12))
        self.L22.append(ht.to_scalar_if_single(l22))
        self.L12q.append(ht.to_scalar_if_single(l12q))
        self.L22q.append(ht.to_scalar_if_single(l22q))

        self.L11_boltz.append(ht.to_scalar_if_single(l11_boltz))
        self.L22_boltz.append(ht.to_scalar_if_single(l22_boltz))
        self.L12_boltz.append(ht.to_scalar_if_single(l12_boltz))

    def DC_bubble_corr(self, nodes, weights, Gammas, omega0, eps,
                       gbx=None, omega_bx=None,
                       gby=None, omega_by=None,
                       gcx=None, omega_cx=None,
                       gcy=None, omega_cy=None,
                       Gamma_ph=None, phonon=False, n_workers=None):
        Ngamma = len(Gammas)

        l11_0 = np.zeros(Ngamma)
        l12_0 = np.zeros_like(l11_0)
        l12q_0 = np.zeros_like(l11_0)

        l11 = np.zeros_like(l11_0)
        l12 = np.zeros_like(l11_0)
        l12q = np.zeros_like(l11_0)

        for g, Gamma in enumerate(Gammas):
            mu_ = self.mu / Gamma
            invt = Gamma / self.Ts[-1]
            
            if not phonon:
                results = ht.compute_chi(omega0, self.Nk, Gamma, mu_, invt, nodes, weights, self.thetas, self.tok_tilde, self.mat_tilde, self.energije, self.rhos_tilde, verbose=False, eps=eps, n_workers=n_workers)
            else:
                results = ht.compute_chi(omega0, self.Nk, Gamma, mu_, invt, nodes, weights, self.thetas, self.tok_tilde, self.mat_tilde, self.energije, self.rhos_tilde, verbose=False, eps=eps, n_workers=n_workers,
                                         gbx=gbx, omega_bx=omega_bx,
                                         gby=gby, omega_by=omega_by,
                                         gcx=gcx, omega_cx=omega_cx,
                                         gcy=gcy, omega_cy=omega_cy,
                                         Gamma_ph=Gamma_ph, phonon=True, include_hartree=self.include_hartree, Vb=self.Vb, Vc=self.Vc)
            Chi_jj0 = - results['chi_jj0'].imag
            dChi_jj  = - results['dchi_jj'].imag
            Chi_jj = Chi_jj0 + dChi_jj
            left, right = ht.find_flat_regime(omega0, Chi_jj0)
            l11_0[g] = ht.get_dc_coefficient(omega0[left:right], Chi_jj0[left:right])[0]
            left, right = ht.find_flat_regime(omega0, Chi_jj)
            l11[g] = ht.get_dc_coefficient(omega0[left:right], Chi_jj[left:right])[0]

            Chi_jEj0 = - results['chi_jEj0'].imag
            dChi_jEj = - results['dchi_jEj'].imag
            Chi_jEj = Chi_jEj0 + dChi_jEj
            left, right = ht.find_flat_regime(omega0, Chi_jEj0)
            l12_0[g] = ht.get_dc_coefficient(omega0[left:right], Chi_jEj0[left:right])[0]
            left, right = ht.find_flat_regime(omega0, Chi_jEj)
            l12[g] = ht.get_dc_coefficient(omega0[left:right], Chi_jEj[left:right])[0]

            Chi_matj0 = - results['chi_matj0'].imag
            dChi_matj = - results['dchi_matj'].imag
            Chi_matj = Chi_matj0 + dChi_matj
            left, right = ht.find_flat_regime(omega0, Chi_matj0)
            l12q_0[g] = ht.get_dc_coefficient(omega0[left:right], Chi_matj0[left:right])[0]
            left, right = ht.find_flat_regime(omega0, Chi_matj)
            l12q[g] = ht.get_dc_coefficient(omega0[left:right], Chi_matj[left:right])[0]

        self.L11_0.append(ht.to_scalar_if_single(l11_0))
        self.L11_corr.append(ht.to_scalar_if_single(l11))
        self.L12_0.append(ht.to_scalar_if_single(l12_0))
        self.L12_corr.append(ht.to_scalar_if_single(l12))
        self.L12q_0.append(ht.to_scalar_if_single(l12q_0))
        self.L12q_corr.append(ht.to_scalar_if_single(l12q))


    def optical_response(self, deg, Gamma, omegas):
        nodes, weights = roots_legendre(deg)
        results = ht.compute_chi(omegas, self.Nk, Gamma, self.mu / Gamma, self.Ts[-1] / Gamma, nodes, weights, self.thetas, self.tok_tilde, self.mat_tilde, self.energije, self.rhos_tilde, verbose=False)
        return results

    def simulate_perturbation(self, A0, t0, sigma, Omega0, dt, t_max, Ncorr, tol, Gamma_, perturbation_operator, measure_provider, do_freeze):
        time, measurement, norma, delta_bs, delta_cs = ht.simulate_pulz(self.K, self.hk0, self.rho, self.phys_parameters, self.include_hartree, perturbation_operator, measure_provider,
                                                                    A0, t0, sigma, Omega0, dt, t_max, do_freeze, Ncorr, tol, self.geom, self.phases, self.g_ffts, Gamma_)
        pulz = ht.A_pulz(time, A0, t0, sigma, Omega0)
        results = {'time' : time,
                   'measurement' : measurement,
                   'pulz' : pulz,
                   'norma' : norma,
                   'delta_bs' : delta_bs,
                   'delta_cs' : delta_cs}
        return results

    def collect_data(self):
        data = {"delta_bs": np.array(self.delta_bs),
                "delta_cs": np.array(self.delta_cs),
                "ns0": np.array(self.ns0),
                "ns1": np.array(self.ns1),
                "gaps": np.array(self.gaps),
                "mus": np.array(self.mus),
                "errors": np.array(self.errors),
                "occupations": np.array(self.occupations),
                "Ts": np.array(self.Ts),
                "mean_energies": np.array(self.mean_energies),
                "L11": np.array(self.L11),
                "L12": np.array(self.L12),
                "L22" : np.array(self.L22),
                "L12q": np.array(self.L12q),
                "L22q" : np.array(self.L22q),
                "L11_boltz": np.array(self.L11_boltz),
                "L12_boltz": np.array(self.L12_boltz),
                "L22_boltz": np.array(self.L22_boltz),
                "L11_0": np.array(self.L11_0),
                "L11_corr": np.array(self.L11_corr),
                "L12_0": np.array(self.L12_0),
                "L12_corr": np.array(self.L12_corr),
                "L12q_0": np.array(self.L12q_0),
                "L12q_corr": np.array(self.L12q_corr),
                "phys_parameters" : np.array(self.phys_parameters),
                "Gammas": self.Gammas,
                "include_hartree" : self.include_hartree}
        return data