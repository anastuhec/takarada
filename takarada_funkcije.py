import numpy as np
import scipy.linalg as LA
import matplotlib

def colorFader(c1, c2, mix=0.5):
    c1 = np.array(matplotlib.colors.to_rgb(c1))
    c2 = np.array(matplotlib.colors.to_rgb(c2))
    return matplotlib.colors.to_hex((1-mix)*c1 + mix*c2)

def E_analytic(K, rho, phys_parameters):
    Nk = len(K)
    Vb = phys_parameters['Vb']
    Vc = phys_parameters['Vc']
    t = phys_parameters['t']
    t_ = phys_parameters['t_']
    epsilon = phys_parameters['epsilon']
    delta_k = -Vb/Nk*np.sum(rho[1,0,:]) - Vc/Nk*np.sum(rho[1,0,:]*np.exp(-1j*K)) * np.exp(-1j*K)
    E_minus = (t-t_)*np.cos(K) - np.sqrt( ((t+t_)*np.cos(K) - epsilon)**2 + np.abs(delta_k)**2 )
    E_plus = (t-t_)*np.cos(K) + np.sqrt( ((t+t_)*np.cos(K) - epsilon)**2 + np.abs(delta_k)**2 )
    return np.vstack([E_minus, E_plus])

def v_analytic(K, rho, phys_parameters):
    Nk = len(K)
    t = phys_parameters['t']
    t_ = phys_parameters['t_']
    epsilon = phys_parameters['epsilon']
    Vb = phys_parameters['Vb']
    Vc = phys_parameters['Vc']

    delta_b = -Vb/Nk*np.sum(rho[1,0,:])
    delta_c = -Vc/Nk*np.sum(rho[1,0,:]*np.exp(-1j*K))
    delta_k = delta_b + delta_c*np.exp(-1j*K)

    vel_minus = -(t-t_)*np.sin(K) - (-((t+t_)*np.cos(K) - epsilon)*(t+t_)*np.sin(K) + (1j*np.exp(1j*K) * delta_b * delta_c.conj() ).real ) / np.sqrt( ((t+t_)*np.cos(K) - epsilon)**2 + np.abs(delta_k)**2)
    vel_plus = -(t-t_)*np.sin(K) + (-((t+t_)*np.cos(K) - epsilon)*(t+t_)*np.sin(K) + (1j*np.exp(1j*K) * delta_b * delta_c.conj() ).real ) / np.sqrt( ((t+t_)*np.cos(K) - epsilon)**2 + np.abs(delta_k)**2)
    return np.vstack([vel_minus, vel_plus])


def rho0(Nk):
    rho = np.zeros((2,2,Nk))
    rho[0,0,:] = 1
    return rho

def h_k(k, rho, K, phys_parameters, eps0, include_hartree):
    t = phys_parameters['t']
    t_ = phys_parameters['t_']
    t12 = phys_parameters['t12']
    epsilon = phys_parameters['epsilon']
    Vb = phys_parameters['Vb']
    Vc = phys_parameters['Vc']
    
    Nk = rho.shape[-1]
    hk = np.zeros((2,2), dtype='complex')
    hk[0,0] += 2*t*np.cos(k) - epsilon
    hk[1,1] += -2*t_*np.cos(k) + epsilon

    ad = t12 * (np.exp(1j*k) + np.exp(-1j*k))
    hk[0,1] += ad
    hk[1,0] += ad.conj()

    if include_hartree == True:
        hk[0,0] += (Vb + Vc)*np.sum(rho[1,1,:])/Nk
        hk[1,1] += (Vb + Vc)*np.sum(rho[0,0,:])/Nk

    ad = - 1/Nk * Vb * np.sum(rho[1,0,:]) - 1/Nk * Vc * np.sum(rho[1,0,:] * np.exp(-1j*K)) * np.exp(-1j*k)
    hk[0,1] += ad
    hk[1,0] += ad.conj()

    if eps0 != 0:
        hk[0,1] += 2*eps0*1j*np.sin(k)
        hk[1,0] += - 2*eps0*1j*np.sin(k)
    return hk

def diagonalize(rho, K, T, mu, phys_parameters, eps0, include_hartree):
    Nk = len(K)
    energije, vecs = np.zeros((2,Nk)), np.zeros((2,2,Nk), dtype='complex')
    fs = np.zeros((2,2,Nk))

    for i in [0, Nk//2]:
        en, v = LA.eigh(h_k(K[i], rho, K, phys_parameters, eps0, include_hartree))
        energije[:,i] = en
        vecs[:,:,i] = v
        if T == 0:
            np.fill_diagonal(fs[:, :, i], np.array([1, 0]))
        else:
            np.fill_diagonal(fs[:,:,i], 1/(1 + np.exp((en - mu)/T)))

    for i in range(1, Nk//2):
        en, v = LA.eigh(h_k(K[i], rho, K, phys_parameters, eps0, include_hartree))
        energije[:,i] = en
        energije[:,-i] = en
        vecs[:,:,i] = v
        vecs[:,:,-i] = v.conj()
        if T == 0:
            np.fill_diagonal(fs[:, :, i], np.array([1, 0]))
            np.fill_diagonal(fs[:, :, -i], np.array([1, 0]))
        else:
            np.fill_diagonal(fs[:,:,i], 1/(1 + np.exp((en - mu)/T)))
            np.fill_diagonal(fs[:,:,-i], 1/(1 + np.exp((en - mu)/T)))
    return energije, vecs, fs

def F(rho, K, T, mu, phys_parameters, eps0, include_hartree):
    _, vecs, fs = diagonalize(rho, K, T, mu, phys_parameters, eps0, include_hartree)
    rho_new = np.einsum('ijk,jmk,mnk->ink', vecs, fs, np.swapaxes(vecs.conj(),0,1))
    return rho_new, np.max(np.abs(rho - rho_new))

def zasedenost(rho):
    return (np.sum(np.diag(np.einsum('ijk->ij', rho)))/(np.prod(rho.shape[-1]))).real

def Rho_next(rho, K, T, mu, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=0.5):
    err, N_iters = 1, 0
    while err > epsilon_threshold and N_iters < maxiter:
        if N_iters < N_epsilon: eps = eps0
        else: eps = 0
        rho_new, err = F(rho, K, T, mu, phys_parameters, eps, include_hartree)
        rho = rho_new * mix + rho * (1 - mix)
        N_iters += 1
    rho, _ = F(rho, K, T, mu, phys_parameters, 0, include_hartree)
    energije, vecs, fs = diagonalize(rho, K, T, mu, phys_parameters, 0, include_hartree)
    return rho, err, energije, vecs, fs, zasedenost(rho)

def Phi(K, rho):
    Nk = rho.shape[-1]
    return [1/Nk * np.sum(rho[0,1,:] * np.exp(-1j*K*delta)).real for delta in [0,1]]

def NewMu(K, rho, T, mu, dmu, phys_parameters, eps0, epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=0.5, n_pass=1e-4, mix2=0.001, mix3=1.5, max_trials=30):
    rho_a, err_a, energije_a, vecs_a, fs_a, n_a = Rho_next(rho, K, T, mu, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=mix)
    if np.abs(n_a - 1) < n_pass and err_a < epsilon_threshold:
        return rho_a, err_a, energije_a, vecs_a, fs_a, n_a, mu
    n_b = Rho_next(rho, K, T, mu + dmu, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=mix)[-1]
    chi = (n_b - n_a)/dmu
    if chi != 0: mu = mu - mix2 * (n_a - 1)/np.abs(chi)

    pogoj = False
    koraki = 0
    if np.abs(chi) > 0: faktor = (n_a - 1)/chi * mix3
    else: faktor = 0.1
    if chi >= 0:
        if n_a >= 1:
            sign = -1
        elif n_a < 1: sign = +1
    elif chi < 0:
        if n_a > 1: sign = +1
        elif n_a < 1: sign = -1
        
    sgns = np.ones(2) * np.sign(n_a - 1)
    ns = np.array([0, n_a])
    mus = [0, mu]
    enough = False
    while sgns[0] == sgns[1]:
        if np.abs(n_a - 1) < n_pass and err_a < epsilon_threshold:
            enough = True
            break
        rho_b, err_b, energije_b, vecs_b, fs_b, n_b = Rho_next(rho, K, T, mu + faktor*koraki*sign, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=mix)
        if np.abs(n_b - 1) < n_pass and err_b < epsilon_threshold:
            return rho_b, err_b, energije_b, vecs_b, fs_b, n_b, mu + faktor*koraki*sign
        ns[0] = n_b
        mus[0] = mu + faktor*koraki*sign
        sgns[1] = np.sign(n_b - 1)
        if sgns[0] != sgns[1]: break
        if n_b < 1 and n_b < ns[1]: sign *= -1
        if n_b > 1 and n_b > ns[1]: sign *= -1
        ns = np.roll(ns, 1)
        mus = np.roll(mus, 1)
        sgns[1] = np.sign(n_b - 1)
        koraki += 1
        if np.abs(n_b - 1) < n_pass and err_b < epsilon_threshold:
            enough = True
            mu_mid = mu + faktor*koraki*sign
            break
        
    mus = np.sort(np.array([mu + faktor*koraki*sign, mu + faktor*(koraki-1)*sign]))
    ns = np.sort(np.array(ns))

    trials = 0
    while pogoj == False:
        if enough == True:
            break   
        mu_mid = (mus[0] + mus[1])/2
        n_mid = Rho_next(rho, K, T, mu_mid, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=mix)[-1]
        if n_mid > 1: mus[1] = mu_mid
        elif n_mid < 1: mus[0] = mu_mid
        if np.abs(n_mid - 1) < n_pass: break
        trials += 1 
        if trials > max_trials: break
    rho, err, energije, vecs, fs, n = Rho_next(rho, K, T, mu_mid, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=mix)
    return rho, err, energije, vecs, fs, n, mu_mid

''' df/domega, f je Fermi-Diracova porazdelitvena funkcija '''
def fd_1(omega, T): return -1/(4*T)/np.cosh(omega/(2*T))**2

def parameters(phys_parameters, mu):
    b = phys_parameters['b']
    t = phys_parameters['t']
    t_ = phys_parameters['t_']
    t12 = phys_parameters['t12']
    epsilon = phys_parameters['epsilon']
    Vb = phys_parameters['Vb']
    Vc = phys_parameters['Vc']

    kinetic = [(1, 1, 1, t),
               (-1, 1, 1, t),
               (1, 2, 2, -t_),
               (-1, 2, 2, -t_ ),
               (0, 1, 2, t12),
               (1, 1, 2, t12),
               (0, 2, 1, t12),
               (-1, 2, 1, t12),
               (0, 1, 1, -epsilon - mu),
               (0, 2, 2, epsilon - mu)]
    interaction = [(0, 1, 2, Vb/2),
                   (0, 2, 1, Vb/2),
                   (1, 1, 2, Vc/2),
                   (-1, 2, 1, Vc/2),]
    pos = {1: 0., 2: b}
    return pos, kinetic, interaction

''' matrix for number density operator '''
def j_tok(K, phys_parameters, mu):
    pos, kinetic, interaction = parameters(phys_parameters, mu)

    Nk = len(K)
    j = np.zeros((2, 2, Nk), dtype='complex')
    for line in kinetic:
        (x, orb1, orb2, t) = list(map(float,line))
        orb1, orb2 = int(orb1), int(orb2)
        ad = 1j * t * np.exp(-1j * K * x) * (pos[orb2] - pos[orb1] - x)
        j[orb1 - 1, orb2 - 1] += ad
        if orb1 != orb2:
            j[orb2 - 1, orb1 - 1] += ad.conjugate()

    j_K = np.zeros((2, 2, Nk), dtype='complex')
    for line in kinetic:
        (x, orb1, orb2, t) = list(map(float,line))
        orb1, orb2 = int(orb1), int(orb2)
        for line_ in kinetic:
            (x_, orb1_, orb2_, t_) = list(map(float,line_))
            orb1_, orb2_ = int(orb1_), int(orb2_)
            if orb2 == orb1_:
                ad = - 1j * 0.5 * t * t_ * np.exp(-1j * K * (x + x_)) * (pos[orb1] - pos[orb2_] + (x+x_))
                j_K[orb1 - 1, orb2_ - 1] += ad

    j_I = np.zeros((2, 2, 2, Nk, Nk), dtype='complex') # spin alpha gammma beta
    for line in kinetic:
        (x, orb1, orb2, t) = list(map(float,line))
        if orb1 == orb2 and x == 0: pass
        else:
            orb1, orb2 = int(orb1), int(orb2)
            for line_ in interaction:
                (x_, orb1_, orb2_, V_) = list(map(float,line_))
                orb1_, orb2_ = int(orb1_), int(orb2_)

                if orb2 == orb2_:
                    for ind_q, q in enumerate(K):
                        ad = -1j * t * V_ * np.exp(-1j * (K*x + q*x_)) * np.exp(-1j*q*x) * (pos[orb1] - pos[orb1_] + (x-x_)) / Nk
                        j_I[orb1 - 1, orb1_ - 1, orb2 - 1, :, ind_q] += ad

                if orb1 == orb2_:
                    for ind_q, q in enumerate(K):
                        ad = 1j * t * V_ * np.exp(-1j * (K*x + q*x_)) * (pos[orb2] - pos[orb1_] - (x+x_)) / Nk
                        j_I[orb1 - 1, orb1_ -1, orb2 -1, :, ind_q] += ad
    return j, j_K, j_I

def spektralna_k(omega, mu, energije_k, Gamma=0.05):
    A = np.zeros((2,2))
    for i in range(2):
        A[i,i] = -1/np.pi * Gamma / ((omega - energije_k[i] + mu)**2 + Gamma**2)
    return A

''' create TNS class '''
class model:
    def __init__(self, Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree):
        self.K = 2*np.pi * np.arange(-Nk/2, Nk/2) / Nk
        self.Nk = Nk
        self.parameters1 = parameters1
        self.parameters2 = parameters2
        self.phys_parameters = phys_parameters
        self.mu = mu0
        self.include_hartree = include_hartree

        self.phis = []
        self.mus = []
        self.errors = []
        self.occupations = []
        self.times_rho = []
        self.times_boltzmann = []
        self.times_kubo = []
        self.Ts = []
        self.betas = []

        self.kubo_L11 = []
        self.kubo_LI = []
        self.kubo_LK = []
        self.boltz_L11 = []

    def GS(self):
        rho, err, energije, vecs, fs, n = Rho_next(rho0(self.Nk), self.K, 0, self.mu, self.phys_parameters, self.parameters1['eps0'],
                                                  self.parameters1['epsilon_threshold'], self.parameters1['N_epsilon'], self.parameters1['maxiter'], self.include_hartree, mix=0.5)
        self.rho = rho
        self.energije = energije
        self.vecs = vecs
        self.fs = fs
        self.phi = Phi(self.K, self.rho)
        self.j_matrix = j_tok(self.K, self.phys_parameters, self.mu)

        #print(f'found ground state, err={err}, n_err={np.abs(n-1)}, phi={self.phi}')

    def next_T(self, T, i, show_print=None) -> None:
        if i == 1: parameters = self.parameters1
        elif i == 2: parameters = self.parameters2

        eps0 = parameters['eps0']
        dmu = parameters['dmu']
        epsilon_threshold = parameters['epsilon_threshold']
        N_epsilon = parameters['N_epsilon']
        maxiter = parameters['maxiter']
        n_pass = parameters['n_pass']

        rho, err, energije, vecs, fs, n, mu = NewMu(self.K, self.rho, T, self.mu, dmu,
                                                        self.phys_parameters, eps0, epsilon_threshold, N_epsilon, maxiter, self.include_hartree, n_pass=n_pass)
        self.rho = rho
        self.energije = energije
        self.fs = fs
        self.vecs = vecs
        self.err = err
        self.n = n
        self.mu = mu
        self.phi = Phi(self.K, self.rho)
        if show_print == None: print(1/T, err, n, self.phi)

    def run(self, Ts, Gamma=0.05, show_print=None):
        for _, T in enumerate(Ts):
            if T == Ts[-1]:
                self.next_T(T, 1, show_print)
                self.Ts.append(T)
                self.betas.append(1/T)
                self.phis.append(self.phi)
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)
                self.j_matrix = j_tok(self.K, self.phys_parameters, self.mu)

                fs = np.zeros((2,2,self.Nk))
                for i in range(self.Nk):
                    np.fill_diagonal(fs[:, :, i], 1/(1 + np.exp((self.energije[:,i] - self.mu)/T)))
            else:
                self.next_T(T, 2, show_print)

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

def find_GS_mu(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree, beta0=40, beta1=30, scale=1.05):
    m = model(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree)
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
    m = model(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree)
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

def Phit(K, rho, U, energije, phys_parameters, mu, omegas, Gamma):
    Nk = len(K)
    j, jK, _ = j_tok(K, phys_parameters, mu)

    phi = np.zeros((len(omegas), Nk), dtype='complex')
    phiK = np.copy(phi)

    for i, omega in enumerate(omegas):
        for ii, _ in enumerate(K):
            u = U[:,:,ii]
            A = spektralna_k(omega, mu, energije[:,ii], Gamma=Gamma)
            phi[i, ii] += np.trace(u @ j[:,:,ii] @ u.conj().T @ A @ u @ j[:,:,ii] @ u.conj().T @ A)
            phiK[i, ii] += np.trace(u @ jK[:,:,ii] @ u.conj().T @ A @ u @ j[:,:,ii] @ u.conj().T @ A)

    phi = np.einsum('uk->u', phi) / Nk
    phiK = np.einsum('uk->u', phiK) / Nk
        
    rezultati = {'phi': phi,
                 'phiK': phiK
                 }

    return rezultati

def Phit_I(K, rho, U, energije, phys_parameters, mu, omegas, Gamma):
    Nk = len(K)
    j, _, jI = j_tok(K, phys_parameters, mu)

    phi3 = np.zeros((len(omegas), Nk), dtype='complex')
    phi5 = np.copy(phi3)
    phi6 = np.copy(phi3)
    phi4 = np.zeros((len(omegas), Nk, Nk), dtype='complex')

    M_3 = np.einsum('acbk, ccu-> abk', jI[:,:,:,:,Nk//2], rho)
    M_4 = - np.einsum('ijlkq, jlk-> ijkq', jI, rho)
    M_6 = np.zeros((2, 2, Nk), dtype='complex')
    M_5 = - np.diag(np.einsum('ijlk,ilk->j', jI[:,:,:,:,Nk//2], rho))

    for i, _ in enumerate(K):
        for ii, _ in enumerate(K):
            ind = (i + ii - Nk//2) % Nk
            M_6[:,:,i] += np.einsum('ijl,ij->jl', jI[:,:,:,i,ii], rho[:,:,ind])

    for i, omega in enumerate(omegas):
        for ii, _ in enumerate(K):
            u = U[:,:,ii]
            A = spektralna_k(omega, mu, energije[:,ii], Gamma=Gamma)

            phi3[i, ii] += np.trace(u @ M_3[:,:,ii] @ u.conj().T @ A @ u @ j[:,:,ii] @ u.conj().T @ A)

            phi5[i, ii] += np.trace(u @ M_5 @ u.conj().T @ A @ u @ j[:,:,ii] @ u.conj().T @ A) 

            phi6[i, ii] += np.trace(u @ M_6[:,:,ii] @ u.conj().T @ A @ u @ j[:,:,ii] @ u.conj().T @ spektralna_k(omega, mu, energije[:,ii]))

            for iii, _ in enumerate(K):
                ind = (ii + iii - Nk//2) % Nk
                u = U[:,:,ind]
                A = spektralna_k(omega, mu, energije[:,ind], Gamma=Gamma)
                phi4[i, ii, iii] += np.trace(u @ M_4[:,:,ii,iii] @ u.conj().T @ A @ u @ j[:,:,ind] @ u.conj().T @ A)


    phi3 = np.einsum('uk->u', phi3) / Nk
    phi5 = np.einsum('uk->u', phi5) / Nk
    phi6 = np.einsum('uk->u', phi6) / Nk
    phi4 = np.einsum('ukq->u', phi4) / Nk
        
    rezultati = {'phi3': phi3,
                 'phi4': phi4,
                 'phi5': phi5,
                 'phi6': phi6}
    if np.allclose(np.einsum('ijk,ijk->', j, rho), 0) == True:
        rezultati['phi1'] = np.zeros(len(omegas))
        rezultati['phi2'] = np.zeros(len(omegas))
    else:
        print('prvi in drugi prispevek sta nenicelna!')
    return rezultati

def phi_mf(K, rho, U, energije, phys_parameters, mu, omegas, Gamma):
    Nk = len(K)
    Nk = len(K)
    Vb = phys_parameters['Vb']
    Vc = phys_parameters['Vc']
    t = phys_parameters['t']
    t_ = phys_parameters['t_']

    delta_b = -Vb/Nk*np.sum(rho[1,0,:])
    delta_c = -Vc/Nk*np.sum(rho[1,0,:]*np.exp(-1j*K))
    delta_k = delta_b + delta_c*np.exp(-1j*K)

    j, _, _ = j_tok(K, phys_parameters, mu)
    j_mf = np.zeros((2,2,Nk), dtype='complex')
    ad = -(t-t_) * (delta_k * np.sin(K) + 1j/4*(Vc/Vb*delta_b * np.exp(-1j*K) - Vb/Vc*delta_c) - 1j/2*(delta_b - delta_c*np.exp(-1j*K))*np.cos(K) )
    j_mf[0,1,:] = ad
    j_mf[1,0,:] = ad.conj()

    phimf = np.zeros((len(omegas), Nk), dtype='complex')
    for i, omega in enumerate(omegas):
        for ii, _ in enumerate(K):
            u = U[:,:,ii]
            A = spektralna_k(omega, mu, energije[:,ii], Gamma=Gamma)
            phimf[i, ii] += np.trace(u @ j_mf[:,:,ii] @ u.conj().T @ A @ u @ j[:,:,ii] @ u.conj().T @ A) 
    phimf = np.einsum('uk->u', phimf) / Nk
    return phimf


def Phi_boltz(K, rho, phys_parameters, energije, mu, omegas, faktor=0.2):
    Nk = len(K)
    phi_boltz = np.zeros(len(omegas))
    if phys_parameters['t12'] == 0:
        vel = v_analytic(K, rho, phys_parameters)
        v_max = np.max(np.abs(vel))
        sigma = np.sqrt(v_max * (omegas[1] - omegas[0]) * (K[1] - K[0])) * faktor
        for i, omega in enumerate(omegas):
            for ind in [0, Nk//2]:
                phi_boltz[i] += 1/(2*np.pi*sigma**2)**0.5 * np.exp(-(omega - energije[0,ind] + mu)**2/(2*sigma**2)) * vel[0,ind]**2
                phi_boltz[i] += 1/(2*np.pi*sigma**2)**0.5 * np.exp(-(omega - energije[1,ind] + mu)**2/(2*sigma**2)) * vel[1,ind]**2
            for ind in range(1,Nk//2):
                phi_boltz[i] += 2 * 1/(2*np.pi*sigma**2)**0.5 * np.exp(-(omega - energije[0,ind] + mu)**2/(2*sigma**2)) * vel[0,ind]**2
                phi_boltz[i] += 2 * 1/(2*np.pi*sigma**2)**0.5 * np.exp(-(omega - energije[1,ind] + mu)**2/(2*sigma**2)) * vel[1,ind]**2
    else:
        vel1 = np.diff(energije[0]) / (K[1] - K[0])
        vel2 = np.diff(energije[1]) / (K[1] - K[0])
        v_max = np.max([np.max(np.abs(vel1)), np.max(np.abs(vel2))])
        sigma = np.sqrt(v_max * (omegas[1] - omegas[0]) * (K[1] - K[0])) * faktor
        for i, omega in enumerate(omegas):
            for ii, _ in enumerate(K[1:]):
                phi_boltz[i] += 1/np.sqrt(2*np.pi*sigma**2)**0.5 * np.exp(-(omega - energije[0,ii] + mu)**2/(2*sigma**2)) * vel1[ii]**2
                phi_boltz[i] += 1/np.sqrt(2*np.pi*sigma**2)**0.5 * np.exp(-(omega - energije[1,ii] + mu)**2/(2*sigma**2)) * vel2[ii]**2
    return phi_boltz / Nk

def Kn_boltz(K, energije, mu, T):
    K0, K1 = 0, 0
    vel1 = np.diff(energije[0]) / (K[1] - K[0])
    vel2 = np.diff(energije[1]) / (K[1] - K[0])
    for ind in range(len(vel1)):
        K0 += -fd_1(energije[0,ind] - mu, T) * vel1[ind]**2 - \
            fd_1(energije[1,ind] - mu, T) * vel2[ind]**2
        K1 += -fd_1(energije[0,ind] - mu, T) * vel1[ind]**2 * (energije[0,ind] - mu) - \
            fd_1(energije[1,ind] - mu, T) * vel2[ind]**2 * (energije[1,ind] - mu)
    return K0, K1