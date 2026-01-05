import numpy as np
import scipy.linalg as LA
from numba import njit, prange
from scipy.optimize import brentq

''' analytic result for energy bands IF t12 = 0 '''
def E_analytic(K, rho, phys_parameters):
    _, t, t_, _, epsilon, epsilon_, Vb, Vc, _ = phys_parameters
    Nk = len(K)
    delta_k = -Vb/Nk*np.sum(rho[1,0,:]) - Vc/Nk*np.sum(rho[1,0,:]*np.exp(-1j*K)) * np.exp(-1j*K)
    E_minus = (t-t_)*np.cos(K) - np.sqrt( ((t+t_)*np.cos(K) - epsilon)**2 + np.abs(delta_k)**2 )
    E_plus = (t-t_)*np.cos(K) + np.sqrt( ((t+t_)*np.cos(K) - epsilon)**2 + np.abs(delta_k)**2 )
    return np.vstack([E_minus, E_plus])

''' analytic result for group velocities IF t12 = 0 '''
def v_analytic(K, rho, phys_parameters):
    _, t, t_, _, epsilon, epsilon_, Vb, Vc, _ = phys_parameters
    Nk = len(K)

    delta_b = -Vb/Nk*np.sum(rho[1,0,:])
    delta_c = -Vc/Nk*np.sum(rho[1,0,:]*np.exp(-1j*K))
    delta_k = delta_b + delta_c*np.exp(-1j*K)

    vel_minus = -(t-t_)*np.sin(K) - (-((t+t_)*np.cos(K) - epsilon)*(t+t_)*np.sin(K) + (1j*np.exp(1j*K) * delta_b * delta_c.conj() ).real ) / np.sqrt( ((t+t_)*np.cos(K) - epsilon)**2 + np.abs(delta_k)**2)
    vel_plus = -(t-t_)*np.sin(K) + (-((t+t_)*np.cos(K) - epsilon)*(t+t_)*np.sin(K) + (1j*np.exp(1j*K) * delta_b * delta_c.conj() ).real ) / np.sqrt( ((t+t_)*np.cos(K) - epsilon)**2 + np.abs(delta_k)**2)
    return np.vstack([vel_minus, vel_plus])

def rho0(Nk):
    rho = np.zeros((2, 2, Nk))
    rho[0,0,:] = 1.
    return rho

def Delta(K, rho, Vb, Vc):
    Nk = len(K)
    deltas = [0., 1.]
    phi_b = np.sum(rho[1,0] * np.exp(1j*K * deltas[0]))
    phi_c = np.sum(rho[1,0]  * np.exp(1j*K * deltas[1]))
    return - np.array([Vb * phi_b, Vc * phi_c]) / Nk

def h_k0(K, phys_parameters):
    _, t, t_, t12, epsilon, epsilon_, _, _, delta = phys_parameters
    Nk = len(K)
    hk = np.zeros((2, 2, Nk), dtype=np.complex128)

    # diagonal hopping and on-site energy
    hk[0,0,:] += 2*t*np.cos(K) - epsilon
    hk[1,1,:] += -2*t_*np.cos(K) + epsilon_

    # off-diagonal hopping
    ad = (t12+delta) + (t12-delta)*np.exp(1j*K)
    hk[0,1,:] += ad
    hk[1,0,:] += ad.conj()
    return hk

def h_k(K, hk0, rho, phys_parameters, eps0, include_hartree):
    _, _, _, _, _, _, Vb, Vc, _ = phys_parameters
    delta_b, delta_c = Delta(K, rho, Vb, Vc)

    Nk = rho.shape[-1]
    hk = hk0.copy()

    # Fock term:
    delta_k = delta_b + delta_c * np.exp(-1j*K)
    hk[0,1,:] += delta_k
    hk[1,0,:] += delta_k.conj()

    # Hartree term:
    if include_hartree == True:
        hk[0,0,:] += (Vb + Vc) * np.sum(rho[1,1,:]) / Nk
        hk[1,1,:] += (Vb + Vc) * np.sum(rho[0,0,:]) / Nk

    # simulate a perturbation to break symmetry
    if eps0 != 0:
        hk[0,1,:] += eps0 * np.exp(-1j*K)
        hk[1,0,:] += - eps0 * np.exp(1j*K)
    return hk

def diagonalize(hamiltonian, K, T, mu):
    Nk = len(K)
    energije, vecs = np.zeros((2,Nk)), np.zeros((2,2,Nk), dtype=np.complex128)
    fs = np.zeros((2,2,Nk))

    for i in [0, Nk//2]:
        en, v = LA.eigh(hamiltonian[:,:,i])
        energije[:,i] = en
        vecs[:,:,i] = v
        if T == 0:
            np.fill_diagonal(fs[:, :, i], np.array([1, 0]))
        else:
            np.fill_diagonal(fs[:,:,i], 1/(1 + np.exp((en - mu)/T)))

    for i in range(1, Nk//2):
        en, v = LA.eigh(hamiltonian[:,:,i])
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

def F(hamiltonian, rho, K, T, mu):
    _, vecs, fs = diagonalize(hamiltonian, K, T, mu)
    rho_new = np.einsum('ijk,jmk,mnk->ink', vecs, fs, np.swapaxes(vecs.conj(),0,1))
    return rho_new, np.max(np.abs(rho - rho_new))

def zasedenost(rho):
    return (np.sum(np.diag(np.einsum('ijk->ij', rho)))/(np.prod(rho.shape[-1]))).real

def Rho_next(hk0, rho, K, T, mu, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=0.5):
    err, N_iters = 1, 0
    while err > epsilon_threshold and N_iters < maxiter:
        if N_iters < N_epsilon: eps = eps0
        else: eps = 0
        rho_new, err = F(h_k(K, hk0, rho, phys_parameters, eps, include_hartree), rho, K, T, mu)
        rho = rho_new * mix + rho * (1 - mix)
        N_iters += 1
    rho, _ = F(h_k(K, hk0, rho, phys_parameters, 0., include_hartree), rho, K, T, mu)
    energije, vecs, fs = diagonalize(h_k(K, hk0, rho, phys_parameters, 0., include_hartree), K, T, mu)
    return rho, err, energije, vecs, fs, zasedenost(rho)

def Rho_next2(hamiltonian, rho, K, T, mu,
             epsilon_threshold, maxiter, mix=0.5):
    err, N_iters = 1, 0
    while err > epsilon_threshold and N_iters < maxiter:
        rho_new, err = F(hamiltonian, rho, K, T, mu)
        rho = rho_new * mix + rho * (1 - mix)
        N_iters += 1
    rho, _ = F(hamiltonian, rho, K, T, mu)
    energije, vecs, fs = diagonalize(hamiltonian, K, T, mu)
    return rho, err, energije, vecs, fs, zasedenost(rho)


def f_newmu(mu, hk0, rho, K, T, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, mix=0.5):
    _, _, _, _, _, n = Rho_next(hk0, rho, K, T, mu, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, mix=mix)
    return n - 1

def NewMu2(mu1, mu2, hk0, rho, K, T, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, mix=0.5, xtol=1e-10, rtol=1e-10, maxiterbrentq=100):
    return brentq(f_newmu, mu1, mu2, args=(hk0, rho, K, T, phys_parameters, eps0, epsilon_threshold, N_epsilon, maxiter, mix), 
                   xtol=xtol, rtol=rtol, maxiter=maxiterbrentq, full_output=False)

@njit(cache=True)
def parameters(b, t, t_, t12, epsilon, epsilon_, Vb, Vc, mu, delta=0):
    kinetic = np.array([
        (1, 0, 0, t),
        (-1, 0, 0, t),
        (1, 1, 1, -t_),
        (-1, 1, 1, -t_),
        (0, 0, 1, t12 + delta),
        (-1, 0, 1, t12 - delta),
        (0, 1, 0, t12 + delta),
        (1, 1, 0, t12 - delta),
        (0, 0, 0, epsilon - mu),
        (0, 1, 1, epsilon_ - mu)
    ])

    interaction = np.array([
        (0, 0, 1, Vb / 2),
        (0, 1, 0, Vb / 2),
        (1, 0, 1, Vc / 2),
        (-1, 1, 0, Vc / 2)
    ])

    pos = np.array([0.0, b])
    return pos, kinetic, interaction

@njit(cache=True)
def j_tok(K, phys_parameters, mu):
    b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta = phys_parameters
    pos, kinetic, _ = parameters(b, t, t_, t12, epsilon, epsilon_, Vb, Vc, mu, delta)
    Nk = len(K)
    j = np.zeros((2, 2, Nk), dtype=np.complex128)
    for line in kinetic:
        x, orb1, orb2, t = line
        x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
        ad = - 1j * t * np.exp(-1j * K * x) * (pos[orb1 - 1] - pos[orb2 - 1] + x)
        j[orb1, orb2] += ad
    return j

def input_data(K, phys_parameters, mu):
    Nk = len(K)
    b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta = phys_parameters
    pos, kinetic, interaction = parameters(b, t, t_, t12, epsilon, epsilon_, Vb, Vc, mu, delta)
    geom = dict()
    geom["kinetic"] = kinetic
    geom["interaction"] = interaction
    geom["pos"] = pos

    phases_kin = np.zeros((len(kinetic), Nk), dtype=np.complex128)
    phases_int = np.zeros((len(interaction), Nk), dtype=np.complex128)
    for l in range(len(kinetic)):
        x = kinetic[l][0]
        phases_kin[l] = np.exp(-1j*K*x)
    for l in range(len(interaction)):
        x = interaction[l][0]
        phases_int[l] = np.exp(-1j*K*x)
    phases = dict()
    phases["int"] = phases_int
    phases["kin"] = phases_kin
    return geom, phases

def convolution(g, rho_fft):
    return np.fft.fftshift(np.fft.ifft(g * rho_fft))

def prepare_densities_fft(rho):
    rho00 = rho[0,0,:]
    rho11 = rho[1,1,:]
    rho01 = rho[0,1,:]
    rho10 = rho[1,0,:]

    return {
        "n0": rho00.sum(),
        "n1": rho11.sum(),
        "n01" : rho01.sum(),
        "n10": rho10.sum(),
        "rho00_fft": np.fft.fft(np.fft.ifftshift(rho00)),
        "rho11_fft": np.fft.fft(np.fft.ifftshift(rho11)),
        "rho01_fft": np.fft.fft(np.fft.ifftshift(rho01)),
        "rho10_fft": np.fft.fft(np.fft.ifftshift(rho10)),
    }

def G_ffts(phases, Nk):
    L = len(phases["kin"])
    M = len(phases["int"])
    g_ffts_M4a1 = np.zeros((L, M, Nk), dtype=np.complex128)
    g_ffts_M4a2 = np.copy(g_ffts_M4a1)
    g_ffts_M4b1 = np.copy(g_ffts_M4a1)
    g_ffts_M4b2 = np.copy(g_ffts_M4a1)

    for l in range(L):
        for m in range(M):
            g = np.conj(phases["kin"][l]) * phases["int"][m]
            g_ffts_M4a1[l,m,:] = np.fft.fft(np.fft.ifftshift(g))
            
            g = np.conj(phases["int"][m]) * phases["kin"][l]
            g_ffts_M4b1[l,m,:] = np.fft.fft(np.fft.ifftshift(g))

            g = phases["int"][m]
            g_ffts_M4a2[l,m,:] = np.fft.fft(np.fft.ifftshift(g))

            g = np.conj(phases["int"][m])
            g_ffts_M4b2[l,m,:] = np.fft.fft(np.fft.ifftshift(g))
    return g_ffts_M4a1, g_ffts_M4a2, g_ffts_M4b1, g_ffts_M4b2

def compute_all_mf_matrices(K, rho, geom, phases, g_ffts):
    Nk = len(K)

    M3  = np.zeros((2,2,Nk), dtype=np.complex128)
    M6  = np.zeros((2,2,Nk), dtype=np.complex128)
    M4a = np.zeros((2,2,Nk), dtype=np.complex128)
    M4b = np.zeros((2,2,Nk), dtype=np.complex128)

    # --- prepare densities ---
    rho00 = rho[0,0,:]
    rho11 = rho[1,1,:]
    rho01 = rho[0,1,:]
    rho10 = rho[1,0,:]

    n = [rho00.sum(), rho11.sum()]

    rho_fft = {
        (0,0): np.fft.fft(np.fft.ifftshift(rho00)),
        (1,1): np.fft.fft(np.fft.ifftshift(rho11)),
        (0,1): np.fft.fft(np.fft.ifftshift(rho01)),
        (1,0): np.fft.fft(np.fft.ifftshift(rho10)),
    }

    g_ffts_M4a1, g_ffts_M4a2, g_ffts_M4b1, g_ffts_M4b2 = g_ffts

    # --- loop over geometry ---
    for l, (x, orb1, orb2, t) in enumerate(geom["kinetic"]):
        orb1, orb2 = int(orb1), int(orb2)
        phase_k = phases["kin"][l]
        fk = t * phase_k / Nk

        for m, (x_, orb1_, orb2_, V_) in enumerate(geom["interaction"]):
            orb1_, orb2_ = int(orb1_), int(orb2_)
            if orb2 == orb2_:
                lega = geom["pos"][orb2] - geom["pos"][orb1_] - x_

                # ---------- M3 ----------
                M3[orb1,orb2] += -1j * t * V_ * lega * phase_k * n[orb1_] / Nk

                # ---------- M6 ----------
                suma = np.sum(rho[orb1,orb2,:] * phase_k)
                M6[orb1_,orb1_] += -1j * t * V_ * lega * suma / Nk

                # ---------- M4a ----------
                #g = -1j * V_ * lega * np.conj(phase_k) * phases["int"][m]
                g_fft = -1j * V_ * lega * g_ffts_M4a1[l,m,:] #np.fft.fft(np.fft.ifftshift(g))
                gh = np.fft.fftshift(
                        np.fft.ifft(g_fft * rho_fft[(orb1,orb1_)]))
                M4a[orb1_,orb2] += fk * gh

                # ---------- M4b ----------
                h = fk * rho[orb1_,orb2,:]
                h_fft = np.fft.fft(np.fft.ifftshift(h))
                g_fft = -1j * V_ * lega * g_ffts_M4b1[l,m,:]
                #g_fft = np.fft.fft(np.fft.ifftshift(
                #            -1j * V_ * lega * np.conj(phases["int"][m]) * phases["kin"][l]
                #        ))
                gh = np.fft.fftshift(np.fft.ifft(g_fft * h_fft))
                M4b[orb1,orb1_] += gh

            if orb1 == orb2_:
                lega = geom["pos"][orb1] - geom["pos"][orb1_] - x_
                
                # ---------- M3 ----------
                M3[orb1,orb2] += +1j * t * V_ * lega * phase_k * n[orb1_] / Nk

                # ---------- M6 ----------
                suma = np.sum(rho[orb1,orb2,:] * phase_k)
                M6[orb1_,orb1_] += +1j * t * V_ * lega * suma / Nk

                # ---------- M4a ----------
                #g = +1j * V_ * lega * phases["int"][m]
                g_fft = +1j * V_ * lega * g_ffts_M4a2[l,m,:] #np.fft.fft(np.fft.ifftshift(g))
                gh = np.fft.fftshift(
                        np.fft.ifft(g_fft * rho_fft[(orb1,orb1_)]))
                M4a[orb1_,orb2] += fk * gh

                # ---------- M4b ----------
                #g = +1j * V_ * lega * np.conj(phases["int"][m])
                g_fft = +1j * V_ * lega * g_ffts_M4b2[l,m,:]#np.fft.fft(np.fft.ifftshift(g))
                h = fk * rho[orb1_,orb2,:]
                h_fft = np.fft.fft(np.fft.ifftshift(h))
                gh = np.fft.fftshift(
                            np.fft.ifft(g_fft * h_fft)
                )
                M4b[orb1,orb1_] += gh

    return M3 + M6 - M4a - M4b

@njit
def spektralna_k(omega, mu, energije_k, Gamma):
    N_orb = len(energije_k)
    A = np.zeros(N_orb)
    for orb in range(N_orb):
        A[orb] = -1/np.pi * Gamma / ( (omega - (energije_k[orb] - mu))**2 + Gamma**2 )
    return A

@njit
def Spektralka(omegas, mu, energije, Gamma):
    Nk = energije.shape[1]
    Nomega = len(omegas)
    A = np.zeros((Nomega, 2, Nk))
    for i in prange(Nomega):
        for m in range(Nk):
            A_k = spektralna_k(omegas[i], mu, energije[:,m], Gamma)
            A[i,:,m] = A_k
    return A

@njit(parallel=True, cache=True)
def phi_Kubo(K, mat1, mat2, spektralka, omegas):
    Nk = len(K)
    phi = np.zeros(len(omegas), dtype=np.complex128)
    A = spektralka

    for m in [0, Nk//2]:
        for a in range(2):
            for b in range(2):
                phi += (mat1[a,b,m] * A[:,b,m] * mat2[b,a,m] * A[:,a,m]).real
    for m in prange(1,Nk//2):
        for a in range(2):
            for b in range(2):
                phi += (mat1[a,b,m] * A[:,b,m] * mat2[b,a,m] * A[:,a,m]).real
    return phi / Nk

@njit(parallel=True, cache=True)
def phi_Kubo_diagonal(K, mat1, mat2, spektralka, omegas):
    Nk = len(K)
    phi = np.zeros(len(omegas), dtype=np.complex128)
    A = spektralka

    for m in [0, Nk//2]:
        for a in range(2):
            phi += (mat1[a,a,m] * A[:,a,m] * mat2[a,a,m] * A[:,a,m]).real
    for m in prange(1,Nk//2):
        for a in range(2):
            phi += (mat1[a,a,m] * A[:,a,m] * mat2[a,a,m] * A[:,a,m]).real
    return phi / Nk

''' df/domega, f je Fermi-Diracova porazdelitvena funkcija '''
@njit(cache=True)
def fd_1(omega, T): return -1/(4*T)/np.cosh(omega/(2*T))**2

def fd(omegas, T): return 1 / (np.exp(omegas/T) + 1)

def delta_approximation(x, width, shape='Gaussian'):
    if shape == 'Gaussian':
        return 1/(2*np.pi*width**2)**0.5 * np.exp(-x**2/(2*width**2))
    elif shape == 'Lorentzian':
        return 1/np.pi * width/(x**2 + width**2)

def phi_Boltzmann(K, rho, phys_parameters, energije, mu, omegas, faktor=0.2, shape='Gaussian'):
    Nk = len(K)
    phi = np.zeros(len(omegas))
    _, _, _, t12, _, _, _, _, _ = phys_parameters
    if t12 == 0:
        vel = v_analytic(K, rho, phys_parameters)
    else:
        vel1 = np.diff(energije[0]) / (K[1] - K[0])
        vel2 = np.diff(energije[1]) / (K[1] - K[0])
        vel = np.zeros((2, Nk))
        vel[0] = np.hstack([[0], vel1])
        vel[1] = np.hstack([[0], vel2])

    v_max = np.max(np.abs(vel))
    sigma = np.sqrt(v_max * (omegas[1] - omegas[0]) * (K[1] - K[0])) * faktor

    for i, omega in enumerate(omegas):
        for j in range(0,Nk//2+1):
            if j in [0,Nk//2]: multiply = 1
            else: multiply = 2
            for alpha in [0,1]:
                phi[i] += multiply * delta_approximation(omega - energije[alpha,j] + mu, sigma, shape) * vel[alpha,j]**2
    return phi / Nk

def Kn_boltz(K, energije, mu, T):
    K0, K1 = 0, 0
    Nk = len(K)
    vel1 = np.diff(energije[0]) / (K[1] - K[0])
    vel2 = np.diff(energije[1]) / (K[1] - K[0])

    for j in [0,Nk//2]:
        K0 += -fd_1(energije[0,j] - mu, T) * vel1[j]**2 - fd_1(energije[1,j] - mu, T) * vel2[j]**2
        K1 += -fd_1(energije[0,j] - mu, T) * vel1[j]**2 * (energije[0,j] - mu) - fd_1(energije[1,j] - mu, T) * vel2[j]**2 * (energije[1,j] - mu)

    for j in range(1,Nk//2):
        K0 += 2 * (-fd_1(energije[0,j] - mu, T) * vel1[j]**2 - fd_1(energije[1,j] - mu, T) * vel2[j]**2)
        K1 += 2 * (-fd_1(energije[0,j] - mu, T) * vel1[j]**2 * (energije[0,j] - mu) - fd_1(energije[1,j] - mu, T) * vel2[j]**2 * (energije[1,j] - mu))

    return K0, K1


sigmas = np.zeros((4, 2, 2), dtype=np.complex128)
sigmas[0] = np.eye(2)
sigmas[1] = np.array([[0,1],[1,0]])
sigmas[2] = np.array([[0,-1j], [1j,0]])
sigmas[3] = np.diag([1,-1])

def rho_operators(K, phys_parameters, include_hartree):

    b, _, _, _, _, _, Vb, Vc, _ = phys_parameters

    if include_hartree == True:
        thetas = np.hstack([[- Vb/2, Vb/2, Vb/2, Vb/2],
                            [- Vc/2, Vc/2, Vc/2, Vc/2]])
        if Vc == 0:
            thetas = thetas[:4]
        nus = [0, 1, 2, 3]
    else:
        thetas = np.array([Vb/2, Vb/2, Vc/2, Vc/2])
        if Vc == 0:
            thetas = thetas[:2]
        nus = [1, 2]

    if Vc == 0:
        deltas = [0]
    else: deltas = [0, 1]

    rhos = np.zeros((len(thetas), 2, 2, len(K)), dtype=np.complex128)
    for i, delta in enumerate(deltas):
        for j, nu in enumerate(nus):
            ind = len(deltas) * i + j
            U_kdelta = np.zeros((2, 2, len(K)), dtype=np.complex128)
            U_kdelta[0,0] = np.exp(-1j*K*delta/2)
            U_kdelta[1,1] = np.exp(1j*K*delta/2)
            Rho = np.einsum('ijx, jl, klx -> ikx', U_kdelta, sigmas[nu], U_kdelta.conj())
            rhos[ind] = Rho
    return rhos, thetas

@njit(parallel=True, cache=True)
def chi_Kubo_omegas(K, mat1, mat2, omegas, energije, fs, Gamma):
    Nk = len(K)
    Nw = len(omegas)
    phi = np.zeros(Nw, dtype=np.complex128)

    for iw in prange(Nw):
        w = omegas[iw]
        suma_w = 0.0 + 0.0j
        for m in prange(Nk):
            for a in range(2):
                for b in range(2):
                    if a != b:
                        suma_w += mat1[a,b,m] * mat2[b,a,m] * (fs[a,a,m] - fs[b,b,m]) / (w + energije[a,m] - energije[b,m] + 1j*2*Gamma)

        phi[iw] = suma_w / Nk
    return phi

@njit(parallel=True, cache=True)
def chi_Kubo_omegas_imag(K, mat1, mat2, omegas, energije, fs, Gamma):
    Nk = len(K)
    Nw = len(omegas)
    phi = np.zeros(Nw, dtype=np.complex128)

    for iw in prange(Nw):
        w = omegas[iw]
        suma_w = 0.0 + 0.0j
        for m in prange(Nk):
            for a in range(2):
                for b in range(2):
                    suma_w += mat1[a,b,m] * mat2[b,a,m]  * (fs[a,a,m] - fs[b,b,m]) * 2*Gamma / ((w + energije[a,m] - energije[b,m])**2 + (2*Gamma)**2)
        phi[iw] = suma_w / Nk
    return phi

def operator_tilde(op_bare, vecs):
    op_tilde = np.empty_like(op_bare, dtype=np.complex128)
    if len(op_tilde.shape) > 3:
        # this means there are multiple operators
        n_ops = op_tilde.shape[0]
        for n in range(n_ops):
            op_tilde[n] = np.einsum('jix, jlx, lmx -> imx', vecs.conj(), op_bare[n], vecs)
    else:
        op_tilde = np.einsum('jix, jlx, lmx -> imx', vecs.conj(), op_bare, vecs)
    return op_tilde

def chi_rho_rho(K, rhos, thetas, vecs, energije, fs, omegas, Gamma):
    thetas = np.diag(thetas)
    rhos_tilde = operator_tilde(rhos, vecs)

    Nop = rhos_tilde.shape[0]
    Nw = len(omegas)

    chi_rho_rho = np.zeros((Nop, Nop, Nw), dtype=np.complex128)
    for i in range(Nop):
        for j in range(Nop):
            chi_rho_rho[i,j] = chi_Kubo_omegas(K, rhos_tilde[i], rhos_tilde[j], omegas, energije, fs, Gamma)

    chi_rho_rho_renormalized = np.empty_like(chi_rho_rho)
    I = np.eye(Nop, dtype=np.complex128)

    for iw in range(Nw):
        inv = LA.inv(I - chi_rho_rho[:,:,iw] @ thetas)
        chi_rho_rho_renormalized[:,:,iw] = inv @ chi_rho_rho[:,:,iw]

    return chi_rho_rho, chi_rho_rho_renormalized

def chi_j_j(K, tok, rhos, thetas, vecs, energije, fs, omegas, Gamma):
    tok_tilde = operator_tilde(tok, vecs)
    thetas = np.diag(thetas)
    rhos_tilde = operator_tilde(rhos, vecs)

    Nop = rhos_tilde.shape[0]
    Nw = len(omegas)

    ''' chi_j_j is bare current-current susceptibility (bubble) '''
    chi_j_j = chi_Kubo_omegas(K, tok_tilde, tok_tilde, omegas, energije, fs, Gamma)

    ''' below we construct dchi_j_j, which is correction of the bubble,
    for this we need rho-current and rho-rho susceptibilities in terms of bubbles '''
    chi_rho_rho = np.zeros((Nop, Nop, Nw), dtype=np.complex128)
    chi_rho_j = np.zeros((Nop, Nw), dtype=np.complex128)
    for i in range(Nop):
        chi_rho_j[i] = chi_Kubo_omegas(K, rhos_tilde[i], tok_tilde, omegas, energije, fs, Gamma)
        for j in range(Nop):
            chi_rho_rho[i,j] = chi_Kubo_omegas(K, rhos_tilde[i], rhos_tilde[j], omegas, energije, fs, Gamma)
    chi_j_rho = chi_rho_j[np.newaxis,:,:]

    inverz = np.zeros((Nop, Nop, Nw), dtype=np.complex128)
    I = np.eye(Nop, dtype=np.complex128)
    for i in range(Nw):
        inverz[:,:,i] = LA.inv(I - chi_rho_rho[:,:,i] @ thetas)

    dchi_j_j = np.einsum('ijo, jl, lmo, mo -> io', chi_j_rho, thetas, inverz, chi_rho_j)[0]

    return chi_j_j, dchi_j_j, chi_rho_rho, chi_rho_j

def A_pulz(t, A0, t0, sigma, Omega):
    return A0 * np.cos(Omega * t) * np.exp(-(t-t0)**2/(2*sigma**2))

@njit(parallel=True)
def evolve_rho_kernel(Hk, rho, dt):
    Nk = rho.shape[2]

    rho_next = np.empty_like(rho)

    for j in prange(Nk):
        hk = Hk[:,:,j]

        # Decompose H = ε I + d·σ, ε I do not actually need
        dx  = 0.5 * (hk[0,1] + hk[1,0]).real
        dy  = -0.5 * (hk[0,1] - hk[1,0]).imag
        dz  = 0.5 * (hk[0,0] - hk[1,1]).real

        norm_d = np.sqrt(dx*dx + dy*dy + dz*dz)

        c = np.cos(norm_d * dt)
        s = np.sin(norm_d * dt) / norm_d

        # Build U explicitly, using: U = cos(|d_k|dt) - isin(|d_k|dt) d_k*sigma / |d_k|
        U00 = c - 1j * s * dz
        U11 = c + 1j * s * dz
        U01 = -1j * s * (dx - 1j*dy)
        U10 = -1j * s * (dx + 1j*dy)

        # Apply U rho
        r00 = rho[0,0,j]
        r01 = rho[0,1,j]
        r10 = rho[1,0,j]
        r11 = rho[1,1,j]

        a00 = U00*r00 + U01*r10
        a01 = U00*r01 + U01*r11
        a10 = U10*r00 + U11*r10
        a11 = U10*r01 + U11*r11

        # Apply U† to U rho --> this is rho_next
        rho_next[0,0,j] = a00*U00.conjugate() + a01*U01.conjugate()
        rho_next[0,1,j] = a00*U10.conjugate() + a01*U11.conjugate()
        rho_next[1,0,j] = a10*U00.conjugate() + a11*U01.conjugate()
        rho_next[1,1,j] = a10*U10.conjugate() + a11*U11.conjugate()

    return rho_next

@njit(parallel=True)
def measure(Nk, Nop, measure_operators, rho_k):
    measurements_k = np.zeros((Nop, Nk), dtype=np.complex128)
    measurements_k = np.zeros((Nop, Nk), dtype=np.complex128)
    for j in prange(Nk):
        for n in range(Nop):
            measurements_k[n, j] = (
                rho_k[0,0,j]*measure_operators[n,0,0,j] +
                rho_k[0,1,j]*measure_operators[n,1,0,j] +
                rho_k[1,0,j]*measure_operators[n,0,1,j] +
                rho_k[1,1,j]*measure_operators[n,1,1,j]
            )
    measurements = measurements_k.sum(axis=1)
    return measurements

def simulate_pulz(K, hk0, rho, phys_parameters, include_hartree, perturbation_operator, measure_provider,
                  A0, t0, sigma, Omega, dt, t_max,
                  do_freeze, Ncorr, tol, geom, phases, g_ffts):
    N_points = int(t_max/dt)
    Nk = len(K)

    if callable(measure_provider):
        dynamic_measure = True
    else:
        dynamic_measure = False
        measure_operators_fixed = measure_provider
        if measure_operators_fixed.ndim == 3:
            measure_operators_fixed = measure_operators_fixed[np.newaxis, ...]

    if dynamic_measure or measure_operators_fixed.ndim == 3:
        Nop = 1
    else:
        Nop = measure_operators_fixed.shape[0]

    if dynamic_measure:
        measure_operators = measure_provider(K, rho, geom, phases, g_ffts)[np.newaxis, ...]
    else:
        measure_operators = measure_operators_fixed
        
    rho0 = np.copy(rho)
    H0 = h_k(K, hk0, rho0, phys_parameters, 0., include_hartree)

    rho_expvals = np.zeros((N_points, Nop), dtype=np.complex128)
    print(measure_operators.shape)
    ts = dt * np.arange(N_points)

    for i in range(N_points):
        if i % 50 == 0:
            print(i/N_points)
        # ---------- predictor ----------
        rho_guess = rho
        A_t = A_pulz(i * dt, A0, t0, sigma, Omega)
        A_half = A_pulz(i * dt + dt/2, A0, t0, sigma, Omega)

        for _ in range(Ncorr):

            if do_freeze:
                H_k  = H0 - A_t * perturbation_operator
                H_k1 = H0 - A_t * perturbation_operator
            else:
                H_k  = h_k(K, hk0, rho,       phys_parameters, 0., include_hartree) - A_t * perturbation_operator
                H_k1 = h_k(K, hk0, rho_guess, phys_parameters, 0., include_hartree) - A_t * perturbation_operator

            H_k12 = 0.5 * (H_k + H_k1) - A_half * perturbation_operator

            rho_new = evolve_rho_kernel(H_k12, rho, dt)

            # ---------- convergence check ----------
            err = np.max(np.abs(rho_new - rho_guess))
            rho_guess = rho_new

            if err < tol:
                break
        
        # accept converged solution
        rho = rho_guess
        if dynamic_measure:
            measure_operators = measure_provider(K, rho, geom, phases, g_ffts)[np.newaxis, ...]
        else:
            measure_operators = measure_operators_fixed

        measurement_t = measure(Nk, Nop, measure_operators, rho_guess) 
        rho_expvals[i] = measurement_t

    return ts, rho_expvals

def susceptibility(time, signal, probe, Gamma, omega_cut, Nk):
    dt = time[1] - time[0]
    window = np.exp(-2 * Gamma * time)
    
    signal_omega = np.fft.fft((signal - signal[0]) * window * dt) / Nk
    probe_omega = np.fft.fft(probe * window * dt)

    omega = 2*np.pi*np.fft.fftfreq(len(time), d=dt)

    pos = (omega > 0) * (omega < omega_cut)
    omega = omega[pos]
    signal_omega = signal_omega[pos]
    probe_omega = probe_omega[pos]

    return omega, signal_omega, probe_omega

def optical_conductivity(time, signal, probe, Gamma, omega_cut, Nk):
    omega, signal_omega, probe_omega = susceptibility(time, signal, probe, Gamma, omega_cut, Nk)
    sigma_omega = signal_omega / (-1j * omega * probe_omega)

    return omega, sigma_omega.real

def sum_rule_rhs(tok, vecs, fs, energije):
    Nk = tok.shape[-1]
    tok_tilde = operator_tilde(tok, vecs)

    suma = 0.
    for n in range(Nk):
        for a in range(2):
            for b in range(2):
                if a != b:
                    suma += -np.abs(tok_tilde[a,b,n])**2 * (fs[b,b,n] - fs[a,a,n]) / (energije[b,n] - energije[a,n])
    suma = np.pi / 2 * suma / Nk

    return suma

def integral_sigma_omega(sigma, omega):
    return np.trapz(sigma.real, omega)
