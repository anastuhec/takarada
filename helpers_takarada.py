import numpy as np
import scipy.linalg as LA
from numba import njit, prange
from scipy.optimize import brentq

# ====== self-consistency calculation ======

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

def f_newmu(mu, hk0, rho, K, T, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=0.5):
    _, _, _, _, _, n = Rho_next(hk0, rho, K, T, mu, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=mix)
    return n - 1

def NewMu2(mu1, mu2, hk0, rho, K, T, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=0.5, xtol=1e-10, rtol=1e-10, maxiterbrentq=100):
    return brentq(f_newmu, mu1, mu2, args=(hk0, rho, K, T, phys_parameters, eps0, epsilon_threshold, N_epsilon, maxiter, include_hartree, mix), 
                   xtol=xtol, rtol=rtol, maxiter=maxiterbrentq, full_output=False)

''' analytic result for energy bands IF t12 = 0 and delta = 0 and epsilon_ = epsilon '''
def E_analytic(K, rho, phys_parameters):
    _, t, t_, _, epsilon, _, Vb, Vc, _ = phys_parameters
    Nk = len(K)
    delta_k = -Vb/Nk*np.sum(rho[1,0,:]) - Vc/Nk*np.sum(rho[1,0,:]*np.exp(-1j*K)) * np.exp(-1j*K)
    E_minus = (t-t_)*np.cos(K) - np.sqrt( ((t+t_)*np.cos(K) - epsilon)**2 + np.abs(delta_k)**2 )
    E_plus = (t-t_)*np.cos(K) + np.sqrt( ((t+t_)*np.cos(K) - epsilon)**2 + np.abs(delta_k)**2 )
    return np.vstack([E_minus, E_plus])



# ====== transport ======

''' operator in band basis obtained from operator in orbital basis '''
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

''' analytic result for group velocities IF t12 = 0 and delta = 0 and epsilon_ = epsilon '''
def v_analytic(K, rho, phys_parameters):
    _, t, t_, _, epsilon, _, Vb, Vc, _ = phys_parameters

    delta_b, delta_c = Delta(K, rho, Vb, Vc)
    delta_k = delta_b + delta_c*np.exp(-1j*K)

    vel_minus = -(t-t_)*np.sin(K) - (-((t+t_)*np.cos(K) - epsilon)*(t+t_)*np.sin(K) + (1j*np.exp(1j*K) * delta_b * delta_c.conj() ).real ) / np.sqrt( ((t+t_)*np.cos(K) - epsilon)**2 + np.abs(delta_k)**2)
    vel_plus = -(t-t_)*np.sin(K) + (-((t+t_)*np.cos(K) - epsilon)*(t+t_)*np.sin(K) + (1j*np.exp(1j*K) * delta_b * delta_c.conj() ).real ) / np.sqrt( ((t+t_)*np.cos(K) - epsilon)**2 + np.abs(delta_k)**2)
    return np.vstack([vel_minus, vel_plus])

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

    return M3, M6, -M4a, -M4b

def compute_mf_matrices(K, rho, geom, phases, g_ffts):
    M3, M6, M4a, M4b = compute_all_mf_matrices(K, rho, geom, phases, g_ffts)
    return M3 + M6 + M4a + M4b

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
                phi += 2 * (mat1[a,b,m] * A[:,b,m] * mat2[b,a,m] * A[:,a,m]).real
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
            phi += 2 * (mat1[a,a,m] * A[:,a,m] * mat2[a,a,m] * A[:,a,m]).real
    return phi / Nk

''' df/domega, f is Fermi-Dirac distribution function '''
@njit(cache=True)
def fd_1(omega, T): return -1/(4*T)/np.cosh(omega/(2*T))**2

def delta_approximation(x, width, shape='Gaussian'):
    if shape == 'Gaussian':
        return 1/(2*np.pi*width**2)**0.5 * np.exp(-x**2/(2*width**2))
    elif shape == 'Lorentzian':
        return 1/np.pi * width/(x**2 + width**2)

def phi_Boltzmann(K, rho, phys_parameters, energije, mu, omegas, faktor=0.2, shape='Gaussian'):
    Nk = len(K)
    phi = np.zeros(len(omegas))
    _, _, _, t12, epsilon, epsilon_, _, _, delta = phys_parameters
    if epsilon == epsilon_ and t12 == 0 and delta == 0:
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
    return K0 / Nk, K1 / Nk

sigmas = np.zeros((4, 2, 2), dtype=np.complex128)
sigmas[0] = np.eye(2)
sigmas[1] = np.array([[0,1],[1,0]])
sigmas[2] = np.array([[0,-1j], [1j,0]])
sigmas[3] = np.diag([1,-1])

def rho_operators(K, phys_parameters, include_hartree):

    _, _, _, _, _, _, Vb, Vc, _ = phys_parameters

    if include_hartree == True:
        thetas = np.array([Vb/2, -Vb/2, -Vb/2, -Vb/2,
                            Vc/2, -Vc/2, -Vc/2, -Vc/2])
        if Vc == 0:
            thetas = thetas[:4]
        nus = [0, 1, 2, 3]
    else:
        thetas = np.array([-Vb/2, -Vb/2, -Vc/2, -Vc/2])
        if Vc == 0:
            thetas = thetas[:2]
        nus = [1, 2]

    if Vc == 0:
        deltas = [0]
    else: deltas = [0, 1]

    rhos = np.zeros((len(thetas), 2, 2, len(K)), dtype=np.complex128)
    for i, delta in enumerate(deltas):
        for j, nu in enumerate(nus):
            ind = len(nus) * i + j
            U_kdelta = np.zeros((2, 2, len(K)), dtype=np.complex128)
            U_kdelta[0,0] = np.exp(-1j*K*delta/2)
            U_kdelta[1,1] = np.exp(1j*K*delta/2)
            Rho = np.einsum('ijx, jl, klx -> ikx', U_kdelta, sigmas[nu], U_kdelta.conj())
            rhos[ind] = Rho
    return rhos, thetas

''' Fermi-Dirac function '''
@njit
def fd(eps, mu, T):
    return 1.0 / (np.exp((eps - mu) / T) + 1.0)

''' a bubble (just propagators, no operator vertices attached) needed for susceptibilities
this form does not perform well numerically in the omega --> 0 limit '''
@njit(cache=True)
def Pi_bubble0(omega, E_mk, E_nk, Gamma, mu, T):
    return - (fd(E_mk, mu, T) - fd(E_nk, mu, T)) / (omega + E_mk - E_nk + 1j*2*Gamma)


''' retarded Green's function '''
@njit
def G_R(epsilon, E, Gamma):
    return 1. /(epsilon - E + 1j*Gamma)

''' advanced Green's function '''
@njit
def G_A(epsilon, E, Gamma):
    return 1. /(epsilon - E - 1j*Gamma)

''' (single-particle) spectral function '''
@njit
def A_k(epsilon, E, Gamma):
    return 1/np.pi * Gamma  / ( (epsilon - E)**2 + Gamma**2 )

''' susceptibility between U and V, using bubble  Pi '''
@njit(cache=True)
def Pi_bubble(omega, E_mk, E_nk, Gamma, mu, T, width, deps):
    eps_min = min(E_mk, E_nk - omega, mu) - width
    eps_max = max(E_mk, E_nk - omega, mu) + width
    Npts = int((eps_max - eps_min) / deps)
    chi_mn = 0.0 + 1j*0.0
    chi_nm = 0.0 + 1j*0.0
    for i in range(Npts):
        eps = eps_min + i * deps
        
        # first term
        f = fd(eps, mu, T)
        # first term, mn
        chi_mn += A_k(eps, E_mk, Gamma) * G_R(eps + omega, E_nk, Gamma) * f
        # first term, nm
        chi_nm += A_k(eps, E_nk, Gamma) * G_R(eps + omega, E_mk, Gamma) * f

        # second term
        fw = fd(eps + omega, mu, T)
        # second term, mn
        chi_mn += A_k(eps + omega, E_nk, Gamma) * G_A(eps, E_mk, Gamma) * fw
        # second term, nm
        chi_nm += A_k(eps + omega, E_mk, Gamma) * G_A(eps, E_nk, Gamma) * fw

    chi_mn = chi_mn * deps
    chi_nm = chi_nm * deps
    return - chi_mn, - chi_nm

@njit
def Pi_bubble_tilde(omega, E_mk, E_nk, Gamma, mu_, invt, L, nodes, weights):
    w = omega / Gamma
    e_mk = E_mk / Gamma
    e_nk = E_nk / Gamma

    e_min = min(e_mk, e_nk - w, mu_) - L
    e_max = max(e_mk, e_nk - w, mu_) + L

    mid = 0.5*(e_max + e_min)
    half = 0.5*(e_max - e_min)
    
    res_mn = 0.0 + 0.0*1j
    res_nm = 0.0 + 0.0*1j

    res_w_mn = 0.0 + 0.0*1j
    res_w_nm = 0.0 + 0.0*1j

    invpi = 1. / np.pi

    n_nodes = len(nodes)
    for i in range(n_nodes):
        e = mid + half * nodes[i]
        ew = e + w
        dm = e - e_mk
        dn = e - e_nk
        dmw = dm + w
        dnw = dn + w
        pref = e - mu_ + 0.5*w

        weight = weights[i]

        f = 1. / (np.exp((e - mu_) * invt) + 1.)
        fw = 1. / (np.exp((ew - mu_) * invt) + 1.)

        A_mk = invpi / (dm*dm + 1.)
        A_nk = invpi / (dn*dn + 1.)
        A_mkw = invpi / (dmw*dmw + 1.)
        A_nkw = invpi / (dnw*dnw + 1.)

        Grnw = 1. / (dnw + 1j)
        Grmw = 1. / (dmw + 1j)

        Gam = 1. / (dm - 1j)
        Gan = 1. / (dn - 1j)

        inte_mn_e = A_mk * Grnw * f + A_nkw * Gam * fw
        inte_nm_e = A_nk * Grmw * f + A_mkw * Gan * fw
    
    
        res_mn += - weight * inte_mn_e
        res_nm += - weight * inte_nm_e

        res_w_mn += - pref * weight * inte_mn_e
        res_w_nm += - pref * weight * inte_nm_e

    return half * res_mn / Gamma, half * res_nm / Gamma, half * res_w_mn, half * res_w_nm

@njit(parallel=True, cache=True)
def Pi_bubble_tilde_w(Nk, omega, energije, Gamma, mu_, invt, L, nodes, weights):
    pi = np.zeros((2,2,2,Nk), dtype=np.complex128)
    pi_eps = np.zeros_like(pi)

    for j in prange(Nk):
        for m in range(2):
            for n in range(m,2):
                pi_mnk, pi_nmk, pie_mnk, pie_nmk = Pi_bubble_tilde(omega, energije[m,j], energije[n,j], Gamma, mu_, invt, L, nodes, weights,)

                pi[m,n,0,j] = pi_mnk
                pi[m,n,1,j] = pi_nmk
                pi_eps[m,n,0,j] = pie_mnk
                pi_eps[m,n,1,j] = pie_nmk

    return pi, pi_eps

@njit(parallel=True, cache=True)
def chi_UV(Nk, U, V, pi_w, pi_eps_w, eps=False):
    chi = np.complex64(0.0)

    for j in prange(Nk):
        for m in range(2):
            for n in range(m,2):
                
                U_mn = U[m,n,j]
                U_nm = U[n,m,j]
                V_mn = V[m,n,j]
                V_nm = V[n,m,j]

                if eps:
                    pi_mn = pi_eps_w[m,n,0,j]
                    pi_nm = pi_eps_w[m,n,1,j]
                else:
                    pi_mn = pi_w[m,n,0,j]
                    pi_nm = pi_w[m,n,1,j]
                
                if m == n:
                    chi += 0.5 * (U_mn * V_nm * pi_mn + U_nm * V_mn * pi_nm)
                else:
                    chi += U_mn * V_nm * pi_mn + U_nm * V_mn * pi_nm
    return chi / Nk


''' current-current, current-density, density-density corrections using a more complicated bubble (Pi_bubble, chi_UV),
including their corrections '''
def chi_jrho2(K, tok_tilde, rhos_tilde, mat_tilde, thetas, energije, mu, T, omega, Gamma, L, nodes, weights):
    thetas = np.diag(thetas)
    Nop = rhos_tilde.shape[0]
    Nk = len(K)
    mu_ = mu / Gamma
    invt = Gamma / T

    pi_w, pi_eps_w = Pi_bubble_tilde_w(Nk, omega, energije, Gamma, mu_, invt, L, nodes, weights)

    ''' chi_j_j is bare current-current susceptibility (bubble) '''
    chi_j_j = chi_UV(Nk, tok_tilde, tok_tilde, pi_w, pi_eps_w, )
    chi_jE_j = chi_UV(Nk, tok_tilde, tok_tilde, pi_w, pi_eps_w, eps=True)
    chi_jE2_j = chi_UV(Nk, mat_tilde, tok_tilde, pi_w, pi_eps_w)
    ''' below we construct dchi_j_j, which is correction of the bubble,
    for this we need rho-current and rho-rho susceptibilities in terms of bubbles '''
    chi_rho_rho = np.zeros((Nop, Nop), dtype=np.complex128)
    chi_rho_j = np.zeros(Nop, dtype=np.complex128)
    chi_j_rho = np.zeros(Nop, dtype=np.complex128)
    chi_jE_rho = np.zeros(Nop, dtype=np.complex128)
    chi_jE2_rho = np.zeros(Nop, dtype=np.complex128)
    for i in range(Nop):
        chi_rho_j[i] = chi_UV(Nk, rhos_tilde[i], tok_tilde, pi_w, pi_eps_w)
        chi_j_rho[i] = chi_UV(Nk, tok_tilde, rhos_tilde[i], pi_w, pi_eps_w)
        chi_jE_rho[i] = chi_UV(Nk, tok_tilde, rhos_tilde[i], pi_w, pi_eps_w, eps=True)
        chi_jE2_rho[i] = chi_UV(Nk, mat_tilde, rhos_tilde[i], pi_w, pi_eps_w)
        for j in range(Nop):
            chi_rho_rho[i,j] = chi_UV(Nk, rhos_tilde[i], rhos_tilde[j], pi_w, pi_eps_w)

    I = np.eye(Nop, dtype=np.complex128)
    chi_rho_rho_renorm = np.empty_like(chi_rho_rho, dtype=np.complex64)
    inv = LA.inv(I - chi_rho_rho @ np.diag(thetas))
    chi_rho_rho_renorm = inv @ chi_rho_rho

    dchi_j_j = chi_j_rho @ thetas @ inv @ chi_rho_j
    dchi_jE_j = chi_jE_rho @ thetas @ inv @ chi_rho_j
    dchi_jE2_j = chi_jE2_rho @ thetas @ inv @ chi_rho_j

    results = {'chi_j_j' : chi_j_j,
               'dchi_j_j' : dchi_j_j, 
               'chi_rho_j' : chi_rho_j,
               'chi_j_rho' : chi_j_rho,
               'chi_jE_j' : chi_jE_j,
               'dchi_jE_j' : dchi_jE_j,
               'chi_jE2_j' : chi_jE2_j,
               'dchi_jE2_j' : dchi_jE2_j
               }

    return results


# ====== response and susceptibility obtained from simulation of a pulse ======

''' Gaussian pulse modulated by a cosine (in practice, however, I choose Omega=0, i.e. the pulse is a Gaussian ''' 
def A_pulz(t, A0, t0, sigma, Omega):
    return A0 * np.cos(Omega * t) * np.exp(-(t-t0)**2/(2*sigma**2))

''' evolving rho(t) into rho(t+dt). for 2-band system, this can be done analytically using decomposition of h_k into a linear combination of Pauli matrices '''
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

@njit
def relax_rho(rho, rho_eq, dt, Gamma):
    decay = np.exp(-Gamma * dt)
    return rho_eq + decay * (rho - rho_eq)

''' expectation value of measure_operators when system is described by density matrix rho'''
@njit(parallel=True)
def measure(Nk, Nop, measure_operators, rho):
    measurements_k = np.zeros((Nop, Nk), dtype=np.complex128)
    measurements_k = np.zeros((Nop, Nk), dtype=np.complex128)
    for j in prange(Nk):
        for n in range(Nop):
            # Tr[rho_k O_k]
            measurements_k[n, j] = (
                rho[0,0,j]*measure_operators[n,0,0,j] +
                rho[0,1,j]*measure_operators[n,1,0,j] +
                rho[1,0,j]*measure_operators[n,0,1,j] +
                rho[1,1,j]*measure_operators[n,1,1,j]
            )
    measurements = measurements_k.sum(axis=1)
    return measurements

''' main function which propagates the system and measures observables (measure_provider)
upon application of a perturbation (generated by perturbation_operator)
* do_freeze=True means that Hartree-Fock is frozen to its equilibrium value, hence we observe no corrections, e.g., in current-current response
* do_freeze=False is the opposite; Hartree-Fock is dynamic, i.e. densities respond to perturbations, and in this response the vertex corrections are captured
'''
def simulate_pulz(K, hk0, rho, phys_parameters, include_hartree,
                  perturbation_operator, measure_provider,
                  A0, t0, sigma, Omega, dt, t_max,
                  do_freeze, Ncorr, tol, geom, phases, g_ffts, Gamma=0.0):

    N_points = int(t_max/dt)
    Nk = len(K)
    rho_eq = np.copy(rho)

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
    ts = dt * np.arange(N_points)

    for i in range(N_points):

        if i % 50 == 0:
            print(i/N_points, flush=True)

        A_t = A_pulz(i * dt, A0, t0, sigma, Omega)
        A_half = A_pulz(i * dt + dt/2, A0, t0, sigma, Omega)

        # Hamiltonian at current density
        if do_freeze:
            H_k0 = H0
        else:
            H_k0 = h_k(K, hk0, rho, phys_parameters, 0., include_hartree)

        # ------------------
        # Predictor
        # ------------------

        if Gamma != 0.0:
            rho_half = relax_rho(rho, rho_eq, dt/2, Gamma)
            rho_pred = evolve_rho_kernel(H_k0 - A_t * perturbation_operator, rho_half, dt)
            rho_pred = relax_rho(rho_pred, rho_eq, dt/2, Gamma)
        else:
            rho_pred = evolve_rho_kernel(H_k0 - A_t * perturbation_operator, rho, dt)

        if do_freeze:
            H_k1 = H0
        else:
            H_k1 = h_k(K, hk0, rho_pred, phys_parameters, 0., include_hartree)

        rho_guess = rho_pred

        # ------------------
        # Corrector iteration
        # ------------------

        for _ in range(Ncorr):

            H_mid = 0.5 * (H_k0 + H_k1) - A_half * perturbation_operator

            if Gamma != 0.0:
                rho_half = relax_rho(rho, rho_eq, dt/2, Gamma)
                rho_new = evolve_rho_kernel(H_mid, rho_half, dt)
                rho_new = relax_rho(rho_new, rho_eq, dt/2, Gamma)
            else:
                rho_new = evolve_rho_kernel(H_mid, rho, dt)

            err = np.max(np.abs(rho_new - rho_guess))
            rho_guess = rho_new

            if err < tol:
                break

            if not do_freeze:
                H_k1 = h_k(K, hk0, rho_guess, phys_parameters, 0., include_hartree)

        rho = rho_guess

        # ------------------
        # Measurements
        # ------------------

        if dynamic_measure:
            measure_operators = measure_provider(K, rho, geom, phases, g_ffts)[np.newaxis, ...]
        else:
            measure_operators = measure_operators_fixed

        measurement_t = measure(Nk, Nop, measure_operators, rho)
        rho_expvals[i] = measurement_t

    return ts, rho_expvals

''' susceptibility obtained from temporal response, using Fourier transform. window exp(-eta*t) is applied '''
def susceptibility(time, signal, probe, eta, omega_cut, Nk):
    dt = time[1] - time[0]
    window = np.exp(- eta * time)
    
    signal_omega = np.fft.fft((signal - signal[0]) * window * dt) / Nk
    probe_omega = np.fft.fft(probe * window * dt)

    omega = 2*np.pi*np.fft.fftfreq(len(time), d=dt)

    pos = (omega > 0) * (omega < omega_cut)
    omega = omega[pos]
    signal_omega = signal_omega[pos]
    probe_omega = probe_omega[pos]

    return omega, signal_omega, probe_omega

''' optical conductivity calculated from susceptibility obtained from temporal response'''
def optical_conductivity(time, signal, probe, eta, omega_cut, Nk):
    omega, signal_omega, probe_omega = susceptibility(time, signal, probe, eta, omega_cut, Nk)
    sigma_omega = signal_omega / (-1j * omega * probe_omega)

    return omega, sigma_omega.real

def sum_rule_rhs(tok_tilde, energije, mu, T):
    Nk = tok_tilde.shape[-1]

    suma = 0.
    for n in range(Nk):
        for a in range(2):
            for b in range(2):
                if a != b:
                    suma += +np.abs(tok_tilde[a,b,n])**2 * Pi_bubble0(0, energije[b,n], energije[a,n], 0., mu, T)
    suma = np.pi / 2 * suma / Nk

    return suma

def integral_sigma_omega(sigma, omega):
    return np.trapezoid(sigma.real, omega)



''' this is just helpers to get local maxima and local minima. I use this to get the envelope of the response '''
def local_minima(arr):
    n = len(arr)
    indices, vals = [], []
    for i in range(n):
        if i > 0 and arr[i] > arr[i - 1]:
            continue
        if i < n - 1 and arr[i] >= arr[i + 1]:
            continue
        indices.append(i)
        vals.append(arr[i])
    return np.array(indices), np.array(vals)

def local_maxima(arr):
    n = len(arr)
    indices, vals = [], []
    for i in range(n):
        if i > 0 and arr[i] <= arr[i - 1]:
            continue
        if i < n - 1 and arr[i] <= arr[i + 1]:
            continue
        indices.append(i)
        vals.append(arr[i])
    return np.array(indices), np.array(vals)
