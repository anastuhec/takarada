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

def h_k(K, rho, phys_parameters, eps0, include_hartree):
    _, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta = phys_parameters
    delta_b, delta_c = Delta(K, rho, Vb, Vc)

    Nk = rho.shape[-1]
    hk = np.zeros((2, 2, Nk), dtype=np.complex128)

    # diagonal hopping and on-site energy
    hk[0,0,:] += 2*t*np.cos(K) - epsilon
    hk[1,1,:] += -2*t_*np.cos(K) + epsilon_

    # off-diagonal hopping
    ad = (t12+delta) + (t12-delta)*np.exp(1j*K)
    hk[0,1,:] += ad
    hk[1,0,:] += ad.conj()

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

def Rho_next(rho, K, T, mu, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=0.5):
    err, N_iters = 1, 0
    while err > epsilon_threshold and N_iters < maxiter:
        if N_iters < N_epsilon: eps = eps0
        else: eps = 0
        rho_new, err = F(h_k(K, rho, phys_parameters, eps, include_hartree), rho, K, T, mu)
        rho = rho_new * mix + rho * (1 - mix)
        N_iters += 1
    rho, _ = F(h_k(K, rho, phys_parameters, 0., include_hartree), rho, K, T, mu)
    energije, vecs, fs = diagonalize(h_k(K, rho, phys_parameters, 0., include_hartree), K, T, mu)
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


def f_newmu(mu, rho, K, T, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, mix=0.5):
    _, _, _, _, _, n = Rho_next(rho, K, T, mu, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, mix=mix)
    return n - 1

def NewMu2(mu1, mu2, rho, K, T, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, mix=0.5, xtol=1e-10, rtol=1e-10, maxiterbrentq=100):
    return brentq(f_newmu, mu1, mu2, args=(rho, K, T, phys_parameters, eps0, epsilon_threshold, N_epsilon, maxiter, mix), 
                   xtol=xtol, rtol=rtol, maxiter=maxiterbrentq, full_output=False)

@njit(cache=True)
def parameters(b, t, t_, t12, epsilon, epsilon_, Vb, Vc, mu, delta=0):
    kinetic = np.array([
        (1, 1, 1, t),
        (-1, 1, 1, t),
        (1, 2, 2, -t_),
        (-1, 2, 2, -t_),
        (0, 1, 2, t12 + delta),
        (-1, 1, 2, t12 - delta),
        (0, 2, 1, t12 + delta),
        (1, 2, 1, t12 - delta),
        (0, 1, 1, epsilon - mu),
        (0, 2, 2, epsilon_ - mu)
    ])

    interaction = np.array([
        (0, 1, 2, Vb / 2),
        (0, 2, 1, Vb / 2),
        (1, 1, 2, Vc / 2),
        (-1, 2, 1, Vc / 2)
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
        j[orb1 - 1, orb2 - 1] += ad
    return j

# this is M^(3)
def mf_matrix1(K, rho, phys_parameters, mu):
    b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta = phys_parameters
    pos, kinetic, interaction = parameters(b, t, t_, t12, epsilon, epsilon_, Vb, Vc, mu, delta)
    Nk = len(K)
    matrix = np.zeros((2, 2, Nk), dtype=np.complex128)

    for alpha in range(1,3):
        for beta in range(1,3):
            for line in kinetic:
                x, orb1, orb2, t = line
                x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
                if orb1 == orb2 and x == 0: pass
                if orb1 == alpha and orb2 == beta:
                    for line_ in interaction:
                        x_, orb1_, orb2_, V_ = line_
                        x_, orb1_, orb2_, V_ = float(x_), int(orb1_), int(orb2_), float(V_)

                        if orb2 == orb2_:
                            suma_n = np.sum(rho[orb1_ - 1, orb1_ - 1, :])
                            lega = pos[orb2 - 1] - pos[orb1_ - 1] - x_
                            matrix[orb1 -1, orb2 - 1] += -1j * t * V_ * lega * np.exp(-1j*K*x) / Nk * suma_n

                        if orb1 == orb2_:
                            suma_n = np.sum(rho[orb1_ - 1, orb1_ - 1, :])
                            lega = pos[orb1 - 1] - pos[orb1_ - 1] - x_
                            matrix[orb1 - 1, orb2 - 1] += 1j * t * V_ * lega * np.exp(-1j*K*x) / Nk * suma_n
    return matrix

# this is M^(6)
def mf_matrix2(K, rho, phys_parameters, mu):
    b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta = phys_parameters
    pos, kinetic, interaction = parameters(b, t, t_, t12, epsilon, epsilon_, Vb, Vc, mu, delta)
    Nk = len(K)
    matrix = np.zeros((2, 2, Nk), dtype=np.complex128)

    for alpha in range(1,3):
        for beta in range(1,3):
            for line in kinetic:
                x, orb1, orb2, t = line
                x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
                if orb1 == orb2 and x == 0: pass
                if orb1 == alpha and orb2 == beta:
                    for line_ in interaction:
                        x_, orb1_, orb2_, V_ = line_
                        x_, orb1_, orb2_, V_ = float(x_), int(orb1_), int(orb2_), float(V_)

                        if orb2 == orb2_:
                            suma_n = np.sum(rho[orb1 - 1, orb2 - 1, :] * np.exp(-1j*K*x))
                            lega = pos[orb2 - 1] - pos[orb1_ - 1] - x_
                            matrix[orb1_ -1, orb1_ - 1] += -1j * t * V_ * lega / Nk * suma_n

                        if orb1 == orb2_:
                            suma_n = np.sum(rho[orb1 - 1, orb2 - 1, :] * np.exp(-1j*K*x))
                            lega = pos[orb1 - 1] - pos[orb1_ - 1] - x_
                            matrix[orb1_ - 1, orb1_ - 1] += 1j * t * V_ * lega  / Nk * suma_n
    return matrix

# this is M^(4)
def mf_matrix3(K, rho, phys_parameters, mu):
    b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta = phys_parameters
    pos, kinetic, interaction = parameters(b, t, t_, t12, epsilon, epsilon_, Vb, Vc, mu, delta)
    Nk = len(K)
    matrix = np.zeros((2, 2, Nk), dtype=np.complex128)

    for alpha in range(1,3):
        for beta in range(1,3):
            for line in kinetic:
                x, orb1, orb2, t = line
                x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
                if orb1 == orb2 and x == 0: pass
                if orb1 == alpha and orb2 == beta:
                    f_k = t * np.exp(-1j*K*x) / Nk

                    for line_ in interaction:
                        x_, orb1_, orb2_, V_ = line_
                        x_, orb1_, orb2_, V_ = float(x_), int(orb1_), int(orb2_), float(V_)

                        if orb2 == orb2_:
                            lega = pos[orb2 - 1] - pos[orb1_ - 1] - x_
                            g = -1j * V_ * lega * np.exp(1j*K*x) * np.exp(-1j*K*x_)
                            h = rho[orb1 - 1, orb1_ - 1, :]

                            g_fft = np.fft.fft(np.fft.ifftshift(g))
                            h_fft = np.fft.fft(np.fft.ifftshift(h))

                            gh = np.fft.ifft(g_fft * h_fft)
                            gh = np.fft.fftshift(gh)

                            matrix[orb1_ -1, orb2 -1] +=  f_k * gh

                        if orb1 == orb2_:
                            lega = pos[orb1 - 1] - pos[orb1_ - 1] - x_
                            g = 1j * V_ * lega * np.exp(-1j*K*x_)
                            h = rho[orb1 - 1, orb1_ - 1, :]

                            g_fft = np.fft.fft(np.fft.ifftshift(g))
                            h_fft = np.fft.fft(np.fft.ifftshift(h))
                            gh = np.fft.ifft(g_fft * h_fft)
                            gh = np.fft.fftshift(gh)
                            matrix[orb1_ -1, orb2 - 1] += f_k * gh
    return -matrix

# this is M^(4)
def mf_matrix4(K, rho, phys_parameters, mu):
    b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta = phys_parameters
    pos, kinetic, interaction = parameters(b, t, t_, t12, epsilon, epsilon_, Vb, Vc, mu, delta)
    Nk = len(K)
    matrix = np.zeros((2, 2, Nk), dtype=np.complex128)
    for alpha in range(1,3):
        for beta in range(1,3):
            for line in kinetic:
                x, orb1, orb2, t = line
                x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
                if orb1 == orb2 and x == 0: pass
                if orb1 == alpha and orb2 == beta:
                    for line_ in interaction:
                        x_, orb1_, orb2_, V_ = line_
                        x_, orb1_, orb2_, V_ = float(x_), int(orb1_), int(orb2_), float(V_)

                        if orb2 == orb2_:
                            lega = pos[orb2 - 1] - pos[orb1_ - 1] - x_
                            g = -1j * V_ * lega * np.exp(-1j*K*x) * np.exp(1j*K*x_)
                            h = t * np.exp(-1j*K*x) * rho[orb1_ - 1, orb2 - 1, :] / Nk

                            g_fft = np.fft.fft(np.fft.ifftshift(g))
                            h_fft = np.fft.fft(np.fft.ifftshift(h))

                            gh = np.fft.ifft(g_fft * h_fft)
                            gh = np.fft.fftshift(gh)

                            matrix[orb1 -1, orb1_ -1] +=  gh

                        if orb1 == orb2_:
                            lega = pos[orb1 - 1] - pos[orb1_ - 1] - x_
                            g = 1j * V_ * lega * np.exp(1j*K*x_)
                            h = t * np.exp(-1j*K*x) * rho[orb1_ - 1, orb2 - 1, :] / Nk

                            g_fft = np.fft.fft(np.fft.ifftshift(g))
                            h_fft = np.fft.fft(np.fft.ifftshift(h))
                            gh = np.fft.ifft(g_fft * h_fft)
                            gh = np.fft.fftshift(gh)
                            matrix[orb1 -1, orb1_ - 1] += gh
    return -matrix

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
    _, _, _, _, _, _, Vb, Vc, _ = phys_parameters

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
def evolve_rho_kernel(Hk, rho, dt, measure_operators, Nop, do_measure):
    Nk = rho.shape[2]

    if do_measure:
        measurements_k = np.zeros((Nop, Nk), dtype=np.complex128)
    else:
        measurements_k = np.zeros((1, 1), dtype=np.complex128)

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

        # measure operators (evaluate expectation values)
        if do_measure:
            for n in range(Nop):
                measurements_k[n, j] = (
                    rho_next[0,0,j]*measure_operators[n,0,0,j] +
                    rho_next[0,1,j]*measure_operators[n,1,0,j] +
                    rho_next[1,0,j]*measure_operators[n,0,1,j] +
                    rho_next[1,1,j]*measure_operators[n,1,1,j]
                )

    if do_measure:
        measurements = measurements_k.sum(axis=1)
    else:
        measurements = np.zeros(Nop, dtype=np.complex128)

    return rho_next, measurements

def simulate_pulz(K, rho, phys_parameters, include_hartree, perturbation_operator, measure_operators,
                  A0, t0, sigma, Omega, dt, t_max):
    N_points = int(t_max/dt)
    if measure_operators.ndim == 3: 
        measure_operators = measure_operators[np.newaxis, ...]
    Nop = measure_operators.shape[0]

    rho_expvals = np.zeros((N_points, Nop), dtype=np.complex128)
    ts = dt * np.arange(N_points)

    for i in range(N_points):
        if i % 50 == 0:
            print(i/N_points)

        A_t = A_pulz(i * dt, A0, t0, sigma, Omega)
        H_k = h_k(K, rho, phys_parameters, 0., include_hartree) - A_t * perturbation_operator
        rho1, _ = evolve_rho_kernel(H_k, rho, dt, measure_operators, Nop, do_measure=False)

        # Hamiltonian with evolved density matrix
        H_k1 = h_k(K, rho1, phys_parameters, 0., include_hartree) - A_t * perturbation_operator

        # Hamiltonian in mid-step
        A_half = A_pulz(i * dt + 0.5 * dt, A0, t0, sigma, Omega)
        H_k12 = 0.5 * (H_k1 + H_k) - A_half * perturbation_operator

        rho_next, measurements = evolve_rho_kernel(H_k12, rho, dt, measure_operators, Nop, do_measure=True)

        rho_expvals[i] = measurements
        rho = rho_next

    return ts, rho_expvals