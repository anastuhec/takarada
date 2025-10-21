import numpy as np
import scipy.linalg as LA
import os
os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5\\parameters")

def Rho0(Ny, Nx):
    rho0 = np.zeros((6, 6, Ny, Nx), dtype='complex')
    for i in [4, 5]: rho0[i, i, :, :] = 1
    return rho0

def Rhoinfty(Ny, Nx):
    rho0 = np.zeros((6, 6, Ny, Nx), dtype='complex')
    for i in range(6): rho0[i, i, :, :] = 2/6
    return rho0

def H_hopping(Kymesh, Kxmesh, a, b, file='parametri-kinetic.txt'):
    Ny, Nx = Kymesh.shape
    hop = np.zeros((6, 6, Ny, Nx), dtype='complex')
    with open(file, 'r') as f:
        for line in f:
            [x, y, orb1, orb2, t] = list(map(float, line.split()))
            orb1, orb2 = int(orb1), int(orb2)
            ad = t * np.exp(-1j*(Kxmesh * x * a + Kymesh * y * b))
            hop[orb1 - 1, orb2 - 1] += ad
            if orb1 != orb2: hop[orb2 - 1, orb1 - 1] += ad.conjugate()
    return hop

def H_perturb(Kymesh, Kxmesh, a, b, file='perturbacija.txt'):
    return H_hopping(Kymesh, Kxmesh, a, b, file=file)

def Delta(Kxmesh, Nk, rho, i, j, x): 
    if type(x) == np.ndarray:
        return np.array([np.sum(rho[i, j] * np.exp(1j * Kxmesh * x1)) for x1 in x]) / Nk
    else: return np.sum(rho[i, j] * np.exp(1j * Kxmesh * x)) / Nk

def Phi(Kxmesh, Nk, rho, a):
    pos12 = np.array([0, a])
    pos34 = np.array([0, -a])
    phi = np.zeros(4, dtype='complex')
    phi[0] = np.sum(Delta(Kxmesh, Nk, rho, 0, 4, pos12))
    phi[1] = np.sum(Delta(Kxmesh, Nk, rho, 1, 4, pos12))
    phi[2] = np.sum(Delta(Kxmesh, Nk, rho, 2, 5, pos34))
    phi[3] = np.sum(Delta(Kxmesh, Nk, rho, 3, 5, pos34))
    return phi

def H_hartree(rho, Nk, U, V, file='parametri-interaction.txt'):
    hartree_k = np.zeros((6,6), dtype='complex')
    with open(file) as f:
        for line in f:
            [orb1, orb2] = list(map(float, line.split()))[-2:]
            orb1, orb2 = int(orb1-1), int(orb2-1)
            if orb1 == orb2: 
                hartree_k[orb1, orb1] += U * np.sum(rho[orb1,orb1])
            else:
                hartree_k[orb1, orb1] += 2 * V * np.sum(rho[orb2, orb2])
                hartree_k[orb2, orb2] += 2 * V * np.sum(rho[orb1, orb1])
    return hartree_k / Nk
    
def H_fock(Kxmesh, Nk, rho, a, V):
    fock = np.zeros(rho.shape, dtype='complex')
    phi = Phi(Kxmesh, Nk, rho, a)
    fock[0,4] = - V * Delta(Kxmesh, Nk, rho, 0, 4, 0) * (1 - np.exp(-1j * Kxmesh * a)) - \
                    V * phi[0] * np.exp(-1j * Kxmesh * a)
    fock[4,0] = fock[0,4].conjugate()
    fock[1,4] = - V * Delta(Kxmesh, Nk, rho, 1, 4, 0) * (1 - np.exp(-1j * Kxmesh * a)) - \
                    V * phi[1] * np.exp(-1j * Kxmesh * a)
    fock[4,1] = fock[1,4].conjugate()
    fock[2,5] = - V * Delta(Kxmesh, Nk, rho, 2, 5, 0) * (1 - np.exp(1j * Kxmesh * a)) - \
                    V * phi[2] * np.exp(1j * Kxmesh * a)
    fock[5,2] = fock[2,5].conjugate()
    fock[3,5] = - V * Delta(Kxmesh, Nk, rho, 3, 5, 0) * (1 - np.exp(1j * Kxmesh * a)) - \
                    V * phi[3] * np.exp(1j * Kxmesh * a)
    fock[5,3] = fock[3,5].conjugate()
    return fock

def H_diagonalize(Ny, Nx, hop, perturb, hartree, fock, T, mu, eps):
    H = hop + fock
    if eps != 0: H += perturb * eps
    energije, vecs = np.zeros((6, Ny, Nx)), np.zeros((6, 6, Ny, Nx), dtype='complex')
    fs = np.zeros((6, 6, Ny, Nx))
    for m in range(Ny):
        for n in range(Nx):
            en, v = LA.eigh(H[:, :, m, n] + hartree)
            energije[:, m, n] = en
            vecs[:, :, m, n] = v
            if T == 0: np.fill_diagonal(fs[:, :, m, n], np.array([1, 1, 0, 0, 0, 0]))
            elif T == 'infty': np.fill_diagonal(fs[:, :, m, n], np.array([1, 1, 1, 1, 1, 1])/3)
            else:
                np.fill_diagonal(fs[:, :, m, n], 1/(1 + np.exp((en - mu)/T)))
    return energije, vecs, fs

def H_diagonalize2(Ny, Nx, hop, perturb, hartree, fock, T, mu, eps):
    H = hop + fock
    if eps != 0: H += perturb * eps
    energije, vecs = np.zeros((6, Ny, Nx)), np.zeros((6, 6, Ny, Nx), dtype='complex')
    fs = np.zeros((6, 6, Ny, Nx))

    for n in range(Nx):
        en, v = LA.eigh(H[:, :, 0, n] + hartree)
        energije[:, 0, n] = en
        vecs[:, :, 0, n] = v
        if T == 0: np.fill_diagonal(fs[:, :, 0, n], np.array([1, 1, 0, 0, 0, 0]))
        elif T == 'infty': np.fill_diagonal(fs[:, :, 0, n], np.array([1, 1, 1, 1, 1, 1])/3)
        else:
            np.fill_diagonal(fs[:, :, 0, n], 1/(1 + np.exp((en - mu)/T)))

        en, v = LA.eigh(H[:, :, Ny//2, n] + hartree)
        energije[:, Ny//2, n] = en
        vecs[:, :, Ny//2, n] = v
        if T == 0: np.fill_diagonal(fs[:, :, Ny//2, n], np.array([1, 1, 0, 0, 0, 0]))
        elif T == 'infty': np.fill_diagonal(fs[:, :, Ny//2, n], np.array([1, 1, 1, 1, 1, 1])/3)
        else:
            np.fill_diagonal(fs[:, :, Ny//2, n], 1/(1 + np.exp((en - mu)/T)))

    for m in range(Ny):
        en, v = LA.eigh(H[:, :, m, 0] + hartree)
        energije[:, m, 0] = en
        vecs[:, :, m, 0] = v
        if T == 0: np.fill_diagonal(fs[:, :, m, 0], np.array([1, 1, 0, 0, 0, 0]))
        elif T == 'infty': np.fill_diagonal(fs[:, :, m, 0], np.array([1, 1, 1, 1, 1, 1])/3)
        else:
            np.fill_diagonal(fs[:, :, m, 0], 1/(1 + np.exp((en - mu)/T)))

        en, v = LA.eigh(H[:, :, m, Nx//2] + hartree)
        energije[:, m, Nx//2] = en
        vecs[:, :, m, Nx//2] = v
        if T == 0: np.fill_diagonal(fs[:, :, m, Nx//2], np.array([1, 1, 0, 0, 0, 0]))
        elif T == 'infty': np.fill_diagonal(fs[:, :, m, Nx//2], np.array([1, 1, 1, 1, 1, 1])/3)
        else:
            np.fill_diagonal(fs[:, :, m, Nx//2], 1/(1 + np.exp((en - mu)/T)))

    for m in range(1,Ny//2):
        for n in range(1,Nx//2):
            en, v = LA.eigh(H[:, :, m, n] + hartree)
            energije[:, m, n] = en
            energije[:, -m, -n] = en
            vecs[:, :, m, n] = v
            vecs[:, :, -m, -n] = v.conj()
            if T == 0:
                np.fill_diagonal(fs[:, :, m, n], np.array([1, 1, 0, 0, 0, 0]))
                np.fill_diagonal(fs[:, :, -m, -n], np.array([1, 1, 0, 0, 0, 0]))
            elif T == 'infty':
                np.fill_diagonal(fs[:, :, m, n], np.array([1, 1, 1, 1, 1, 1])/3)
                np.fill_diagonal(fs[:, :, -m, -n], np.array([1, 1, 1, 1, 1, 1])/3)
            else:
                np.fill_diagonal(fs[:, :, m, n], 1/(1 + np.exp((en - mu)/T)))
                np.fill_diagonal(fs[:, :, -m, -n], 1/(1 + np.exp((en - mu)/T)))

            en, v = LA.eigh(H[:, :, m, n + Nx//2] + hartree)
            energije[:, m, n + Nx//2] = en
            energije[:, -m, Nx//2 - n] = en
            vecs[:, :, m, n + Nx//2] = v
            vecs[:, :, -m, Nx//2 - n] = v.conj()
            if T == 0:
                np.fill_diagonal(fs[:, :, m, n + Nx//2], np.array([1, 1, 0, 0, 0, 0]))
                np.fill_diagonal(fs[:, :, -m, Nx//2 - n], np.array([1, 1, 0, 0, 0, 0]))
            elif T == 'infty':
                np.fill_diagonal(fs[:, :, m, n + Nx//2], np.array([1, 1, 1, 1, 1, 1])/3)
                np.fill_diagonal(fs[:, :, -m, Nx//2 - n], np.array([1, 1, 1, 1, 1, 1])/3)
            else:
                np.fill_diagonal(fs[:, :, m, n + Nx//2], 1/(1 + np.exp((en - mu)/T)))
                np.fill_diagonal(fs[:, :, -m, Nx//2 - n], 1/(1 + np.exp((en - mu)/T)))

    return energije, vecs, fs

def F(Ny, Nx, rho, hop, perturb, hartree, fock, T, mu, eps=0, colors='no', occupation='no', vectors='no'):
    energije, vecs, fs = H_diagonalize2(Ny, Nx, hop, perturb, hartree, fock, T, mu, eps)
    rho_new = np.einsum('ijkl,jmkl,mnkl-> inkl', vecs, fs, np.swapaxes(vecs.conj(), 0, 1))
    if colors == 'yes':
        barve = np.einsum('ijkl->jkl', np.abs(vecs[:4, :, :, :])**2)
        return rho_new, energije, np.max(np.abs(rho - rho_new)), barve
    if occupation == 'no': return rho_new, energije, np.max(np.abs(rho - rho_new))
    elif occupation == 'yes' and vectors == 'yes':
        return rho_new, energije, fs, vecs, np.max(np.abs(rho - rho_new))
    
def Occupation(rho):
    return (np.sum(np.diag(np.einsum('ijkl->ij', rho)))/(np.prod(rho.shape[-2:]))).real
    
def GS(Kxmesh, rho, hop, perturb, hartree, fock, mu, eps0, a, U, V, epsilon, maxiter=1000, N_epsilon=5):
    Ny, Nx = Kxmesh.shape
    err, N_iters = 1, 0
    while err > epsilon and N_iters < maxiter:
        if N_iters < N_epsilon: eps = eps0
        else: eps = 0
        rho, _, err = F(Ny, Nx, rho, hop, perturb, hartree, fock, 0, mu, eps=eps)
        fock = H_fock(Kxmesh, Nx*Ny, rho, a, V)
        hartree = H_hartree(rho, Nx*Ny, U, V)
        N_iters += 1
    rho, energije, fs, vecs, err = F(Ny, Nx, rho, hop, perturb, hartree, fock, 0, mu, eps=eps, occupation='yes', vectors='yes')
    print(f'Found ground state with error={err}, occupation error={np.abs(Occupation(rho)-2)}')
    return rho, energije, fs, vecs, err, Occupation(rho), fock, hartree

''''''''''''''''''''''''''

def Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, maxiter, mix, epsilon, eps0=0.0, N_epsilon=5):
    Ny, Nx = Kxmesh.shape
    err, N_iters = 1, 0
    while err > epsilon and N_iters < maxiter:
        if N_iters < N_epsilon: eps = eps0
        else: eps0 = 0
        rho_new, _, err = F(Ny, Nx, rho, hop, perturb, hartree, fock, T, mu, eps=eps)
        rho = rho_new * mix + rho * (1 - mix)
        fock = H_fock(Kxmesh, Nx*Ny, rho, a, V)
        hartree = H_hartree(rho, Nx*Ny, U, V)
        rho = rho_new * mix + rho * (1 - mix)
        N_iters += 1
    rho, energije, fs, vecs, err = F(Ny, Nx, rho, hop, perturb, hartree, fock, T, mu, occupation='yes', vectors='yes')
    return rho, energije, fs, vecs, fock, hartree, err, Occupation(rho)

def NewMu(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, dmu, maxiter, maxiter_last, eps_last, mix, mix2, mix3, n_pass, max_trials, faktor1=0.001):
    rho_a, energije_a, fs_a, vecs_a, fock_a, hartree_a, err_a, n_a = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu, maxiter, mix, eps_last)
    #if np.abs(n_a - 2) < n_pass and err_a < eps_last:
    #    return rho_a, energije_a, fs_a, vecs_a, fock_a, hartree_a, err_a, n_a, mu
    n_b = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu + dmu, maxiter, mix, eps_last)[-1]
    chi = (n_b - n_a)/dmu
    if chi != 0: mu = mu - mix2 * (n_a - 2)/np.abs(chi)

    pogoj = False
    koraki = 0
    if np.abs(chi) > 0: faktor = (n_a - 2)/chi * mix3
    else: faktor = faktor1
    if chi >= 0:
        if n_a >= 2:
            sign = -1
        elif n_a < 2: sign = +1
    elif chi < 0:
        if n_a >= 2: sign = +1
        elif n_a < 2: sign = -1
        
    sgns = np.ones(2) * np.sign(n_a - 2)
    ns = np.array([0, n_a])
    mus = [0, mu]
    enough = False
    while sgns[0] == sgns[1]:
        if np.abs(n_a - 2) < n_pass and err_a < eps_last:
            enough = True
            break
        rho_b, energije_b, fs_b, vecs_b, fock_b, hartree_b, err_b, n_b = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu + faktor*koraki*sign, maxiter, mix, eps_last)
        if np.abs(n_b - 2) < n_pass and err_b < eps_last: return rho_b, energije_b, fs_b, vecs_b, fock_b, hartree_b, err_b, n_b,  mu + faktor*koraki*sign
        ns[0] = n_b
        mus[0] = mu + faktor*koraki*sign
        sgns[1] = np.sign(n_b - 2)
        if sgns[0] != sgns[1]: break
        if n_b < 2 and n_b < ns[1]: sign *= -1
        if n_b > 2 and n_b > ns[1]: sign *= -1
        ns = np.roll(ns, 1)
        mus = np.roll(mus, 1)
        sgns[1] = np.sign(n_b - 2)
        koraki +=1
        if np.abs(n_b - 2) < n_pass and err_b < eps_last:
            enough = True
            mu_mid = mu + faktor*koraki*sign
            break
        
    mus = np.sort(np.array([mu + faktor*koraki*sign, mu + faktor*(koraki-1)*sign]))
    ns = np.sort(np.array(ns))

    trials = 0
    while pogoj == False:
        mu_mid = (mus[0] + mus[1])/2
        if enough == True:
            break   
        n_mid = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu_mid, maxiter, mix, eps_last)[-1]
        if n_mid > 2: mus[1] = mu_mid
        elif n_mid < 2: mus[0] = mu_mid
        if np.abs(n_mid - 2) < n_pass: break
        trials += 1 
        if trials > max_trials: break
    rho, energije, fs, vecs, fock, hartree, err, n = Rho_next(Kxmesh, rho, hop, perturb, hartree, fock, a, U, V, T, mu_mid, maxiter_last, mix, eps_last)
    return rho, energije, fs, vecs, fock, hartree, err, n, mu_mid