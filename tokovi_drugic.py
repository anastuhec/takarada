import numpy as np
import os, scipy, mpmath

os.chdir("C:\\Users\\anast\\OneDrive\\Namizje\\1m\\poletje\\ta2nise5\\parameters")

''' matrix for number density operator '''
def j_tok(Kymesh, Kxmesh, a, b, b2, file='parametri-kinetic.txt'):
    pos = {1: np.array([-a/4, b/2 - b2]), 2: np.array([-a/4,b2]), 3 : np.array([a/4,-b2]), 4: np.array([a/4, -b/2 + b2]), 5: np.array([a/4,b/4]), 6: np.array([-a/4, -b/4])}
    Ny, Nx = Kymesh.shape
    jx = np.zeros((6, 6, Ny, Nx), dtype='complex')
    jy = np.copy(jx)

    with open(file, 'r') as f:
        for line in f:
            [x, y, orb1, orb2, t] = list(map(float, line.split()))
            orb1, orb2 = int(orb1), int(orb2)
            if orb1 == orb2 and (x,y) == (0,0): pass # this is onsite energy, does not contribute to j
            else:
                osnova = 1j * t * np.exp(-1j * (Kxmesh * x * a + Kymesh * y * b))
                position = pos[orb2] - pos[orb1] - np.array([x*a, y*b])
                ad_x = osnova * position[0]
                ad_y = osnova * position[1]

                jx[orb1 - 1, orb2 - 1] += ad_x
                if orb1 != orb2: jx[orb2 - 1, orb1 - 1] += ad_x.conjugate() 
                jy[orb1 - 1, orb2 - 1] += ad_y
                if orb1 != orb2: jy[orb2 - 1, orb1 - 1] += ad_y.conjugate()
    jmatrix = np.zeros((2,6,6,Ny,Nx), dtype='complex')
    jmatrix[0] = jx
    jmatrix[1] = jy
    return jmatrix

''' matrix for energy density operator -- contribution from hop-hop '''
def j_1(Kymesh, Kxmesh, a, b, b2, rho, Nk, U, V, mu=0., file='hop.txt'):
    pos = {1: np.array([-a/4, b/2 - b2]), 2: np.array([-a/4,b2]), 3 : np.array([a/4,-b2]), 4: np.array([a/4, -b/2 + b2]), 5: np.array([a/4,b/4]), 6: np.array([-a/4, -b/4])}
    Ny, Nx = Kymesh.shape
    jx = np.zeros((6, 6, Ny, Nx), dtype='complex')
    jy = np.copy(jx)
    with open(file, 'r') as f1:
        for line in f1:
            [x, y, orb1, orb2, t] = list(map(float, line.split()))
            if orb1 == orb2 and (x,y) == (0,0):
                t += - mu + U*np.sum(rho[int(orb1)-1,int(orb1)-1])/Nk
                if orb1 == 1 or orb1 == 2:
                    t += 2*V*np.sum(rho[4,4])/Nk
                if orb1 == 5:
                    t += 2*V*np.sum(rho[0,0])/Nk + 2*V*np.sum(rho[1,1])/Nk
                if orb1 == 3 or orb1 == 4:
                    t += 2 * V * np.sum(rho[5,5])/Nk
                if orb1 == 6:
                    t += 2*V*np.sum(rho[2,2])/Nk + 2*V*np.sum(rho[3,3])/Nk

            orb1, orb2 = int(orb1), int(orb2)

            with open(file, 'r') as f2:
                for line_ in f2:
                    [x_, y_, orb1_, orb2_, t_] = list(map(float, line_.split()))
                    if orb1_ == orb2_ and (x_,y_) == (0,0):
                        t_ += t_ - mu + U*np.sum(rho[int(orb1_)-1,int(orb1_)-1])/Nk
                        if orb1_ == 1 or orb1_ == 2:
                            t_ += 2*V*np.sum(rho[4,4])/Nk
                        if orb1_ == 5:
                            t_ += 2*V*np.sum(rho[0,0])/Nk + 2*V*np.sum(rho[1,1])/Nk
                        if orb1_ == 3 or orb1_ == 4:
                            t_ += 2*V*np.sum(rho[5,5])/Nk
                        if orb1_ == 6:
                            t_ += 2*V*np.sum(rho[2,2])/Nk + 2*V*np.sum(rho[3,3])/Nk

                    orb1_, orb2_ = int(orb1_), int(orb2_)
                    if orb2 == orb1_:
                        osnova = - 1j * t * t_ * 0.5 * np.exp(-1j * (Kxmesh * (x + x_) * a + Kymesh * (y + y_) * b))
                        position = (pos[orb1] - pos[orb2_] + np.array([(x+x_)*a, (y+y_)*b]))
                        ad_x = osnova * position[0]
                        ad_y = osnova * position[1]
                        jx[orb1 - 1, orb2_ - 1] += ad_x
                        jy[orb1 - 1, orb2_ - 1] += ad_y
    jmatrix = np.zeros((2,6,6,Ny,Nx), dtype='complex')
    jmatrix[0] = jx
    jmatrix[1] = jy
    return jmatrix

def Delta(Kxmesh, Kymesh, Nk, rho, i, j, x): 
    if type(x) == np.ndarray:
        return np.array([np.sum(rho[i, j] * np.exp(-1j * Kxmesh * x1[0] - 1j * Kymesh * x1[1])) for x1 in x]) / Nk
    else: return np.sum(rho[i, j] * np.exp(-1j * Kxmesh * x[0] - 1j * Kymesh * x[1])) / Nk

def j_MF(Kymesh, Kxmesh, rho, V, a, b, b2, file1='hop.txt', file2='int.txt'):
    pos = {1: np.array([-a/4, b/2 - b2]), 2: np.array([-a/4, b2]), 3 : np.array([a/4,-b2]), 4: np.array([a/4, -b/2 + b2]), 5: np.array([a/4,b/4]), 6: np.array([-a/4, -b/4])}
    Ny, Nx = Kymesh.shape
    Nk = Ny*Nx
    jxy = np.zeros((2, 6, 6, Ny, Nx), dtype='complex')
    with open(file1, 'r') as f1:
        for line in f1: 
            [x, y, orb1, orb2, t] = list(map(float, line.split()))
            orb1, orb2 = int(orb1), int(orb2)
            with open(file2, 'r') as f2:
                for line in f2:
                    [x_, y_, orb1_, orb2_, utez] = list(map(float, line.split()))
                    orb1_, orb2_ = int(orb1_), int(orb2_)
                    if orb2 == orb2_:
                        position = (pos[orb1] - pos[orb1_] + np.array([(x - x_)*a, (y - y_)*b]))/2
                        osnova = 1j * t * utez * V * Delta(Kxmesh, Kymesh, Nk, rho, orb1_ -1, orb1_ - 1, [0,0]) * np.exp(-1j * (Kxmesh * x * a + Kymesh * y * b))
                        for nu in [0,1]:
                            jxy[nu, orb1 - 1, orb2 - 1] += - 2 * position[nu] * osnova

                        osnova = 1j * t * utez * V * Delta(Kxmesh, Kymesh, Nk, rho, orb1 - 1, orb2 - 1, [x*a, y*b])
                        for nu in [0,1]:
                            jxy[nu, orb1_ - 1, orb1_ - 1] += - 2 * position[nu] * osnova

                        osnova = 1j * t * utez * V * Delta(Kxmesh, Kymesh, Nk, rho, orb1_ -1, orb2 - 1, [x_*a, y_*b]) * np.exp(-1j * (Kxmesh * (x-x_) * a + Kymesh * (y-y_) * b))
                        for nu in [0,1]:
                            jxy[nu, orb1 - 1, orb1_ - 1] += position[nu] * osnova

                        osnova = 1j * t * utez * V * Delta(Kxmesh, Kymesh, Nk, rho, orb1 - 1, orb1_ -1, [(x-x_)*a, (y-y_)*b]) * np.exp(-1j * (Kxmesh * x_ * a + Kymesh * y_ * b))
                        for nu in [0,1]:
                            jxy[nu, orb1_ - 1, orb2 - 1] += position[nu] * osnova

                    if orb1 == orb2_:
                        position = (pos[orb2] - pos[orb1_] - np.array([(x+x_)*a, (y+y_)*b]))/2
                        osnova = 1j * t * utez * V * Delta(Kxmesh, Kymesh, Nk, rho, orb1_ -1, orb1_ - 1, [0,0]) * np.exp(-1j * (Kxmesh * x * a + Kymesh * y * b))
                        for nu in [0,1]:
                            jxy[nu, orb1 - 1, orb2 - 1] += 2 * position[nu] * osnova

                        osnova = 1j * t * utez * V * Delta(Kxmesh, Kymesh, Nk, rho, orb1 -1, orb2 - 1, [x*a, y*b])
                        for nu in [0,1]:
                            jxy[nu, orb1_ - 1, orb1_ - 1] += 2 * position[nu] * osnova

                        osnova = 1j * t * utez * V * Delta(Kxmesh, Kymesh, Nk, rho, orb1_ -1, orb2 - 1, [(x+x_)*a, (y+y_)*b]) * np.exp(1j * (Kxmesh * x_ * a + Kymesh * y_ * b))
                        for nu in [0,1]:
                            jxy[nu, orb1 - 1, orb1_ - 1] += - position[nu] * osnova

                        osnova = 1j * t * utez * V * Delta(Kxmesh, Kymesh, Nk, rho, orb1 -1, orb1_ - 1, [-x_*a, -y_*b]) * np.exp(-1j * (Kxmesh * (x+x_) * a + Kymesh * (y+y_) * b))
                        for nu in [0,1]:
                            jxy[nu, orb1_ - 1, orb2 - 1] += - position[nu] * osnova
    return jxy     

''' tensor for energy density operator -- contribution from hop-interaction'''
def j_2_kq(Kymesh, Kxmesh, qy, qx, V, a, b, b2, file1='hop.txt', file2='int.txt'):
    pos = {1: np.array([-a/4, b/2 - b2]), 2: np.array([-a/4,b2]), 3 : np.array([a/4,-b2]), 4: np.array([a/4, -b/2 + b2]), 5: np.array([a/4,b/4]), 6: np.array([-a/4, -b/4])}
    Ny, Nx = Kymesh.shape
    jx = np.zeros((6, 6, 6, Ny, Nx), dtype='complex')
    jy = np.copy(jx)
    with open(file1, 'r') as f1:
        for line in f1:
            [x, y, orb1, orb2, t] = list(map(float, line.split()))
            orb1, orb2 = int(orb1), int(orb2)

            with open(file2, 'r') as f2:
                for line in f2:
                    [x_, y_, orb1_, orb2_, utez] = list(map(float, line.split()))
                    orb1_, orb2_ = int(orb1_), int(orb2_)

                    if orb2 == orb2_:
                        osnova = - t * utez * V * np.exp(-1j * (Kxmesh * x * a + Kymesh * y * b)) * np.exp(1j * (qx * x_ * a + qy * y_ * b)) * np.exp(-1j*(qx * x * a + qy * y * b))
                        position = (pos[orb1] - pos[orb1_] + np.array([(x - x_)*a, (y - y_)*b]))
                        jx[orb1 - 1, orb1_ - 1, orb2 - 1] += osnova * position[0]
                        jy[orb1 - 1, orb1_ - 1, orb2 - 1] += osnova * position[1]

                    if orb1 == orb2_:
                        osnova = t * utez * V * np.exp(-1j * (Kxmesh * x * a + Kymesh * y * b)) * np.exp(1j * (qx * x_ * a + qy * y_ * b))
                        position = (pos[orb2] - pos[orb1_] + np.array([(-x - x_)*a, (-y - y_)*b]))
                        jx[orb1 - 1, orb1_ - 1, orb2 - 1] += osnova * position[0]
                        jy[orb1 - 1, orb1_ - 1, orb2 - 1] += osnova * position[1]
    return 1j*jx, 1j*jy

def j_2(Kymesh, Kxmesh, V, a, b, b2):
    Ny, Nx = Kymesh.shape
    Ky = Kymesh[:,0]
    Kx = Kxmesh[0]
    jx = np.zeros((6, 6, 6, Ny, Nx, Ny, Nx), dtype='complex')
    jy = np.copy(jx)
    for i_ in range(Ny):
        for j_ in range(Nx):
            qy, qx = Ky[i_], Kx[j_]
            jx[:,:,:,:,:,i_,j_], jy[:,:,:,:,:,i_,j_] = j_2_kq(Kymesh, Kxmesh, qy, qx, V, a, b, b2)
    return jx, jy

def new_indices(Ny, Nx):
    New_ks_indices = np.zeros((Ny, Nx, Ny, Nx, 2))
    for i in range(Ny):
        for j in range(Nx):
            for i_ in range(Ny):
                for j_ in range(Nx):
                    ind = (i + i_ - Ny//2 ) % Nx
                    if ind == Ny: ind = 0
                    New_ks_indices[i,j,i_,j_,0] = int(ind)

                    ind = (j + j_ - Nx // 2) % Nx
                    New_ks_indices[i,j,i_,j_,1] = int(ind)
    return New_ks_indices

def Spektralna_k(omega, mu, energije_k, Gamma):
    N_orbitals = energije_k.shape[0]
    A = np.zeros((N_orbitals, N_orbitals))
    for i in range(N_orbitals):
        A[i,i] = 1/np.pi * Gamma / ((omega - (energije_k[i] - mu))**2 + Gamma**2 ) 
    return A

''' df/domega, f je Fermi-Diracova porazdelitvena funkcija '''
def fd_1(omega, T): return -1/(4*T)/np.cosh(omega/(2*T))**2


def helper_phi(omegas, j_tilde, energije, mu, Gamma, simplify=None):
    transportna = np.zeros((2, omegas.shape[0]), dtype='complex')
    for q, omega in enumerate(omegas):
        A = Spektralna_k(omega, mu, energije, Gamma)
        for nu in range(2):
            if simplify == 'yes':
                for orb in range(6):
                    transportna[nu][q] += 2 * j_tilde[nu,orb,orb]**2 * A[orb,orb]**2
            else: transportna[nu][q] += 2 * np.trace(j_tilde[nu] @ A @ j_tilde[nu] @ A)
    return transportna

def transportna_phi(Kymesh, vecs, energije, j_matrix, mu, omegas, Gamma=0.01, simplify=None):
    Ny, Nx = Kymesh.shape
    transportna = np.zeros((2, omegas.shape[0]), dtype='complex')
    j_matrix_new = np.zeros(j_matrix.shape, dtype='complex')

    for n in range(Nx):
        vec = vecs[:,:,0,n]
        j_tilde = np.einsum('ij,hjl,lk->hik', vec.conj().T, j_matrix[:,:,:,0,n], vec)
        j_matrix_new[:,:,:,0,n] = j_tilde
        transportna += helper_phi(omegas, j_tilde, energije[:,0,n], mu, Gamma, simplify)

    for m in range(Ny):
        vec = vecs[:,:,m,0]
        j_tilde = np.einsum('ij,hjl,lk->hik', vec.conj().T, j_matrix[:,:,:,m,0], vec)
        j_matrix_new[:,:,:,m,0] = j_tilde
        transportna += helper_phi(omegas, j_tilde, energije[:,m,0], mu, Gamma, simplify)

    for m in range(1,Ny//2):
        for n in range(1,Nx//2):
            vec = vecs[:,:,m,n]
            j_tilde = np.einsum('ij,hjl,lk->hik', vec.conj().T, j_matrix[:,:,:,m,n], vec)
            j_matrix_new[:,:,:,m,n] = j_tilde
            j_matrix_new[:,:,:,-m,-n] = -j_tilde.conj()
            transportna += 2 * helper_phi(omegas, j_tilde, energije[:,m,n], mu, Gamma, simplify)

            vec = vecs[:,:,m,n+Nx//2]
            j_tilde = np.einsum('ij,hjl,lk->hik', vec.conj().T, j_matrix[:,:,:,m,n+Nx//2], vec)
            j_matrix_new[:,:,:,m,n+Nx//2] = j_tilde
            j_matrix_new[:,:,:,-m,Nx//2-n] = -j_tilde.conj()
            transportna += 2 * helper_phi(omegas, j_tilde, energije[:,m,n+Nx//2], mu, Gamma, simplify)
    
    return transportna.real, j_matrix_new

def transportna_phi_Boltzmann(Kymesh, Kxmesh, energije, mu, omegas, faktor=1):

    energije_new = np.copy(energije)

    Ny, Nx = Kymesh.shape
    dKy, dKx = np.diff(Kymesh[:,0])[0], np.diff(Kxmesh[0])[0]
    velocity_new_y, velocity_new_x = np.zeros(energije.shape), np.zeros(energije.shape)

    mask = np.zeros(energije[1].shape)
    ind1, ind2 = np.where(energije[1] - energije[0] < 0.005) 
    for i in range(len(ind1)):
        mask[ind1[i], ind2[i]] = 1
        energije_new[1,ind1[i], ind2[i]] = energije[0,ind1[i], ind2[i]]
        energije_new[0,ind1[i], ind2[i]] = energije[1,ind1[i], ind2[i]]
    for j in range(3,Nx//2):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[0] != 0:
                energije_new[1,:enice[-1],j] = energije[0,:enice[-1],j]
                energije_new[0,:enice[-1],j] = energije[1,:enice[-1],j]
    for j in range(Nx//2+1, Nx-5):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[-1] != 0:
                energije_new[1,enice[0]:,j] = energije[0,enice[0]:,j]
                energije_new[0,enice[0]:,j] = energije[1,enice[0]:,j]

    mask = np.zeros(energije[1].shape)
    ind1, ind2 = np.where(energije[3] - energije[2] < 0.001) 
    for i in range(len(ind1)):
        mask[ind1[i], ind2[i]] = 1
    for j in range(Nx//4,Nx//2):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[0] != 0:
                energije_new[3,:enice[-1],j] = energije[2,:enice[-1],j]
                energije_new[2,:enice[-1],j] = energije[3,:enice[-1],j]
    for j in range(Nx//2, Nx - Nx//4):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[-1] != 0:
                energije_new[3,enice[0]:,j] = energije[2,enice[0]:,j]
                energije_new[2,enice[0]:,j] = energije[3,enice[0]:,j]

    mask = np.zeros(energije[1].shape)
    ind1, ind2 = np.where(energije[5] - energije[4] < 0.001) 
    for i in range(len(ind1)):
        mask[ind1[i], ind2[i]] = 1
    for j in range(Nx//4,Nx//2):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[0] != 0:
                energije_new[5,:enice[-1],j] = energije[4,:enice[-1],j]
                energije_new[4,:enice[-1],j] = energije[5,:enice[-1],j]
    for j in range(Nx//2, Nx - Nx//4):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[-1] != 0:
                energije_new[5,enice[0]:,j] = energije[4,enice[0]:,j]
                energije_new[4,enice[0]:,j] = energije[5,enice[0]:,j]

    maksimumi_y, maksimumi_x = [], []
    for i in range(6):
        gr = np.gradient(energije[i])
        velocity_new_y[i, :, :], velocity_new_x[i, :, :] = gr[0]/dKy, gr[1]/dKx
        maksimumi_y.append(np.max(np.abs(velocity_new_y[i,:,:])))
        maksimumi_x.append(np.max(np.abs(velocity_new_x[i,:,:])))

    domega = omegas[1] - omegas[0]
    transportna = np.zeros((2, omegas.shape[0]))
    v_max = np.array([np.max(maksimumi_y), np.max(maksimumi_x)])
    sigma = faktor * np.array([np.sqrt(v_max[0] * domega * dKy), np.sqrt(v_max[1] * domega * dKx)])
    for i in range(Ny):
        for j in range(Nx//2):
            for m in range(6):
                ''' factor 2 is for spin degeneracy,
                 second factor 2 if due to inversion symmetry of the Brillouin zone '''
                transportna[0] += 2 * 2 *  1/(2*np.pi*sigma[0]**2)**0.5 * np.exp(-(omegas - energije[m,i,j] + mu)**2/(2*sigma[0]**2)) * velocity_new_x[m,i,j]**2
                transportna[1] += 2 * 2 * 1/(2*np.pi*sigma[1]**2)**0.5 * np.exp(-(omegas - energije[m,i,j] + mu)**2/(2*sigma[1]**2)) * velocity_new_y[m,i,j]**2
    jnew = np.zeros((2, 6, Ny, Nx))
    jnew[0] = velocity_new_x
    jnew[1] = velocity_new_y
    return transportna, jnew

def transportna_phiE(Kymesh, vecs, energije, j_matrix, j1_matrix, j2_matrix, mu, omegas, Gamma=0.05):
    Ny, Nx = Kymesh.shape
    transportna_K = np.zeros((2, omegas.shape[0]), dtype='complex')
    transportna_I = np.copy(transportna_K)
    for i in range(Ny):
        for j in range(Nx):
            j_tilde_K =  np.einsum('ij,hjl,lk->hik', vecs[:,:,i,j].conj().T, j1_matrix[:,:,:,i,j], vecs[:,:,i,j])
            j_tilde_I = np.einsum('ij,hjl,lk->hik', vecs[:,:,i,j].conj().T, j2_matrix[:,:,:,i,j], vecs[:,:,i,j])
            j_tilde = np.einsum('ij,hjl,lk->hik', vecs[:,:,i,j].conj().T, j_matrix[:,:,:,i,j], vecs[:,:,i,j])
            for m, omega in enumerate(omegas):
                A = Spektralna_k(omega, mu, energije[:,i,j], Gamma)
                for nu in range(2):
                    ''' factor 2 is for spin degeneracy '''
                    transportna_K[nu][m] += 2*np.trace(j_tilde_K[nu] @ A @ j_tilde[nu] @ A)
                    transportna_I[nu][m] += 2*np.trace(j_tilde_I[nu] @ A @ j_tilde[nu] @ A)
    return transportna_K, transportna_I

def L11(phi, omegas, T):
    return np.sum(phi * (-fd_1(omegas, T)) ) * (omegas[1] - omegas[0])

def L12(phiK, phiI, phiI_full, phi, omegas, T):
    l12_k = np.sum(phiK * (-fd_1(omegas, T))) * (omegas[1] - omegas[0])
    l12_i = np.sum(phiI * (-fd_1(omegas, T))) * (omegas[1] - omegas[0])
    l12_i_full = np.sum(phiI_full * (-fd_1(omegas, T))) * (omegas[1] - omegas[0])
    l12_neint = np.sum((omegas) * phi * (-fd_1(omegas, T))) * (omegas[1] - omegas[0])
    return l12_k, l12_i, l12_i_full, l12_neint

def transportna_I_full(Kymesh, vecs, energije, fs, j2_matrix_full, j_matrix, mu, omega_max=1, domega=0.01, Gamma=0.05):
    Ny, Nx = Kymesh.shape

    omegas = np.arange(-omega_max, omega_max, domega)
    
    transportna = np.zeros((2, omegas.shape[0]), dtype='complex') # xx, xy, yx, yy
    '''for i in range(Ny):
        for j in range(Nx):
            for i_ in range(Ny):
                for j_ in range(Nx):
                    ind_y = (i + i_ - Ny//2 ) % Ny
                    ind_x = (j + j_ - Nx//2) % Nx
                    M_1_full = np.einsum('ij,j,jb,habi->hab', vecs[:,:,i,j], np.diag(fs[:,:,i,j]), vecs[:,:,i,j].conj().T, j2_matrix_full[:,:,:,:,i,j,i_,j_]) + \
                        + np.einsum('aj,j,ji,hbai->hab', vecs[:,:,i,j], np.diag(fs[:,:,i,j]), vecs[:,:,i,j].conj().T, j2_matrix_full[:,:,:,:,i,j,i_,j_].conj())
                    M_1_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,ind_y,ind_x].conj().T, M_1_full, vecs[:,:,ind_y,ind_x])
                    j_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,ind_y,ind_x].conj().T, j_matrix[:,:,:,ind_y,ind_x], vecs[:,:,ind_y,ind_x])
                    
                    for m, omega in enumerate(omegas):
                        A = Spektralna_k(omega, mu, energije[:,i,j], Gamma)
                        transportna[0][0][m] += -2*np.trace(M_1_tilde[0] @ A @ j_tilde[0] @ A)
                        transportna[1][0][m] += -2*np.trace(M_1_tilde[1] @ A @ j_tilde[1] @ A)'''

    for i in range(Ny):
        for j in range(Nx):
            M_2_full = np.einsum('ijkl,jjkl,jikl,haib->hab', np.swapaxes(vecs.conj(), 0, 1), fs, vecs, j2_matrix_full[:,:,:,:,i,j]) + \
                    np.einsum('ijkl, jjkl, jikl, hbia->hab', np.swapaxes(vecs.conj(), 0, 1), fs, vecs, j2_matrix_full[:,:,:,:,i,j].conj())
            M_2_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,i,j].conj().T, M_2_full, vecs[:,:,i,j])
            j_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,i,j].conj().T, j_matrix[:,:,:,i,j], vecs[:,:,i,j])
            for m, omega in enumerate(omegas):
                A = Spektralna_k(omega, mu, energije[:,i,j], Gamma)
                transportna[0][m] += 4*np.trace(M_2_tilde[0] @ A @ j_tilde[0] @ A)
                transportna[1][m] += 4*np.trace(M_2_tilde[1] @ A @ j_tilde[1] @ A)

    '''M_3_full = np.einsum('ijmn,jjmn,jlmn,hlaimn->ha', vecs, fs, np.swapaxes(vecs.conj(), 0, 1), j2_matrix_full[:,:,:,:,:,:,Ny//2,Nx//2]) +\
            np.einsum('ijmn,jjmn,jlmn,hialmn->ha', vecs, fs, np.swapaxes(vecs.conj(), 0,1), j2_matrix_full[:,:,:,:,:,:,Ny//2,Nx//2].conj())
    for i in range(Ny):
        for j in range(Nx):
            M_3_tilde = np.einsum('ij,hj,jk->hik', vecs[:,:,i,j].conj().T, M_3_full, vecs[:,:,i,j])
            j_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,i,j].conj().T, j_matrix[:,:,:,i,j], vecs[:,:,i,j])
            for m, omega in enumerate(omegas):
                A = Spektralna_k(omega, mu, energije[:,i,j], Gamma)
                transportna[0][2][m] += 4*np.trace(M_3_tilde[0] @ A @ j_tilde[0] @ A)
                transportna[1][2][m] += 4*np.trace(M_3_tilde[1] @ A @ j_tilde[1] @ A)'''
                
    '''for i in range(Ny):
        for j in range(Nx):
            M_4_tilde_sum = np.zeros((2,6,6), dtype='complex')
            for i_ in range(Ny):
                for j_ in range(Nx):
                    ind_y = (i + i_ - Ny//2 ) % Ny
                    ind_x = (j + j_ - Nx//2) % Nx
                    
                    M_4_full = np.einsum('ij,ai,ii,hjab->hab', vecs[:,:,ind_y,ind_x].conj().T, vecs[:,:,ind_y,ind_x], fs[:,:,ind_y,ind_x], j2_matrix_full[:,:,:,:,i,j,i_,j_]) +\
                            np.einsum('ib,ji,ii,hjba->hab', vecs[:,:,ind_y,ind_x].conj().T, vecs[:,:,ind_y,ind_x], fs[:,:,ind_y,ind_x], j2_matrix_full[:,:,:,:,i,j,i_,j_].conj())
                    M_4_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,ind_y,ind_x].conj().T, M_4_full, vecs[:,:,ind_y,ind_x])
                    M_4_tilde_sum += M_4_tilde
            j_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,i,j].conj().T, j_matrix[:,:,:,i,j], vecs[:,:,i,j])
            for m, omega in enumerate(omegas):
                A = Spektralna_k(omega, mu, energije[:,i,j], Gamma)
                transportna[0][3][m] += -2*np.trace(M_4_tilde[0] @ A @ j_tilde[0] @ A)
                transportna[1][3][m] += -2*np.trace(M_4_tilde[1] @ A @ j_tilde[1] @ A)'''
    return omegas, transportna / 2 / (Ny*Nx)
    

def transportna_I_full_alternative(Kymesh, vecs, energije, fs, j2_matrix_full, j_matrix, mu, omega_max=1, domega=0.01, Gamma=0.05):
    Ny, Nx = Kymesh.shape
    omegas = np.arange(-omega_max, omega_max, domega)
    
    transportna = np.zeros((2, 4, omegas.shape[0]), dtype='complex') # xx, xy, yx, yy
    for i in range(Ny):
        for j in range(Nx):
            for i_ in range(Ny):
                for j_ in range(Nx):
                    ind_y = (i + i_ - Ny//2 ) % Ny
                    ind_x = (j + j_ - Nx//2) % Nx
                    M_1_full = np.einsum('ij,j,jb,habi->hab', vecs[:,:,i,j], np.diag(fs[:,:,i,j]), vecs[:,:,i,j].conj().T, j2_matrix_full[:,:,:,:,i,j,i_,j_]) + \
                        + np.einsum('aj,j,ji,hbai->hab', vecs[:,:,i,j], np.diag(fs[:,:,i,j]), vecs[:,:,i,j].conj().T, j2_matrix_full[:,:,:,:,i,j,i_,j_].conj())
                    M_1_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,ind_y,ind_x].conj().T, M_1_full, vecs[:,:,ind_y,ind_x])
                    j_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,ind_y,ind_x].conj().T, j_matrix[:,:,:,ind_y,ind_x], vecs[:,:,ind_y,ind_x])
                    for m, omega in enumerate(omegas):
                        A = Spektralna_k(omega, mu, energije[:,i,j], Gamma)
                        transportna[0][0][m] += -2*np.trace(M_1_tilde[0] @ A @ j_tilde[0] @ A)
                        transportna[1][0][m] += -2*np.trace(M_1_tilde[1] @ A @ j_tilde[1] @ A)

    for i in range(Ny):
        for j in range(Nx):
            M_2_full = np.einsum('ijkl,jjkl,jikl,haib->hab', vecs, fs, np.swapaxes(vecs.conj(), 0, 1), j2_matrix_full[:,:,:,:,i,j,Ny//2,Nx//2]) + \
                    np.einsum('ijkl, jjkl, jikl, hbia->hab', vecs, fs, np.swapaxes(vecs.conj(), 0, 1), j2_matrix_full[:,:,:,:,i,j,Ny//2,Nx//2].conj())
            M_2_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,i,j].conj().T, M_2_full, vecs[:,:,i,j])
            j_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,i,j].conj().T, j_matrix[:,:,:,i,j], vecs[:,:,i,j])
            for m, omega in enumerate(omegas):
                A = Spektralna_k(omega, mu, energije[:,i,j], Gamma)
                transportna[0][1][m] += 4*np.trace(M_2_tilde[0] @ A @ j_tilde[0] @ A)
                transportna[1][1][m] += 4*np.trace(M_2_tilde[1] @ A @ j_tilde[1] @ A)

    M_3_full = np.einsum('ijmn,limn,hjalmn->ha', np.swapaxes(vecs.conj(), 0, 1), vecs, j2_matrix_full[:,:,:,:,:,:,Ny//2,Nx//2]) +\
            np.einsum('ijmn,limn,hlajmn->ha', np.swapaxes(vecs.conj(), 0,1), vecs, j2_matrix_full[:,:,:,:,:,:,Ny//2,Nx//2].conj())
    for i in range(Ny):
        for j in range(Nx):
            M_3_tilde = np.einsum('ij,hj,jk->hik', vecs[:,:,i,j].conj().T, M_3_full, vecs[:,:,i,j])
            j_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,i,j].conj().T, j_matrix[:,:,:,i,j], vecs[:,:,i,j])
            for m, omega in enumerate(omegas):
                A = Spektralna_k(omega, mu, energije[:,i,j], Gamma)
                transportna[0][2][m] += 4*np.trace(M_3_tilde[0] @ A @ j_tilde[0] @ A)
                transportna[1][2][m] += 4*np.trace(M_3_tilde[1] @ A @ j_tilde[1] @ A)

    for i in range(Ny):
        for j in range(Nx):
            M_4_tilde_sum = np.zeros((2,6,6), dtype='complex')
            for i_ in range(Ny):
                for j_ in range(Nx):
                    ind_y = (i + i_ - Ny//2 ) % Ny
                    ind_x = (j + j_ - Nx//2) % Nx
                    
                    M_4_full = np.einsum('ij,ai,hjab->hab', vecs[:,:,ind_y,ind_x].conj().T, vecs[:,:,ind_y,ind_x], j2_matrix_full[:,:,:,:,i,j,i_,j_]) +\
                            np.einsum('ib,ji,hjba->hab', vecs[:,:,ind_y,ind_x].conj().T, vecs[:,:,ind_y,ind_x], j2_matrix_full[:,:,:,:,i,j,i_,j_].conj())
                    M_4_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,ind_y,ind_x].conj().T, M_4_full, vecs[:,:,ind_y,ind_x])
                    M_4_tilde_sum += M_4_tilde
            
            j_tilde = np.einsum('ij,hjk,kl->hil', vecs[:,:,i,j].conj().T, j_matrix[:,:,:,i,j], vecs[:,:,i,j])
            for m, omega in enumerate(omegas):
                A = Spektralna_k(omega, mu, energije[:,i,j], Gamma)
                transportna[0][3][m] += -2*np.trace(M_4_tilde_sum[0] @ A @ j_tilde[0] @ A)
                transportna[1][3][m] += -2*np.trace(M_4_tilde_sum[1] @ A @ j_tilde[1] @ A)
    return omegas, transportna / 2 / (Ny*Nx)


def L_Boltzmann(Kymesh, Kxmesh, energije, mu, T):
    energije_new = np.copy(energije)

    Ny, Nx = Kymesh.shape
    dKy, dKx = np.diff(Kymesh[:,0])[0], np.diff(Kxmesh[0])[0]
    velocity_y, velocity_x = np.zeros(energije.shape), np.zeros(energije.shape)
    velocity_new_y, velocity_new_x = np.copy(velocity_y), np.copy(velocity_x)

    for i in range(6):
        gr = np.gradient(energije[i])
        velocity_y[i, :, :], velocity_x[i, :, :] = gr[0]/dKy, gr[1]/dKx

    mask = np.zeros(energije[1].shape)
    ind1, ind2 = np.where(energije[1] - energije[0] < 0.005) 
    for i in range(len(ind1)):
        mask[ind1[i], ind2[i]] = 1
        energije_new[1,ind1[i], ind2[i]] = energije[0,ind1[i], ind2[i]]
        energije_new[0,ind1[i], ind2[i]] = energije[1,ind1[i], ind2[i]]
    for j in range(3,Nx//2):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[0] != 0:
                energije_new[1,:enice[-1],j] = energije[0,:enice[-1],j]
                energije_new[0,:enice[-1],j] = energije[1,:enice[-1],j]
    for j in range(Nx//2+1, Nx-5):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[-1] != 0:
                energije_new[1,enice[0]:,j] = energije[0,enice[0]:,j]
                energije_new[0,enice[0]:,j] = energije[1,enice[0]:,j]

    mask = np.zeros(energije[1].shape)
    ind1, ind2 = np.where(energije[3] - energije[2] < 0.001) 
    for i in range(len(ind1)):
        mask[ind1[i], ind2[i]] = 1
    for j in range(Nx//4,Nx//2):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[0] != 0:
                energije_new[3,:enice[-1],j] = energije[2,:enice[-1],j]
                energije_new[2,:enice[-1],j] = energije[3,:enice[-1],j]
    for j in range(Nx//2, Nx - Nx//4):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[-1] != 0:
                energije_new[3,enice[0]:,j] = energije[2,enice[0]:,j]
                energije_new[2,enice[0]:,j] = energije[3,enice[0]:,j]

    mask = np.zeros(energije[1].shape)
    ind1, ind2 = np.where(energije[5] - energije[4] < 0.001) 
    for i in range(len(ind1)):
        mask[ind1[i], ind2[i]] = 1
    for j in range(Nx//4,Nx//2):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[0] != 0:
                energije_new[5,:enice[-1],j] = energije[4,:enice[-1],j]
                energije_new[4,:enice[-1],j] = energije[5,:enice[-1],j]
    for j in range(Nx//2, Nx - Nx//4):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[-1] != 0:
                energije_new[5,enice[0]:,j] = energije[4,enice[0]:,j]
                energije_new[4,enice[0]:,j] = energije[5,enice[0]:,j]

    for i in range(6):
        gr = np.gradient(energije_new[i])
        velocity_new_y[i, :, :], velocity_new_x[i, :, :] = gr[0]/dKy, gr[1]/dKx

    L11 = np.zeros(2)
    L12 = np.copy(L11)
    L11_new, L12_new = np.copy(L11), np.copy(L11)
    for i in range(Ny):
        for j in range(Nx//2):
            for m in range(6):
                L11[0] += 2 * 2 * velocity_x[m,i,j]**2 * (-fd_1(energije[m,i,j] - mu, T) ) 
                L11[1] += 2 * 2 * velocity_y[m,i,j]**2 * (-fd_1(energije[m,i,j] - mu, T) )
                L12[0] += 2 * 2 * velocity_x[m,i,j]**2 * (energije[m,i,j] - mu) * (-fd_1(energije[m,i,j] - mu, T))
                L12[1] += 2 * 2 * velocity_y[m,i,j]**2 * (energije[m,i,j] - mu) * (-fd_1(energije[m,i,j] - mu, T))

                L11_new[0] += 2 * 2 * velocity_new_x[m,i,j]**2 * (-fd_1(energije_new[m,i,j] - mu, T) ) 
                L11_new[1] += 2 * 2 * velocity_new_y[m,i,j]**2 * (-fd_1(energije_new[m,i,j] - mu, T) )
                L12_new[0] += 2 * 2 * velocity_new_x[m,i,j]**2 * (energije_new[m,i,j] - mu) * (-fd_1(energije_new[m,i,j] - mu, T))
                L12_new[1] += 2 * 2 * velocity_new_y[m,i,j]**2 * (energije_new[m,i,j] - mu) * (-fd_1(energije_new[m,i,j] - mu, T))
    return L11, L12, L11_new, L12_new