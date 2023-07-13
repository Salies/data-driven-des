import numpy as np
def DMD(data, r):
    """Dynamic Mode Decomposition (DMD) algorithm."""
    
    ## Build data matrices
    X1 = data[:, : -1]
    X2 = data[:, 1 :]
    ## Perform singular value decomposition on X1
    u, s, v = np.linalg.svd(X1, full_matrices = False)
    ## Compute the Koopman matrix
    A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * np.reciprocal(s[: r])
    ## Perform eigenvalue decomposition on A_tilde
    Phi, Q = np.linalg.eig(A_tilde)
    ## Compute the coefficient matrix
    Psi = X2 @ v[: r, :].conj().T @ np.diag(np.reciprocal(s[: r])) @ Q
    #A = Psi @ np.diag(Phi) @ np.linalg.pinv(Psi)
    
    return A_tilde, Phi, Psi

x = np.linspace( -10, 10, 200)
t = np.linspace(0, 4 * np.pi , 61)
dt = t[1] - t[0]
S, T = np.meshgrid(x, t)
# Criando os modos para testar o potencial de captura do DMD
# Essa primeira função é puramente espacial e real
X1 = 0.8 * np.sin(S) * (1 + 0 * T)
# Essa segunda função, por sua vez, é espaço-temporal e complexa
X2 = 1.0 / np.cosh(S + 3) * np.exp(2.3j * T)
# Nosso sinal de teste é a soma dessas duas funções
# Testaremos então a capacidade do DMD de capturar as duas funções
X = X1 + X2

A_ti, Phi, A = DMD(X, 2)
print(A_ti.shape)
print(Phi.shape)
print(A.shape)