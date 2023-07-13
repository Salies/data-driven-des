import numpy as np

class SINDy:
    def __init__(self, x, dx, candidates):
        self.candidates = candidates
        self.dx = dx
        # Construindo theta
        # Cada coluna representa uma função candidata
        # Cada linha é uma entrada de dados
        qtd_candidates = len(candidates)
        m = x.shape[0]
        self.theta = np.empty((m, qtd_candidates))
        for i, candidate in enumerate(candidates):
            self.theta[:,i] = candidate(*x.T)

    def sparsify(self, l, n_iter):
        n = self.dx.shape[1]
        # Tenta encontrar os n_candidatas * n_col_dados possíveis coeficientes que melhor se ajustam aos dados
        Xi = np.linalg.lstsq(self.theta, self.dx, rcond=None)[0]

        for _ in range(n_iter):
            # Esparsifica coeficientes pequenos
            # Lambda é o parâmetro de esparsificação
            # (aqui chamado de l pois lambda é palavra reservada)
            # A esparsificação, nos exemplos mais simples (Lorenz, SIR) serve apenas para descartar
            # coeficientes que de fato não fazem parte da dinâmica. Em casos mais complexos,
            # a esparsificação ajuda a descartar coeficientes que pouco influenciam na dinâmica,
            # melhorando a eficiência computacional do modelo.
            smallinds = np.abs(Xi) < l # Encontra os coeficientes pequenos
            Xi[smallinds] = 0 # Esparsifica-os
            # Para cada dimensão do sistema
            for i in range(n):
                # Pega os índices não-pequenos
                biginds = smallinds[:,i] == False
                # Atualiza Xi apenas com os coeficientes que não foram esparsificados
                Xi[biginds,i] = np.linalg.lstsq(self.theta[:,biginds], self.dx[:,i],rcond=None)[0]
                
        self.xi = Xi
        self.not_sparse_idx = [np.nonzero(self.xi[:,i])[0] for i in range(n)]
        # Define uma função para calcular a saída baseada nos coeficientes esparsificados
        # e nas funções candidatas equivalentes.
        def handle_input(x):
            res = np.zeros(x.shape)
            for i in range(n):
                for j in self.not_sparse_idx[i]:
                    res[i] += self.xi[j,i] * self.candidates[j](*x.T)
            return res
        self.handle_input = handle_input

    def model(self, x, t0):
        return self.handle_input(x)
    
class DMD:
    def __init__(self, X, X_shifted):
        # implementar o dmd resumidamente aquio
        pass