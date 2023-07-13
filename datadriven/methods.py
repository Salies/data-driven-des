import numpy as np

class SINDy:
    def __init__(self, x, dx, candidates):
        self.candidates = candidates
        self.dx = dx
        # Construindo theta
        qtd_candidates = len(candidates)
        m = x.shape[0]
        self.theta = np.zeros((m, qtd_candidates))
        for i, candidate in enumerate(candidates):
            self.theta[:,i] = candidate(*x.T)

    def sparsify(self, l, n_iter):
        n = self.dx.shape[1]
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

    def model(self):
        # A quantidade de argumentos na função retornada é a qtd de colunas de xi
        # (ou seja, a quantidade de colunas da matriz de entrada)
        func_array = []
        for i in range(self.xi.shape[1]):
            idx_not_sparse = np.nonzero(self.xi[:,i])[0]
            coeffs = self.xi[idx_not_sparse,i]
            funcs = [self.candidates[j] for j in idx_not_sparse]
            print(coeffs, funcs)
            # constói uma função que retorna a soma dos termos
            #func = lambda x: sum([coeffs[k]*funcs[k](*x.T) for k in range(len(coeffs))])
            #print(func(np.array([1.0,0.5,0.2])))
            #func_array.append(lambda x: sum((coeffs[k]*funcs[k](*x.T) for k in range(len(coeffs)))))
            def func(x):
                return lambda x: sum((coeffs[k]*funcs[k](*x.T) for k in range(len(coeffs))))
            func_array.append(func)
        print(len(func_array))
        print(func_array[2]()(np.array([1.0,0.5,0.2])))
        # Criando uma função base
        self.func_array = func_array
        final_func = lambda x, t0: np.array((func_array[i](x) for i in range(len(func_array)))).T
        return final_func