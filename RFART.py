import numpy as np
import collections
    

class RFART:
    def __init__(self, alpha = 0, beta = 1.0, rho = 0.8, thr = 0.9, delta = 1.01 , gamma = 0.95):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.thr = thr
        self.delta = delta
        self.gamma = gamma
        
        self.W = np.zeros(0)
        self.L = np.zeros(0)
        self.M = 0
        
        
    def fit(self, X, y):
        if X.shape[0] == 0 or X.shape[0] == 0:
            return
        
        self.M = X.shape[1]
        U = self.complement(X)
        R = self.train(U, y, self.rho, self.thr)
        
        
        self.W = np.array(list(R.keys()))
        self.L = np.array(list(R.values()))
        
    
    def predict(self, X, rho = None):
        if X.shape[0] == 0 or X.shape[0] == 0:
            return np.zeros(0)
        
        rho = rho if rho else self.rho
        
        N = X.shape[0]
        U = self.complement(X)
        
        t = np.zeros(N)
        
        W_norm = self.norm(self.W)
        J = self.W.shape[0]
        
        h = collections.Counter(self.L)
        K = int(max(h.keys())) + 1
        
        for k, times in h.items():
            h[k] = float(times) / self.L.shape[0]
    
        
        for i in range(N):
            v_inter = self.inter(U[i], self.W)
            T = self.choice(v_inter, W_norm)

            if max(T) >= rho:
                j = np.argmax(T)
                t[i] = self.L[j]
            else:
                v = np.zeros(K)
                for j in range(J):
                    v[int(self.L[j])] += T[j] * h[self.L[j]]
                t[i] = np.argmax(v)
        
        return t
        
    
    def train(self, U, y, rho = 1, thr = 0):
        W, c = self.fart(U, rho)

        J = W.shape[0]
        
        R = {}
        
        for j in range(J):
            idxs = np.where(c == j)[0]
            if idxs.shape[0] == 0:
                continue
            
            U_j = U[idxs]
            y_j = y[idxs]
            
            p, l = self.purity(y_j)
            if p < thr:
                R_tmp = self.train(U_j, y_j, min(1, rho * self.delta), thr * self.gamma)
                R = {**R, **R_tmp}

            else:
                R[tuple(W[j])]=l 
        return R
    
            
    def purity(self, y):
        cnt = collections.Counter(y)
        return cnt.most_common(1)[0][1] / float(y.shape[0]), cnt.most_common(1)[0][0]
    
    
    def fart(self, U, rho = 1):
        N = U.shape[0]
        
        W = U[0].reshape(1, U.shape[1])
        W_norm = self.norm(W)
        
        
        c = np.zeros(N)
        
        for i in range(N):
            v_inter = self.inter(U[i], W)
            T = self.choice(v_inter, W_norm)

            j = np.argmax(T)

            if T[j] >= rho:
                W[j] = self.beta * v_inter[j] + (1 - self.beta) * W[j]
                W_norm[j] = self.norm(W[j])
                c[i] = j
            else:
                W = np.vstack([W, U[i]])
                W_norm = np.append(W_norm, self.M)
                c[i] = W.shape[0] - 1
                
        return W, c
        
        
        
    def norm(self, X):
        return sum(X.T)
    
    def inter(self, x, W):
        return np.minimum(x, W)
    
    
    def choice(self, v_inter, W_norm):
        return self.norm(v_inter) / (self.alpha + W_norm)
    
    
    def complement(self, X):
        
        X_norm = self.norm(X)
        idxs = np.where(X_norm == 0)
        X_norm[idxs] = 1
                
        B = X / X_norm.reshape(X.shape[0],1)
        
        
        U = np.hstack((X, 1-X))
        
        return U
        
        

        
