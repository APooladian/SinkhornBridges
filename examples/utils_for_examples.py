import numpy as np
import ot

def sinkhorn_potentials(source,target,eps,dim,n,iters_max=None):
    a, b = np.ones((n,)) / n, np.ones((n,)) / n 
    M = ot.dist(source.T, target.T)/2.
    _,log_ = ot.sinkhorn(a, b, M, eps,method='sinkhorn_log',log=True, stopThr=5e-4,
                                            numItermax=iters_max,print_period=50)
    logv_opt = log_['log_v']
    logu_opt = log_['log_u']
    return logu_opt, logv_opt

class ent_drift():
    def __init__(self, data, pot, eps):
        self.data = data
        self.pot = pot
        self.eps = eps

    def estimator(self,x,t):
        M = ot.dist(x,self.data)/(2*t)
        K = -M/self.eps + self.pot
        gammaz = -np.max(K,axis=1)
        K_shift = K + gammaz.reshape(-1,1)
        exp_ = np.exp(K_shift)
        top_ = exp_ @ self.data
        bot_ = exp_.sum(axis=1)
        entmap = top_/bot_.reshape(-1,1)

        return (-x + entmap)/(t)
    
    def __call__(self, x, t):
        return self.estimator(x, t)
