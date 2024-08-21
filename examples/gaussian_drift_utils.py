def gen_gaussian_samples(mean,cov,Nsamples):
    return np.random.multivariate_normal(mean, cov, size=Nsamples).T

def Depsilon(A,B,eps):
    dim = A.shape[0]
    Asqrt = sqrtm(A)
    return sqrtm((Asqrt @ B @ Asqrt) + ((eps**2)/4)*np.eye(dim))

def Cepsilon(A,B,eps):
    dim = A.shape[0]
    Deps = Depsilon(A,B,eps)
    Asqrt = sqrtm(A)
    Asqrtinv = inv(Asqrt)
    return (Asqrt @ Deps @ Asqrtinv) - (eps/2)*np.eye(dim)

def Sigma_t(A,B,eps,Ceps,t):
    return ((1-t)**2)*A + (t**2)*B + (1-t)*t*(Ceps + Ceps.T) + eps*t*(1-t)*np.eye(dim)

def drift_t(A,B,eps,t):
    dim = A.shape[0]
    Ceps = Cepsilon(A,B,eps)
    Σ_t = Sigma_t(A,B,eps,Ceps,t)
    Pt = t*B + (1-t)*Ceps
    Qt = (1-t)*A + t*Ceps
    St = Pt - Qt.T - (eps*t)*np.eye(dim)
    return St.T @ inv(Σ_t)
