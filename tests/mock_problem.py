"""Analytic stand-in for the laminate hierarchy: same interface, vector QoI,
per-level bias that shrinks with 'refinement'. No dolfinx needed."""
import numpy as np, scipy.stats as stats, tinyDA as tda

class MockLaminate:
    """Cheap surrogate for LinearElasticity with a 3-component QoI."""
    def __init__(self, level):
        self.bias = 0.4 * 0.25 ** level          # discretisation error, shrinks with level
    def __call__(self, p):
        E1, E2 = p
        # 'displacement' observations: smooth, monotone in 1/E
        obs = np.array([1.0/E1, 1.0/E2, 0.5/E1 + 0.5/E2]) * 80.0 + self.bias*0.01
        qoi = np.array([
            -2.4 / E1 - 1.1 / E2 + self.bias * 0.02,   # tip deflection
            6.0 + 0.01 * E1 + 0.004 * E2 + self.bias,  # mean von Mises
            0.09 / E1 + 0.04 / E2 + self.bias * 0.001, # compliance
        ])
        return obs, qoi

TRUE = np.array([80.0, 40.0]); SD = 0.005

def build_problem(n_levels=3, seed=678):
    st = np.random.get_state(); np.random.seed(seed)
    try:
        levels = [MockLaminate(i) for i in range(n_levels)]
        x_true, qoi_true = MockLaminate(5)(TRUE)
        data = x_true + np.random.normal(scale=SD, size=x_true.shape[0])
        prior = tda.JointPrior([stats.uniform(loc=50, scale=50),
                                stats.uniform(loc=0.1, scale=50)])
    finally:
        np.random.set_state(st)
    class P: pass
    pr = P(); pr.levels=levels; pr.data=data; pr.prior=prior
    pr.true_parameters=TRUE; pr.noise_sd=SD; pr.qoi_true=qoi_true
    def posteriors(idx, adaptive_coarse=True):
        cov = SD**2*np.eye(data.size); out=[]
        for pos,i in enumerate(idx):
            fin = pos==len(idx)-1
            ll = tda.GaussianLogLike(data,cov) if (fin or not adaptive_coarse) else tda.AdaptiveGaussianLogLike(data,cov)
            out.append(tda.Posterior(prior, ll, levels[i]))
        return out
    pr.posteriors = posteriors
    pr.map_estimate = tda.get_MAP(posteriors([0], adaptive_coarse=False)[0])
    return pr
