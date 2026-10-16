"""Problem setup for the Fig. 9 runs: cells 1-12 of 02_predator_prey_variance_reduction.ipynb.

The code below is copied from the notebook unchanged (only the data plot, cell 8, is left out), so the
model, data, priors, posteriors and MAP points are exactly the notebook's. Importing this module runs it:

    from fig9_setup import *

The data are generated under np.random.seed(987) as in the notebook, so every script and every array task
sees the same data and the same posteriors. Seed the sampler separately afterwards.
"""

# --- notebook cell 1: imports ---------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import arviz as az

from scipy.integrate import solve_ivp

import tinyDA as tda


# --- notebook cell 2: seed for the data -----------------------------------------------------

np.random.seed(987)


# --- notebook cell 3: the original model ----------------------------------------------------

class PredatorPreyModel: 
    def __init__(self, datapoints):
        
        # set the timesteps, where we are collecting the model output
        self.datapoints = datapoints
        
        # set the span of the integration.
        self.t_span = [0, self.datapoints[-1]]
        
    def dydx(self, t, y, a, b, c, d):
        # Lotka-Volterra Model model, see e.g. https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
        return np.array([a*y[0] - b*y[0]*y[1], c*y[0]*y[1] - d*y[1]])

    def __call__(self, parameters):
        
        # extract the parameters, and take the exponential to keep them positive
        P_0, Q_0, a, b, c, d = np.exp(parameters)
        
        # solve the initial value problem.
        self.y = solve_ivp(lambda t, y: self.dydx(t, y, a, b, c, d), self.t_span, np.array([P_0, Q_0]), t_eval=self.datapoints) 
        
        # return the results, only if the integration succeeded.
        if self.y.success:
            return self.y.y.flatten(), self.y.y[1,:].mean()
        else:
            return np.nan, np.nan         # (model output, QoI)


# --- notebook cell 4: two free parameters ---------------------------------------------------

class ReducedPredatorPreyModel(PredatorPreyModel):
    def __init__(self, datapoints, free, fixed_parameters):
        super().__init__(datapoints)

        # which of the six (log-)parameters are sampled, and the values of the others
        self.free = free
        self.fixed_parameters = fixed_parameters

    def __call__(self, parameters):
        full_parameters = self.fixed_parameters.copy()
        full_parameters[self.free] = parameters
        return super().__call__(full_parameters)


# --- notebook cell 5: true parameters -------------------------------------------------------

names = ['N_0', 'P_0', 'a', 'b', 'c', 'd']

# set the true parameters
P_0 = 10
Q_0 = 5
a = 3.0
b = 0.7
c = 0.2
d = 1.0

# collect the parameters in a vector and take the logarithm.
# we sample the log of the parameters and take the exponential
# inside the model to keep the parameters positive.
true_parameters_all = np.log(np.array([P_0, Q_0, a, b, c, d]))

# which parameters are sampled; here the predation parameters a and b
free = [2, 3]
true_parameters = true_parameters_all[free]

# set the integration points.
t_span = [0,12]
n_eval = 1000
t_eval = np.linspace(t_span[0], t_span[1], n_eval)

# initalise the true model and solve it
my_model = PredatorPreyModel(t_eval)
y_true = my_model(true_parameters_all)[0]
Q_true = my_model(true_parameters_all)[1]


# --- notebook cell 6: forward models --------------------------------------------------------

# fine model
n_data_l2 = 25 # number of datapoints
t_eval_l2 = np.linspace(t_span[0], t_span[1], n_data_l2) # datapoints
my_model_l2 = ReducedPredatorPreyModel(t_eval_l2, free, true_parameters_all) # initialise model

# coarse model (stops integrating early, at t = 8)
n_data_l1 = 17 # number of datapoints
t_eval_l1 = t_eval_l2[:n_data_l1] # datapoints (the first 17 of the fine model)
my_model_l1 = ReducedPredatorPreyModel(t_eval_l1, free, true_parameters_all) # initialise model

# coarse model (stops integrating even earlier, at t = 4)
n_data_l0 = 9 # number of datapoints
t_eval_l0 = t_eval_l2[:n_data_l0] # datapoints (the first 9 of the fine model)
my_model_l0 = ReducedPredatorPreyModel(t_eval_l0, free, true_parameters_all) # initialise model


# --- notebook cell 7: data ------------------------------------------------------------------

# set the noise level
sigma = 1.0

noise_l2 = np.random.normal(scale=sigma, size=(t_eval_l2.size,2)) # fine noise
data_l2 = my_model_l2(true_parameters)[0] + np.hstack((noise_l2[:,0], noise_l2[:,1])) # noisy fine data.

noise_l1 = np.hstack((noise_l2[:n_data_l1,0], noise_l2[:n_data_l1,1])) # coarse noise
data_l1 = my_model_l1(true_parameters)[0] + noise_l1 # noisy coarse data

noise_l0 = np.hstack((noise_l2[:n_data_l0,0], noise_l2[:n_data_l0,1])) # coarse noise
data_l0 = my_model_l0(true_parameters)[0] + noise_l0 # noisy coarse data


# --- notebook cell 9: priors ----------------------------------------------------------------

class LogTransformed:
    # the distribution of log(X), for a positive random variable X ~ dist
    def __init__(self, dist):
        self.dist = dist

    def logpdf(self, x):
        return self.dist.logpdf(np.exp(x)) + x

    def rvs(self, size=None):
        return np.log(self.dist.rvs(size=size))

# priors from the paper, on the parameters themselves
priors_all = [stats.norm(10.8, 1),
              stats.norm(5.3, 1),
              stats.norm(2.5, 0.5),
              stats.invgamma(1, scale=0.5),
              stats.invgamma(1, scale=0.5),
              stats.norm(1.2, 0.3)]

my_prior = tda.JointPrior([LogTransformed(priors_all[i]) for i in free])


# --- notebook cell 10: likelihoods -----------------------------------------------------------

# define the likelihood
cov_likelihood_l2 = sigma**2*np.eye(data_l2.size)
cov_likelihood_l1 = sigma**2*np.eye(data_l1.size)
cov_likelihood_l0 = sigma**2*np.eye(data_l0.size)

my_loglike_l2 = tda.GaussianLogLike(data_l2, cov_likelihood_l2)
my_loglike_l1 = tda.GaussianLogLike(data_l1, cov_likelihood_l1)
my_loglike_l0 = tda.GaussianLogLike(data_l0, cov_likelihood_l0)


# --- notebook cell 11: posteriors ------------------------------------------------------------

# set up the link factories
my_posterior_l2 = tda.Posterior(my_prior, my_loglike_l2, my_model_l2)
my_posterior_l1 = tda.Posterior(my_prior, my_loglike_l1, my_model_l1)
my_posterior_l0 = tda.Posterior(my_prior, my_loglike_l0, my_model_l0)

my_posteriors = [my_posterior_l0, my_posterior_l1, my_posterior_l2]


# --- notebook cell 12: MAP points ------------------------------------------------------------

# get the maximum a posteriori point on each level, starting from the true parameters so the result is
# the same every time
MAP = [tda.get_MAP(my_posteriors[l], initial_parameters=true_parameters, method='Nelder-Mead',
                   options={'xatol': 1e-8, 'fatol': 1e-10}) for l in range(3)]
for l in range(3):
    print(f'level {l} MAP: ' + '  '.join(f'{names[i]}={v:.4f}' for i, v in zip(free, np.exp(MAP[l]))))
