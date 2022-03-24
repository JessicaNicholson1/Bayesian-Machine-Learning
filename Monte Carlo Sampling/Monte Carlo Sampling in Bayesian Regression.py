%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import mc_setup as mc
import random

# Parameters that determine the generated data

sig_gen = 0.2  # Standard deviation
s2_gen  = sig_gen**2  # Variance
r_gen   = 1  # Basis function width used to generate the data
x_max   = 10  # x-limit of the data

N_train = 30
N_test  = 250

# Parameters that determine the basis set used for modelling
# - note that the length scale "r" will be varied

M = 16 # Number of functions, spaced equally
centres = np.linspace(0, x_max, M)

"""
Generate Data
We synthesise two data sets:
a training set of size  𝑁=30  with added noise of std. dev.  𝜎=0.2 
a test set of size  𝑁=250  with no noise, representing the true function
The "input"  𝑥  data lies in the range  [0,10] .

Generate Basis
For our linear (in-the-parameters) model, we utilise a Gaussian RBF basis function set, where  𝜙𝑚(𝑥;𝑐𝑚,𝑟)=exp{−(𝑥−𝑐𝑚)2/𝑟2} . 
Each basis function has its own center; these are equally spaced and will remain fixed. 
The length scale, or width, 𝑟 is common across basis functions, and will be varied as part of the lab.
""""

# Generate training data
seed = 4
Data = lab2.DataGenerator(m=9, r=r_gen, noise=sig_gen, rand_offset=seed)
x_train, t_train = Data.get_data('TRAIN', N_train)
x_test, t_test = Data.get_data('TEST', N_test)

# Demonstrate use of basis
r = r_gen * 0.5  # Example model uses basis functions that are too narrow
RBF = lab2.RBFGenerator(centres, r) # centres was fixed earlier
PHI_train = RBF.evaluate(x_train)
PHI_test = RBF.evaluate(x_test)

# Find posterior mean for fixed guesses for alpha and s2
alph = 1e-12
s2 = 0.1**2
mu, _ = lab2.compute_posterior(PHI_train, t_train, alph, s2)
y_test = PHI_test @ mu

# Show the training data and generating function, plus our mean fit
lab2.plot_regression(x_train, t_train, x_test, t_test, y_test)
plt.title("Data, Underlying Function & Example Predictor")
pass 

""" Computing & Visualising the Hyperparameter Posterior """

# Compute the hyperparameter posterior
def log_prob_alph_r_given_t(alph, r, s2, x, t, centres):
        
    formula = lab2.RBFGenerator(centres, width = r)
    PHI     = formula.evaluate(x)
    
    first   = lab2.compute_log_marginal(PHI, t, alph, s2)
    
    
    return first
  
log_prob_alph_r_given_t(alph, r, s2, x_train, t_train, centres)


# Visualise the hyperparameter posterior and find its maximum
# Output should be a visualisation of p(alpha,r|t) and log p(alpha,r|t) 
# plus a print out of the most probable alpha and r values

l_alphas = np.linspace(-9,6, 250)
l_rs     = np.linspace(-2,2, 250)

alphas   = 10 ** l_alphas
rs       = 10 ** l_rs

empty = np.zeros((len(alphas), len(rs)))

for i, alph in enumerate(alphas):
    for j, r in enumerate(rs):
        empty[i][j] = log_prob_alph_r_given_t(alph, r, s2_gen, x_train, t_train, centres)
        
max_ar = np.where(empty == empty.max())
max_la = l_alphas[max_ar[0]]
max_lr = l_rs[max_ar[1]]  
max_a  = alphas[max_ar[0]]
max_r  = rs[max_ar[1]]

print("log maximum a:", max_la.sum())
print("log maximum r:", max_lr.sum())
print("maximum     a:", max_a.sum())
print("maximum     r:", max_r.sum()) 

plt.figure(figsize = (16, 6))

plt.subplot(1, 2,1)
plt.title(r"$\log\ p(\mathbf{t}|\alpha,r)$", fontsize = 13)
plt.contourf(l_alphas, l_rs, empty.T)
plt.xlabel(r"log $\alpha$", fontsize = 15)
plt.ylabel("log r", fontsize = 15)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title(r"$p(\mathbf{t}|\alpha,r)$", fontsize = 13)
plt.contourf(l_alphas, l_rs, np.exp(empty.T))
plt.xlabel(r"log $\alpha$",fontsize = 15)
plt.colorbar()
plt.show()

"""Importance Sampling
This task focuses on implementing the importance sampling algorithm, 
and then applying it to find an expected value for basis function width (r)  
"""

# Importance sampler function

def importance(num_samples, log_pstar, log_qstar, qrvs, fun):
    
    samples = qrvs(num_samples)
    log_p   = np.array([log_pstar(sample) for sample in samples])
    log_q   = np.array([log_qstar(sample) for sample in samples])
    
    log_w   = (log_p - log_q)[:,None]
    w       = np.exp(log_w)
    wf      = w * fun(samples)
    
    return wf.cumsum(axis = 0) / w.cumsum(axis = 0)
  
# Test the importance sampler on three proposal distributions
# For each of the three different proposal distributions: 
# - chart the convergence of the expectation
# - print out the final expectation <r>

import random
np.random.seed(0)
random.seed(0)

log_pstar = lambda x: log_prob_alph_r_given_t(10**x[0], 10**x[1], s2_gen, x_train, t_train, centres)
fun       = lambda x: 10**x

#uniform  𝑄(log𝛼,log𝑟)
uniform_log_qstar = lambda x: stats.uniform(-9, 15).logpdf(x[0]) + stats.uniform(-2, 4).logpdf(x[1])
uniform_qrvs      = lambda x: np.c_[stats.uniform(-9, 15).rvs(x), stats.uniform(-2, 4).rvs(x)]

#Gaussian centred at the midpoint of the visualisation
gauss_cent_log_qstar = lambda x: stats.multivariate_normal.logpdf(x, (-1.5,0), (2.5**2, (4/6)**2))
gauss_cent_qrvs      = lambda x: np.c_[stats.multivariate_normal((-1.5,0), (2.5**2, (4/6)**2)).rvs(x)]

#Gaussian located at the maximum point 
gauss_max_log_qstar = lambda x: stats.multivariate_normal.logpdf(x, (max_la.sum(),max_lr.sum()), (2**2, 0.2**2))
gauss_max_qrvs      = lambda x: np.c_[stats.multivariate_normal((max_la.sum(),max_lr.sum()), (2**2, 0.2**2)).rvs(x)]

#results
N            = 5000
uniform      = importance(N, log_pstar, uniform_log_qstar,   uniform_qrvs,   fun)
gauss_centre = importance(N, log_pstar, gauss_cent_log_qstar,gauss_cent_qrvs,fun)
gauss_max    = importance(N, log_pstar, gauss_max_log_qstar, gauss_max_qrvs, fun)

print(f"Uniform final expectation         <r> : {uniform[-1][-1]:6.2f}")
print(f"Gaussian Center final expectation <r> : {gauss_centre[-1][-1]:6.2f}")
print(f"Gaussian Max final expectation    <r> : {gauss_max[-1][-1]:6.2f}")


import matplotlib.pyplot as plt

titles  = ["Uniform Expectation", "Gaussian Centre Expectation", "Gaussian Max Expectation"]
samples = [uniform, gauss_centre, gauss_max]
fig, ax = plt.subplots(1, 3, figsize = (16, 4), sharey = True)

for i, (title, sample) in enumerate(zip(titles, samples)):
    ax[i].set_title(title, size = 13)
    ax[i].set_xlabel("Samples", size = 15)
    if i == 0: ax[i].set_ylabel("log r", size = 15)
    ax[i].plot(sample[5:])
    
# Compare the convergence of the three samplers above by assessing the results over multiple runs
# Output should be a graph showing the variance of the Expectations against sample number, 
# as computed over multiple repetitions, with all three proposals on the same axis.

rep = 100
N   = 1000
uniform2      = np.array([importance(N, log_pstar, uniform_log_qstar,   uniform_qrvs,   fun) for _ in range(rep)])
gauss_centre2 = np.array([importance(N, log_pstar, gauss_cent_log_qstar,gauss_cent_qrvs,fun) for _ in range(rep)])
gauss_max2    = np.array([importance(N, log_pstar, gauss_max_log_qstar, gauss_max_qrvs, fun) for _ in range(rep)])

titles  = ["Uniform Expectation", "Gaussian Centre Expectation", "Gaussian Max Expectation"]
samples = [uniform2, gauss_centre2, gauss_max2]
fig, ax = plt.subplots(1, 3, figsize = (16, 4), sharey = True)

for i, (title, sample) in enumerate(zip(titles, samples)):
    ax[i].set_title(title, size = 13)
    ax[i].set_xlabel("Samples", size = 15)
    if i == 0: ax[i].set_ylabel("Variance", size = 15)
    ax[i].semilogy(np.var(sample,0))

""" MCMC & The Metropolis Algorithm """"

def metropolis(num_samples, pstar, qrvs, x0):
    pts      = x0.copy()
    rejected = []
    samples  = [pts]
    
    for i in range(num_samples - 1):
        pn = qrvs(pts)
        
        val = np.exp(pstar(pn) - pstar(pts))
        
        if val >= 1:
            pts    = pn
        else:
            u = np.random.uniform()
            if u < val:
                pts    = pn
            else:
                rejected.append(pn)
        samples.append(pts)
            
    return rejected, samples
  
# Apply the Metropolis sampler to compute  ⟨𝑟⟩  again

# For each of the three proposals, the output should be
# - print out of the length-scale used for the proposal
# - print out of the acceptance rate
# - plot of convergence: <r> against sample number
# - print out of the final value of <r>
# - replot the previous visualisation, and overlay the samples (as a scatter plot on top)

np.random.seed(2)
random.seed

log_pstar = lambda x: log_prob_alph_r_given_t(10**x[0], 10**x[1], s2_gen, x_train, t_train, centres)

#Gaussian 1
"""one too narrow, which has a very high acceptance rate (anything over 80%), and which "random walks" """
gauss_qrvs_1 = lambda x: stats.multivariate_normal.rvs(size=1, mean=x, cov=[0.001] * 2)

#Gaussian 2
"""one too broad, which has a low acceptance rate (say, around 5%)"""
gauss_qrvs_2 = lambda x: stats.multivariate_normal.rvs(size=1, mean=x, cov=[0.6] * 2)

#Gaussian 3
"""one "just right", with acceptance rate between 20% and 30%"""
gauss_qrvs_3 = lambda x: stats.multivariate_normal.rvs(size=1, mean=x, cov=[0.09] * 2)

#calculate acceptance rates & expected value for each gaussian, plot graphs
"""acceptance = (num_samples-len(rejected)) / num_samples
   expected value of  ⟨𝑟⟩  under the posterior  𝑝(𝛼,𝑟|𝐭)"""

N  = 5000
x0 = np.zeros(2)

gaussians = [gauss_qrvs_1, gauss_qrvs_2, gauss_qrvs_3]
lengths   = ["0.01", "0.5", "0.09"]
titles    = ["Gaussian Narrow", "Gaussian Broad", "Gaussian Just Right"]
     
    
for gauss, leng in zip(gaussians, lengths):
    reject, samples = metropolis(N, log_pstar, gauss, x0)
    accept  = (N - len(reject)) / N
    
    burn_out      = int(0.05* len(samples))
    expectation   = 10**(np.array(samples)[burn_out:].cumsum(0) / np.arange(1, N - burn_out +1)[:,None])
    
    print("------------------------------")
    print(f"Gaussian Acceptance Rate: { accept:6.2f}, Lengthscale: {leng}")
    print(f"final expectation   <r> : { expectation[1][-1]:6.2f}")
    print("------------------------------")
        
## plot the usual graph showing convergence: i.e. a graph of  ⟨𝑟⟩  with increasing sample number
plt.figure(figsize = (9,7))
plt.title("Final Expectation (r)", fontsize = 13)
plt.plot(expectation[:,0])
plt.ylabel("log r", fontsize = 15)
plt.xlabel("Number of Runs", fontsize = 15)
plt.show()

## re-create the earlier visualisation of  𝑝(𝛼,𝑟|𝐭)  and overlay the samples. 
plt.figure(figsize = (10,8))
plt.title(r"$\log\ p(\mathbf{t}|\alpha,r)$ with Samples", fontsize = 13)
plt.contourf(l_alphas, l_rs, empty.T)
plt.scatter(np.array(samples)[:,0],np.array(samples)[:,1], s = 0.1, c = 'r')
plt.xlabel(r"log $\alpha$", fontsize = 15)
plt.ylabel("log r", fontsize = 15)
plt.colorbar()
plt.show()

# Extend your Metropolis sampler to estimate the noise variance  𝜎2
# Output should be
# - print out of the length-scale used for the proposal
# - print out of the acceptance rate
# - plot of convergence: <s2> against sample number
# - print out of the final estimate of noise standard deviation

np.random.seed(2)
random.seed

log_pstar2 = lambda x: log_prob_alph_r_given_t(10**x[0], 10**x[1], 10**x[2], x_train, t_train, centres)

#Gaussian 1
"""one too narrow, which has a very high acceptance rate (anything over 80%), and which "random walks" """
gauss_qrvs_1 = lambda x: stats.multivariate_normal.rvs(size=1, mean=x, cov=[0.001] * 3)

#Gaussian 2
"""one too broad, which has a low acceptance rate (say, around 5%)"""
gauss_qrvs_2 = lambda x: stats.multivariate_normal.rvs(size=1, mean=x, cov=[0.6] * 3)

#Gaussian 3
"""one "just right", with acceptance rate between 20% and 30%"""
gauss_qrvs_3 = lambda x: stats.multivariate_normal.rvs(size=1, mean=x, cov=[0.09] * 3)

#calculate acceptance rates & expected value for each gaussian, plot graphs
"""acceptance = (num_samples-len(rejected)) / num_samples
   expected value of  ⟨𝑟⟩  under the posterior  𝑝(𝛼,𝑟|𝐭)"""

N  = 10000
x0 = np.zeros(3)

gaussians = [gauss_qrvs_1, gauss_qrvs_2, gauss_qrvs_3]
lengths   = ["0.01", "0.5", "0.09"]
titles    = ["Gaussian Narrow", "Gaussian Broad", "Gaussian Just Right"]
    
for gauss, leng in zip(gaussians, lengths):
    reject, samples = metropolis(N, log_pstar2, gauss, x0)
    accept  = (N - len(reject)) / N
    
    burn_out      = int(0.05* len(samples))
    expectation   = 10**(np.array(samples)[burn_out:].cumsum(0) / np.arange(1, N - burn_out +1)[:,None])
    
print("------------------------------")
print(f"Lengthscale:", leng)
print(f"Acceptance Rate: {accept:6f}")
print(f"Noise Standard Deviation: {np.sqrt(expectation[:,2][-1]):6f}")
print("------------------------------")
        
## plot the usual graph showing convergence: i.e. a graph of  ⟨𝑟⟩  with increasing sample number
plt.figure(figsize = (9,7))
plt.title("convergence <s2>", fontsize = 13)
plt.plot(expectation[:,1])
plt.ylabel("log r", fontsize = 15)
plt.xlabel("Number of Runs", fontsize = 15)
plt.show()

""" Extend the Metropolis algorithm to sample over all the unknowns in the model: that is,
not just 𝛼, 𝜎, 𝑟 but also the weights 𝐰. This is slightly artificial, in that we don't really 
need to sample the weights in this model (the posterior over 𝐰 is analytically computable), 
but it demonstates the principle. In a neural network model (which will be nonlinear in 𝐰 ), 
we will need to sample"""

# Be "strictly Bayesian" and sample all the model unknowns to derive the mean predictor


N      = 10000 
qrvs   = lambda x: stats.multivariate_normal.rvs(size=1, mean=x, cov=[0.001] * 19)
values = np.log10(np.random.uniform(0,5,19))

def logpstar(values):

    a  = 10 ** values[0]
    r  = 10 ** values[1]
    s2 = 10 ** values[2]
    
    w = values[3:]
    N, M = x_train.shape
    
    formula   = lab2.RBFGenerator(centres, width = r)
    PHI_train = formula.evaluate(x_train)
    y_pred    = PHI_train @ w

    first   = -N/2 * (np.log(2 * np.pi * s2)) - ((0.5 * s2)*(t_train - y_pred)**2).sum()
    second  = -M/2 * (np.log(2 * np.pi / a)) - ((a/2) * (w**2)).sum()
    
    final = first + second 
    
    return final 
    
def metropolis2(num_samples, pstar, qrvs, x0):
    pts      = x0.copy()
    rejected = []
    samples  = [pts]
    
    for i in range(num_samples - 1):
        pn    = qrvs(pts)
        
        val   = np.exp(pstar(pn) - pstar(pts))
        
        if val >= 1:
            pts    = pn
        else:
            u = np.random.uniform()
            if u < val:
                pts    = pn
            else:
                rejected.append(pn)
        samples.append(pts)
            
    return rejected, samples

rejected,samples = metropolis2(N, log_pstar, qrvs, values)

y_pred = []

for sample in samples:
    log_params = sample[:3]
    w          = sample[3:]
    params     = 10**log_params
    
    RBF      = lab2.RBFGenerator(centres, params[1])
    PHI_test = RBF.evaluate(x_test)
    
    y_pred.append(PHI_test @ w)
    
y_test = np.mean(y_pred, axis = 0)


lab2.plot_regression(x_train, t_train, x_test, t_test, y_test)
plt.title("Mean Predictor")
plt.show()
