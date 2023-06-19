import math
import numpy as np
from scipy.optimize import brent

# Implementation copied from https://github.com/gonzfe05/HCOPE/tree/master

def inverted_MPeB(X,n,c,mu_):
	'''Returns confidence level given a lower bound

	Parameters
	----------
	X : Vector of n integers.
		Independently distributed random variables
	n : int
		Length of vectors
	c : int.
		Threshold. Must be empirically fixed
	mu_ : int
		Lower bound.

	Returns
	-------
	float
		Confidence level for the threshold
	'''
	Y = [min([X_i,c]) for X_i in X]
	Y2 = [(Y_i/c) for Y_i in Y]
	k1 = (7*n) / (3*(n-1))
	k3 = mu_ * (1/c) * n - sum([Y_i/c for Y_i in Y])
	k2_1 = n * sum([Y2_i**2 for Y2_i in Y2])
	k2_2 = sum(Y2) ** 2
	k2_diff = k2_1 - k2_2
	k2 = math.sqrt( (2 * k2_diff) / (n-1) )
	z_factor1 = k2**2 - 4*k1*k3
	if z_factor1 > 0:
		z = (-k2 + math.sqrt(z_factor1)) / (2*k1)
		if z > 0:
			return 1-min(1,2*math.exp(-(z**2)))
		else:
			return 0
	else:
		return 0

def lower_bound(c,X,n,delta):
	'''Returns likelihood of threshold c value

	Parameters
	----------
	c : int.
		Threshold. Must be empirically fixed
	X : Vector of n integers.
		Independently distributed random variables
	n : int
		Length of X vector
	delta : int
		1 - Confidence level for the threshold

	Returns
	-------
	float
		Likelihood of c threshold
	'''
	Y = [min([X_i,c]) for X_i in X]
	Y2 = [y/c for y in Y]
	
	f1 = (n/c)**(-1)
	f2 = sum(Y2)
	f3 = (7*n*math.log(2/delta)) / (3*(n - 1))
	f4_1 = (2*math.log(2/delta)) / (n-1)
	f4_2 = n*sum([y_i**2 for y_i in Y2])-sum(Y2)**2
	f5 = math.sqrt(f4_1*f4_2)
	f6 = f2-f3-f5
	return f1*f6

def c_like(c,X_pre,n_pre,n_post,delta):
    '''Returns likelihood of threshold c value

	Parameters
	----------
	c : int.
		Threshold. Must be empirically fixed
	X_pre : Vector of n_pre integers.
		Independently distributed random variables
	n_pre : int
		Length of X_pre vector
	n_post : int
		Length of X_post vector
	delta : int
		1 - Confidence level for the threshold

	Returns
	-------
	float
		Likelihood of c threshold
	'''
    Y = [min([X_i,c]) for X_i in X_pre]
    f1 = sum(Y)/n_pre
    f2_1 = 7*c*math.log(2/delta)
    f2_2 = 3*(n_post - 1)
    f2 = f2_1 / f2_2
    f3_1 = math.log(2/delta) / n_post
    f3_2 = 2 / (n_pre*(n_pre-1))
    f3_3 = n_pre * sum([Y_i**2 for Y_i in Y]) - sum(Y)**2
    # Calculation can be unstable when there are a significant number of zeros
    f3_3 = np.max([f3_3, 0])
    f3 = math.sqrt(f3_1*f3_2*f3_3)
    return -(f1-f2-f3)


def optimize_threshold(X_pre,n_pre,n_post,delta):
	'''Returns likelihood of threshold c value

	Parameters
	----------
	X_pre : Vector of n_pre integers.
		Independently distributed random variables
	n_pre : int
		Length of X_pre vector
	n_post : int
		Length of X_post vector
	delta : int
		1 - Confidence level for the threshold

	Returns
	-------
	float
		Optimal c threhold
	'''
	c_optim, fval, iters_, funcalls = brent(c_like,(X_pre,n_pre,n_post,delta),
                                         brack=(0.00001,1000),full_output=True)
	return c_optim

def generate_bounds(X,delta):
    #Divide dataset for optimization
    N = len(X)
    mean = np.mean(X)
    n_pre = round(N*0.05)
    n_post = N-n_pre
    np.random.shuffle(X)
    X_pre, X_post = X[:n_pre], X[n_pre:]
    #optimize threshold
    c_opt = optimize_threshold(X_pre,n_pre,n_post,delta)
    #bounds
    lower_bounds = []
    for i in [0.9,0.95,0.99]:
        temp_lb = lower_bound(c_opt,X_post,n_post,i)
        lower_bounds.append(temp_lb)
    #calculate bounds
    confidences = []
    bounds = []
    for i in np.arange(0, mean, 0.5):
        bounds.append(i)
        temp_conf = inverted_MPeB(X_post,n_post,c_opt,i)
        confidences.append(temp_conf)
    return X_post, lower_bounds, bounds, confidences

def get_lower_bound(X,delta):
    #Divide dataset for optimization
    N = len(X)
    mean = np.mean(X)
    n_pre = round(N*0.05)
    n_post = N-n_pre
    np.random.shuffle(X)
    X_pre, X_post = X[:n_pre], X[n_pre:]
    #optimize threshold
    c_opt = optimize_threshold(X_pre,n_pre,n_post,delta)
    #bounds
    return lower_bound(c_opt,X_post,n_post,1-delta)
