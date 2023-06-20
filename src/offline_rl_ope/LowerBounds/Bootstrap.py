from scipy.stats import bootstrap
import numpy as np

def get_lower_bound(X,delta, random_state=1, n_resamples:float=0.05, 
                    batch:int=5):
    n_resamples = int(np.round(n_resamples*len(X), decimals=0))
    res = bootstrap(data=(X,), statistic=np.mean, confidence_level=1-delta, 
                    method="BCa", random_state=random_state, 
                    n_resamples=n_resamples, batch=batch)
    return res.confidence_interval.low