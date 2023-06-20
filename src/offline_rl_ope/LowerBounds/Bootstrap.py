from scipy.stats import bootstrap
import numpy as np

def get_lower_bound(X,delta, random_state=1):
    res = bootstrap(data=(X,), statistic=np.mean, confidence_level=1-delta, 
                    method="BCa", random_state=random_state)
    return res.confidence_interval.low