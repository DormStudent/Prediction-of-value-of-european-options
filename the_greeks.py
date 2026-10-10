import numpy as np
from scipy.stats import norm
def delta(u,s,t,r,v,option_type):
    d1 = (np.log(u/s)+(r+v**2/2)*t)/(v*np.sqrt(t))
    if option_type == "call":
         delta = norm.cdf(d1)
    if option_type == "put":
         delta = norm.cdf(d1) - 1
    return delta
