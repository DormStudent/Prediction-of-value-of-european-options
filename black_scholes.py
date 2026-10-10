import numpy as np
from scipy.stats import norm
def black_scholes(u,s,t,r,v,option_type):
    if option_type == "call":
        d1 = (np.log(u/s)+(r+v**2/2)*t)/(v*np.sqrt(t))
        d2 = d1 - v*np.sqrt(t)
        C = u*norm.cdf(d1) - s*np.exp(-r*t)*norm.cdf(d2)
        return C
    if option_type == "put":
        d1 = (np.log(u/s)+(r+v**2/2)*t)/(v*np.sqrt(t))
        d2 = d1 - v*np.sqrt(t)
        P = s*np.exp(-r*t)*norm.cdf(-d2) - u*norm.cdf(-d1)
        return P