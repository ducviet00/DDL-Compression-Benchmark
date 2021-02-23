import hashlib
import time
import optparse
import scipy.stats as stats
def gen_threshold_from_normal_distribution(p_value, mu, sigma):
    zvalue = stats.norm.ppf((1-p_value)/2)
    return mu+zvalue*sigma, mu-zvalue*sigma

