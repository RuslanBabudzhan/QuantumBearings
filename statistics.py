
from pyentrp import entropy as pent
import antropy as ent
from hurst import compute_Hc
import numpy as np


def stat_range(data):
    return max(data) - min(data)


def sample_entropy(data):
    return float(pent.sample_entropy(data, 1))


def shannon_entropy(data):
    return pent.shannon_entropy(data)


def energy(data):
    return sum(np.abs(data)**2)


def hurst(data):
    h, _, _ = compute_Hc(data, kind='change')
    return h


def petrosian_fd(data):
    return ent.petrosian_fd(data)


def zero_crossing(data):
    return ent.num_zerocross(data)


def higuchi_fd(data):
    return ent.higuchi_fd(data)


def activity(data):
    activity, _ = ent.hjorth_params(data)
    return activity


def complexity(data):
    _, complexity = ent.hjorth_params(data)
    return complexity


def crest_factor(data):
    return np.max(np.abs(data))/np.sqrt(np.mean(np.square(data)))





