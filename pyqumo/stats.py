import numpy as np


def get_cv(m1: float, m2: float) -> float:
    """Compute coefficient of variation.
    """
    return (m2 - m1**2)**0.5 / m1


def get_skewness(m1: float, m2: float, m3: float) -> float:
    """Compute skewness.
    """
    var = m2 - m1**2
    std = var**0.5
    return (m3 - 3*m1*var - m1**3) / (var * std)


def get_noncentral_m3(mean: float, cv: float, skew: float) -> float:
    """Compute non-central third moment if mean, CV and skew provided.
    """
    m1 = mean
    std = cv * mean
    var = std**2
    return skew * var * std + 3 * m1 * var + m1**3


def get_noncentral_m2(mean: float, cv: float) -> float:
    """Compute non-central from the mean value and coef. of variation.
    """
    return (mean * cv)**2 + mean**2


def moment(source, maxn=1, minn=1):
    """Computes moments from a given source.

    Args:
        source (array-like, distribution or arrival):
            data to compute moments for. Treated as a set of samples
            if an array-like, otherwise is expected to have a `moment(k)`
            method (e.g. distribution or arrival process)
        maxn (int): maximum number of moment to compute.
        minn (int): minimum number of moment to compute

    Returns:
        ndarray containing computed moments
    """
    if hasattr(source, 'moment'):
        return np.asarray([source.moment(k) for k in range(1, maxn + 1)])
    if not isinstance(source, np.ndarray):
        source = np.asarray(source)
    ret = [np.power(source, k).mean() for k in range(minn, maxn + 1)]
    return np.asarray(ret)


def lag(source, maxn=1):
    """Computes lags from a given source.

    Args:
        source (array-like or arrival):
            data to compute moments for. Treated as a set of samples
            if an array-like, otherwise is expected to have a `lag(k)`
            method (e.g. arrival process)
        maxn (int): a number of lags to compute.

    Returns:
        ndarray containing computed lags
    """
    if hasattr(source, 'lag'):
        return np.asarray([source.lag(k) for k in range(1, maxn + 1)])
    else:
        n = len(list(source))
        maxn = min(maxn, n - 1)  # need at least k + 1 samples for k-th lag
        moments = moment(source, 2)
        m1 = moments[0]
        sigma2 = moments[1] - moments[0] ** 2
        ret = []
        for k in range(1, maxn + 1):
            values = [(source[i] - m1) * (source[i + k] - m1)
                      for i in range(n - k)]
            ret.append(sum(values) / ((n - k) * sigma2))
        return np.asarray(ret)


def normalize_moments(moments, k=None):
    # TODO: add unit test
    """
    Normalizes moments using a given or computed coefficient.

    Args:
        moments (array-like): an array of the first moments
        k (scalar, callable or None): a coefficient. If None, it will be $M1$
            if only one moment is given or $\frac{M2}{M1}$. If scalar,
            this scalar will be used. If callable, it will be called as
            `k(moments)` to get the coefficient.

    Returns: tuple (ndarray, mu), where the first component is normalized
     moments, the second is the coefficient being used.
    """
    if k is None:
        try:
            mu = moments[1] / moments[0]
        except IndexError:
            mu = moments[0]
    else:
        try:
            mu = k(moments)
        except TypeError:
            mu = k
    ret = list(moments)
    for i in range(len(ret)):
        ret[i] /= pow(mu, i + 1)
    return np.asarray(ret), mu
