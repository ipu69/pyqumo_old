from typing import Optional

import numpy as np

from pyqumo.arrivals import MarkovArrival
from pyqumo.errors import BoundsError
from pyqumo.fitting import fit_acph2, fit_mern2, fit_map_horvath05
from pyqumo.random import PhaseType, HyperErlang
from pyqumo.stats import get_noncentral_m2, get_noncentral_m3


def random_phase_type(
        avg: float, min_cv: float, max_cv: float, max_skew: float,
        max_order: Optional[int] = None) -> PhaseType:
    """
    Generate random phase-type distribution with randomly chosen CV and skew.

    # TODO: write tests

    Parameters
    ----------
    avg : float
    min_cv : float
    max_cv : float
    max_skew : float
    max_order : int, optional (default: None)
        if given, any generated distribution will be cut to keep average
        value, but to have number of states less or equal to `max_order`.

    Returns
    -------
    dist : PhaseType
    """
    cv = np.random.uniform(min_cv, max_cv)
    skew = np.random.uniform(cv - 1/cv, max_skew)
    moments = np.asarray([
        avg,
        get_noncentral_m2(avg, cv),
        get_noncentral_m3(avg, cv, skew)
    ])
    try:
        dist = fit_acph2(moments, strict=True)[0]
    except (BoundsError, ZeroDivisionError):
        dist = fit_mern2(moments, strict=False, max_shape_inc=1)[0]

        # If order is too large, we won't try to tune CV or skew.
        # Instead, we select a random shape and scale rate and shape.
        if max_order is not None and dist.order > max_order:
            bad_dist = dist  # store distribution
            new_shapes = np.random.randint(
                max_order // 4,
                max_order // 2 + 1, 2)
            k = new_shapes.astype(np.float32) / bad_dist.shapes
            dist = HyperErlang(
                shapes=new_shapes,
                params=(bad_dist.params * k),
                probs=bad_dist.probs)

            # Check the new distribution has good enough rate:
            if (err := abs(dist.rate - bad_dist.rate) / bad_dist.rate) > .01:
                print(f"!!! old rate was {bad_dist:.3f}, "
                      f"new rate is {dist.rate:.3f} [error = {err:.3f}]")
                assert False

    return dist.as_ph().scale(dist.mean / avg)


def random_markov_arrival(
        ph: PhaseType, min_lag1: float, max_lag1: float) -> MarkovArrival:
    """
    Generate random MAP with a given stationary PH and random lag-1.

    # TODO: write tests

    Parameters
    ----------
    ph : PhaseType
        stationary PH-distribution
    min_lag1 : float
    max_lag1 : float

    Returns
    -------
    MarkovArrival
    """
    lag = np.random.uniform()
    arrival = fit_map_horvath05(ph, lag, n_iters=3)[0]
    return arrival.scale(arrival.mean / ph.mean)
