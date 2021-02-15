from dataclasses import dataclass

import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyqumo.arrivals import Poisson
from pyqumo.random import Exponential, Distribution
from pyqumo.cqumo.sim import simulate_gg1n


@dataclass
class GG1Props:
    arrival: Distribution
    service: Distribution
    queue_capacity: int

    # System and queue sizes:
    system_size_avg: float
    system_size_std: float
    queue_size_avg: float
    queue_size_std: float

    # Loss probability and utilization:
    loss_prob: float
    utilization: float

    # Departure process properties:
    departure_rate: float

    # Response and wait time:
    response_time_avg: float
    wait_time_avg: float

    # Test parameters:
    tol: float = 1e-1
    max_packets: int = int(1e5)


@pytest.mark.parametrize('props', [
    GG1Props(
        arrival=Poisson(2), service=Poisson(5), queue_capacity=4,
        system_size_avg=0.642, system_size_std=0.981,
        queue_size_avg=0.2444, queue_size_std=0.6545,
        loss_prob=0.0062, utilization=0.3975, departure_rate=1.9877,
        response_time_avg=0.323, wait_time_avg=0.123),
    GG1Props(
        arrival=Exponential(42), service=Exponential(34),
        queue_capacity=7,
        system_size_avg=5.3295, system_size_std=5.6015**0.5,
        queue_size_avg=4.3708, queue_size_std=5.2010**0.5,
        loss_prob=0.2239, utilization=0.9587, departure_rate=32.5959,
        response_time_avg=0.163, wait_time_avg=0.134,
        max_packets=int(1e5)
    ),
    GG1Props(
        arrival=Poisson(1), service=Exponential(2),
        queue_capacity=np.inf,
        system_size_avg=1, system_size_std=2.0**0.5,
        queue_size_avg=0.5, queue_size_std=1.25**0.5,
        loss_prob=0, utilization=0.5, departure_rate=1.0,
        response_time_avg=1.0, wait_time_avg=0.5, max_packets=int(1e5)
    )
])
def test_gg1(props):
    tol = props.tol
    results = simulate_gg1n(props.arrival, props.service, props.queue_capacity,
                            max_packets=props.max_packets)
    desc = f"arrival: {props.arrival}, " \
           f"service: {props.service}, " \
           f"queue capacity: {props.queue_capacity}"

    # Check system and queue sizes:
    assert_allclose(results.system_size.mean, props.system_size_avg, rtol=tol,
                    err_msg=f"system size average mismatch ({desc})")
    assert_allclose(results.system_size.std, props.system_size_std, rtol=tol,
                    err_msg=f"system size std.dev. mismatch ({desc})")
    assert_allclose(results.queue_size.mean, props.queue_size_avg, rtol=tol,
                    err_msg=f"queue size average mismatch ({desc})")
    assert_allclose(results.queue_size.std, props.queue_size_std, rtol=tol,
                    err_msg=f"queue size std.dev. mismatch ({desc})")

    # Loss probability and utilization:
    assert_allclose(results.loss_prob, props.loss_prob, rtol=tol,
                    err_msg=f"loss probability mismatch ({desc})")
    assert_allclose(results.utilization, props.utilization, rtol=tol,
                    err_msg=f"utilization mismatch ({desc})")

    # Departure process:
    assert_allclose(1/results.departures.avg, props.departure_rate, rtol=tol,
                    err_msg=f"departure rate mismatch ({desc})")

    # Wait and response time:
    assert_allclose(results.response_time.avg, props.response_time_avg,
                    rtol=tol, err_msg=f"response time mismatch ({desc})")
    assert_allclose(results.wait_time.avg, props.wait_time_avg, rtol=tol,
                    err_msg=f"waiting time mismatch ({desc})")
