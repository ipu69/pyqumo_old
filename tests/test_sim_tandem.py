from dataclasses import dataclass
from typing import Union, Sequence

import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyqumo.arrivals import Poisson, MarkovArrival
from pyqumo.random import HyperExponential, PhaseType, Distribution, Exponential
from pyqumo.sim.tandem import simulate


@dataclass
class TandemProps:
    arrival: Union[Distribution, Sequence[Distribution]]
    service: Union[Distribution, Sequence[Distribution]]
    queue_capacity: int
    num_stations: int

    # System and queue sizes:
    system_size_avg: Sequence[float]
    system_size_std: Sequence[float]
    queue_size_avg: Sequence[float]
    queue_size_std: Sequence[float]
    busy_avg: Sequence[float]
    busy_std: Sequence[float]

    # Loss probability and utilization:
    drop_prob: Sequence[float]
    delivery_prob: Sequence[float]
    utilization: Sequence[float]

    # Intervals:
    departure_avg: Sequence[float]
    arrival_avg: Sequence[float]
    response_time_avg: Sequence[float]
    wait_time_avg: Sequence[float]
    delivery_delay_avg: Sequence[float]

    # Test parameters:
    tol: float = 1e-1
    max_packets: int = int(2e5)


# noinspection DuplicatedCode
@pytest.mark.parametrize('props', [
    TandemProps(
        arrival=Poisson(2), service=Poisson(5),
        queue_capacity=4, num_stations=1,
        # System and queue sizes:
        system_size_avg=[0.642], system_size_std=[0.981],
        queue_size_avg=[0.2444], queue_size_std=[0.6545],
        busy_avg=[0.3975], busy_std=[0.4894],
        # Scalar probabilities and rates:
        drop_prob=[0.0062], delivery_prob=[0.9938], utilization=[0.3975],
        # Intervals:
        departure_avg=[0.5031], arrival_avg=[0.5], response_time_avg=[0.323],
        wait_time_avg=[0.123], delivery_delay_avg=[0.323]),
    TandemProps(
        arrival=Poisson(1), service=Poisson(2),
        queue_capacity=np.inf, num_stations=3,
        # System and queue sizes:
        system_size_avg=[1, 1, 1], system_size_std=[1.414, 1.414, 1.414],
        queue_size_avg=[0.5, 0.5, 0.5], queue_size_std=[1.11, 1.11, 1.11],
        busy_avg=[0.5, 0.5, 0.5], busy_std=[0.5, 0.5, 0.5],
        # Scalar probabilities and rates:
        drop_prob=[0, 0, 0], delivery_prob=[1, 1, 1], utilization=[.5, .5, .5],
        # Intervals:
        departure_avg=[1, 1, 1], arrival_avg=[1, 1, 1],
        response_time_avg=[1, 1, 1], wait_time_avg=[.5, .5, .5],
        delivery_delay_avg=[3, 0, 0]),
    TandemProps(
        arrival=[Poisson(1), Poisson(2), Poisson(7)],
        service=[Exponential(10), Exponential(6), Exponential(40)],
        queue_capacity=np.inf, num_stations=3,
        # System and queue sizes:
        system_size_avg=[1/9, 1, 1/3],
        system_size_std=[np.sqrt(10)/9, np.sqrt(2), 2/3],
        queue_size_avg=[1/90, 1/2, 1/12],
        queue_size_std=[np.sqrt(109)/90, np.sqrt(5)/2, np.sqrt(19)/12],
        busy_avg=[1/10, 1/2, 1/4],
        busy_std=[3/10, 1/2, np.sqrt(3)/4],
        # Scalar probabilities and rates:
        drop_prob=[0, 0, 0], delivery_prob=[1, 1, 1], utilization=[.1, .5, .25],
        # Intervals:
        departure_avg=[1, 1/3, 1/10],
        arrival_avg=[1, 1/3, 1/10],
        response_time_avg=[1/9, 1/3, 1/30],
        wait_time_avg=[1/90, 1/6, 1/120],
        delivery_delay_avg=[43/90, 11/30, 1/30]),
    TandemProps(
        arrival=[MarkovArrival.poisson(1),
                 None,
                 None,
                 HyperExponential([4.0], [1.0])],
        service=PhaseType.exponential(10),
        queue_capacity=np.inf, num_stations=4,
        # System and queue sizes:
        system_size_avg=[1/9, 1/9, 1/9, 1],
        system_size_std=[(10**.5)/9, (10**.5)/9, (10**.5)/9, 2**.5],
        queue_size_avg=[1/90, 1/90, 1/90, 1/2],
        queue_size_std=[(109**.5)/90, (109**.5)/90, (109**.5)/90, (5**.5)/2],
        busy_avg=[.1, .1, .1, .5],
        busy_std=[.3, .3, .3, .5],
        # Scalar probabilities and rates:
        drop_prob=[0, 0, 0, 0], delivery_prob=[1, 1, 1, 1],
        utilization=[.1, .1, .1, .5],
        # Intervals:
        departure_avg=[1, 1, 1, .2], arrival_avg=[1, 1, 1, .2],
        response_time_avg=[1/9, 1/9, 1/9, .2],
        wait_time_avg=[1/90, 1/90, 1/90, .1],
        delivery_delay_avg=[8/15, 0, 0, .2],
        max_packets=int(3e4), tol=.2
    ),
])
def test_mm1_tandem(props):
    tol = props.tol
    ret = simulate(props.arrival, props.service, props.queue_capacity,
                   num_stations=props.num_stations,
                   max_packets=props.max_packets)

    # Check system and queue sizes:
    for i in range(props.num_stations):
        desc = f"station: {i}, arrival: {props.arrival}, " \
               f"service: {props.service}, " \
               f"length: {props.num_stations}, " \
               f"queue capacity: {props.queue_capacity}"

        # System size, queue size, busy rate
        assert_allclose(
            ret.system_size[i].mean, props.system_size_avg[i],
            rtol=tol, err_msg=f"system size average mismatch ({desc})")
        assert_allclose(
            ret.system_size[i].std, props.system_size_std[i],
            rtol=tol, err_msg=f"system size std.dev. mismatch ({desc})")
        assert_allclose(
            ret.queue_size[i].mean, props.queue_size_avg[i],
            rtol=tol, err_msg=f"queue size average mismatch ({desc})")
        assert_allclose(
            ret.queue_size[i].std, props.queue_size_std[i],
            rtol=tol, err_msg=f"queue size std.dev. mismatch ({desc})")
        assert_allclose(
            ret.busy[i].mean, props.busy_avg[i],
            rtol=tol, err_msg=f"busy rate average mismatch ({desc})")
        assert_allclose(
            ret.busy[i].std, props.busy_std[i],
            rtol=tol, err_msg=f"busy rate std.dev. mismatch ({desc})")

        # Loss probability and utilization:
        assert_allclose(
            ret.drop_prob[i], props.drop_prob[i],
            rtol=tol, err_msg=f"drop probability mismatch ({desc})")
        assert_allclose(
            ret.delivery_prob[i], props.delivery_prob[i],
            rtol=tol, err_msg=f"delivery probability mismatch ({desc})")
        assert_allclose(
            ret.get_utilization(i), props.utilization[i],
            rtol=tol, err_msg=f"utilization mismatch ({desc})")

        # Intervals:
        assert_allclose(
            ret.departures[i].avg, props.departure_avg[i],
            rtol=tol, err_msg=f"average departure interval mismatch ({desc})")
        assert_allclose(
            ret.arrivals[i].avg, props.arrival_avg[i],
            rtol=tol, err_msg=f"average arrival interval mismatch ({desc})")
        assert_allclose(
            ret.response_time[i].avg, props.response_time_avg[i],
            rtol=tol, err_msg=f"average response time mismatch ({desc})")
        assert_allclose(
            ret.wait_time[i].avg, props.wait_time_avg[i],
            rtol=tol, err_msg=f"waiting time mismatch ({desc})")
        assert_allclose(
            ret.delivery_delays[i].avg, props.delivery_delay_avg[i],
            rtol=tol, err_msg=f"average delivery delays mismatch ({desc})")
