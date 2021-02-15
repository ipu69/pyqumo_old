from dataclasses import dataclass
from typing import Optional, Sequence

import pytest
from numpy.testing import assert_allclose

from pyqumo.queues import MM1Queue, BasicQueueingSystem, MM1NQueue, MapPh1NQueue
from pyqumo.arrivals import Poisson, MarkovArrival


from pyqumo.random import PhaseType


@dataclass
class QueueProps:
    # Classes:
    arrival_class: Optional[type] = None
    service_class: Optional[type] = None
    departure_class: Optional[type] = None
    # Rates:
    arrival_rate: Optional[float] = None
    service_rate: Optional[float] = None
    departure_rate: Optional[float] = None
    # Probabilities and utilization:
    system_size_pmf: Sequence[float] = ()
    queue_size_pmf: Sequence[float] = ()
    utilization: Optional[float] = None
    system_size_avg: Optional[float] = None
    system_size_var: Optional[float] = None
    queue_size_avg: Optional[float] = None
    queue_size_var: Optional[float] = None
    # Loss probability and bandwidth
    loss_prob: Optional[float] = None
    bandwidth: Optional[float] = None
    # Response and wait time
    response_time: Optional[float] = None
    wait_time: Optional[float] = None
    # Tolerance
    tol: float = 1e-2


# noinspection DuplicatedCode
@pytest.mark.parametrize('queue, props, string', [
    # M/M/1 queues:
    (
        MM1Queue(2.0, 5.0), QueueProps(
            arrival_class=Poisson, service_class=Poisson,
            departure_class=Poisson,
            arrival_rate=2, service_rate=5, departure_rate=2,
            system_size_pmf=[0.6, 0.24, 0.096, 0.0384, 0.0154],
            system_size_avg=0.6667, system_size_var=1.1111,
            queue_size_pmf=[0.84, 0.096, 0.0384, 0.0154],
            queue_size_avg=0.2667, queue_size_var=0.5511,
            utilization=0.4, loss_prob=0.0, bandwidth=2,
            response_time=1/3, wait_time=2/15,
        ), '(MM1Queue: arrival_rate=2, service_rate=5)'
    ), (
        MM1Queue(1.0, 2.0), QueueProps(
            arrival_rate=1, service_rate=2, departure_rate=1,
            system_size_pmf=[0.5, 0.25, 0.125, 0.0625],
            system_size_avg=1, system_size_var=2.0,
            queue_size_pmf=[0.75, 0.125, 0.0625, 0.03125],
            queue_size_avg=0.5, queue_size_var=1.25,
            utilization=0.5, loss_prob=0.0, bandwidth=1,
            response_time=1.0, wait_time=0.5,
        ), '(MM1Queue: arrival_rate=1, service_rate=2)'
    ),
    # M/M/1/N queues:
    (
        MM1NQueue(2, 5, queue_capacity=4), QueueProps(
            arrival_class=Poisson, service_class=Poisson,
            departure_class=MarkovArrival,
            arrival_rate=2, service_rate=5, departure_rate=1.9877,
            system_size_pmf=[0.6025, 0.2410, 0.0964, 0.0385, 0.0154, 0.0062],
            system_size_avg=0.6420, system_size_var=0.9624,
            queue_size_pmf=[0.8434, 0.0964, 0.0385, 0.0154, 0.0062],
            queue_size_avg=0.2444, queue_size_var=0.4284,
            utilization=0.3975, loss_prob=0.0062, bandwidth=1.9877,
            response_time=0.323, wait_time=0.123,
        ), '(MM1NQueue: arrival_rate=2, service_rate=5, capacity=5)'
    ), (
        # Queue with arrival rate > service rate (valid for finite queue):
        MM1NQueue(42, 34, queue_capacity=7), QueueProps(
            arrival_rate=42, service_rate=34, departure_rate=32.5959,
            system_size_pmf=[
                0.04129548, 0.05101206, 0.0630149, 0.07784193, 0.09615768,
                0.11878301, 0.14673196, 0.18125713, 0.22390586
            ], system_size_avg=5.3295, system_size_var=5.6015,
            queue_size_pmf=[
                0.09230753, 0.0630149, 0.07784193, 0.09615768, 0.11878301,
                0.14673196, 0.18125713, 0.22390586
            ], queue_size_avg=4.3708, queue_size_var=5.2010,
            utilization=0.9587, loss_prob=0.2239, bandwidth=32.5959,
            response_time=0.163, wait_time=0.134,
        ), '(MM1NQueue: arrival_rate=42, service_rate=34, capacity=8)'
    ),
    # MAP/PH/1/N representation of M/M/1/N queue:
    (
        MapPh1NQueue(
            MarkovArrival.poisson(2),
            PhaseType.exponential(5),
            queue_capacity=4
        ), QueueProps(
            arrival_rate=2, service_rate=5, departure_rate=1.9877,
            system_size_pmf=[0.6025, 0.2410, 0.0964, 0.0385, 0.0154, 0.0062],
            system_size_avg=0.6420, system_size_var=0.9624,
            queue_size_pmf=[0.8434, 0.0964, 0.0385, 0.0154, 0.0062],
            queue_size_avg=0.2444, queue_size_var=0.4284,
            utilization=0.3975, loss_prob=0.0062, bandwidth=1.9877,
            response_time=0.323, wait_time=0.123,
        ), "(MapPh1NQueue: arrival=(MAP: d0=[[-2]], d1=[[2]]), "
           "service=(GI: f=(PH: s=[[-5]], p=[1])), capacity=5)"
    ), (
        MapPh1NQueue(
            MarkovArrival.poisson(42),
            PhaseType.exponential(34),
            queue_capacity=7
        ), QueueProps(
            arrival_rate=42, service_rate=34, departure_rate=32.5959,
            system_size_pmf=[
                0.04129548, 0.05101206, 0.0630149, 0.07784193, 0.09615768,
                0.11878301, 0.14673196, 0.18125713, 0.22390586
            ], system_size_avg=5.3295, system_size_var=5.6015,
            queue_size_pmf=[
                0.09230753, 0.0630149, 0.07784193, 0.09615768, 0.11878301,
                0.14673196, 0.18125713, 0.22390586
            ], queue_size_avg=4.3708, queue_size_var=5.2010,
            utilization=0.9587, loss_prob=0.2239, bandwidth=32.5959,
            response_time=0.163, wait_time=0.134,
        ), "(MapPh1NQueue: arrival=(MAP: d0=[[-42]], d1=[[42]]), "
           "service=(GI: f=(PH: s=[[-34]], p=[1])), capacity=8)"
    )
])
def test_basic_props(
        queue: BasicQueueingSystem,
        props: QueueProps,
        string: str
):
    """
    Test basic properties of the queueing system model.
    """
    tol = props.tol

    # 1) Validate classes:
    if props.arrival_class is not None:
        assert isinstance(queue.arrival, props.arrival_class)
    if props.service_class is not None:
        assert isinstance(queue.service, props.service_class)
    if props.departure_class is not None:
        assert isinstance(queue.departure, props.departure_class)

    # 2) Validate arrival, service and departure rates:
    if props.arrival_rate is not None:
        assert_allclose(queue.arrival.rate, props.arrival_rate, rtol=tol,
                        err_msg=f'arrival rate mismatch for {string}')
    if props.service_rate is not None:
        assert_allclose(queue.service.rate, props.service_rate, rtol=tol,
                        err_msg=f'service rate mismatch for {string}')
    if props.departure_rate is not None:
        assert_allclose(queue.departure.rate, props.departure_rate, rtol=tol,
                        err_msg=f'departure rate mismatch for {string}')

    # 3) Validate system and queue size probabilities, properties and
    #    utilization coefficient, bandwidth and loss probability:
    if (n := len(props.system_size_pmf)) > 0:
        assert_allclose(
            [queue.system_size.pmf(i) for i in range(n)], props.system_size_pmf,
            rtol=tol, err_msg=f'system size PMF mismatch for {string}')
    if (n := len(props.queue_size_pmf)) > 0:
        assert_allclose(
            [queue.queue_size.pmf(i) for i in range(n)], props.queue_size_pmf,
            rtol=tol, err_msg=f'queue size PMF mismatch for {string}')
    if props.system_size_avg is not None:
        assert_allclose(queue.system_size.mean, props.system_size_avg, rtol=tol,
                        err_msg=f'average system size mismatch for {string}')
    if props.system_size_var is not None:
        assert_allclose(queue.system_size.var, props.system_size_var, rtol=tol,
                        err_msg=f'system size variance mismatch for {string}')
    if props.queue_size_avg is not None:
        assert_allclose(queue.queue_size.mean, props.queue_size_avg, rtol=tol,
                        err_msg=f'average queue size mismatch for {string}')
    if props.queue_size_var is not None:
        assert_allclose(queue.queue_size.var, props.queue_size_var, rtol=tol,
                        err_msg=f'queue size variance mismatch for {string}')
    if props.utilization is not None:
        assert_allclose(queue.utilization, props.utilization, rtol=tol,
                        err_msg=f'utilization mismatch for {string}')
    if props.loss_prob is not None:
        assert_allclose(queue.loss_prob, props.loss_prob, rtol=tol,
                        err_msg=f'loss probability mismatch for {string}')
    if props.bandwidth is not None:
        assert_allclose(queue.bandwidth, props.bandwidth, rtol=tol,
                        err_msg=f'bandwidth mismatch for {string}')

    # 4) Validate response and waiting time:
    if props.wait_time is not None:
        assert_allclose(queue.wait_time, props.wait_time, rtol=tol,
                        err_msg=f'waiting time mismatch for {string}')
    if props.response_time is not None:
        assert_allclose(queue.response_time, props.response_time, rtol=tol,
                        err_msg=f'response time mismatch for {string}')

    # 4) Validate string representation:
    assert str(queue) == string
