from functools import cached_property, lru_cache
from typing import Union, Callable, Mapping

import numpy as np

from pyqumo.matrix import cbdiag
from pyqumo.random import Distribution, CountableDistribution, PhaseType
from pyqumo.arrivals import Poisson, MarkovArrival, RandomProcess, \
    GIProcess


class BasicQueueingSystem:
    def __init__(
            self,
            arrival: Distribution,
            service: Distribution,
            queue_capacity: int = np.inf,
            precision: float = 1e-9
    ):
        """
        Queueing system constructor.
        """
        self._arrival = arrival
        self._service = service
        self._queue_capacity = queue_capacity
        self._precision = precision

    @property
    def arrival(self) -> RandomProcess:
        return self._arrival

    @property
    def service(self) -> RandomProcess:
        return self._service

    @property
    def lambda_(self):
        return self.arrival.rate

    @property
    def mu(self):
        return self.service.rate

    @property
    def queue_capacity(self) -> int:
        return self._queue_capacity

    @cached_property
    def capacity(self):
        return self._queue_capacity + 1

    @property
    def utilization(self) -> float:
        return 1 - self.get_system_size_prob(0)

    @cached_property
    def bandwidth(self) -> float:
        return (1 - self.loss_prob) * self.lambda_

    @cached_property
    def system_size(self) -> CountableDistribution:
        props = self._get_system_size_props()
        m1, var = props.get('avg', None), props.get('var', None)
        moments = []
        if m1 is not None:
            moments = [m1, var + m1**2] if var is not None else [m1]
        return CountableDistribution(
            self.get_system_size_prob,
            precision=self._precision,
            moments=moments)

    @cached_property
    def queue_size(self) -> CountableDistribution:
        props = self._get_queue_size_props()
        m1, var = props.get('avg', None), props.get('var', None)
        moments = []
        if m1 is not None:
            moments = [m1, var + m1**2] if var is not None else [m1]

        def fn(x: int) -> float:
            if x > 0:
                return self.get_system_size_prob(x + 1)
            if x == 0:
                return self.get_system_size_prob(0) + \
                       self.get_system_size_prob(1)
            return 0.0

        return CountableDistribution(
            fn, precision=self._precision, moments=moments)

    def _get_system_size_props(self) -> Mapping[str, float]:
        """
        This helper can return props of the system size distribution: avg, var.

        By default, returns an empty dictionary. If overridden, may return
        a dictionary with keys 'avg', 'var'. If they found, they will be
        used to compute 1-st and 2-nd moments precisely.
        """
        return {}

    def _get_queue_size_props(self) -> Mapping[str, float]:
        """
        This helper can return props of the queue size distribution: avg, var.

        By default, returns an empty dictionary. If overridden, may return
        a dictionary with keys 'avg', 'var'. If they found, they will be
        used to compute 1-st and 2-nd moments precisely.
        """
        return {}

    @property
    def wait_time(self):
        return self.response_time - self.service.mean

    @property
    def departure(self) -> RandomProcess:
        raise NotImplementedError

    @property
    def get_system_size_prob(self) -> Callable[[int], float]:
        raise NotImplementedError

    @property
    def response_time(self) -> float:
        raise NotImplementedError

    @property
    def loss_prob(self) -> float:
        raise NotImplementedError


class MM1Queue(BasicQueueingSystem):
    def __init__(self, arrival_rate: float, service_rate: float,
                 precision: float = 1e-9):
        arrival = Poisson(arrival_rate)
        service = Poisson(service_rate)
        super().__init__(arrival, service, precision=precision)

    @cached_property
    def departure(self):
        return Poisson(self.lambda_)

    @cached_property
    def get_system_size_prob(self) -> Callable[[int], float]:
        rho = self.lambda_ / self.mu
        if 0 <= rho <= 1:
            return lambda size: (1 - rho) * pow(rho, size) if size >= 0 else 0
        raise ValueError(f"no system size distribution, utilization = {rho}")

    def _get_system_size_props(self) -> Mapping[str, float]:
        rho = self.lambda_ / self.mu
        if rho >= 1:
            return {'avg': np.inf, 'var': np.inf}
        avg = rho / (1 - rho)
        var = rho / (1 - rho)**2
        return {'avg': avg, 'var': var}

    def _get_queue_size_props(self) -> Mapping[str, float]:
        rho = self.lambda_ / self.mu
        if rho >= 1:
            return {'avg': np.inf, 'var': np.inf}
        avg = rho**2 / (1 - rho)
        m2 = rho**2 * (1 + rho) / (1 - rho)**2
        var = m2 - avg**2
        return {'avg': avg, 'var': var}

    @cached_property
    def response_time(self):
        service_rate_diff = self.service.rate - self.arrival.rate
        if service_rate_diff < 1e-12:
            return np.inf
        return 1 / service_rate_diff

    @cached_property
    def loss_prob(self) -> float:
        return 0.0

    def __repr__(self):
        return f"(MM1Queue: " \
               f"arrival_rate={self.lambda_:.3g}, " \
               f"service_rate={self.mu:.3g}" \
               f")"


class MM1NQueue(BasicQueueingSystem):
    def __init__(self, arrival_rate: float,
                 service_rate: float,
                 queue_capacity: int,
                 precision: float = 1e-9):
        if abs(np.math.modf(queue_capacity)[0]) > 1e-12 or queue_capacity <= 0:
            raise ValueError(f"positive integer expected, "
                             f"but {queue_capacity} found")
        arrival = Poisson(arrival_rate)
        service = Poisson(service_rate)
        super().__init__(arrival, service, queue_capacity=queue_capacity,
                         precision=precision)

    @cached_property
    def departure(self):
        n = self.capacity
        a = self.arrival.rate
        b = self.service.rate
        d0 = cbdiag(n + 1, [
            (0, np.asarray([[-(a + b)]])),
            (1, np.asarray([[a]]))
        ])
        d0[0, 0] += b
        d0[n, n] += a
        d1 = cbdiag(n + 1, [(-1, np.asarray([[b]]))])
        return MarkovArrival(d0, d1)

    @cached_property
    def get_system_size_prob(self) -> Callable[[int], float]:
        rho = self.arrival.rate / self.service.rate
        p0 = (1 - rho) / (1 - rho**(self.capacity + 1))
        return lambda x: rho**x * p0 if 0 <= x <= self.capacity else 0.0

    def _get_system_size_props(self) -> Mapping[str, float]:
        """
        Compute system size average and variance for M/M/1/N.

        Average: :math:`m_1 = [1 - (N+1) r^N + N r^{N+1}] / (1 - r)^2`

        M2: :math:`m_2 = (1+r)/(1-r) m_1 - N(N+1) r^{N+1} / (1 - r^{N+1})`

        Variance: :math:`Var(N_s) = m_2 - m_1^2`

        Here `r = arrival.rate / service.rate` (usually written as rho).

        Returns
        -------
        values : dict
            Contains 'avg', 'm2' and 'var'
        """
        rho = self.arrival.rate / self.service.rate
        n = self.capacity
        p0 = self.get_system_size_prob(0)
        rho_n = pow(rho, n)    # r^n
        rho_np1 = rho_n * rho  # r^{n+1}
        avg = p0 * rho * (1 - (n + 1) * rho**n + n * rho**(n+1)) / (1 - rho)**2
        m2 = ((1 + rho) / (1 - rho) * avg -
              n * (n + 1) * rho**(n + 1) / (1 - rho**(n + 1)))
        var = m2 - avg**2
        return {'avg': avg, 'm2': m2, 'var': var}

    def _get_queue_size_props(self) -> Mapping[str, float]:
        rho = self.arrival.rate / self.service.rate
        n = self.capacity
        ns = self._get_system_size_props()
        p0 = self.get_system_size_prob(0)
        avg = ns['avg'] - (1 - p0)
        m2 = ns['m2'] - 2 * ns['avg'] + (1 - p0)
        var = ns['var'] - 2 * p0 * ns['avg'] + p0 * (1 - p0)

        return {'avg': avg, 'm2': m2, 'var': var}

    @cached_property
    def response_time(self) -> float:
        p1 = self.get_system_size_prob(1)
        r = self.queue_capacity
        lambda_, mu = self.lambda_, self.mu
        rho = lambda_ / mu

        k1 = (mu - rho**r * ((r + 1) * mu - r * lambda_)) / (mu - lambda_)**2
        k2 = p1 / (1 - self.loss_prob)

        return k1 * k2 + self.service.mean

    @cached_property
    def loss_prob(self) -> float:
        return self.get_system_size_prob(self.capacity)

    def __repr__(self):
        return f"(MM1NQueue: " \
               f"arrival_rate={self.lambda_:.3g}, " \
               f"service_rate={self.mu:.3g}, " \
               f"capacity={self.capacity}" \
               f")"


class MapPh1NQueue(BasicQueueingSystem):
    def __init__(self,
                 arrival: MarkovArrival,
                 service: PhaseType,
                 queue_capacity: int):
        if abs(np.math.modf(queue_capacity)[0]) > 1e-12 or queue_capacity <= 0:
            raise ValueError(f"positive integer expected, "
                             f"but {queue_capacity} found")
        super().__init__(arrival, GIProcess(service), 
                         queue_capacity=queue_capacity, precision=1e-20)

    def _get_casted_arrival_and_service(self) \
            -> (MarkovArrival, PhaseType):
        """
        Returns (arrival, service), casted to MarkovArrival and PH.

        This method is mostly needed to avoid problems with Python linter,
        that warns about wrong types (RandomProcess instead of
        MarkovArrivalProcess for arrival, and RandomProcess instead of
        GenericIndependentProcess for service).

        However, most of the methods are cached, so this doesn't add much
        overhead job.
        """
        arrival = self.arrival
        assert isinstance(arrival, MarkovArrival)
        service_process = self.service
        assert isinstance(service_process, GIProcess)
        service = service_process.dist
        assert isinstance(service, PhaseType)
        return arrival, service

    @cached_property
    def departure(self) -> MarkovArrival:
        arrival, service = self._get_casted_arrival_and_service()

        # Aliasing matrices from arrival MAP and service PH
        d0 = arrival.d0
        d1 = arrival.d1
        w = arrival.order
        iw = np.eye(w)
        s = service.s
        tau = service.init_probs
        v = service.order
        iv = np.eye(v)
        ev = np.ones((v, 1))
        m = self.capacity - 1
        b = v * w
        ob = np.zeros((b, b))

        # Building blocks
        d0_iv = np.kron(d0, iv)
        d1_iv = np.kron(d1, iv)
        d0_s = np.kron(d0, iv) + np.kron(iw, s)
        ct = np.kron(-s.dot(ev), tau)
        iw_ct = np.kron(iw, ct)
        r0 = np.kron(d1, np.kron(tau, ev))
        ra = np.kron(d0 + d1, iv) + np.kron(iw, s)

        # Building departure D0 and D1
        d0_dep = cbdiag(self.capacity, ((0, d0_s), (1, d1_iv)))
        d0_dep[m*b:, m*b:] = ra
        d0_left_col = np.vstack((d0_iv,) + (ob,) * self.capacity)
        d0_top_row = np.hstack((r0,) + (ob,) * m)
        d0_dep = np.hstack((d0_left_col, np.vstack((d0_top_row, d0_dep))))
        D1_dep = cbdiag(self.capacity + 1, ((-1, iw_ct),))

        return MarkovArrival(d0_dep, D1_dep)

    @cached_property
    def get_system_size_prob(self) -> Callable[[int], float]:
        arrival, service = self._get_casted_arrival_and_service()
        departure = self.departure

        b = arrival.order * service.order
        ctmc = departure.ctmc
        pmf = ctmc.steady_pmf

        return lambda x: ctmc.steady_pmf[x*b: (x+1)*b].sum() \
            if 0 <= x <= self.capacity else 0.0

    @cached_property
    def response_time(self):
        return self.system_size.mean / self.bandwidth
    
    @cached_property
    def arrival_pmf_regarding_system_size(self):
        """
        Build (N, W) matrix where K-th row is arrival MAP PMF for system size K.

        Let arrival MAP has W states and PH - V states. Let
        :math:`Psi_k = [p_0, p_1, ..., p_{W-1}]` - distribution of arrival MAP
        when there are k packets in the system.

        States in (departure) CTMC are arranged as :math:`N = kVW + iV + j`,
        where `i` - MAP state number (0-based) and `j` - PH state
        (also 0-based)

        To compute :math:`Psi_{k,i}` we need to find sum:
        :math:`Theta[kVW+iV] + Theta[kVW+iV + 1] + ... + Theta[kVW+iV + V-1]`

        Returns
        -------
        psi : np.ndarray
            A 2D matrix with K-th row representing PMF of arrival MAP when
            there are K packets in the system
        """
        arrival, service = self._get_casted_arrival_and_service()
        v, w = service.order, arrival.order
        theta = self.departure.ctmc.steady_pmf
        psi = []

        for k in range(self.capacity + 1):
            pmf = np.zeros(w)
            for i in range(w):
                base = k * v * w + i * v
                pmf[i] = theta[base:(base + v)].sum()
            psi.append(pmf)

        return np.asarray(psi)

    @cached_property
    def loss_prob(self) -> float:
        arrival, _ = self._get_casted_arrival_and_service()
        psi = self.arrival_pmf_regarding_system_size
        return psi[self.capacity].dot(arrival.d1).sum() / self.arrival.rate

    def __repr__(self):
        return f"(MapPh1NQueue: " \
               f"arrival={self.arrival}, " \
               f"service={self.service}, " \
               f"capacity={self.capacity}" \
               f")"
