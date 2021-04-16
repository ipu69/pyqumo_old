from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional, Callable

from pyqumo import MarkovArrival, PhaseType, rel_err, MapPh1NQueue, BoundsError, \
    Erlang, HyperExponential, Exponential
from pyqumo.fitting import fit_acph2, fit_mern2, fit_map_horvath05


def get_complexity(arrival_order: int, service_order: int, capacity: int,
                   net_size: int) -> int:
    """
    Compute tandem network complexity as `W(V(M+2))^N`.
    """
    return arrival_order * (service_order * (capacity + 2))**net_size


@dataclass
class SolveResults:
    skipped: bool
    delay: Optional[float] = None
    delivery_prob: Optional[float] = None
    last_system_size: Optional[float] = None
    elapsed: Optional[float] = None
    max_inp_order: Optional[int] = None
    max_out_order: Optional[int] = None
    m1_err: Optional[float] = None
    cv_err: Optional[float] = None
    skew_err: Optional[float] = None
    lag1_err: Optional[float] = None

    def update_m1_err(self, x: float):
        if self.m1_err is None or self.m1_err < x:
            self.m1_err = x

    def update_cv_err(self, x: float):
        if self.cv_err is None or self.cv_err < x:
            self.cv_err = x

    def update_skew_err(self, x: float):
        if self.skew_err is None or self.skew_err < x:
            self.skew_err = x

    def update_lag1_err(self, x: float):
        if self.lag1_err is None or self.lag1_err < x:
            self.lag1_err = x


def solve_iterative(
        arrival: Optional[MarkovArrival] = None,
        service: Optional[PhaseType] = None,
        capacity: Optional[int] = None,
        net_size: Optional[int] = None,
        reducer: Optional[Callable[[MarkovArrival], MarkovArrival]] = None,
        reduce_arrival: bool = False,
        reduce_departure: bool = False,
        max_precise_order: int = 8000) -> SolveResults:
    """
    Solve MAP/PH/1/N -> */PH/1/N -> ... */PH/1/N model analytically.

    If `reducer` is not None, then this function is applied
    to each departure process prior to sending it to the arrival to the
    next station.

    If `reduce_arrival = True`, then `reducer()` is applied
    to the first arrival as well (`inp.arrival`).

    Parameters
    ----------
    arrival : MarkovArrival, optional
    service : PhaseType, optional
    capacity : int, optional
    net_size : int, optional
    reducer : None or Callable[[MarkovArrival], MarkovArrival]
        if not None, this function is applied to each departure
    reduce_arrival : bool, optional (default: False)
        if True, reduce arrival process as well
    max_precise_order : int (default: 8000)
        if looking for precise solution, will ignore matrices with number
        of rows (or columns) larger then this value

    Returns
    -------
    SolveResults
    """
    # Create solution that will be filled later, and start measuring time.
    t_start = perf_counter()
    solution = SolveResults(
        skipped=False, delay=0.0, delivery_prob=1.0, m1_err=0.0,
        cv_err=0.0, skew_err=0.0, lag1_err=0.0, max_inp_order=0,
        max_out_order=0)

    # Если нужно предобработать входной поток, делаем это
    _inp_arrival = arrival
    if reduce_arrival:
        # FIXME: another bad code
        try:
            arrival = reducer(_inp_arrival)
        except Exception:
            solution.skipped = True
            return solution
        # --- end of bad code (.. khm ..)
        solution.m1_err = rel_err(arrival.mean, _inp_arrival.mean)
        solution.cv_err = rel_err(arrival.cv, _inp_arrival.cv)
        solution.skew_err = rel_err(arrival.skewness, _inp_arrival.skewness)
        solution.lag1_err = rel_err(arrival.lag(1), _inp_arrival.lag(1))
    else:
        arrival = _inp_arrival

    # Since we now the maximum order, try to avoid useless iterations for
    # large tasks:
    complexity = get_complexity(arrival.order, service.order, capacity,
                                net_size)
    if not reduce_departure and complexity > max_precise_order:
        return SolveResults(skipped=True, max_out_order=complexity)

    # Итерационно рассчитываем характеристики сети
    solution.max_inp_order = arrival.order
    sta_index = 0
    while sta_index < net_size:
        # If arrival MAP matrix is too large, then abort execution:
        if not reduce_departure and arrival.order > max_precise_order:
            return SolveResults(skipped=True, max_inp_order=arrival.order)

        # Обновляем, если надо, максимальный размер входа:
        solution.max_inp_order = max(solution.max_inp_order, arrival.order)

        # Строим очередной узел
        system = MapPh1NQueue(arrival, service, capacity)
        dep = system.departure

        # FIXME: better find _real_ reasons for these errors.
        # --- from here: we check whether some error appeared and, if
        # so, reject to solve this problem by returning 'skipped = True'.
        # Most probably, these errors appear due to floating point arithmetics
        # and precision on large matrix operations. Another possible source
        # is in solving optimization problems when building departure
        # approximations.
        # For now, just skip such entries.
        try:
            # If departure matrix is too bad (e.g., rate is negative), skip:
            if (isinstance(dep.rate, complex) or
                    isinstance(dep.cv, complex) or
                    isinstance(dep.skewness, complex) or
                    isinstance(system.loss_prob, complex) or
                    isinstance(system.response_time, complex) or
                    dep.rate < 0 or dep.cv < 0 or system.loss_prob < 0 or
                    system.response_time < 0):
                solution.skipped = True
        except ValueError:
            solution.skipped = True
        if solution.skipped:
            solution.max_out_order = max(solution.max_out_order, dep.order)
            return solution

        # Рассчитываем и накапливаем задержку, вероятность доставки
        # и обновляем размер системы.
        solution.delay += system.response_time
        solution.delivery_prob *= 1 - system.loss_prob
        solution.last_system_size = system.system_size.mean
        solution.max_out_order = max(solution.max_out_order, dep.order)

        # Если нужно аппроксимировать выход, делаем это.
        # Иначе используем выход в качестве нового входа.
        if reduce_departure:
            # FIXME: another bad code
            try:
                arrival = reducer(dep)
            except Exception:
                solution.skipped = True
                return solution
            # --- end of bad code (.. khm ..)
            solution.update_m1_err(rel_err(arrival.mean, dep.mean))
            solution.update_cv_err(rel_err(arrival.cv, dep.cv))
            solution.update_skew_err(rel_err(arrival.skewness, dep.skewness))
            solution.update_lag1_err(rel_err(arrival.lag(1), dep.lag(1)))
        else:
            arrival = system.departure

        # Переходим к следующей станции
        sta_index += 1

    # Замеряем время завершения
    solution.elapsed = perf_counter() - t_start
    return solution


def reduce_map(
        arrival: MarkovArrival,
        num_moments: int = 3,
        use_lag: bool = False,
        tol: float = .01) -> MarkovArrival:
    """
    Find another MAP matching the given number of moments and, optionally,
    lag-1 correlation coefficient.

    Parameters
    ----------
    arrival : MarkovArrival
        the arrival process to reduce
    num_moments : 1, 2 or 3 (default: 3)
        number of moments to match
    use_lag : bool (default: False)
        flag indicating whether to try to fit lag-1 autocorrelation.
    tol : float (default: .01)
        when fitting, if cv differs on this value from 1.0, exponential
        distribution is used.

    Returns
    -------
    reduced : MarkovArrival
    """
    m1 = arrival.mean
    if num_moments == 1:
        ph = PhaseType.exponential(1 / m1)
    elif num_moments == 2:
        cv = arrival.cv
        std = arrival.std
        if cv < 0.99:
            ph = Erlang.fit(m1, std).as_ph()
        elif cv > 1.01:
            ph = HyperExponential.fit(m1, std).as_ph()
        else:
            ph = Exponential(1 / m1).as_ph()
    elif num_moments == 3:
        moments = [arrival.moment(i) for i in range(1, 4)]
        try:
            ph = fit_acph2(moments, strict=True)[0]
        except BoundsError:
            dist = fit_mern2(moments, strict=False)[0]
            ph = dist.as_ph()
    else:
        raise ValueError(f"expected num_moments = 1, 2 or 3, but "
                         f"{num_moments} found")
    # Fit lag, if needed:
    if use_lag:
        return fit_map_horvath05(ph, arrival.lag(1))[0]
    return MarkovArrival.phase_type(ph.s, ph.p)
