from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional, Callable

from pyqumo import MarkovArrival, PhaseType, rel_err, MapPh1NQueue


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
        inp: Dict,
        reducer: Optional[Callable[[MarkovArrival], MarkovArrival]] = None,
        reduce_arrival: bool = False,
        arrival: Optional[MarkovArrival] = None,
        service: Optional[PhaseType] = None,
        capacity: Optional[int] = None,
        net_size: Optional[int] = None) -> SolveResults:
    """
    Solve MAP/PH/1/N -> */PH/1/N -> ... */PH/1/N model analytically.

    If `reducer` is not None, then this function is applied
    to each departure process prior to sending it to the arrival to the
    next station.

    If `reduce_arrival = True`, then `reducer()` is applied
    to the first arrival as well (`inp.arrival`).

    Parameters
    ----------
    inp : dict-like, optional
        assumed that `inp` has at least `arrival`, `service`,
        `net_size` and `capacity` fields. Also can contain 'complexity',
        which values could be 'complex' or 'simple'.
    arrival : MarkovArrival, optional
    service : PhaseType, optional
    capacity : int, optional
    net_size : int, optional
    reducer : None or Callable[[MarkovArrival], MarkovArrival]
        if not None, this function is applied to each departure
    reduce_arrival : bool, optional (default: False)
        if True, reduce arrival process as well

    Returns
    -------
    SolveResults
    """
    is_precise = reducer is None
    is_complex = inp.get('complexity', 'simple') != 'simple'

    # To solve the problem precisely (without departure processes
    # approximations), we use only those inputs which are marked as simple.
    if is_precise and is_complex:
        return SolveResults(skipped=True)

    # Create solution that will be filled later, and start measuring time.
    t_start = perf_counter()
    solution = SolveResults(
        skipped=False, delay=0.0, delivery_prob=1.0, m1_err=0.0,
        cv_err=0.0, skew_err=0.0, lag1_err=0.0, max_inp_order=0,
        max_out_order=0)

    # Если нужно предобработать входной поток, делаем это
    _inp_arrival = arrival or inp['arrival']
    if reduce_arrival:
        arrival = reducer(_inp_arrival)
        solution.m1_err = rel_err(arrival.mean, _inp_arrival.mean)
        solution.cv_err = rel_err(arrival.cv, _inp_arrival.cv)
        solution.skew_err = rel_err(arrival.skewness, _inp_arrival.skewness)
        solution.lag1_err = rel_err(arrival.lag(1), _inp_arrival.lag(1))
    else:
        arrival = _inp_arrival

    service = service or inp['service']
    capacity = capacity if capacity is not None else inp['capacity']
    net_size = net_size if net_size is not None else inp['net_size']

    print("going to solve problem with complexity: ", get_complexity(
        arrival.order, service.order, capacity, net_size
    ))

    # Итерационно рассчитываем характеристики сети
    solution.max_inp_order = arrival.order
    sta_index = 0
    while sta_index < net_size:
        # Обновляем, если надо, максимальный размер входа:
        solution.max_inp_order = max(solution.max_inp_order, arrival.order)

        # Строим очередной узел
        system = MapPh1NQueue(arrival, service, capacity)
        dep = system.departure

        # Рассчитываем и накапливаем задержку, вероятность доставки
        # и обновляем размер системы.
        solution.delay += system.response_time
        solution.delivery_prob *= 1 - system.loss_prob
        solution.last_system_size = system.system_size.mean
        solution.max_out_order = max(solution.max_out_order, dep.order)

        # Если нужно аппроксимировать выход, делаем это.
        # Иначе используем выход в качестве нового входа.
        if reducer is not None:
            arrival = reducer(dep)
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
