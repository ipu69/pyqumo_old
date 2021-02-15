import numpy as np
from libcpp.vector cimport vector
from pyqumo.cqumo.sim cimport SimData, NodeData, simMM1, VarData, simGG1, \
    makeDblFn, DblFn
from pyqumo.sim.helpers import Statistics
from pyqumo.sim.gg1 import Results as GG1Results
from pyqumo.sim.tandem import Results as TandemResults
from pyqumo.random import CountableDistribution, Exponential


# noinspection PyUnresolvedReferences
cdef vector_asarray(vector[double] vect):
    cdef int n = vect.size()
    ret = np.zeros(n)
    for i in range(n):
        ret[i] = vect[i]
    return ret


cdef _build_statistics(VarData* cs):
    return Statistics(avg=cs.mean, std=cs.std, var=cs.var, count=cs.count)


# noinspection PyUnresolvedReferences
cdef _build_gg1_results(const SimData& sim_data):
    cdef int addr = 0
    cdef NodeData data = sim_data.nodeData.at(0)
    results = GG1Results()
    results.system_size = CountableDistribution(
        vector_asarray(data.systemSize.pmf()))
    results.queue_size = CountableDistribution(
        vector_asarray(data.queueSize.pmf()))
    results.busy = CountableDistribution(
        vector_asarray(data.serverSize.pmf()))
    results.loss_prob = data.lossProb
    results.departures = _build_statistics(&data.departures)
    results.response_time = _build_statistics(&data.responseTime)
    results.wait_time = _build_statistics(&data.waitTime)
    results.real_time = sim_data.realTimeMs
    return results


cdef _build_tandem_results(const SimData& simData, int numStations):
    cdef int addr
    cdef NodeData nodeData
    results = TandemResults(num_stations=numStations)
    for addr in range(numStations):
        nodeData = simData.nodeData.at(addr)
        results.system_size.append(
            CountableDistribution(vector_asarray(nodeData.systemSize.pmf())))
        results.queue_size.append(
            CountableDistribution(vector_asarray(nodeData.queueSize.pmf())))
        results.busy.append(
            CountableDistribution(vector_asarray(nodeData.serverSize.pmf())))
        results.drop_prob.append(nodeData.dropProb)
        results.delivery_prob.append(nodeData.deliveryProb)
        results.departures.append(_build_statistics(&nodeData.departures))
        results.wait_time.append(_build_statistics(&nodeData.waitTime))
        results.response_time.append(_build_statistics(&nodeData.responseTime))
        results.delivery_delays.append(_build_statistics(&nodeData.delays))
    results.real_time = simData.realTimeMs
    return results

cdef double _call_pyobject(void *context):
    # noinspection PyBroadException
    try:
        evaluable = <object>context
        return evaluable.eval()
    except:
        return -1


cdef call_simGG1(
        void* pyArrival,
        void* pyService,
        int queue_capacity,
        int max_packets):
    cdef DblFn cArrival = makeDblFn(_call_pyobject, pyArrival)
    cdef DblFn cService = makeDblFn(_call_pyobject, pyService)
    cdef SimData c_ret = simGG1(
        cArrival, cService, queue_capacity, max_packets)
    result: Results = _build_gg1_results(c_ret)
    return result


cdef call_simMM1(
        double arrival_rate,
        double service_rate,
        int queue_capacity,
        int max_packets):
    cdef SimData c_ret = simMM1(
        arrival_rate, service_rate, queue_capacity, max_packets)
    result: Results = _build_gg1_results(c_ret)
    return result


cdef call_simTandem(
        void* pyArrival,
        vector[void*] pyServices,
        int queue_capacity,
        int max_packets):
    cdef DblFn cArrival = makeDblFn(_call_pyobject, pyArrival)
    cdef vector[DblFn] cServices
    cdef unsigned i
    for i in range(pyServices.size()):
        cServices.push_back(makeDblFn(_call_pyobject, pyServices[i]))
    cdef SimData c_ret = simTandem(
        cArrival, cServices, queue_capacity, max_packets)
    return _build_tandem_results(c_ret, pyServices.size())

def simulate_mm1n(
        arrival_rate: float,
        service_rate: float,
        queue_capacity: int,
        max_packets: int = 100000
) -> GG1Results:
    """
    Wrapper for C++ implementation of M/M/1/N or M/M/1 model.

    Returns results in the same dataclass as defined for G/G/1 model
    in `pyqumo.sim.gg1.Results`.

    Parameters
    ----------
    arrival_rate : float
    service_rate : float
    queue_capacity : int
    max_packets : int, optional
        By default 100'000

    Returns
    -------
    results : Results
    """
    return call_simMM1(arrival_rate, service_rate, queue_capacity, max_packets)


def simulate_gg1n(
        arrival,
        service,
        queue_capacity: int,
        max_packets: int = 100000
) -> GG1Results:
    """
    Wrapper for C++ implementation of G/G/1/N or G/G/1 model.

    Returns results in the same dataclass as defined for G/G/1 model
    in `pyqumo.sim.gg1.Results`.

    Parameters
    ----------
    arrival : callable `() -> double`
    service : callable `() -> double`
    queue_capacity : int
    max_packets : int, optional
        By default 100'000

    Returns
    -------
    results : Results
    """
    cdef void* pyArrival = <void*>arrival.rnd
    cdef void* pyService = <void*>service.rnd
    if queue_capacity == np.inf:
        queue_capacity = -1
    return call_simGG1(pyArrival, pyService, queue_capacity, max_packets)


def simulate_tandem(
        arrival,
        services,
        queue_capacity: int,
        max_packets: int = 100000
) -> TandemResults:
    """
    Simulate tandem using C++ implementation. No cross-traffic supported.

    Example
    -------
    >>> simulate_tandem(
    >>>     Erlang(2, 1), 
    >>>     [Exponential(5), Exponential(6), Exponential(3)], 
    >>>     10,  # queue capacity
    >>>     1000000)  # number of packets (1 million)

    Results returned are defined in pyqumo.sim.tandem.Results.

    Parameters
    ----------
    arrival : Distribution instance
    services : list of Distribution instances, this list size is the number
        of nodes in the tandem network
    queue_capacity: int or np.inf
    max_packets: 
    """
    cdef void* pyArrival = <void*>arrival.rnd
    cdef vector[void*] pyServices
    cdef unsigned i = 0
    for i in range(len(services)):
        pyServices.push_back(<void*>services[i].rnd)
    if queue_capacity == np.inf:
        queue_capacity = -1
    return call_simTandem(pyArrival, pyServices, queue_capacity, max_packets)
