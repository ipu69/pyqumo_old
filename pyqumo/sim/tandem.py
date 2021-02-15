from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple, Sequence, Callable, Union
import numpy as np
from tabulate import tabulate

from pyqumo.matrix import str_array
from pyqumo.random import CountableDistribution, Distribution
from pyqumo.sim.helpers import Statistics, build_statistics, Queue, \
    TimeSizeRecords, FiniteFifoQueue, InfiniteFifoQueue, Server


class Packet:
    """
    Packet representation for tandem G/G/1/N model.

    Stores timestamps, indexes and flags:
    - the first station the packet entered: source
    - when the packet arrived: arrived_time_list[n]
    - when the packet started serve: service_started_time_list[n]
    - when the packet finished serving: service_finished_time_list[n]
    - when the packet was dropped (if was): drop_time
    - where the packet was dropped (if was): drop_node
    - whether the packet was dropped: was_dropped
    - whether the packet was delivered: was_delivered
    - where to deliver: target

    In all lists index is the number of station.
    """
    def __init__(self, source: int, target: int, num_stations: int):
        """
        Create the packet.

        Parameters
        ----------
        source : int
            Node index where the packet was created
        target : int
            Node index where the packet should be delivered
        num_stations : int
            Number of stations in the network
        """

        def create_list() -> List[Optional[float]]:
            """Helper for empty list creation."""
            return [None] * num_stations

        self.source: int = source
        self.target: int = target

        # Information about packet arrivals at each node:
        self.arrived: List[Optional[float]] = create_list()

        # Information about service start, finish and success, per node
        self.service_started: List[Optional[float]] = create_list()
        self.service_finished: List[Optional[float]] = create_list()
        self.was_served: List[bool] = [False] * num_stations

        # Information about whether, where and when the packet was dropped
        self.was_dropped: bool = False
        self.drop_time: Optional[float] = None
        self.drop_node: Optional[int] = None

        # Information about whether and when the packet was delivered
        self.was_delivered: bool = False
        self.delivery_time: Optional[float] = None


class Records:
    """
    Statistical records.
    """
    def __init__(self, num_stations: int):
        """
        Create records.

        Parameters
        ----------
        num_stations : int
        """
        self._packets: List[Packet] = []
        self._system_sizes: List[TimeSizeRecords] = [
            TimeSizeRecords() for _ in range(num_stations)
        ]
        self._num_stations = num_stations

    def get_system_size(self, node: int) -> TimeSizeRecords:
        """
        Get the system size recorded for the given node.

        Parameters
        ----------
        node : int

        Returns
        -------
        records : TimeSizeRecords
        """
        return self._system_sizes[node]

    def add_packet(self, packet: Packet) -> None:
        self._packets.append(packet)

    @property
    def packets(self) -> List[Packet]:
        """
        Get packets list.
        """
        return self._packets

    @property
    def num_stations(self) -> int:
        """
        Get number of stations.
        """
        return self._num_stations


CountDistList = List[Optional[CountableDistribution]]


class Results:
    """
    Results returned from G/G/1/N model simulation.

    Discrete stochastic properties like system size, queue size and busy
    periods are represented with `CountableDistribution`. Continuous properties
    are not fitted into any kind of distribution, they are represented with
    `Statistics` tuples.

    Utilization coefficient, as well as loss probability, are just floating
    point numbers.

    To pretty print the results one can make use of `tabulate()` method.
    """
    def __init__(self, records: Optional[Records] = None, 
                 num_stations: Optional[int] = None):
        """
        Create results.

        Parameters
        ----------
        records : Records, optional
        num_stations: int, optional
        """
        self.real_time = 0.0
        self.system_size: List[CountableDistribution] = []
        self.queue_size: List[CountableDistribution] = []
        self.busy: List[CountableDistribution] = []
        self.drop_prob: List[float] = []
        self.delivery_prob: List[float] = []
        self.departures: List[Statistics] = []
        self.arrivals: List[Statistics] = []
        self.wait_time: List[Statistics] = []
        self.response_time: List[Statistics] = []
        self.delivery_delays: List[Statistics] = []

        self._num_stations = records.num_stations if records is not None \
            else num_stations

        if records is None:
            return

        def idiv(x: int, y: int, default: float = 0.0) -> float:
            return x / y if y != 0 else default

        for i in range(records.num_stations):
            #
            # 1) Build system size, queue size and busy (server size)
            #    distributions for each node. To do this, we need PMFs.
            #    Queue size PMF and busy PMF can be computed from system size PMF.
            #
            system_size_pmf = list(records.get_system_size(i).pmf)
            num_states = len(system_size_pmf)
            p0 = system_size_pmf[0]
            p1 = system_size_pmf[1] if num_states > 1 else 0.0

            queue_size_pmf = [p0 + p1] + system_size_pmf[2:]
            server_size_pmf = [p0, sum(system_size_pmf[1:])]

            self.system_size.append(CountableDistribution(system_size_pmf))
            self.queue_size.append(CountableDistribution(queue_size_pmf))
            self.busy.append(CountableDistribution(server_size_pmf))

            #
            # 2) For future estimations, we need packets and some filters.
            #    Group all of them here.
            #
            packets = records.packets
            arrived_packets = [p for p in packets if p.arrived[i] is not None]
            served_packets = [p for p in arrived_packets if p.was_served[i]]
            dropped_here_packets = [
                p for p in arrived_packets
                if p.was_dropped and p.drop_node == i]
            source_packets = [p for p in packets if p.source == i]
            delivered_packets = [p for p in source_packets if p.was_delivered]
            dropped_from_packets = [p for p in source_packets if p.was_dropped]

            #
            # 3) Build scalar statistics.
            #
            num_arrived = len(arrived_packets)
            num_dropped_here = len(dropped_here_packets)
            num_dropped_from = len(dropped_from_packets)
            num_delivered = len(delivered_packets)

            # Drop - packet arrived here (at i), but was not queued and dropped
            p_drop = idiv(num_dropped_here, num_arrived)

            # Delivery - packet was generated here (at i) and was delivered
            # (at some another station j).
            # Here we ignore packets those were generated here, but didn't
            # finish transmission till the simulation end.
            p_delivery = idiv(num_delivered, num_delivered + num_dropped_from,
                              default=1.0)

            # Record 'em!
            self.drop_prob.append(p_drop)
            self.delivery_prob.append(p_delivery)

            #
            # 4) Build various intervals statistics: departures, waiting times,
            #    response times.
            #
            departures = _get_packet_intervals(
                served_packets, i, lambda pkt, node: pkt.service_finished[node]
            )
            arrivals = _get_packet_intervals(
                arrived_packets, i, lambda pkt, node: pkt.arrived[node]
            )
            self.departures.append(build_statistics(departures))
            self.arrivals.append(build_statistics(arrivals))
            self.response_time.append(build_statistics([
                p.service_finished[i] - p.arrived[i] for p in served_packets]))
            self.wait_time.append(build_statistics([
                p.service_started[i] - p.arrived[i] for p in served_packets]))
            self.delivery_delays.append(build_statistics([
                p.delivery_time - p.arrived[i] for p in delivered_packets]))

    def get_utilization(self, node: int) -> float:
        """
        Get utilization coefficient, that is `Busy = 1` probability.
        """
        return self.busy[node].pmf(1)

    def tabulate(self) -> str:
        """
        Build a pretty formatted table with all key properties.
        """
        items = [
            ('Number of stations', self._num_stations),
            ('Loss probability', self.drop_prob),
        ]

        for node in range(self._num_stations):
            items.append((f'[[ STATION #{node} ]]', ''))

            ssize = self.system_size[node]
            qsize = self.queue_size[node]
            busy = self.busy[node]

            ssize_pmf = [ssize.pmf(x) for x in range(ssize.truncated_at + 1)]
            qsize_pmf = [qsize.pmf(x) for x in range(qsize.truncated_at + 1)]
            busy_pmf = [busy.pmf(x) for x in range(busy.truncated_at + 1)]

            items.extend([
                ('System size PMF', str_array(ssize_pmf)),
                ('System size average', ssize.mean),
                ('System size std.dev.', ssize.std),
                ('Queue size PMF', str_array(qsize_pmf)),
                ('Queue size average', qsize.mean),
                ('Queue size std.dev.', qsize.std),
                ('Busy PMF', str_array(busy_pmf)),
                ('Utilization', self.get_utilization(node)),
                ('Drop probability', self.drop_prob[node]),
                ('Delivery probability', self.delivery_prob[node]),
                ('Departures, average', self.departures[node].avg),
                ('Departures, std.dev.', self.departures[node].std),
                ('Response time, average', self.response_time[node].avg),
                ('Response time, std.dev.', self.response_time[node].std),
                ('Wait time, average', self.wait_time[node].avg),
                ('Wait time, std.dev.', self.wait_time[node].std),
                ('End-to-end delays, average', self.delivery_delays[node].avg),
                ('End-to-end delays, std.dev.', self.delivery_delays[node].std),
            ])
        return tabulate(items, headers=('Param', 'Value'))


@dataclass
class Params:
    """
    Model parameters: arrival and service processes, queue capacity and limits.
    """
    arrivals: List[Optional[Distribution]]
    services: List[Distribution]
    num_stations: int
    queue_capacity: int = np.inf
    max_packets: int = 1000000
    max_time: float = np.inf


class Event(Enum):
    STOP = 0
    ARRIVAL = 1
    SERVICE_END = 2


class System:
    """
    System state representation.

    This object takes care of the queue, current time, next arrival and
    service end time, server status and any other kind of dynamic information,
    except for internal state of arrival or service processes.
    """
    def __init__(self, params: Params):
        """
        Constructor.

        Parameters
        ----------
        params : Params
            Model parameters
        """
        def create_queue() -> Queue[Packet]:
            if params.queue_capacity < np.inf:
                return FiniteFifoQueue(params.queue_capacity)
            return InfiniteFifoQueue()

        def create_server() -> Server[Packet]:
            return Server()

        n: int = params.num_stations
        self._queues = [create_queue() for _ in range(n)]
        self._servers = [create_server() for _ in range(n)]
        self._time: float = 0.0
        self._service_ends: List[Optional[float]] = [None] * n
        self._next_arrivals: List[Optional[float]] = [None] * n
        self._stopped: bool = False

    @property
    def time(self):
        return self._time

    @property
    def stopped(self):
        return self._stopped

    def get_queue(self, node: int) -> Queue[Packet]:
        return self._queues[node]

    def get_server(self, node: int) -> Server[Packet]:
        return self._servers[node]

    def reset_time(self):
        self._time = 0.0

    def stop(self):
        self._stopped = True

    def get_size(self, node: int):
        """
        Get system size, that is queue size plus one (busy) or zero (empty).
        """
        return self._servers[node].size + self._queues[node].size

    def schedule(self, event: Event, interval: float, node: int = 0):
        if interval < 0:
            raise RuntimeError(f"non-negative interval expected, "
                               f"but {interval} found")
        fire_time = self._time + interval
        if event == Event.ARRIVAL:
            self._next_arrivals[node] = fire_time
        else:
            assert event == Event.SERVICE_END, event
            self._service_ends[node] = fire_time

    def next_event(self) -> Tuple[Event, int]:
        """
        Get next event type and move time.

        Returns
        -------
        pair : tuple of event and station index
        """
        def find_min(values: Sequence[Optional[float]]) -> \
                Tuple[Optional[int], float]:
            """Helper to get minimum timestamp and its index."""
            min_value: float = np.inf
            min_index: Optional[int] = None
            for index_, value_ in enumerate(values):
                if value_ is not None and value_ < min_value:
                    min_value = value_
                    min_index = index_
            return min_index, min_value

        min_arrival_node, min_arrival_time = find_min(self._next_arrivals)
        min_service_node, min_service_time = find_min(self._service_ends)

        if min_arrival_node is None and min_service_node is None:
            return Event.STOP, 0

        if min_arrival_time >= min_service_time:
            self._time = min_service_time
            self._service_ends[min_service_node] = None
            return Event.SERVICE_END, min_service_node

        # Otherwise, arrival happened:
        self._time = min_arrival_time
        self._next_arrivals[min_arrival_node] = None
        return Event.ARRIVAL, min_arrival_node


def simulate(
        arrivals: Union[Distribution, Sequence[Optional[Distribution]]],
        services: Union[Distribution, Sequence[Distribution]],
        queue_capacity: int = np.inf,
        num_stations: int = 1,
        cross_traffic: bool = False,
        max_time: float = np.inf,
        max_packets: int = 1000000
) -> Results:
    """
    Run simulation model of G/G/1/N system.

    Simulation can be stopped in two ways: by reaching maximum simulation time,
    or by reaching the maximum number of generated packets. By default,
    simulation is limited with the maximum number of packets only (1 million).

    Queue is expected to have finite capacity.

    Arrival and service time processes can be of any kind, including Poisson
    or MAP. To use a PH or normal distribution, a GenericIndependentProcess
    model with the corresponding distribution may be used.

    Parameters
    ----------
    arrivals : RandomProcess or sequence of RandomProcess
        Arrival random process or sequence of RandomProcess
    services : RandomProcess
        Service time random process.
    queue_capacity : int
        Queue capacity.
    num_stations : int
        Number of stations in the network
    cross_traffic : bool, optional
        If arrivals is a single random var, and cross_traffic = True,
        then a separate arrival will be attached to each queue.
        Default: False.
    max_time : float, optional
        Maximum simulation time (default: infinity).
    max_packets
        Maximum number of simulated packets (default: 1'000'000)

    Returns
    -------
    results : Results
        Simulation results.
    """
    arrivals_: List[Optional[Distribution]] = []
    if isinstance(arrivals, Distribution):
        if cross_traffic:
            arrivals_ = [arrivals.copy() for _ in range(num_stations)]
        else:
            arrivals_ = [None] * num_stations
            arrivals_[0] = arrivals
    else:
        arrivals_ = [x.copy() if x is not None else None for x in arrivals]

    if isinstance(services, Distribution):
        services_ = [services.copy() for _ in range(num_stations)]
    else:
        services_ = [x.copy() for x in services]

    if (n := len(arrivals_)) != num_stations:
        raise ValueError(f"expected {num_stations} arrivals, but {n} found")
    if (n := len(services_)) != num_stations:
        raise ValueError(f"expected {num_stations} services, but {n} found")

    params = Params(
        arrivals=arrivals_,
        services=services_,
        queue_capacity=queue_capacity,
        num_stations=num_stations,
        max_packets=max_packets,
        max_time=max_time
    )
    system = System(params)
    records = Records(num_stations)

    # Initialize:
    system.reset_time()
    for node, arrival in enumerate(params.arrivals):
        if arrival is not None:
            system.schedule(Event.ARRIVAL, arrival(), node)
    for node in range(params.num_stations):
        records.get_system_size(node).add(0.0, 0)

    # Run!
    max_time = params.max_time
    while not system.stopped:
        # Extract new event:
        event, node = system.next_event()

        # Check whether event is scheduled too late, and we need to stop:
        if system.time > max_time:
            system.stop()
            continue

        # Process event
        if event == Event.ARRIVAL:
            _handle_arrival(node, system, params, records)
        elif event == Event.SERVICE_END:
            _handle_service_end(node, system, params, records)
        elif event == Event.STOP:
            system.stop()

    return Results(records)


def _process_packet(node: int, packet: Packet, system: System, params: Params,
                    records: Records):
    """
    Helper for processing a new packet at the given node.

    Parameters
    ----------
    node : int
    packet : Packet
    system : System
    params : Params
    records : Records
    """
    time_now = system.time
    packet.arrived[node] = time_now

    # If server is ready, start serving. Otherwise, push the packet into
    # the queue. If the queue was full, mark the packet is being dropped
    # for further analysis.
    server = system.get_server(node)
    queue = system.get_queue(node)
    if server.ready:
        # start serving immediately
        server.serve(packet)
        packet.service_started[node] = time_now
        system.schedule(Event.SERVICE_END, params.services[node](), node)
        records.get_system_size(node).add(time_now, system.get_size(node))

    elif queue.push(packet):
        # packet was queued
        records.get_system_size(node).add(time_now, system.get_size(node))

    else:
        # mark packet as being dropped
        packet.was_dropped = True
        packet.drop_node = node
        packet.drop_time = time_now


def _handle_arrival(node: int, system: System, params: Params,
                    records: Records):
    """
    Handle new packet arrival event.

    First of all, a new packet is created. Then we check whether the
    system is empty. If it is, this new packet starts serving immediately.
    Otherwise, it is added to the queue.

    If the queue was full, the packet is dropped. To mark this, we set
    `dropped` flag in the packet to `True`.

    In the end we schedule the next arrival. We also check whether the
    we have already generated enough packets. If so, `system.stopped` flag
    is set to `True`, so on the next main loop iteration the simulation
    will be stopped.

    Parameters
    ----------
    node : int
    system : System
    params : Params
    records : Records
    """
    num_packets_built = len(records.packets)
    if num_packets_built >= params.max_packets:
        # If too many packets were generated, ask to stop:
        system.stop()

    # Create and record new packet:
    packet = Packet(
        source=node,
        target=(params.num_stations - 1),
        num_stations=params.num_stations)
    records.add_packet(packet)

    # Process packet arrival at the node
    _process_packet(node, packet, system, params, records)

    # Schedule next arrival
    system.schedule(Event.ARRIVAL, params.arrivals[node](), node)


def _handle_service_end(node: int, system: System, params: Params,
                        records: Records):
    """
    Handle end of the packet service.

    If the queue is empty, the server becomes idle. Otherwise, it starts
    serving the next packet from the queue.

    The packet that left the server is marked as `served = True`.

    Parameters
    ----------
    system : System
    params : Params
    records : Records
    """
    time_now = system.time
    server = system.get_server(node)
    queue = system.get_queue(node)

    # Extract the packet from the server and mark it as served:
    packet = server.pop()
    packet.service_finished[node] = time_now
    packet.was_served[node] = True

    # If this node is the target, then packet is delivered.
    # Otherwise, pass it to the next node.
    if packet.target == node:
        packet.was_delivered = True
        packet.delivery_time = time_now
    elif node < params.num_stations - 1:
        _process_packet(node + 1, packet, system, params, records)
    else:
        raise RuntimeError("shouldn't arrive here: packet reached network"
                           "end, and this is not its destination!!")

    # Start serving next packet, if exists:
    if (packet := queue.pop()) is not None:
        server.serve(packet)
        packet.service_started[node] = time_now
        system.schedule(Event.SERVICE_END, params.services[node](), node)

    # Record new system size:
    records.get_system_size(node).add(time_now, system.get_size(node))


def _get_packet_intervals(
        packets: Sequence[Packet],
        node: int,
        getter: Callable[[Packet, int], float]
) -> np.ndarray:
    """
    Build departures intervals sequence.

    Parameters
    ----------
    packets : sequence of Packet
    node : int

    Returns
    -------
    intervals : np.ndarray
    """
    prev_time = 0.0
    intervals = []
    for packet in packets:
        if packet.was_served[node]:
            new_time = getter(packet, node)
            intervals.append(new_time - prev_time)
            prev_time = new_time
    return np.asarray(intervals)
