# from dataclasses import dataclass
#
#
# @dataclass
# class Packet:
#     source: int
#     created_at: float
#     arrived_at: float
#     service_started_at: float
#
#     def __init__(self, source: int, created_at: float):
#         self._source = source
#         self._created_at = created_at
#         self.arrived_at = None
#         self.service_started_at = None
#
#     @property
#     def source(self) -> int:
#         return self._source
#
#     @property
#     def created_at(self) -> float:
#         return self._created_at
#
#     def __str__(self):
#         return f'(Packet: src={self.source}, t={self.created_at:.3g})'
#
#
# def _handle_arrival
#
#
# class QueueingSystem(Model):
#     """QueueingSystem model represents a simple queue */*/1/N or */*/1,
#     where any distribution can be used for arrival and service times.
#
#     This model consists of four children:
#     - `queue`: a model representing the packets queue (`Queue`)
#     - `source`: a model representing the packets source (`Source`)
#     - `server`: a model representing the packets server (`Server`)
#     - `sink`: a model which collects served packets (`Sink`)
#     """
#
#     def __init__(self, sim):
#         super().__init__(sim)
#
#         arrival = sim.params.arrival
#         service = sim.params.service
#         queue_capacity = sim.params.queue_capacity
#
#         self.children['queue'] = Queue(sim, queue_capacity)
#         self.children['source'] = Source(sim, arrival, index=0)
#         self.children['server'] = Server(sim, service)
#         self.children['sink'] = Sink(sim)
#
#         # Building connections:
#         self.source.connections['queue'] = self.queue
#         self.queue.connections['server'] = self.server
#         self.server.connections['queue'] = self.queue
#         self.server.connections['next'] = self.sink
#
#         # Statistics:
#         self.system_size_trace = Trace()
#         self.system_size_trace.record(self.sim.stime, 0)
#         self.system_wait_intervals = Statistic()
#
#     @property
#     def queue(self):
#         return self.children['queue']
#
#     @property
#     def source(self):
#         return self.children['source']
#
#     @property
#     def server(self):
#         return self.children['server']
#
#     @property
#     def sink(self):
#         return self.children['sink']
#
#     @property
#     def system_size(self):
#         return self.queue.size + (1 if self.server.busy else 0)
#
#     def update_system_size(self, index=0):
#         assert index == 0
#         self.system_size_trace.record(self.sim.stime, self.system_size)
#
#     def add_system_wait_interval(self, value, index=0):
#         assert index == 0
#         self.system_wait_intervals.append(value)
#
#
# class QueueingTandemNetwork(Model):
#     """Queueing tandem network model.
#     """
#
#     def __init__(self, sim):
#         super().__init__(sim)
#
#         arrivals = sim.params.arrivals
#         services = sim.params.services
#         queue_capacity = sim.params.queue_capacity
#         n = sim.params.num_stations
#         active_sources = {i for i, ar in enumerate(arrivals) if ar is not None}
#
#         if n < 1:
#             raise ValueError('num_stations must be >= 1')
#
#         self.queues, self.sources, self.servers = [], [], []
#         for i in range(n):
#             self.queues.append(Queue(sim, queue_capacity, i))
#             self.servers.append(Server(sim, services[i], i))
#             if i in active_sources:
#                 self.sources.append(Source(sim, arrivals[i], i))
#             else:
#                 self.sources.append(None)
#         self.sink = Sink(sim)
#         self.children['queue'] = self.queues
#         self.children['server'] = self.servers
#         self.children['sink'] = self.sink
#         self.children['sources'] = [src for src in self.sources if src]
#
#         # Connecting modules:
#         for i in range(n):
#             self.queues[i].connections['server'] = self.servers[i]
#             self.servers[i].connections['queue'] = self.queues[i]
#             if self.sources[i]:
#                 self.sources[i].connections['queue'] = self.queues[i]
#             if i < n - 1:
#                 self.servers[i].connections['next'] = self.queues[i + 1]
#             else:
#                 self.servers[i].connections['next'] = self.sink
#
#         # Statistics:
#         self.system_size_trace = [Trace() for _ in range(n)]
#         for i in range(n):
#             self.system_size_trace[i].record(sim.stime, 0)
#         self.system_wait_intervals = [Statistic() for _ in range(n)]
#
#     def get_system_size(self, index):
#         return self.queues[index].size + (1 if self.servers[index].busy else 0)
#
#     def update_system_size(self, index):
#         self.system_size_trace[index].record(
#             self.sim.stime, self.get_system_size(index)
#         )
#
#     def add_system_wait_interval(self, value, index):
#         self.system_wait_intervals[index].append(value)
#
#
# class QueueingTandemNetworkWithFixedService(Model):
#     """Queueing tandem network model with equal service times at servers.
#     """
#
#     def __init__(self, sim):
#         super().__init__(sim)
#
#         arrivals = sim.params.arrivals
#         service = sim.params.service
#         queue_capacity = sim.params.queue_capacity
#         n = sim.params.num_stations
#         active_sources = {i for i, ar in enumerate(arrivals) if ar is not None}
#
#         if n < 1:
#             raise ValueError('num_stations must be >= 1')
#
#         self.queues, self.sources, self.servers = [], [], []
#         for i in range(n):
#             self.queues.append(Queue(sim, queue_capacity, i))
#             self.servers.append(FixedServer(sim, service, i))
#             if i in active_sources:
#                 self.sources.append(Source(sim, arrivals[i], i))
#             else:
#                 self.sources.append(None)
#         self.sink = Sink(sim)
#         self.children['queue'] = self.queues
#         self.children['server'] = self.servers
#         self.children['sink'] = self.sink
#         self.children['sources'] = [src for src in self.sources if src]
#
#         # Connecting modules:
#         for i in range(n):
#             self.queues[i].connections['server'] = self.servers[i]
#             self.servers[i].connections['queue'] = self.queues[i]
#             if self.sources[i]:
#                 self.sources[i].connections['queue'] = self.queues[i]
#             if i < n - 1:
#                 self.servers[i].connections['next'] = self.queues[i + 1]
#             else:
#                 self.servers[i].connections['next'] = self.sink
#
#         # Statistics:
#         self.system_size_trace = [Trace() for _ in range(n)]
#         for i in range(n):
#             self.system_size_trace[i].record(sim.stime, 0)
#         self.system_wait_intervals = [Statistic() for _ in range(n)]
#
#     def get_system_size(self, index):
#         return self.queues[index].size + (1 if self.servers[index].busy else 0)
#
#     def update_system_size(self, index):
#         self.system_size_trace[index].record(
#             self.sim.stime, self.get_system_size(index)
#         )
#
#     def add_system_wait_interval(self, value, index):
#         self.system_wait_intervals[index].append(value)
#
#
# class Queue(Model):
#     """Queue module represents the packets queue, stores only current size.
#
#     Connections: server
#
#     Methods and properties:
#     - push(): increase the queue size
#     - pop(): decrease the queue size
#     - size: get current queue size
#
#     Statistics:
#     - size_trace: Trace, holding the history of the queue size updates
#     - num_arrived: `int`
#     - num_dropped: `int`
#     - drop_ratio: `float`
#     """
#
#     def __init__(self, sim, capacity, index=0):
#         super().__init__(sim)
#         self.__capacity = capacity
#         self.packets = deque()
#         self.index = index
#         # Statistics:
#         self.size_trace = Trace()
#         self.size_trace.record(self.sim.stime, 0)
#         self.num_dropped = 0
#         self.arrival_intervals = Intervals()
#         self.arrival_intervals.record(sim.stime)
#         self.num_arrived = 0
#         self.wait_intervals = Statistic()
#
#     @property
#     def capacity(self):
#         return self.__capacity
#
#     @property
#     def size(self):
#         return len(self.packets)
#
#     @property
#     def drop_ratio(self):
#         if self.num_arrived == 0:
#             return 0
#         return self.num_dropped / self.num_arrived
#
#     def handle_message(self, message, connection=None, sender=None):
#         self.push(message)
#
#     def push(self, packet):
#         self.arrival_intervals.record(self.sim.stime)
#         self.num_arrived += 1
#         server = self.connections['server'].module
#
#         packet.arrived_at = self.sim.stime
#
#         if self.size == 0 and not server.busy:
#             server.serve(packet)
#             self.wait_intervals.append(self.sim.stime - packet.arrived_at)
#
#         elif self.capacity is None or self.size < self.capacity:
#             self.packets.append(packet)
#             self.size_trace.record(self.sim.stime, self.size)
#
#         else:
#             self.num_dropped += 1
#
#         self.parent.update_system_size(self.index)
#
#     def pop(self):
#         packet = self.packets.popleft()
#         self.size_trace.record(self.sim.stime, self.size)
#         self.wait_intervals.append(self.sim.stime - packet.arrived_at)
#         return packet
#
#     def __str__(self):
#         return f'Queue({self.size})'
#
#
# class Source(Model):
#     """Source module represents the traffic source with exponential intervals.
#
#     Connections: queue
#
#     Handlers:
#     - handle_timeout(): called upon next arrival timeout
#
#     Statistics:
#     - intervals: `Intervals`, stores a set of inter-arrival intervals
#
#     Parent: `QueueingSystem`
#     """
#
#     def __init__(self, sim, arrival, index):
#         super().__init__(sim)
#         self.arrival = arrival
#         self.index = index
#
#         # Statistics:
#         self.intervals = Intervals()
#         self.num_generated = 0
#         self.delays = Statistic()
#
#         # Initialize:
#         self._schedule_next_arrival()
#
#     def handle_timeout(self):
#         packet = Packet(self, self.sim.stime)
#         self.connections['queue'].send(packet)
#         self._schedule_next_arrival()
#         self.num_generated += 1
#
#     def _schedule_next_arrival(self):
#         self.intervals.record(self.sim.stime)
#         self.sim.schedule(self.arrival(), self.handle_timeout)
#
#
# class Server(Model):
#     """Server module represents a packet server with exponential service time.
#
#     Connections: queue, sink
#
#     Handlers:
#     - on_service_end(): called upon service timeout
#
#     Methods:
#     - start_service(): start new packet service; generate error if busy.
#
#     Statistics:
#     - delays: `Statistic`, stores a set of service intervals
#     - busy_trace: `Trace`, stores a vector of server busy status
#     - num_served: `int`
#
#     Parent: `QueueingSystem`
#     """
#
#     def __init__(self, sim, service_time, index=0):
#         super().__init__(sim)
#         self.service_time = service_time
#         self.packet = None
#         self.index = index
#
#         # Statistics:
#         self.service_intervals = Statistic()
#         self.busy_trace = Trace()
#         self.busy_trace.record(self.sim.stime, 0)
#         self.departure_intervals = Intervals()
#         self.departure_intervals.record(sim.stime)
#         self.num_served = 0
#
#     @property
#     def busy(self):
#         return self.packet is not None
#
#     @property
#     def ready(self):
#         return self.packet is None
#
#     def handle_service_end(self):
#         assert self.busy
#         stime = self.sim.stime
#
#         self.connections['next'].send(self.packet)
#         self.parent.add_system_wait_interval(
#             stime - self.packet.arrived_at, self.index
#         )
#
#         self.packet = None
#
#         self.busy_trace.record(stime, 0)
#         self.departure_intervals.record(stime)
#         self.num_served += 1
#
#         # Requesting next packet from the queue:
#         queue = self.connections['queue'].module
#         if queue.size > 0:
#             self.serve(queue.pop())
#
#         self.parent.update_system_size(self.index)
#
#     def _start_service(self):
#         delay = self.service_time()
#         self.sim.schedule(delay, self.handle_service_end)
#         return delay
#
#     def serve(self, packet):
#         assert not self.busy
#         self.packet = packet
#         delay = self._start_service()
#         self.service_intervals.append(delay)
#         self.busy_trace.record(self.sim.stime, 1)
#
#
# class FixedServer(Server):
#     def __init__(self, sim, service_time, index=0):
#         super().__init__(sim, service_time, index)
#
#     def _start_service(self):
#         if self.packet.service_time is None:
#             self.packet.service_time = self.service_time()
#         self.sim.schedule(self.packet.service_time, self.handle_service_end)
#         return self.packet.service_time
#
#
# class Sink(Model):
#     """Sink module represents the traffic sink and counts arrived packets.
#
#     Methods:
#     - receive_packet(): called when the server finishes serving packet.
#     """
#
#     def __init__(self, sim):
#         super().__init__(sim)
#         # Statistics:
#         self.arrival_intervals = Intervals()
#         self.arrival_intervals.record(self.sim.stime)
#
#     def handle_message(self, message, connection=None, sender=None):
#         assert isinstance(message, Packet)
#         self.arrival_intervals.record(self.sim.stime)
#         message.source.delays.append(self.sim.stime - message.created_at)
#
#
# #############################################################################
# # SHORTCUTS
# #############################################################################
# def tandem_queue_network(arrivals, services, queue_capacity, stime_limit):
#     assert len(arrivals) == len(services)
#     num_stations = len(arrivals)
#
#     sr = simulate(QueueingTandemNetwork, stime_limit=stime_limit, params={
#         'arrivals': arrivals,
#         'services': services,
#         'queue_capacity': queue_capacity,
#         'num_stations': num_stations,
#     })
#
#     simret_class = namedtuple('SimRet', ['nodes'])
#     node_class = namedtuple('Node', [
#         'delay', 'queue_size', 'system_size', 'busy', 'arrivals', 'departures',
#         'service', 'num_served', 'num_arrived', 'num_dropped', 'drop_ratio',
#         'queue_wait', 'system_wait',
#     ])
#
#     active_nodes = {i for i in range(num_stations) if arrivals[i] is not None}
#
#     nodes = [
#         node_class(
#             delay=(sr.data.sources[i].delays if i in active_nodes else None),
#             queue_size=sr.data.queues[i].size_trace,
#             system_size=sr.data.system_size_trace[i],
#             busy=sr.data.servers[i].busy_trace,
#             arrivals=sr.data.queues[i].arrival_intervals.statistic(),
#             departures=sr.data.servers[i].departure_intervals.statistic(),
#             service=sr.data.servers[i].service_intervals,
#             num_served=sr.data.servers[i].num_served,
#             num_arrived=sr.data.queues[i].num_arrived,
#             num_dropped=sr.data.queues[i].num_dropped,
#             drop_ratio=sr.data.queues[i].drop_ratio,
#             queue_wait=sr.data.queues[i].wait_intervals,
#             system_wait=sr.data.system_wait_intervals[i],
#         ) for i in range(num_stations)
#     ]
#     return simret_class(nodes=nodes)
#
#
# def tandem_queue_network_with_fixed_service(arrivals, service, queue_capacity,
#                                             stime_limit):
#     num_stations = len(arrivals)
#
#     sr = simulate(
#         QueueingTandemNetworkWithFixedService, stime_limit=stime_limit, params={
#             'arrivals': arrivals,
#             'service': service,
#             'queue_capacity': queue_capacity,
#             'num_stations': num_stations,
#         })
#
#     simret_class = namedtuple('SimRet', ['nodes'])
#     node_class = namedtuple('Node', [
#         'delay', 'queue_size', 'system_size', 'busy', 'arrivals', 'departures',
#         'service', 'num_served', 'num_arrived', 'num_dropped', 'drop_ratio',
#         'queue_wait', 'system_wait',
#     ])
#
#     active_nodes = {i for i in range(num_stations) if arrivals[i] is not None}
#
#     nodes = [
#         node_class(
#             delay=(sr.data.sources[i].delays if i in active_nodes else None),
#             queue_size=sr.data.queues[i].size_trace,
#             system_size=sr.data.system_size_trace[i],
#             busy=sr.data.servers[i].busy_trace,
#             arrivals=sr.data.queues[i].arrival_intervals.statistic(),
#             departures=sr.data.servers[i].departure_intervals.statistic(),
#             service=sr.data.servers[i].service_intervals,
#             num_served=sr.data.servers[i].num_served,
#             num_arrived=sr.data.queues[i].num_arrived,
#             num_dropped=sr.data.queues[i].num_dropped,
#             drop_ratio=sr.data.queues[i].drop_ratio,
#             queue_wait=sr.data.queues[i].wait_intervals,
#             system_wait=sr.data.system_wait_intervals[i],
#         ) for i in range(num_stations)
#     ]
#     return simret_class(nodes=nodes)
