from collections import namedtuple
from typing import Optional, Sequence, TypeVar, Generic, List
import numpy as np

from pyqumo.matrix import str_array


Statistics = namedtuple('Statistics', ['avg', 'var', 'std', 'count'])


class TimeSizeRecords:
    """
    Recorder for time-size statistics.

    Key feature of time-size records is that size varies on a natural numbers
    axis, so it is not float, negative, or something else. Thus, we
    store durations of each size in an array. If new value is larger then
    this array, the array is extended.

    Array of durations is easily converted into PMF just dividing it on the
    total time. Assuming initial time is always zero, total time is
    the time when the last update was recorded.

    One more thing to mention is that when recording values, actually previous
    value is recorded: when we call `add(ti, vi)`, it means that at `ti`
    new value became `vi`. However, we store information that the _previous_
    value `v{i-1}` was kept for `t{i} - t{i-1}` interval. Thus, we store
    the previous value (by default, zero) and previous update time.
    """
    def __init__(self, init_value: int = 0, init_time: float = 0.0):
        self._durations: List[float] = [0.0]
        self._updated_at: float = init_time
        self._curr_value: int = init_value
        self._init_time = init_time

    def add(self, time: float, value: int):
        """
        Record new value at the given time.

        When called, duration of the _previous_ value is actually incremented:
        if, say, system size at time `t2` became equal `v2`, then we need
        to store information, that for interval `t2 - t1` value _was_ `v1`.

        New value and update time are stored, so in the next `add()` call
        they will be used to save interval of current value.

        Parameters
        ----------
        time : float
        value : int
        """
        prev_value = self._curr_value
        self._curr_value = value
        num_cells = len(self._durations)
        if prev_value >= num_cells:
            num_new_cells = prev_value - num_cells + 1
            self._durations.extend([0.0] * num_new_cells)
        self._durations[prev_value] += time - self._updated_at
        self._updated_at = time

    @property
    def pmf(self) -> np.ndarray:
        """
        Normalize durations to get PMF.
        """
        return (np.asarray(self._durations) /
                (self._updated_at - self._init_time))

    def __repr__(self):
        return f"(TimeSizeRecords: durations={str_array(self._durations)})"


def build_statistics(intervals: Sequence[float]) -> Statistics:
    """
    Build Statistics object from the given sequence of intervals.

    Parameters
    ----------
    intervals : 1D array_like

    Returns
    -------
    statistics : Statistics
    """
    if len(intervals) > 0:
        avg = np.mean(intervals)
        var = np.var(intervals, ddof=1)  # unbiased estimate
        std = var**0.5
    else:
        avg = 0.0
        var = 0.0
        std = 0.0
    return Statistics(avg=avg, var=var, std=std, count=len(intervals))


T = TypeVar('T')


class Queue(Generic[T]):
    """
    Abstract base class for the queues used in simulation models.

    Queues accept two methods:

    - `push(value: T) -> bool`
    - `pop() -> [T]`

    Push operation adds an item to the queue and returns true or false
    depending on whether the item was actually queued.

    Any queue will also implement four properties:

    - `capacity: int`
    - `size: int`
    - `empty: bool`
    - `full: bool`
    """
    @property
    def size(self) -> int:
        """
        Get the number of items in the queue.
        """
        raise NotImplementedError

    @property
    def capacity(self) -> int:
        """
        Get the maximum number of items in the queue.
        """
        raise NotImplementedError

    def push(self, item: T) -> bool:
        """
        Add an item to the queue.

        Parameters
        ----------
        item : T
            An item to add to the queue

        Returns
        -------
        success : bool
            True, if the item was really added.
        """
        raise NotImplementedError

    def pop(self) -> Optional[T]:
        """
        Extract an item from the queue.

        Returns
        -------
        item : T or None
            If queue failed to extract an item, it should return None
        """
        raise NotImplementedError

    def __len__(self):
        """
        Get the number of items in the queue (alias to size property).
        """
        return self.size

    @property
    def empty(self):
        """
        Check whether the queue is empty, i.e. number of items is zero.
        """
        return self.size == 0

    @property
    def full(self):
        """
        Check whether the queue is full, i.e. number of items equals capacity.
        """
        return self.size >= self.capacity


class FiniteFifoQueue(Queue[T]):
    """
    Finite queue representing a simple FIFO container.
    """

    def __init__(self, capacity: int):
        """
        Create a queue.

        Parameters
        ----------
        capacity : int
            Specifies maximum queue size
        """
        self.__items: List[T] = [None] * capacity
        self.__capacity = capacity
        self.__size = 0
        self.__head = 0
        self.__end = 0

    @property
    def capacity(self) -> int:
        return self.__capacity

    @property
    def size(self) -> int:
        return self.__size

    def push(self, item: T) -> bool:
        if self.full:
            return False
        self.__items[self.__end] = item
        self.__end = (self.__end + 1) % self.capacity
        self.__size += 1
        return True

    def pop(self) -> Optional[T]:
        if self.empty:
            return None
        item = self.__items[self.__head]
        self.__items[self.__head] = None
        self.__head = (self.__head + 1) % self.capacity
        self.__size -= 1
        return item

    def __repr__(self) -> str:
        """
        Get string representation of the queue.
        """
        if self.__head < self.__end:
            items = self.__items[self.__head:self.__end]
        elif self.__head >= self.__end and self.__size > 0:
            items = self.__items[self.__head:] + self.__items[:self.__end]
        else:
            items = []
        items_str = [str(item) for item in items]
        return f"(FiniteFifoQueue: q=[{', '.join(items_str)}], " \
               f"capacity={self.capacity}, size={self.size})"


class InfiniteFifoQueue(Queue[T]):
    """
    Infinite queue with FIFO order.
    """
    def __init__(self):
        self.__items = []

    @property
    def capacity(self):
        return np.inf

    @property
    def size(self):
        return len(self.__items)

    def push(self, item: T) -> bool:
        self.__items.append(item)
        return True

    def pop(self) -> Optional[T]:
        item: Optional[T] = None
        if len(self.__items) > 0:
            item = self.__items[0]
            self.__items = self.__items[1:]
        return item

    def __repr__(self):
        items = ', '.join([str(item) for item in self.__items])
        return f"(InfiniteFifoQueue: q=[{items}], size={self.size})"


class Server(Generic[T]):
    """
    Simple server model. Just stores a packet of type T and can be empty.
    """
    def __init__(self):
        self._packet: Optional[T] = None

    @property
    def ready(self) -> bool:
        """
        Check whether server is not serving any packet.
        """
        return self._packet is None

    @property
    def busy(self) -> bool:
        """
        Check whether server is serving a packet.
        """
        return self._packet is not None

    @property
    def size(self) -> int:
        """
        Get the number of packets being served (1 or 0)
        """
        return 1 if self._packet is not None else 0

    def pop(self) -> T:
        """
        Move from the server the packet being served.

        After this call busy server becomes ready, and the packet that was
        served is returned in the result.

        If the server was ready (empty), `RuntimeError` exception is thrown.

        Returns
        -------
        packet : T
            A packet that was served.
        """
        if self._packet is None:
            raise RuntimeError("attempted to pop from an empty server")
        packet = self._packet
        self._packet = None
        return packet

    def serve(self, packet: T) -> None:
        """
        Serve a new packet.

        If the server was ready, it starts serving the packet. That is,
        this packet is stored, server becomes busy and its size becomes equal
        one.

        If the server was already busy, `RuntimeError` is raised.

        Parameters
        ----------
        packet : T
        """
        if self._packet is not None:
            raise RuntimeError("attempted to put a packet to a busy server")
        self._packet = packet

    def __str__(self):
        suffix = "" if self._packet is None else f", packet={self._packet}"
        return f"(Server: busy={self.busy}{suffix})"
