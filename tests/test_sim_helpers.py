import pytest
import numpy as np

from pyqumo.sim.helpers import FiniteFifoQueue, Queue, InfiniteFifoQueue, Server


# ###########################################################################
# TEST FiniteFifoQueue
# ###########################################################################
@pytest.mark.parametrize('capacity', [0, 5])
def test_finite_fifo_queue_props(capacity):
    """
    Validate capacity, empty and full properties of a queue with finite cap.
    """
    queue: Queue[int] = FiniteFifoQueue(capacity)
    assert queue.capacity == capacity
    assert queue.size == 0
    assert queue.empty

    if capacity == 0:
        assert queue.full

    # Fill half of the queue with values, check it is neither empty nor full:
    half_capacity = capacity // 2
    for i in range(half_capacity):
        queue.push(i)

    assert queue.size == half_capacity
    if half_capacity > 0:
        assert not queue.empty
        assert not queue.full

    # Fill the rest of the queue and make sure it becomes full then:
    for _ in range(half_capacity, capacity):
        queue.push(0)
    assert queue.size == capacity
    assert queue.full


def test_finite_fifo_queue_push_pop_order():
    """
    Validate push() and pop() works in FIFO order.
    """
    # Put 10 [OK] => 20 [OK] => 30 [LOST]
    queue: Queue[int] = FiniteFifoQueue(2)
    assert queue.push(10)
    assert queue.push(20)
    assert not queue.push(30)
    # queue = [10, 20]
    # Pop 10 => { queue = [20] } => push 40 => { queue = [20, 40] }
    assert queue.pop() == 10
    assert queue.push(40)
    # queue = [20, 40]
    # Pop 20 => Pop 40 => Pop [NONE]
    assert queue.pop() == 20
    assert queue.pop() == 40
    assert queue.pop() is None
    assert queue.empty


def test_finite_fifo_queue_str():
    """
    Validate __repr__() method of the FiniteFifoQueue.
    """
    queue: Queue[int] = FiniteFifoQueue(5)
    assert str(queue) == "(FiniteFifoQueue: q=[], capacity=5, size=0)"
    queue.push(34)
    queue.push(42)
    assert str(queue) == "(FiniteFifoQueue: q=[34, 42], capacity=5, size=2)"
    queue.push(1)
    queue.push(2)
    queue.push(3)
    assert str(queue) == "(FiniteFifoQueue: q=[34, 42, 1, 2, 3], " \
                         "capacity=5, size=5)"


# ###########################################################################
# TEST InfiniteFifoQueue
# ###########################################################################
def test_infinite_fifo_queue():
    """
    Validate basic properties and push/pop to the infinite FIFO queue.
    """
    queue: Queue[int] = InfiniteFifoQueue()
    assert queue.capacity == np.inf
    assert queue.size == 0
    assert queue.empty
    assert not queue.full

    # Add some elements:
    queue.push(1)
    queue.push(2)
    assert queue.size == 2
    assert not queue.full
    assert not queue.empty
    assert str(queue) == "(InfiniteFifoQueue: q=[1, 2], size=2)"

    # Pop:
    assert queue.pop() == 1
    assert queue.pop() == 2
    assert queue.pop() is None
    assert queue.empty
    assert str(queue) == "(InfiniteFifoQueue: q=[], size=0)"

    # Push many elements:
    num_elements = 1000
    for i in range(num_elements):
        item = (i + 42) * 10
        queue.push(item)
    assert queue.size == num_elements
    assert not queue.full
    assert queue.pop() == 420


# ############################################################################
# TEST Server
# ############################################################################
def test_server():
    """
    Validate Server operations.
    """
    server: Server[int] = Server()
    assert str(server) == "(Server: busy=False)"
    assert server.ready
    assert not server.busy
    assert server.size == 0

    # Push a packet
    server.serve(10)
    assert not server.ready
    assert server.busy
    assert server.size == 1
    assert str(server) == "(Server: busy=True, packet=10)"

    # Check that pushing another packet raises error:
    with pytest.raises(RuntimeError):
        server.serve(20)

    # Pop a packet:
    assert server.pop() == 10
    assert str(server) == "(Server: busy=False)"
    assert server.ready
    assert not server.busy
    assert server.size == 0
