import nengo
from collections import deque


class IdealDelay(nengo.synapses.Synapse):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        return {}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        # buffer the input signal based on the delay length
        buffer = deque([0] * int(self.delay / dt))

        def delay_func(t, x):
            buffer.append(x.copy())
            return buffer.popleft()

        return delay_func


class IdealDelayLowpass(nengo.Lowpass):
    def __init__(self, tau, delay, **kwargs):
        super().__init__(tau, **kwargs)
        self.delay = delay

    def make_step(self, shape_in, shape_out, dt, rng, state):
        # buffer the input signal based on the delay length
        buffer = deque([0] * int(self.delay / dt))
        step = super().make_step(shape_in, shape_out, dt, rng, state)

        def delay_func(t, x):
            buffer.append(x.copy())
            return step(t, buffer.popleft())

        return delay_func
