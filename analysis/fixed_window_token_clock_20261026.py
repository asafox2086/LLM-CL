"""Timing callback for future fixed GPU windows; no model, I/O or Torch import.

Reuse the timing convention in edge_visual_gpu_probe_20261019.py: skip the
prompt callback, synchronize each generated token, decode rate excludes the
first token. Caller starts before input preparation, after initial GPU sync.
This module is not attached to any running/frozen experiment.
"""
import math
import time


class FixedWindowTokenClock:
    def __init__(self, *, synchronize, expected_tokens=32, clock=time.perf_counter):
        if expected_tokens < 2:
            raise ValueError('Decode timing requires at least two output tokens')
        self.synchronize = synchronize
        self.clock = clock
        self.expected_tokens = expected_tokens
        self.started = None
        self.prompt_seen = False
        self.times = []
        self.ended = False

    def start(self):
        if self.started is not None:
            raise RuntimeError('Clock cannot be reused across requests')
        self.synchronize()
        self.started = self.clock()

    def put(self, value):
        if self.started is None or self.ended:
            raise RuntimeError('Token callback outside active request')
        if not self.prompt_seen:
            if value.ndim != 2 or value.shape[0] != 1:
                raise ValueError('Expected batch-one prompt callback')
            self.prompt_seen = True
            return
        if value.numel() != 1:
            raise ValueError('Only batch-one single-token greedy callbacks supported')
        if len(self.times) >= self.expected_tokens:
            raise ValueError('Output exceeds fixed token window')
        self.synchronize()
        self.times.append(self.clock())

    def end(self):
        if self.started is None or self.ended:
            raise RuntimeError('Invalid end callback')
        self.ended = True

    def result(self):
        if not self.ended or not self.prompt_seen or len(self.times) != self.expected_tokens:
            raise ValueError('Incomplete fixed token window; do not report partial timing')
        ttft = self.times[0] - self.started
        decode = self.times[-1] - self.times[0]
        if not math.isfinite(ttft) or ttft <= 0 or not math.isfinite(decode) or decode <= 0:
            raise ValueError('Invalid timing interval')
        if any(b < a for a, b in zip(self.times, self.times[1:])):
            raise ValueError('Non-monotonic timestamps')
        return dict(ttft_seconds=ttft, decode_tokens_per_second=(len(self.times)-1)/decode,
                    generated_tokens=len(self.times), decode_interval_seconds=decode,
                    ttft_includes_input_preparation=True, synchronized_token_callbacks=True,
                    interpretation='instrumented fixed-window wall time, not production throughput')
