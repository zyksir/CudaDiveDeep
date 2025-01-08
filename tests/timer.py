# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
import torch

class CudaEventTimer(object):
    def __init__(self, start_event: torch.cuda.Event, end_event: torch.cuda.Event):
        self.start_event = start_event
        self.end_event = end_event

    def get_elapsed_msec(self):
        torch.cuda.current_stream().wait_event(self.end_event)
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)

class Timer:
    def __init__(self, use_host_timer=False):
        self.started_ = False
        self.use_host_timer = use_host_timer
        self.start_event = None
        self.start_time = 0.0

    def start(self):
        """Start the timer."""
        assert not self.started_, f"{self.name_} timer has already been started"
        if self.use_host_timer:
            self.start_time = time.time()
        else:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, "timer is not started"
        self.started_ = False
        if self.use_host_timer:
            end_time = time.time()
            return (end_time - self.start_time)*1000
        else:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            event_timer = CudaEventTimer(self.start_event, end_event)
            self.start_event = None
            return event_timer.get_elapsed_msec()
