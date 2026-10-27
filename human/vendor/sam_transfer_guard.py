"""Make SAM2 offloaded tensors ready before CPU consumers can access them.

Local instance adapter only: no weights, inference equations or shared package edits.
"""
from functools import wraps

import torch


def install_transfer_guard(predictor):
    if getattr(predictor, "_fd_transfer_guard_installed", False):
        raise RuntimeError("Transfer guard already installed")

    def wrap(method):
        @wraps(method)
        def ready(*args, **kwargs):
            state = kwargs.get("inference_state", args[0] if args else None)
            if state is None:
                raise ValueError("SAM inference state is required")
            result = method(*args, **kwargs)
            if torch.device(state["storage_device"]).type == "cpu":
                # SAM performs nonblocking D2H; its next consolidation can run on CPU.
                torch.cuda.current_stream(state["device"]).synchronize()
            return result
        return ready

    predictor._run_single_frame_inference = wrap(predictor._run_single_frame_inference)
    predictor._run_memory_encoder = wrap(predictor._run_memory_encoder)
    predictor._fd_transfer_guard_installed = True
    return predictor
