"""
Lightweight utilities to record intermediate activations from Evo-2 (or any
PyTorch model) without modifying its source.  Typical usage::

    with ActivationRecorder(model, patterns=[r"blocks\.5\.mlp\.l2"]) as rec:
        _ = model(inputs)
        acts = rec.activations  # dict[layer_name] -> Tensor

patterns can be a single regex string / compiled pattern or a list thereof.
During the with-block forward pass, any sub-module whose fully-qualified name
matches **all** supplied patterns is hooked and its output is stored (detached)
in `rec.activations`.
"""
from __future__ import annotations

import contextlib
import re
from typing import Dict, List, Pattern, Union

import torch
import torch.nn as nn

__all__ = [
    "ActivationRecorder",
    "register_activation_hooks",
]

_Pattern = Union[str, Pattern[str]]


class ActivationRecorder(contextlib.AbstractContextManager):
    """Context manager that records forward activations for matching modules."""

    def __init__(
        self,
        model: nn.Module,
        patterns: List[_Pattern] | _Pattern = ".*",
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        if isinstance(patterns, (str, re.Pattern)):
            self.patterns: List[_Pattern] = [patterns]
        else:
            self.patterns = list(patterns)
        self.device = torch.device(device) if device is not None else None

        self.activations: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _match(self, name: str) -> bool:
        """Return True if *name* matches *all* supplied patterns."""
        for p in self.patterns:
            if isinstance(p, str):
                if not re.fullmatch(p, name):
                    return False
            else:  # compiled regex
                if p.fullmatch(name) is None:
                    return False
        return True

    def _hook_fn(self, name: str):
        def _fn(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            detached = output.detach()
            if self.device is not None:
                detached = detached.to(self.device)
            else:
                detached = detached.cpu()
            self.activations[name] = detached

        return _fn

    def _register(self) -> None:
        for name, mod in self.model.named_modules():
            if name == "":  # skip top-level container to avoid duplicate capture
                continue
            if self._match(name):
                handle = mod.register_forward_hook(self._hook_fn(name))
                self._hooks.append(handle)

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self):
        self._register()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        # Returning False will re-raise any exception that happened inside the ctx.
        return False


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def register_activation_hooks(
    model: nn.Module,
    patterns: List[_Pattern] | _Pattern = ".*",
    device: torch.device | str | None = None,
) -> ActivationRecorder:
    """Return an *ActivationRecorder* and register hooks immediately."""
    return ActivationRecorder(model, patterns, device) 