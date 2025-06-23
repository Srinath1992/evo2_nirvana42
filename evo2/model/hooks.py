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
        """Check if a module name should be recorded.

        The *patterns* argument passed to the recorder can be a single regex
        or a list of regexes.  We consider a module a *match* only if its
        fully-qualified name satisfies **all** of those regex patterns—this
        gives you fine-grained control when several sub-patterns are needed
        (e.g. layer type *and* index).

        Parameters
        ----------
        name : str
            The ``named_modules`` key for the sub-module currently being
            inspected.

        Returns
        -------
        bool
            ``True`` if *name* should have a forward hook attached.
        """
        for p in self.patterns:
            if isinstance(p, str):
                if not re.fullmatch(p, name):
                    return False
            else:  # compiled regex
                if p.fullmatch(name) is None:
                    return False
        return True

    def _hook_fn(self, name: str):
        """Factory that builds the real forward-hook function.

        We need a *separate* hook function per matched sub-module so that the
        captured activations can be keyed by the correct name.  This little
        closure does exactly that: it keeps *name* in its scope and hands a
        callable to PyTorch's ``register_forward_hook``.
        """
        def _fn(_module, _inputs, output):
            """The function PyTorch executes right after the module's forward.

            * Detaches the tensor from the graph so we don't keep gradients
              around.
            * Optionally moves it to the user-requested ``device`` (default:
              CPU) to free up GPU VRAM.
            * Stashes it in ``self.activations`` under the sub-module's name.
            """
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
        """Iterate through *all* sub-modules and attach hooks to the matches."""
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
        """Context-manager entry: attach hooks and return *self*."""
        self._register()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Remove hooks on exit so the model behaves normally afterwards."""
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
    """Convenience helper so you don't have to use *with* if you don't want.

    Example
    -------
    >>> rec = register_activation_hooks(model, r"blocks\.0")
    >>> _ = model(batch)
    >>> print(rec.activations.keys())
    >>> rec.__exit__(None, None, None)  # manually dispose the hooks
    """
    return ActivationRecorder(model, patterns, device) 