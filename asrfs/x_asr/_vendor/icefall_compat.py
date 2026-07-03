"""Compatibility shims for the vendored icefall zipformer recipe.

This module replaces the handful of ``icefall.utils`` / recipe ``train.py`` symbols
that the vendored files import, so that ``asrfs.x_asr._vendor`` has no dependency on
the ``icefall`` package (which pulls in ``k2`` at import time).

Provenance (all at pinned commit 7a35ca20d7d2224eead3d2de353f031cb4f6307a):
  * ``make_pad_mask``                -- verbatim from ``icefall/utils.py`` (L1449-1486)
  * ``get_parameter_groups_with_lrs``-- verbatim from ``icefall/utils.py`` (L1583-1652)
  * ``set_batch_count``              -- from ``egs/librispeech/ASR/zipformer/train.py``
                                        (L125-133), with the DDP branch dropped (see VENDOR.md)
  * ``torch_autocast``               -- thin shim (torch >= 2.3 unified ``torch.amp`` API)

See ``VENDOR.md`` for the full surgery log.
"""

import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import List

import torch
import torch.nn as nn


@contextmanager
def torch_autocast(device_type="cuda", **kwargs):
    """Shim for ``icefall.utils.torch_autocast``.

    icefall's version branches on the torch version; torch >= 2.3 (this project pins
    torch 2.12) uses the unified ``torch.amp.autocast`` API directly.
    """
    with torch.amp.autocast(device_type=device_type, **kwargs):
        yield


def set_batch_count(model: nn.Module, batch_count: float) -> None:
    """Drive ScheduledFloat / warmup schedules by stamping ``batch_count`` on modules.

    Copied from the recipe's ``train.py`` (L125-133). The original accepted a DDP-wrapped
    model and unwrapped it via ``isinstance(model, DDP)``; that branch is dropped here since
    the vendored package drives a plain ``nn.Module`` (see VENDOR.md).
    """
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def make_pad_mask(
    lengths: torch.Tensor,
    max_len: int = 0,
    pad_left: bool = False,
) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
      pad_left:
        If ``False`` (default), padding is on the right.
        If ``True``, padding is on the left.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expanded_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    if pad_left:
        mask = expanded_lengths < (max_len - lengths).unsqueeze(1)
    else:
        mask = expanded_lengths >= lengths.unsqueeze(-1)

    return mask


def get_parameter_groups_with_lrs(
    model: nn.Module,
    lr: float,
    include_names: bool = False,
    freeze_modules: List[str] = [],
) -> List[dict]:
    """
    This is for use with the ScaledAdam optimizers (more recent versions that accept lists of
    named-parameters; we can, if needed, create a version without the names).

    It provides a way to specify learning-rate scales inside the module, so that if
    any nn.Module in the hierarchy has a floating-point parameter 'lr_scale', it will
    scale the LR of any parameters inside that module or its submodules.  Note: you
    can set module parameters outside the __init__ function, e.g.:
      >>> a = nn.Linear(10, 10)
      >>> a.lr_scale = 0.5

    Returns: a list of dicts, of the following form:
      if include_names == False:
        [  { 'params': [ tensor1, tensor2, ... ], 'lr': 0.01 },
           { 'params': [ tensor3, tensor4, ... ], 'lr': 0.005 },
         ...   ]
      if include_names == true:
        [  { 'named_params': [ (name1, tensor1, (name2, tensor2), ... ], 'lr': 0.01 },
           { 'named_params': [ (name3, tensor3), (name4, tensor4), ... ], 'lr': 0.005 },
         ...   ]

    """
    named_modules = list(model.named_modules())

    # flat_lr_scale just contains the lr_scale explicitly specified
    # for each prefix of the name, e.g. 'encoder.layers.3', these need
    # to be multiplied for all prefix of the name of any given parameter.
    flat_lr_scale = defaultdict(lambda: 1.0)
    names = []
    for name, m in model.named_modules():
        names.append(name)
        if hasattr(m, "lr_scale"):
            flat_lr_scale[name] = m.lr_scale

    # lr_to_parames is a dict from learning rate (floating point) to: if
    # include_names == true, a list of (name, parameter) for that learning rate;
    # otherwise a list of parameters for that learning rate.
    lr_to_params = defaultdict(list)

    for name, parameter in model.named_parameters():
        split_name = name.split(".")
        # caution: as a special case, if the name is '', split_name will be [ '' ].
        prefix = split_name[0]
        if prefix == "module":  # DDP
            module_name = split_name[1]
            if module_name in freeze_modules:
                logging.info(f"Remove {name} from parameters")
                continue
        else:
            if prefix in freeze_modules:
                logging.info(f"Remove {name} from parameters")
                continue
        cur_lr = lr * flat_lr_scale[prefix]
        if prefix != "":
            cur_lr *= flat_lr_scale[""]
        for part in split_name[1:]:
            prefix = ".".join([prefix, part])
            cur_lr *= flat_lr_scale[prefix]
        lr_to_params[cur_lr].append((name, parameter) if include_names else parameter)

    if include_names:
        return [{"named_params": pairs, "lr": lr} for lr, pairs in lr_to_params.items()]
    else:
        return [{"params": params, "lr": lr} for lr, params in lr_to_params.items()]
