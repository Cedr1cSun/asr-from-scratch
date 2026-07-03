# Vendored icefall Zipformer recipe

## Source & license

- **Upstream:** [k2-fsa/icefall](https://github.com/k2-fsa/icefall)
- **Pinned commit:** `7a35ca20d7d2224eead3d2de353f031cb4f6307a`
- **License:** Apache License 2.0. Copyright Xiaomi Corp. (authors incl. Daniel Povey,
  Zengwei Yao, Wei Kang). Every vendored file keeps its original Apache-2.0 header.
  See the upstream repo-root `LICENSE` for the full text.

These files are vendored verbatim from the pinned commit and then subjected to the
minimal **import surgery** logged below so the package imports with only `torch`
(+ `torchaudio` for the loss in Task 2) — no `k2`, no `lhotse`, no `icefall`.
**No logic or numeric changes** were made; every rerouted call has a numerically
equivalent in-file pure-torch counterpart.

## File inventory

| vendored file            | upstream path (relative to repo root)                                  |
| ------------------------ | ---------------------------------------------------------------------- |
| `zipformer.py`           | `egs/librispeech/ASR/zipformer/zipformer.py`                           |
| `scaling.py`             | `egs/librispeech/ASR/zipformer/scaling.py`                             |
| `subsampling.py`         | `egs/librispeech/ASR/zipformer/subsampling.py`                         |
| `decoder.py`             | `egs/librispeech/ASR/zipformer/decoder.py`                             |
| `joiner.py`              | `egs/librispeech/ASR/zipformer/joiner.py`                              |
| `optim.py`               | `egs/librispeech/ASR/zipformer/optim.py`                               |
| `encoder_interface.py`   | `egs/librispeech/ASR/transducer_stateless/encoder_interface.py` (symlink target) |
| `icefall_compat.py`      | **not vendored** — see below; functions copied from `icefall/utils.py` + `.../zipformer/train.py` |

`icefall_compat.py` provides the four symbols the vendored files (and Task 2) need
from `icefall.utils` / the recipe `train.py`, none of which pull in `k2`:

- `make_pad_mask` — **verbatim** from `icefall/utils.py` (L1449-1486).
- `get_parameter_groups_with_lrs` — **verbatim** from `icefall/utils.py` (L1583-1652),
  brought over with its `logging` / `collections.defaultdict` / `typing.List` deps.
- `torch_autocast` — thin shim over `torch.amp.autocast` (icefall's own version just
  version-branches; this project pins torch 2.12 which uses the unified API directly).
- `set_batch_count` — from `egs/librispeech/ASR/zipformer/train.py` (L125-133), copied
  with the DDP-unwrap branch dropped (see surgery item 7).

## Import surgery log

Format: **file — location — what — why**. Every edit is marked in-code with a
`# VENDOR surgery:` comment where practical.

### 1. `scaling.py` — module imports (orig L23, L27, L29)

- **What:** Deleted `import k2`. Changed
  `from torch.cuda.amp import custom_bwd, custom_fwd` →
  `from torch.amp import custom_bwd as _custom_bwd, custom_fwd as _custom_fwd`, then bound
  `custom_fwd = functools.partial(_custom_fwd, device_type="cuda")` /
  `custom_bwd = functools.partial(_custom_bwd, device_type="cuda")` (added `import functools`).
  Changed `from icefall.utils import torch_autocast` →
  `from asrfs.x_asr._vendor.icefall_compat import torch_autocast`.
- **Why:** `k2` is a heavy compiled dependency we do not install; it is used in this file
  only for the Swoosh activations, each of which already has an in-file pure-torch
  equivalent (items 2-4). `torch.cuda.amp.custom_{fwd,bwd}` were removed/deprecated in
  torch ≥ 2.4; the `torch.amp` equivalents require a `device_type` keyword, bound via
  `functools.partial` so the existing bare `@custom_fwd` / `@custom_bwd` decorator sites
  (4 of them) are untouched. `icefall.utils` imports `k2` at module top, so it is replaced
  by the local compat shim.

### 2. `scaling.py` — `SwooshL.forward` (orig L1407-1411)

- **What:** `k2.swoosh_l_forward(x)` → `SwooshLForward(x)` (no-grad branch);
  `k2.swoosh_l(x)` → `SwooshLFunction.apply(x)` (requires-grad branch). Dropped the
  now-redundant commented-out `# return SwooshLFunction.apply(x)` hint line.
- **Why:** `SwooshLForward` (plain pure-torch fn) and `SwooshLFunction` (torch.autograd
  custom fwd/bwd) are defined in this same file and compute the identical activation /
  gradient; upstream even left `SwooshLFunction.apply(x)` as a commented alternative.
  k2's kernel was only a speed/memory optimization.

### 3. `scaling.py` — `SwooshR.forward` (orig L1481-1485)

- **What:** `k2.swoosh_r_forward(x)` → `SwooshRForward(x)`; `k2.swoosh_r(x)` →
  `SwooshRFunction.apply(x)`. Dropped the redundant commented hint line.
- **Why:** Same as item 2, for the Swoosh-R activation.

### 4. `scaling.py` — `ActivationDropoutAndLinearFunction` fused kernel (orig L1540-1543, L1559-1562) + new helpers

- **What:** In `.forward`, the activation dict `{"SwooshL": k2.swoosh_l_forward,
  "SwooshR": k2.swoosh_r_forward}` → `{"SwooshL": SwooshLForward, "SwooshR": SwooshRForward}`.
  In `.backward`, `{"SwooshL": k2.swoosh_l_forward_and_deriv,
  "SwooshR": k2.swoosh_r_forward_and_deriv}` → `{"SwooshL": SwooshLForwardAndDeriv,
  "SwooshR": SwooshRForwardAndDeriv}`. Added two module-level helpers
  `SwooshLForwardAndDeriv` / `SwooshRForwardAndDeriv` (right after `SwooshRForward`) that
  return `(activation(x), d/dx activation)` where the derivative is the exact analytic
  gradient: `d/dx swoosh_l = sigmoid(x-4) - 0.08`, `d/dx swoosh_r = sigmoid(x-1) - 0.08`.
- **Why:** This is the recipe's memory-efficient fused activation+dropout+linear used on the
  **training** path (`ActivationDropoutAndLinear.forward` calls it when `self.training`).
  Rerouting the k2 entries to the in-file pure-torch forward + analytic derivative preserves
  the exact forward math, the custom memory-efficient backward, AND the dropout behaviour —
  i.e. numerically equivalent, no training-path behaviour change. (The alternative of forcing
  the module's non-fused eval fallback was rejected: that fallback skips dropout, which would
  silently change training numerics.) The fused `Function` class itself is otherwise kept
  intact.

### 5. `zipformer.py` — imports (orig L27, L28/31/34, L50)

- **What:** `from encoder_interface import EncoderInterface` →
  `from asrfs.x_asr._vendor.encoder_interface import EncoderInterface`; the three
  `from scaling import (...)` → `from asrfs.x_asr._vendor.scaling import (...)`;
  `from icefall.utils import torch_autocast` → `from asrfs.x_asr._vendor.icefall_compat import torch_autocast`.
- **Why:** Repoint sibling-module imports (upstream relies on the recipe dir being on
  `sys.path`) to absolute paths inside this package; replace the `k2`-pulling `icefall.utils`.

### 6. `subsampling.py` (orig L23) / `decoder.py` (orig L20) / `joiner.py` (orig L19) — scaling import

- **What:** `from scaling import ...` → `from asrfs.x_asr._vendor.scaling import ...` in each.
- **Why:** Same absolute-import repoint as item 5. These three files have no other external
  deps beyond `torch` + `scaling`.

### 7. `optim.py` — lhotse import + self-test block (orig L24, L953-1237)

- **What:** Deleted `from lhotse.utils import fix_random_seed`. Deleted the trailing
  self-test block: `_test_eden()`, `_test_scaled_adam()`, and the `if __name__ == "__main__":`
  driver, and the unused `Eve` baseline optimizer class (only referenced by the deleted
  `_test_scaled_adam`; zero residual references) — everything from `def _test_eden():` to EOF.
  File went 1237 → 950 lines.
- **Why:** `lhotse` and `fix_random_seed` are used **only** inside those `__main__`
  self-tests; `ScaledAdam` / `Eden` / `LRScheduler` themselves have no `lhotse` dependency.
  Removing the block drops the only `lhotse` reference and the git-subprocess `__main__` driver.

### 8. `icefall_compat.py` — `set_batch_count` (from `train.py` L125-133)

- **What:** Copied `set_batch_count` from the recipe `train.py`, dropping the
  `if isinstance(model, DDP): model = model.module` unwrap and the `Union[nn.Module, DDP]`
  type hint; the loop body (`module.batch_count = ...`, `module.name = name`) is verbatim.
- **Why:** The vendored package drives a plain `nn.Module`; keeping the DDP branch would add
  an unnecessary `torch...DistributedDataParallel` import. Behaviour on a non-DDP module is
  identical.

### Not vendored (documented for Task 2)

- `model.py` — the recipe's `AsrModel` imports `k2` (add_sos / rnnt_loss_* / pruning) and
  `lhotse` (SpecAugment). Task 2 reimplements the thin RNN-T plumbing on top of
  `torchaudio.functional.rnnt_loss` instead.
- `icefall/utils.py` — imports `k2` at module top; only the three pure-torch functions above
  were copied into `icefall_compat.py`.
