"""
training/local_trainer.py

The shared local training-step logic, per training-phase-decisions.md
Sections 2-4 and the discussion chat's patch spec (base file md5
9f8a903308cfb9f08223cd4398659cee). Reused identically by
train_centralized.py (Experiment 1, no FL, no proximal term) and
federated/client.py (Experiments 2-8, each client's local training
within a round) -- exactly one implementation of "what a training
epoch actually does," so the two call sites can't silently drift apart
on batching, clipping, or loss composition.

PATCHED (this version) -- three real, confirmed bugs fixed:

1. FedProx's proximal term was silently dropped on the DP path. Adding
   it to the loss (correct on the non-DP path) never reaches Opacus's
   .grad_sample, since the term is a direct function of the parameters
   and never flows through any model layer's forward pass -- so it
   never triggers GradSampleModule's hooks, and DPOptimizer.step()
   (which reads only .grad_sample) silently ignored it. PROVEN
   empirically, not just reasoned about: DP update shift was exactly
   0.000000 with vs without the term, on both a tiny model and the
   real CoAtNet+LoRA model. Fixed: _add_fedprox_gradient_dp() adds the
   term's closed-form gradient directly to .grad AFTER
   DPOptimizer.pre_step() has already clipped and noised the task
   loss's contribution -- safe because the added quantity depends only
   on current and round-start-global weights, never on any training
   example, so it's pure post-processing of an already-private
   quantity, and correctly bypasses clipping/noising rather than being
   silently dropped by it.

2. Mixup is removed entirely, from all 8 experiments (not just the DP
   ones). With Mixup, each "sample" the model trains on is actually a
   blend of two real examples -- the per-example sensitivity bound
   Opacus's accounting relies on no longer holds, so any epsilon
   reported for experiments 4-8 would be nominal, not real. This
   project's central result is the epsilon-vs-utility curve itself, so
   a nominal epsilon doesn't just weaken that result, it invalidates
   the specific claim those experiments exist to make. Dropped in
   experiments 1-3 too, so the Experiment-3-vs-4 comparison still
   changes exactly one thing (DP on/off), not two. (Also sidesteps an
   unrelated practical issue: timm's Mixup asserts an even batch size,
   which BatchMemoryManager's variable-size chunks won't reliably
   provide.)

3. Class imbalance now goes through a single class-weighted,
   label-smoothed loss (weighted_label_smoothed_ce() below), used
   identically on both paths, replacing Mixup's label smoothing.
   Explicitly NOT nn.CrossEntropyLoss(weight=...): that computes a
   weighted MEAN (dividing by the sum of weights of samples actually
   present in the batch), which makes any one sample's effective
   gradient depend on which OTHER labels happen to be in the same
   batch -- breaking the per-sample independence Opacus's per-sample
   clipping and mean-loss scaling both assume. Divides by batch size
   instead (plain .mean() over per-sample weighted losses), always.
   WeightedRandomSampler is also gone from every experiment's
   dataloader construction (not this file's concern directly, but the
   reason class-weighting replaces it here): Opacus replaces whatever
   sampler you give it with its own uniform Poisson subsampling
   regardless, so a WeightedRandomSampler on the DP path would have
   been silently overridden anyway -- moving to loss-based weighting
   keeps the imbalance-handling mechanism identical across all 8
   experiments instead of differing by path.

Also fixed, smaller:
- M1: passing a scheduler together with a DP-wrapped optimizer now
  raises ValueError at the start of the function, rather than silently
  stepping the schedule on every micro-batch (DPOptimizer.pre_step()
  returns False -- no real step -- on most micro-batches within a
  logical batch, so a step-level scheduler would fire far more often
  than real steps actually happen). Locked design has centralized
  (where a step-level schedule is used) as non-DP, so this combination
  should never legitimately occur.
- M2: compute_selection_metric() now passes labels=np.arange(5),
  zero_division=0 to f1_score() -- without this, a validation split
  with zero examples of some DR grade (plausible for a small FL client,
  or an unlucky batch) would silently change which classes the macro
  average is computed over, making the metric not comparable across
  epochs/experiments, rather than correctly including that class at 0.
- M3 (superseded by M4): the final, partial gradient-accumulation group
  (non-DP path) used to be rescaled by accumulation_steps/k.
- M4: non-DP gradient accumulation is now EXAMPLE-weighted. Each
  micro-batch backpropagates (its mean loss) x (its example count); at
  step time the accumulated gradient is divided by the group's total
  example count, the FedProx gradient is added ONCE, then clip and step.
  So a short final micro-batch (drop_last=False) no longer gets the same
  weight as a full one, and a short final group needs no special
  rescale. For full groups of equal-size micro-batches this equals the
  old per-micro-batch division up to float rounding. Training
  and validation loss are now per-example means, not means of batch
  means. The DP path is unchanged.

Implements, from the locked decisions:
- Class-weighted, label-smoothed training loss (weighted_label_
  smoothed_ce, below) -- class_weights is caller-supplied (this file
  doesn't compute class counts); recommended the caller normalizes
  weights so their expectation under the client's own training label
  distribution is 1, keeping LR and max_grad_norm comparable to an
  unweighted run.
- The Section 4 batching split: manual gradient accumulation for the
  non-DP experiments (1-3) vs. Opacus's BatchMemoryManager cadence
  (now via explicit pre_step()/original_optimizer.step(), to support
  fix 1 above) for the DP experiments (4-8), auto-detected via
  isinstance(optimizer, DPOptimizer) rather than a caller-supplied
  flag -- effective batch size IS the DP sampling rate for 4-8, so a
  caller passing the wrong cadence for what it actually built wouldn't
  just be a training bug, it would silently corrupt the reported
  epsilon.
- clip_grad_norm_(max_norm=1.0) on the non-DP path only. On the DP
  path, Opacus's own per-sample clipping (max_grad_norm, set in
  privacy_calibration.py) is the actual clipping mechanism -- applying
  both would be redundant at best (confirmed: a manual clip there
  wouldn't double-clip anything, since DPOptimizer.step() overwrites
  .grad from .grad_sample regardless -- it would just be inert, wasted
  compute) and could interfere with Opacus's internal accounting at
  worst.
- freeze_bn_eval() (from models/coatnet_lora.py) re-asserted
  immediately after every model.train() call, per that function's own
  docstring.
- Optional step-level LR scheduler hook (scheduler=None by default),
  stepped immediately after every REAL optimizer.step() -- non-DP path
  only now (see M1 above); for train_centralized.py's locked
  "step-level warmup+cosine" schedule. FL's "round-level" schedule
  stays external (client.py/server.py, stepped between rounds).
- Early-stopping/round-selection metric is val macro-F1 (Section 2:
  "patience 5 on val macro-F1"), not val loss -- with ~73.6% No-DR,
  loss keeps improving as the model over-fits the majority class while
  macro-F1 can plateau or worsen as minority classes stall. Using loss
  here would quietly defeat the reason macro-F1 was chosen. Section 2
  never pinned down FL round-selection's metric explicitly -- locked
  here to the same macro-F1, via the same helper, for consistency.
- FedProx's proximal term -- (mu/2) * ||w_local - w_global||^2 over the
  trainable (LoRA + head) parameters only. Computed differently per
  path: added to the loss before backward() on the non-DP path
  (_fedprox_proximal_term); added as a raw gradient after
  DPOptimizer.pre_step() on the DP path (_add_fedprox_gradient_dp) --
  see fix 1 above for why these can't be the same code path. Optional
  (global_params=None means no FedProx, which is what
  train_centralized.py uses).

INTERFACE CHANGES vs. the previous version (see also the interface
report delivered alongside this file): mixup_fn removed; class_weights
(required) and label_smoothing (default 0.1) added; scheduler now
raises ValueError if combined with a DP optimizer instead of silently
being accepted; new module-level function weighted_label_smoothed_ce();
new internal function _add_fedprox_gradient_dp(). train_centralized.py
has since been moved to this interface (class_weights, label_smoothing,
no mixup_fn).

Explicitly OUT of scope: optimizer/privacy_engine construction
(privacy_calibration.py), checkpointing (checkpoint.py), and every
metric beyond the one macro-F1 number both call sites need for a
stop/continue or round-selection decision -- QWK, AUC-ROC, per-class
recall, confusion matrices, per-client breakdowns all belong in
evaluation/, built from the same raw logits/targets this file returns,
not duplicated here.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score

from models.coatnet_lora import NUM_DR_GRADES, freeze_bn_eval

logger = logging.getLogger(__name__)

__all__ = [
    "weighted_label_smoothed_ce",
    "run_training_epoch",
    "run_validation_epoch",
    "compute_selection_metric",
]

try:
    from opacus.optimizers import DPOptimizer
except ImportError:
    # Opacus is confirmed installed on the pod (training-phase-
    # decisions.md Section 8.5), so this should never actually trigger
    # there. Kept optional so this file stays importable (for syntax-
    # level editing) in a local Windows/VS Code environment without
    # Opacus installed -- the isinstance check below just never matches
    # in that case, since is_dp can't be True without the real class.
    DPOptimizer = None
    logger.warning(
        "opacus.optimizers.DPOptimizer could not be imported -- DP-path "
        "detection in run_training_epoch() is disabled. This is only "
        "expected outside the pod environment."
    )


def weighted_label_smoothed_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Class-weighted, label-smoothed cross-entropy -- the one loss
    function used on the training path for all 8 experiments, replacing
    Mixup's SoftTargetCrossEntropy (see module docstring, fix 3).

    per_sample = F.cross_entropy(logits, targets, reduction="none",
                                  label_smoothing=label_smoothing)
    loss = (class_weights[targets] * per_sample).mean()

    Deliberately NOT nn.CrossEntropyLoss(weight=...): that normalizes
    by the SUM of weights of samples actually present in the batch
    (a weighted mean), which makes any one sample's effective gradient
    depend on which other labels happen to share its batch -- breaking
    the per-sample independence Opacus's per-sample clipping and
    mean-loss scaling both assume. This function always divides by
    batch size (plain .mean()), never by the sum of weights, on both
    the DP and non-DP paths.

    Label smoothing happens only here, in exactly one place -- do not
    add it anywhere else in the training pipeline.

    Args:
        logits: raw model output, shape (N, num_classes).
        targets: hard integer labels, shape (N,).
        class_weights: a tensor of shape (num_classes,), already on the
            same device as `logits`. This function does not compute
            class counts -- that's the caller's job. Recommended: the
            caller normalizes weights so their expectation under the
            client's own training label distribution is 1 (e.g.
            weights = (1/class_counts); weights *= num_classes /
            weights[labels].sum()-style normalization, or equivalent) --
            this keeps the overall loss/gradient magnitude comparable to
            an unweighted run, so the locked LR and max_grad_norm values
            stay meaningful rather than needing their own re-tuning
            purely because of the weighting scheme.
        label_smoothing: locked at 0.1 (the same value Mixup's
            label_smoothing= used to provide) -- kept as a parameter,
            not hardcoded, consistent with this project's general
            preference for explicit-over-assumed values.
    """
    per_sample = F.cross_entropy(
        logits, targets, reduction="none", label_smoothing=label_smoothing
    )
    return (class_weights[targets] * per_sample).mean()


def _fedprox_proximal_term(
    model: nn.Module, global_params: List[torch.Tensor], proximal_mu: float
) -> torch.Tensor:
    """FedProx's proximal regularization term, over trainable params only.
    NON-DP PATH ONLY -- see _add_fedprox_gradient_dp() for the DP path,
    which cannot use this function (module docstring, fix 1).

    The frozen backbone is identical between local and global by
    construction (it never changes), so including it would add nothing
    but wasted compute -- only LoRA A/B and the head participate.

    global_params must be an ordered snapshot of
    `[p.detach().clone() for p in model.parameters() if p.requires_grad]`
    taken at the START of the current FL round, in that exact order.
    federated/client.py owns capturing this snapshot before any local
    steps run. Order and count are checked explicitly rather than
    relying on zip()'s silent truncation -- a mismatch here would
    silently compute a meaningless proximal term against the wrong
    parameter pairing, corrupting every FedProx experiment identically.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) == len(global_params), (
        f"Trainable parameter count ({len(trainable_params)}) doesn't "
        f"match the global_params snapshot length ({len(global_params)}) "
        f"-- these must correspond 1:1, in order, or the proximal term "
        f"is meaningless. Confirm the snapshot was taken with exactly "
        f"this same `[p for p in model.parameters() if p.requires_grad]` "
        f"ordering, on the same model."
    )
    squared_diff_sum = sum(
        (local_p - global_p.to(local_p.device)).pow(2).sum()
        for local_p, global_p in zip(trainable_params, global_params)
    )
    return (proximal_mu / 2.0) * squared_diff_sum


def _add_fedprox_gradient_dp(
    model: nn.Module, global_params: List[torch.Tensor], proximal_mu: float
) -> None:
    """Add FedProx's proximal-term GRADIENT directly to .grad, for the
    DP path only, and ONLY immediately after
    optimizer.pre_step() has just returned True (i.e. .grad already
    holds the clipped+noised task-loss gradient for this logical batch).

    Fix for the bug flagged in an earlier version of this file: adding
    the proximal term to the LOSS (as _fedprox_proximal_term does for
    the non-DP path) never reaches Opacus's .grad_sample at all -- the
    term is a direct function of the parameters, never flows through
    any model layer's forward pass, so it never triggers
    GradSampleModule's hooks, and gets silently dropped when
    DPOptimizer.step() overwrites .grad from .grad_sample. Proven
    empirically: DP update shift was exactly 0.000000 with vs without
    the proximal term, on both a tiny model and the real CoAtNet+LoRA
    model, confirming the drop.

    The fix instead adds this term's closed-form gradient --
    d/dp[(mu/2)*||p-g||^2] = mu*(p-g) -- directly to .grad AFTER
    pre_step() has finished clipping and noising the task loss's
    contribution. Safe from a DP-accounting standpoint because the
    added quantity depends ONLY on the current (public) model
    parameters and the round-start global snapshot -- never on any
    training example -- so it is pure post-processing of an
    already-private quantity: it consumes no privacy budget, and
    deliberately bypasses clipping/noising (which exist specifically to
    bound the influence of per-sample DATA, not of a data-independent
    regularization term).

    No autograd involved here -- the gradient is added directly as a
    closed-form expression, not computed via backward(), since it's
    cheap and avoids any question of how autograd would interact with
    Opacus's hooks for this term (the same question that caused the
    original bug).
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) == len(global_params), (
        f"Trainable parameter count ({len(trainable_params)}) doesn't "
        f"match the global_params snapshot length ({len(global_params)}) "
        f"-- these must correspond 1:1, in order, or the proximal term "
        f"is meaningless. Confirm the snapshot was taken with exactly "
        f"this same `[p for p in model.parameters() if p.requires_grad]` "
        f"ordering, on the same model."
    )
    with torch.no_grad():
        for local_p, global_p in zip(trainable_params, global_params):
            if local_p.grad is None:
                # Shouldn't happen immediately after pre_step() returns
                # True (it sets .grad for every trainable parameter) --
                # fail loudly rather than silently skip this parameter's
                # proximal contribution.
                raise RuntimeError(
                    "_add_fedprox_gradient_dp: a trainable parameter has "
                    "no .grad after optimizer.pre_step() returned True "
                    "-- expected pre_step() to have set .grad for every "
                    "trainable parameter by this point. Investigate "
                    "before proceeding; do not silently skip this "
                    "parameter."
                )
            local_p.grad.add_(proximal_mu * (local_p.detach() - global_p.to(local_p.device)))


def _non_dp_group_step(
    model: nn.Module,
    optimizer,
    scheduler,
    group_examples: int,
    global_params: Optional[List[torch.Tensor]],
    proximal_mu: float,
) -> None:
    """Finish one non-DP logical batch (M4). .grad holds the SUM over the
    group's micro-batches of (mean loss x example count); dividing by the
    group's example count gives the example-weighted mean gradient. The
    FedProx gradient is added once per optimizer update (not once per
    micro-batch), then clip, step, scheduler, zero_grad."""
    if group_examples <= 0:
        optimizer.zero_grad()
        return
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(group_examples)
    if global_params is not None:
        _fedprox_proximal_term(model, global_params, proximal_mu).backward()
    clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad),
        max_norm=1.0,
    )
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()


def run_training_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    device: torch.device,
    *,
    accumulation_steps: int,
    class_weights: torch.Tensor,
    label_smoothing: float = 0.1,
    global_params: Optional[List[torch.Tensor]] = None,
    proximal_mu: float = 0.0,
    scheduler=None,
) -> float:
    """One local training epoch. Used as-is by both train_centralized.py
    (global_params=None -- no FedProx, since it's not FL) and
    federated/client.py (global_params=<round-start snapshot>,
    proximal_mu=0.01 per Section 2).

    Args:
        model: the peft-wrapped model from create_coatnet_lora_model().
        dataloader: yields (images, hard_label_targets) batches. No
            Mixup anywhere in this project anymore (module docstring,
            fix 2) -- targets are always plain integer labels.
        optimizer: either a plain torch optimizer (experiments 1-3) or
            an Opacus-wrapped DPOptimizer (4-8) -- which one determines
            the step cadence automatically, see module docstring.
        accumulation_steps: micro-batches per effective/logical batch.
            Used ONLY on the non-DP path (ignored, with a debug log, on
            the DP path, where BatchMemoryManager governs the real
            batching upstream). No default -- effective batch size /
            physical batch size hasn't been empirically confirmed yet
            (open item), so this must come from the caller, not be
            silently assumed here.
        class_weights: tensor of shape (num_classes,) -- see
            weighted_label_smoothed_ce()'s docstring for the expected
            normalization convention. Moved to `device` once here, not
            per micro-batch.
        label_smoothing: passed through to weighted_label_smoothed_ce().
            Locked at 0.1; kept as a parameter rather than hardcoded.
        global_params: trainable-parameter snapshot for FedProx's
            proximal term, or None for no proximal term at all.
        proximal_mu: FedProx's mu. Ignored unless global_params is set.
        scheduler: an LR scheduler to call .step() on immediately after
            every REAL optimizer.step() -- i.e. step-level, not
            epoch-level. Non-DP path ONLY -- raises ValueError if
            provided together with a DP-wrapped optimizer (see M1 in
            the module docstring for why). Use this for
            train_centralized.py's locked "step-level warmup+cosine"
            schedule. Leave as None for FL callers (client.py) -- FL's
            schedule is round-level, stepped externally between rounds,
            and has nothing to do with this function.

    Returns:
        Mean task loss per example (excluding the proximal term, if
        any) over the epoch -- a training-progress signal, NOT the
        early-stopping criterion. Use compute_selection_metric() on a
        run_validation_epoch() result for that.
    """
    if global_params is not None:
        assert proximal_mu > 0.0, (
            "global_params was provided but proximal_mu is 0 -- likely "
            "a caller bug (the proximal term would be computed and then "
            "silently multiplied by zero, wasting compute for no "
            "effect). Pass a nonzero proximal_mu, or omit global_params "
            "entirely for a plain, non-FedProx training step."
        )

    is_dp = DPOptimizer is not None and isinstance(optimizer, DPOptimizer)

    if is_dp and scheduler is not None:
        raise ValueError(
            "run_training_epoch: a scheduler was provided together with "
            "a DP-wrapped optimizer. DPOptimizer.pre_step() returns "
            "False (no real step) on every micro-batch except the last "
            "one in a logical batch, so a scheduler stepped after a "
            "real step would fire far more often than real optimizer "
            "steps actually happen -- not a clean step-level schedule. "
            "Locked design has centralized training (the only caller "
            "that uses a step-level schedule) as non-DP, and FL's "
            "round-level schedule is stepped externally, outside this "
            "function entirely -- so this combination should never "
            "legitimately occur. Treat this as a caller bug, not a case "
            "to work around here."
        )

    if is_dp and global_params is not None:
        logger.debug(
            "run_training_epoch: FedProx + DP optimizer -- adding the "
            "proximal term's gradient directly to .grad after "
            "DPOptimizer.pre_step() (post clip+noise), not to the loss. "
            "See _add_fedprox_gradient_dp()'s docstring for why this is "
            "the correct mechanism on this path."
        )
    elif is_dp:
        logger.debug(
            "run_training_epoch: DP optimizer detected -- accumulation_steps "
            "(%d) is ignored; BatchMemoryManager governs the real batching.",
            accumulation_steps,
        )

    class_weights = class_weights.to(device)

    model.train()
    # Re-assert immediately after model.train() -- see freeze_bn_eval()'s
    # own docstring in models/coatnet_lora.py for why this call, right
    # here, is what actually matters (model.train() just flipped every
    # BatchNorm, frozen ones included, back into train() mode).
    freeze_bn_eval(model)

    total_task_loss = 0.0
    total_examples = 0
    group_examples = 0
    group_batches = 0
    if not is_dp:
        optimizer.zero_grad()

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        batch_n = int(targets.shape[0])

        outputs = model(images)
        task_loss = weighted_label_smoothed_ce(
            outputs, targets, class_weights, label_smoothing=label_smoothing
        )

        if is_dp:
            # Task loss only -- the proximal term (if any) is added
            # directly to .grad AFTER pre_step(), never into the loss,
            # on this path. See _add_fedprox_gradient_dp()'s docstring.
            task_loss.backward()
            if optimizer.pre_step():
                # pre_step() returns True only on the micro-batch that
                # completes a full logical batch -- clip, noise,
                # scaling, and the accountant step hook have already
                # run by the time it returns, and .grad now holds the
                # final, privatized task-loss gradient for this logical
                # batch. No manual clip_grad_norm_ here: Opacus's own
                # per-sample clipping (inside pre_step()) is the real
                # clipping mechanism on this path.
                if global_params is not None:
                    _add_fedprox_gradient_dp(model, global_params, proximal_mu)
                optimizer.original_optimizer.step()
            optimizer.zero_grad()
        else:
            # M4: example-weighted accumulation; the FedProx gradient is
            # added once per update in _non_dp_group_step().
            (task_loss * batch_n).backward()
            group_examples += batch_n
            group_batches += 1
            if group_batches == accumulation_steps:
                _non_dp_group_step(
                    model, optimizer, scheduler, group_examples,
                    global_params, proximal_mu,
                )
                group_examples = 0
                group_batches = 0

        if batch_n:
            total_task_loss += task_loss.item() * batch_n
            total_examples += batch_n

    if not is_dp and group_batches > 0:
        # Final, shorter group (dataset size not a multiple of
        # accumulation_steps x batch size): same example-weighted step.
        _non_dp_group_step(
            model, optimizer, scheduler, group_examples,
            global_params, proximal_mu,
        )

    return total_task_loss / max(total_examples, 1)


def run_validation_epoch(
    model: nn.Module, dataloader, device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Forward-only validation pass -- no backward, no optimizer, plain
    unweighted cross-entropy (progress signal only, not the selection
    criterion -- see compute_selection_metric()). Unchanged by this
    patch: never used Mixup or class weights to begin with.

    Returns (val_loss, all_logits, all_targets):
      - val_loss: per-example mean cross-entropy over the validation set,
        on hard labels (sum over examples / example count, so a short
        final batch is weighted correctly).
      - all_logits: raw, pre-softmax model output, shape (N, 5).
        Deliberately raw rather than hard predictions, so both
        compute_selection_metric() below AND the future evaluation/
        module (QWK, AUC-ROC, per-class F1, confusion matrix, per-client
        breakdown) can derive whatever they need -- softmax
        probabilities for AUC-ROC, argmax class labels for everything
        else -- without a second forward pass over the validation set.
      - all_targets: hard DR-grade labels, shape (N,).

    model.eval() puts every submodule into eval mode uniformly -- unlike
    training, there's no frozen-vs-trainable distinction to preserve
    here, so freeze_bn_eval() isn't needed on this path.
    """
    model.eval()

    total_loss = 0.0
    num_examples = 0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)

            total_loss += F.cross_entropy(outputs, targets, reduction="sum").item()
            num_examples += int(targets.shape[0])
            all_logits.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    val_loss = total_loss / max(num_examples, 1)
    return (
        val_loss,
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_targets, axis=0),
    )


def compute_selection_metric(logits: np.ndarray, targets: np.ndarray) -> float:
    """The one number both train_centralized.py's early-stopping check
    (patience 5) and federated/server.py's round-selection need: val
    macro-F1 -- per Section 2's explicit "patience 5 on val macro-F1,"
    not val loss. With ~73.6% No-DR, loss keeps improving as the model
    over-fits the majority class, while macro-F1 (weighting all 5
    classes equally) can plateau or worsen as minority classes stall --
    exactly why macro-F1 was the locked criterion in the first place;
    using loss here would quietly defeat that choice. FL round-selection
    wasn't pinned to a specific metric in Section 2 -- locked here to
    the same macro-F1, via this same helper, so both call sites can't
    independently drift on `average=` or edge-case handling over time.

    PATCHED (M2): labels=np.arange(NUM_DR_GRADES) and zero_division=0
    are now passed explicitly. Without the former, f1_score()'s default
    behavior computes the macro average only over whichever classes
    happen to appear in this particular batch of targets/predictions --
    a validation split (or an unlucky small FL client) with zero
    examples of some DR grade would silently change the denominator the
    average is computed over, making the metric not comparable across
    epochs or experiments. zero_division=0 makes the 0/0 case for a
    genuinely-absent-and-never-predicted class explicit (0, not a
    warning) instead of relying on sklearn's default behavior.

    Deliberately the only metric computed here -- not QWK, not per-class
    recall, not a confusion matrix. Both call sites need exactly this
    one number for a stop/continue or round-selection decision; the
    richer diagnostic metrics belong in evaluation/, built from the same
    raw logits/targets a run_validation_epoch() caller already has on
    hand, not duplicated here.
    """
    predictions = logits.argmax(axis=1)
    return f1_score(
        targets, predictions, average="macro",
        labels=np.arange(NUM_DR_GRADES), zero_division=0,
    )