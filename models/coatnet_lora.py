"""
models/coatnet_lora.py

Model factory for CoAtNet-1 + LoRA, per training-phase-decisions.md
Section 1 (and 1.1's resolution contingency). This is the single
factory reused identically by the LoRA rank/alpha sweep, the
centralized baseline (Experiment 1), and all 7 federated experiments
(2-8) -- nothing about model *structure* may differ between callers,
only lora_rank/lora_alpha (during the sweep) and img_size (only if the
Section 1.1 contingency is ever triggered).

Locked decisions this file implements:
- coatnet_1_rw_224 via timm, backbone 100% frozen (42M params).
- LoRA on `qkv` + `proj` inside Attention2d, stages 2-3 only (paper's
  S3/S4) -- discovered at runtime by walking the actual module tree,
  not hardcoded as a name-suffix pattern. A plain
  target_modules=["qkv", "proj"] list would let peft's suffix matching
  also catch an unrelated module that happens to be named `proj`
  elsewhere (e.g. a patch-embed projection); this file instead builds
  the exact, fully-qualified module names to target, so that risk
  doesn't apply regardless of what else the architecture contains.
- Classification head is ClassifierHead (NOT NormMlpClassifierHead --
  this corrects training-phase-decisions.md itself; coatnet_1_rw_224's
  cfg has no head_hidden_size, so timm builds a plain pool/drop/fc/
  flatten head, not the Norm-MLP variant used by non-"rw" coatnet_1).
  The actual trainable classifier is the nested head.fc (Linear(768,
  5), 3,845 params) -- NOT the top-level `head` attribute itself, which
  also contains global_pool/drop/flatten with no trainable parameters
  of their own. model.norm is a SEPARATE top-level module (the final
  LayerNorm2d) and stays frozen -- it is not part of the head at all.
  The exact qualified name was an open item (confirm via
  model.get_classifier() on-pod) -- resolved here by discovering it at
  call time via an identity walk over named_modules() (not just
  top-level named_children(), since head.fc is nested), rather than
  assuming "head".
- Frozen BatchNorm submodules must be forced into eval() and kept there
  every training step (Section 4, privacy-math reason, not something
  Opacus's ModuleValidator catches). freeze_bn_eval() below is the
  reusable call local_trainer.py must make every step, not just once
  here at setup.
- img_size is a config parameter, defaulting to 224, never hardcoded
  (Section 1.1) -- so the 384-resolution contingency, if the decision
  gate after Experiments 1 & 2 ever triggers it, is a one-line change
  at the call site, not a rewrite of this file.

Explicitly OUT of scope for this file (per the locked Opacus wrapping
order in Section 4 -- freeze params -> force frozen submodules eval()
-> peft.get_peft_model() -> build optimizer -> privacy_engine.make_
private()): optimizer construction and privacy_engine.make_private()
both happen strictly AFTER this file's peft wrapping, in the calling
code (local_trainer.py / the Opacus setup), never here.

Revised after validation against a real local environment (timm==
1.0.28, peft==0.21.2 -- the pod locks timm==1.0.29; re-run the __main__
smoke test there before trusting this on-pod too). Three real bugs
fixed, none of them judgment calls:
1. _discover_head_attr_name() searched only named_children() (direct
   top-level children), so it could never find head.fc (nested one
   level deeper) -- every real build would have hit its own
   ModelBuildError. Fixed to search named_modules() instead.
2. verify_lora_injection()'s trainable-parameter bound
   (`trainable < 200_000`) was wrong -- doesn't hold across the locked
   rank sweep {4, 8, 16, 32} (fails already at rank 8). See that
   function's docstring for the replacement and why a proposed exact
   formula (38592 * rank + 3845) was deliberately NOT adopted verbatim.
3. verify_lora_injection() called model.train() and ran a dummy forward
   WITHOUT re-calling freeze_bn_eval() first -- exactly the bug this
   file's own freeze_bn_eval() docstring warns against, just happening
   inside this file's own verification function. A random-noise dry
   run would have corrupted the pretrained backbone's frozen BatchNorm
   running statistics on every single verification call. Fixed.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
import timm
from peft import LoraConfig, PeftModel, get_peft_model

logger = logging.getLogger(__name__)

__all__ = [
    "ModelBuildError",
    "create_coatnet_lora_model",
    "freeze_bn_eval",
    "verify_lora_injection",
    "split_trainable_parameters",
]

MODEL_NAME = "coatnet_1_rw_224"
NUM_DR_GRADES = 5
LORA_STAGE_INDICES = (2, 3)  # paper's S3/S4; stages 0-1 are pure MBConv, no attention


class ModelBuildError(RuntimeError):
    """Raised when the expected CoAtNet-1 module structure isn't found.

    Deliberately loud rather than defensive-by-silently-falling-back: a
    wrong match here (e.g. matching the wrong module's `proj`, or
    finding zero Attention2d blocks because a timm version renamed
    something) would corrupt every one of the 8 experiments identically
    and invisibly. Every raise site below includes what WAS found, so a
    real timm-version drift is diagnosable from the message alone,
    without needing to reproduce the failure interactively.
    """


def _get_attention2d_class():
    """Import timm's Attention2d, with an actionable error if it's moved.

    Confirmed in training-phase-decisions.md Section 1 as living in
    timm's maxxvit.py, against timm==1.0.29 (the pod's installed
    version). Imported lazily, inside this function, so importing this
    module doesn't hard-fail in an environment with a different timm
    version -- the failure surfaces only when actually building a model.
    """
    try:
        from timm.models.maxxvit import Attention2d
        return Attention2d
    except ImportError as exc:
        raise ModelBuildError(
            "Could not import Attention2d from timm.models.maxxvit. "
            "This factory was written against timm==1.0.29 "
            "(training-phase-decisions.md Section 1). If the installed "
            "timm version has moved or renamed this class, the LoRA "
            "target-module discovery below needs updating before "
            "proceeding -- do not fall back to a guessed module path; "
            "verify the new location against the installed source first."
        ) from exc


def _discover_lora_target_modules(model: nn.Module) -> List[str]:
    """Find the exact, fully-qualified qkv/proj module names to LoRA-wrap.

    Walks model.named_modules() for Attention2d instances whose
    qualified name falls under "stages.2." or "stages.3." (the locked
    target stages), then collects each such block's `qkv` and `proj`
    children by their full dotted name.

    Returns exact full names -- not name-suffix patterns -- specifically
    so peft's matching can't accidentally also catch an unrelated module
    elsewhere that happens to be named `proj` (e.g. a patch-embed
    projection): passing exact fully-qualified names sidesteps that
    ambiguity entirely, regardless of what else exists in the network.

    Also asserts the negative case: zero Attention2d modules should
    exist outside stages 2-3, matching Section 1's stated assumption
    that only S3/S4 use attention (stages 0-1 are pure MBConv).
    """
    Attention2d = _get_attention2d_class()

    in_target_stage = lambda name: any(  # noqa: E731
        name.startswith(f"stages.{i}.") for i in LORA_STAGE_INDICES
    )

    attn_blocks = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Attention2d) and in_target_stage(name)
    ]
    if not attn_blocks:
        raise ModelBuildError(
            f"Found zero Attention2d modules under stages "
            f"{LORA_STAGE_INDICES} in {MODEL_NAME}. Expected S3/S4 to "
            f"contain attention blocks per training-phase-decisions.md "
            f"Section 1 -- either stage indexing has changed in the "
            f"installed timm version, or model construction produced an "
            f"unexpected architecture. Do not proceed without resolving "
            f"this; do not silently target a different stage."
        )

    stray = [
        name
        for name, module in model.named_modules()
        if isinstance(module, Attention2d) and not in_target_stage(name)
    ]
    if stray:
        raise ModelBuildError(
            f"Found Attention2d modules outside the expected stages "
            f"{LORA_STAGE_INDICES}: {stray}. This contradicts Section "
            f"1's locked assumption that only S3/S4 use attention -- "
            f"verify against the installed timm version before "
            f"proceeding; targeting LoRA at the wrong stages would "
            f"invalidate every experiment identically."
        )

    target_modules: List[str] = []
    for block_name, block in attn_blocks:
        found_qkv = found_proj = False
        for child_name, _child in block.named_children():
            if child_name == "qkv":
                target_modules.append(f"{block_name}.qkv")
                found_qkv = True
            elif child_name == "proj":
                target_modules.append(f"{block_name}.proj")
                found_proj = True
        if not (found_qkv and found_proj):
            raise ModelBuildError(
                f"Attention2d block '{block_name}' is missing a qkv "
                f"and/or proj child (found qkv={found_qkv}, "
                f"proj={found_proj}). Section 1 confirmed both should "
                f"exist (fused-QKV structure) -- this block's internal "
                f"layout doesn't match what was verified; do not "
                f"proceed without checking why."
            )

    return target_modules


def _discover_head_attr_name(model: nn.Module) -> str:
    """Find the qualified module name peft's modules_to_save must target.

    coatnet_1_rw_224's classifier is NOT a direct top-level attribute --
    model.get_classifier() returns model.head.fc (a plain nn.Linear(768,
    5) nested inside ClassifierHead, which also holds non-parametric
    global_pool/drop/flatten). An earlier version of this function
    searched only model.named_children() (direct children: stem,
    stages, norm, head) and so could never find it, since head.fc isn't
    one of those -- every call would have hit the ModelBuildError below
    unconditionally. Fixed to search the full module tree via
    named_modules(), which does reach head.fc, and returns its real
    qualified dotted name ("head.fc") rather than a bare top-level name.

    peft's modules_to_save matching accepts a dotted/qualified name (it
    matches by name.endswith(target)), so passing "head.fc" through
    unchanged to LoraConfig(modules_to_save=[...]) works correctly --
    nothing downstream needs to change for a nested name vs. a top-level
    one.

    Still fully runtime-discovered, not hardcoded as "head.fc" -- the
    identity check against get_classifier()'s actual return value is
    what makes this correct regardless of what a future timm version
    calls it or how deeply it's nested, consistent with this file's
    "discover, don't assume" approach everywhere else.
    """
    classifier = model.get_classifier()
    if classifier is None:
        raise ModelBuildError(
            f"{MODEL_NAME}.get_classifier() returned None -- cannot "
            f"discover the qualified module name for peft's "
            f"modules_to_save. Inspect model.named_modules() by hand "
            f"before proceeding."
        )
    for name, module in model.named_modules():
        if module is classifier and name:
            return name
    raise ModelBuildError(
        f"get_classifier() returned a module that could not be found "
        f"by identity anywhere in {MODEL_NAME}.named_modules(). "
        f"peft's modules_to_save needs a real qualified attribute path "
        f"to this exact object -- work out the correct target by hand "
        f"before proceeding; do not guess a plausible-looking name."
    )


def freeze_bn_eval(model: nn.Module) -> int:
    """Force every genuinely-frozen BatchNorm submodule into eval() mode.

    Per Section 4: live batch statistics on a frozen BatchNorm break
    DP-SGD's per-example gradient independence assumption downstream,
    even though Opacus's ModuleValidator won't catch it (it only checks
    requires_grad, never .training mode). A plain model.train() call at
    the top of every epoch/step flips ALL BatchNorm submodules back into
    train() mode -- including these -- so **this must be called again
    after every such model.train() call**, not just once at setup.

    Scoped to modules whose own parameters (if any) are ALL currently
    requires_grad=False, rather than hardcoding "stages 0-2" by index --
    this stays correct even if stage numbering or BatchNorm placement
    ever shifts. Two call sites, two different roles:
      1. Here, in create_coatnet_lora_model(), called on the frozen
         base_model BEFORE peft wraps anything -- at this point
         everything is frozen, so the scoping is inert (every BatchNorm
         qualifies) but the call still matters, since it's what actually
         sets .eval() for the first time.
      2. In local_trainer.py, called on the wrapped model every training
         step, AFTER model.train(). By then the head is trainable via
         modules_to_save, so the scoping becomes load-bearing: it
         correctly skips any BatchNorm inside the head (leaving it in
         train() mode, as a trainable layer should be) while still
         re-forcing eval() on the genuinely frozen backbone BatchNorm
         that model.train() just flipped back. In practice the head
         (ClassifierHead: global_pool/drop/fc/flatten) contains no
         BatchNorm at all, so this distinction is currently moot -- but
         the scoping is what makes that not an assumption this function
         has to rely on.

    Returns the number of BatchNorm submodules set to eval(), so callers
    can assert it's nonzero at least once during setup.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            params = list(module.parameters(recurse=False))
            if all(not p.requires_grad for p in params):
                module.eval()
                count += 1
    return count


def create_coatnet_lora_model(
    *,
    lora_rank: int,
    lora_alpha: int,
    num_classes: int = NUM_DR_GRADES,
    img_size: int = 224,
    pretrained: bool = True,
    lora_dropout: float = 0.0,
) -> PeftModel:
    """Build one CoAtNet-1 + LoRA model, per training-phase-decisions.md
    Section 1. The single factory for the rank/alpha sweep, the
    centralized baseline, and all 7 federated experiments -- nothing
    about model structure may differ between callers.

    Args:
        lora_rank: LoRA rank. No default -- the sweep protocol (rank in
            {4, 8, 16, 32}) hasn't concluded (open item #1), so callers
            must choose explicitly rather than silently inherit a
            placeholder that could end up in a real experiment by
            accident.
        lora_alpha: LoRA alpha. Locked ratio is alpha = 2 * lora_rank
            during the sweep; pass that ratio explicitly rather than
            having this function compute it, so the sweep script stays
            the one place that ratio is decided.
        num_classes: DR severity grades, 0-4. Locked at 5.
        img_size: Input resolution. Locked at 224 for all 8 experiments
            (Section 1.1) -- exists as a parameter so a future 384
            contingency is a one-line call-site change, not a rewrite.
            Do not change the default without the Section 1.1 decision
            gate (post Experiments 1 & 2) actually being triggered.
        pretrained: Load timm's ImageNet-1k weights. Locked True --
            LoRA's B matrix is zero-initialized specifically so the
            model starts exactly at the pretrained backbone (Section 2's
            LR reasoning depends on this holding).
        lora_dropout: Not discussed in the decisions doc; defaults to
            0.0 rather than silently picking a nonzero value nobody
            signed off on.

    Returns:
        A peft PeftModel wrapping the frozen CoAtNet-1 backbone, with
        LoRA adapters on qkv/proj in stages 2-3 and a trainable head.
    """
    base_model = timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
    )

    # Step 1 (Section 4 wrapping order): freeze everything.
    for param in base_model.parameters():
        param.requires_grad = False

    # Step 2: force frozen BatchNorm into eval() BEFORE peft wraps
    # anything -- the privacy-math reason, not a ModuleValidator
    # requirement. See freeze_bn_eval()'s docstring for why this call
    # site's scoping is currently inert (everything is still frozen at
    # this point) versus its later, load-bearing call in local_trainer.py.
    frozen_bn_count = freeze_bn_eval(base_model)
    if frozen_bn_count == 0:
        logger.warning(
            "%s: freeze_bn_eval() found zero frozen BatchNorm modules "
            "to set to eval(). If this model is expected to have "
            "frozen BatchNorm in its early MBConv stages (Section 4), "
            "this is unexpected -- verify before proceeding to the DP "
            "experiments (4-8), where this matters for correctness.",
            MODEL_NAME,
        )

    target_modules = _discover_lora_target_modules(base_model)
    head_attr_name = _discover_head_attr_name(base_model)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=[head_attr_name],
        bias="none",
    )

    # Step 3: peft.get_peft_model() last. Optimizer construction and
    # privacy_engine.make_private() both happen AFTER this, in the
    # calling code -- never in this file.
    model = get_peft_model(base_model, lora_config)

    logger.info(
        "%s: LoRA on %d modules (rank=%d, alpha=%d) in stages %s; "
        "head attribute '%s' trainable via modules_to_save; "
        "img_size=%d, pretrained=%s.",
        MODEL_NAME,
        len(target_modules),
        lora_rank,
        lora_alpha,
        LORA_STAGE_INDICES,
        head_attr_name,
        img_size,
        pretrained,
    )
    return model


def verify_lora_injection(model: PeftModel, lora_rank: int, img_size: int = 224) -> None:
    """Structural verification of a just-built model, as one callable
    instead of a manual one-off. Meant to run once right after
    create_coatnet_lora_model(), on the pod, as the actual verification
    step -- everything above is written defensively, but this is what
    confirms it worked against the real installed timm/peft versions,
    not just against the reasoning that went into writing it.

    Args:
        lora_rank: the rank the model was actually built with --
            required (not re-derived from the model) so Check 2 below
            can verify the ACTUAL injected rank matches what was
            requested, not just that rank-shaped parameters exist.
        img_size: must match whatever the model was built with, for the
            dry-run dummy input in Check 4.

    Raises AssertionError with a specific, actionable message on any
    failure. Does not catch or downgrade failures -- a wrong result
    here means stop, not proceed to training.

    Checks, in order:
      1. Every non-LoRA/head-named parameter is actually frozen.
      2. Every trainable parameter is LoRA (in stages 2-3, at the
         requested rank) or head -- nothing else -- plus a loose
         supplementary magnitude sanity net.
      3. A dry run (model.train() + freeze_bn_eval() + forward +
         backward) does not mutate any frozen BatchNorm's running
         statistics, and gradients land exactly where expected.
    """
    # Check 1: every parameter NOT matching the LoRA/head naming
    # convention must be frozen. This is the structural check a loose
    # total-parameter-count bound (the previous, now-removed
    # `trainable < 200_000`) could never guarantee on its own: that
    # bound could pass even with one unintended trainable backbone
    # parameter hiding among millions of frozen ones, as long as the
    # TOTAL stayed small by coincidence. This check can't be fooled
    # that way -- it inspects every single parameter's identity, not a
    # sum.
    for name, param in model.named_parameters():
        is_lora_or_head = "lora_A" in name or "lora_B" in name or "modules_to_save" in name
        if not is_lora_or_head:
            assert not param.requires_grad, (
                f"Parameter '{name}' is trainable (requires_grad=True) "
                f"but doesn't match the LoRA (lora_A/lora_B) or head "
                f"(modules_to_save) naming convention -- an unexpected "
                f"backbone parameter is trainable. Investigate before "
                f"proceeding."
            )

    # Check 2: every trainable parameter IS LoRA-or-head (the other
    # direction of Check 1 -- together these fully account for every
    # parameter in the model, trainable or not), LoRA parameters are
    # confined to stages 2-3, and -- replacing the previous broken
    # `trainable < 200_000` bound -- every LoRA parameter's shape
    # actually contains lora_rank in one of its dimensions, directly
    # confirming the INJECTED rank matches the REQUESTED rank.
    #
    # A precise closed-form bound was proposed instead (trainable ==
    # 38592 * lora_rank + 3845, derived from "16 Attention2d blocks in
    # stages 2-3"). Deliberately not adopted verbatim: that 38592
    # constant depends on CoAtNet-1's exact per-stage block count and
    # channel widths -- exactly the kind of fact this file otherwise
    # insists on discovering at runtime rather than hardcoding (see
    # _discover_lora_target_modules()). A silently-wrong hardcoded
    # constant here would be the same class of bug the 200_000 bound
    # already was, just with a different, equally unverifiable number
    # -- and it can't be checked against the real timm/peft source in
    # this environment. The per-parameter rank check below verifies the
    # same underlying concern (did LoRA actually get injected at the
    # requested rank, and nowhere else) directly from real runtime
    # shapes, with no constant to go stale.
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_lora_param = "lora_A" in name or "lora_B" in name
        is_head_param = "modules_to_save" in name
        assert is_lora_param or is_head_param, (
            f"Trainable parameter '{name}' doesn't match peft's LoRA "
            f"(lora_A/lora_B) or modules_to_save naming convention -- "
            f"something unexpected is trainable. Walk "
            f"model.named_modules() by hand before proceeding."
        )
        if is_lora_param:
            assert any(f"stages.{i}." in name for i in LORA_STAGE_INDICES), (
                f"LoRA parameter '{name}' is outside stages "
                f"{LORA_STAGE_INDICES} -- expected LoRA only on "
                f"qkv/proj within stages 2-3."
            )
            assert lora_rank in param.shape, (
                f"LoRA parameter '{name}' has shape {tuple(param.shape)}, "
                f"which doesn't contain the requested lora_rank="
                f"{lora_rank} in any dimension -- the actually-injected "
                f"rank doesn't match what was requested. Investigate "
                f"before proceeding; do not assume this is fine because "
                f"training 'ran'."
            )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable > 0, "Zero trainable parameters -- LoRA/head wrapping did not take effect."
    # Loose supplementary sanity net, NOT the primary defense (Checks 1
    # and 2 above are, and can't be fooled by a small-looking total) --
    # backbone is ~42M params; even the largest locked rank (32) should
    # stay a small fraction of that.
    assert trainable < total * 0.10, (
        f"Trainable parameter count is {trainable:,} ({trainable / total:.1%} "
        f"of {total:,} total) -- unexpectedly high even allowing for the "
        f"full locked rank sweep. Checks 1 and 2 above already confirmed "
        f"WHICH parameters are trainable is correct, so this is purely "
        f"about HOW MANY -- double check lora_rank={lora_rank} is what "
        f"you intended."
    )

    # Check 3: a dry run must not corrupt the pretrained backbone. An
    # earlier version of this function called model.train() and ran a
    # forward pass WITHOUT re-calling freeze_bn_eval() first -- exactly
    # the bug freeze_bn_eval()'s own docstring warns every OTHER caller
    # about, just missed here. model.train() flips every BatchNorm
    # (frozen ones included) back into train() mode; a random-noise
    # dummy forward would then blend garbage statistics into the
    # pretrained backbone's running_mean/running_var on every single
    # verification call. Fixed: re-assert eval() immediately, then
    # explicitly confirm (not just trust) that nothing moved.
    frozen_bn_modules = [
        module for module in model.modules()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        and all(not p.requires_grad for p in module.parameters(recurse=False))
    ]
    model.train()
    freeze_bn_eval(model)
    bn_snapshots = [
        (
            m.running_mean.clone() if m.running_mean is not None else None,
            m.running_var.clone() if m.running_var is not None else None,
            m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None,
        )
        for m in frozen_bn_modules
    ]

    dummy_input = torch.randn(2, 3, img_size, img_size)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()

    for module, (running_mean, running_var, num_batches_tracked) in zip(
        frozen_bn_modules, bn_snapshots
    ):
        assert not module.training, (
            "A frozen BatchNorm module is in train() mode after the dry "
            "run -- freeze_bn_eval() should have kept it in eval(). A "
            "frozen BatchNorm left in train mode during real training "
            "would corrupt the pretrained backbone's running statistics; "
            "investigate before proceeding."
        )
        if running_mean is not None:
            assert torch.equal(module.running_mean, running_mean), (
                "A frozen BatchNorm's running_mean changed during the "
                "dry run despite being in eval() mode -- this should be "
                "impossible; investigate before trusting this model."
            )
        if running_var is not None:
            assert torch.equal(module.running_var, running_var), (
                "A frozen BatchNorm's running_var changed during the dry "
                "run despite being in eval() mode -- investigate before "
                "trusting this model."
            )
        if num_batches_tracked is not None:
            assert torch.equal(module.num_batches_tracked, num_batches_tracked), (
                "A frozen BatchNorm's num_batches_tracked changed during "
                "the dry run despite being in eval() mode -- investigate "
                "before trusting this model."
            )

    missing_grad = [
        name for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert not missing_grad, (
        f"{len(missing_grad)} trainable parameter(s) got no gradient "
        f"from the dry-run backward pass: {missing_grad[:5]}"
        f"{'...' if len(missing_grad) > 5 else ''}. Something in the "
        f"forward path isn't connected to these parameters."
    )

    unexpected_grad = [
        name for name, param in model.named_parameters()
        if not param.requires_grad and param.grad is not None
    ]
    assert not unexpected_grad, (
        f"{len(unexpected_grad)} FROZEN parameter(s) received a "
        f"gradient anyway: {unexpected_grad[:5]}"
        f"{'...' if len(unexpected_grad) > 5 else ''}. This would mean "
        f"the backbone isn't actually frozen -- do not proceed."
    )

    model.zero_grad(set_to_none=True)
    logger.info(
        "verify_lora_injection: passed all checks (%d trainable params, "
        "rank=%d confirmed injected, %d frozen BatchNorm modules "
        "untouched, gradients flowed correctly, nothing frozen leaked).",
        trainable, lora_rank, len(frozen_bn_modules),
    )


def split_trainable_parameters(
    model: PeftModel,
) -> "tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]":
    """Split trainable parameters into (lora_params, head_params), for
    building AdamW's two locked param groups (LR 1e-4 LoRA / 1e-3 head).

    Uses the same peft naming convention verify_lora_injection() already
    relies on (lora_A/lora_B for LoRA, modules_to_save for the head) --
    reusing it here rather than re-deriving the same classification a
    third time (train_centralized.py and federated/client.py both need
    this split for their optimizer construction).

    Raises ModelBuildError if any trainable parameter matches neither
    pattern (would silently exclude it from both param groups, meaning
    a part of the model that's supposed to train would get no LR at
    all) or if either list ends up empty.
    """
    lora_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []
    unmatched: List[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(param)
        elif "modules_to_save" in name:
            head_params.append(param)
        else:
            unmatched.append(name)

    if unmatched:
        raise ModelBuildError(
            f"{len(unmatched)} trainable parameter(s) matched neither "
            f"the LoRA (lora_A/lora_B) nor head (modules_to_save) naming "
            f"convention: {unmatched[:5]}{'...' if len(unmatched) > 5 else ''}. "
            f"These would silently get no param group and no LR -- "
            f"resolve this before building the optimizer."
        )
    if not lora_params:
        raise ModelBuildError("split_trainable_parameters: found zero LoRA parameters.")
    if not head_params:
        raise ModelBuildError("split_trainable_parameters: found zero head parameters.")

    return lora_params, head_params


if __name__ == "__main__":
    # Minimal standalone smoke test -- runnable the moment this one file
    # lands on the pod, before local_trainer.py or any data-loading code
    # exists. Not a substitute for the LoRA rank/alpha sweep (that's
    # step-zero notebook work, over real data, per Section 1) -- just
    # confirms this factory builds a structurally correct model at all,
    # against the actually-installed timm/peft versions.
    logging.basicConfig(level=logging.INFO)
    logger.info("Smoke test: building %s with an arbitrary sweep-range rank/alpha (8/16)...", MODEL_NAME)
    smoke_model = create_coatnet_lora_model(lora_rank=8, lora_alpha=16)
    verify_lora_injection(smoke_model, lora_rank=8)
    logger.info("Smoke test passed.")