"""Supply-side optimizers: fused AdamW and hybrid Muon (Newton-Schulz 5) + AdamW."""
from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn as nn

OPTIMIZERS: tuple[str, ...] = ("adamw", "muon")
DEFAULT_MUON_LR: float = 0.02
DEFAULT_MUON_MOMENTUM: float = 0.95
DEFAULT_MUON_NS_STEPS: int = 5


def zeropower_via_newtonschulz5(
    grad: torch.Tensor,
    *,
    steps: int = DEFAULT_MUON_NS_STEPS,
    eps: float = 1e-7,
) -> torch.Tensor:
    """5-step quintic Newton-Schulz orthogonalization approximating polar factor U @ V^T.

    Operates in bfloat16 on CUDA for throughput and float32 on CPU for numerical fidelity.
    """
    if grad.ndim != 2:
        raise ValueError(f"Newton-Schulz orthogonalization requires a 2D tensor, got {grad.ndim}D")
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_dtype = grad.dtype
    x = grad.bfloat16() if grad.is_cuda else grad.float()
    transposed = False
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    x = x / (x.norm() + float(eps))
    for _ in range(int(steps)):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(dtype=orig_dtype)


def _is_excluded_from_muon(param_name: str, param: torch.Tensor) -> bool:
    """Exclude 1D tensors, token/structure embeddings, and output readout heads from Muon."""
    if param.ndim != 2:
        return True
    lname = param_name.lower()
    excluded_tokens = (
        "tok_emb",
        "aa_emb",
        "embed",
        "lm_head",
        "struct_3di_emb",
        "struct_3di_head",
        "pair_dist_emb",
        "contact_head",
        "geom_proj",
    )
    return any(tok in lname for tok in excluded_tokens)


class MuonWithAdamW(torch.optim.Optimizer):
    """Hybrid optimizer applying Newton-Schulz 5 Muon to hidden 2D matrices and AdamW to the rest.

    - Hidden 2D weight matrices (attention & SwiGLU projections, excluding embeddings & heads):
      Nesterov momentum (``momentum=0.95``, ``nesterov=True``) + 5-step Newton-Schulz
      orthogonalization with aspect-ratio scaling ``sqrt(max(1, d_out / d_in))`` and decoupled
      weight decay.
    - Input embeddings, output heads, and 1D RMSNorm gains:
      Standard decoupled AdamW (``betas=(0.9, 0.95)``, ``eps=1e-8``).

    Each parameter group carries a ``peak_lr`` attribute so external LR schedules can scale
    both Muon (e.g. ``lr=0.02``) and AdamW (e.g. ``base_lr``) proportionally via
    ``set_optimizer_lr(optimizer, current_adamw_lr, peak_adamw_lr)``.
    """

    def __init__(
        self,
        param_groups: Iterable[dict[str, Any]],
        *,
        lr: float = 1e-3,
        muon_lr: float = DEFAULT_MUON_LR,
        momentum: float = DEFAULT_MUON_MOMENTUM,
        nesterov: bool = True,
        ns_steps: int = DEFAULT_MUON_NS_STEPS,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "peak_lr": float(lr),
            "use_muon": False,
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "ns_steps": int(ns_steps),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
        }
        super().__init__(param_groups, defaults)
        self.base_adamw_peak_lr = float(lr)
        self.base_muon_peak_lr = float(muon_lr)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = bool(group.get("use_muon", False))
            lr = float(group["lr"])
            wd = float(group.get("weight_decay", 0.0))

            if use_muon:
                momentum = float(group.get("momentum", DEFAULT_MUON_MOMENTUM))
                nesterov = bool(group.get("nesterov", True))
                ns_steps = int(group.get("ns_steps", DEFAULT_MUON_NS_STEPS))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.is_sparse:
                        raise RuntimeError("MuonWithAdamW does not support sparse gradients")
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    update = g.add(buf, alpha=momentum) if nesterov else buf
                    orth = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    d_out, d_in = int(p.shape[0]), int(p.shape[1])
                    aspect_scale = math.sqrt(max(1.0, float(d_out) / float(max(1, d_in))))
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(orth, alpha=-lr * aspect_scale)
            else:
                beta1, beta2 = group.get("betas", (0.9, 0.95))
                eps = float(group.get("eps", 1e-8))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.is_sparse:
                        raise RuntimeError("MuonWithAdamW does not support sparse gradients")
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    step_t = int(state["step"])
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)

                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    bias_c1 = 1.0 - beta1 ** step_t
                    bias_c2 = 1.0 - beta2 ** step_t
                    step_size = lr / bias_c1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def build_named_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str = "adamw",
    peak_lr: float,
    weight_decay: float,
    device: torch.device,
    muon_lr: float = DEFAULT_MUON_LR,
    muon_lr_scale: float | None = None,
    muon_momentum: float = DEFAULT_MUON_MOMENTUM,
    muon_nesterov: bool = True,
    muon_ns_steps: int = DEFAULT_MUON_NS_STEPS,
) -> torch.optim.Optimizer:
    """Create either fused AdamW or hybrid Muon + AdamW optimizer."""
    opt_key = str(optimizer_name).strip().lower()
    if opt_key not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}; expected {OPTIMIZERS}")

    if opt_key == "adamw":
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
        param_groups = [
            {"params": decay_params, "weight_decay": float(weight_decay), "peak_lr": float(peak_lr)},
            {"params": no_decay_params, "weight_decay": 0.0, "peak_lr": float(peak_lr)},
        ]
        use_fused = device.type == "cuda"
        return torch.optim.AdamW(
            param_groups,
            lr=float(peak_lr),
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )

    # Muon on hidden 2D matrices + AdamW on embeddings, heads, and 1D RMSNorm gains
    eff_muon_lr = (
        float(peak_lr) * float(muon_lr_scale)
        if muon_lr_scale is not None
        else float(muon_lr)
    )
    muon_params: list[nn.Parameter] = []
    adamw_decay_params: list[nn.Parameter] = []
    adamw_no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not _is_excluded_from_muon(name, param):
            muon_params.append(param)
        elif param.ndim >= 2:
            adamw_decay_params.append(param)
        else:
            adamw_no_decay_params.append(param)

    groups: list[dict[str, Any]] = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "use_muon": True,
                "lr": eff_muon_lr,
                "peak_lr": eff_muon_lr,
                "momentum": float(muon_momentum),
                "nesterov": bool(muon_nesterov),
                "ns_steps": int(muon_ns_steps),
                "weight_decay": float(weight_decay),
            }
        )
    if adamw_decay_params:
        groups.append(
            {
                "params": adamw_decay_params,
                "use_muon": False,
                "lr": float(peak_lr),
                "peak_lr": float(peak_lr),
                "weight_decay": float(weight_decay),
            }
        )
    if adamw_no_decay_params:
        groups.append(
            {
                "params": adamw_no_decay_params,
                "use_muon": False,
                "lr": float(peak_lr),
                "peak_lr": float(peak_lr),
                "weight_decay": 0.0,
            }
        )
    return MuonWithAdamW(
        groups,
        lr=float(peak_lr),
        muon_lr=eff_muon_lr,
        momentum=float(muon_momentum),
        nesterov=bool(muon_nesterov),
        ns_steps=int(muon_ns_steps),
        weight_decay=float(weight_decay),
    )


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    current_lr: float,
    *,
    peak_lr: float,
) -> None:
    """Scale each parameter group's LR proportionally to its group-specific peak_lr."""
    ratio = float(current_lr) / float(peak_lr) if float(peak_lr) > 0.0 else 0.0
    for group in optimizer.param_groups:
        g_peak = group.get("peak_lr")
        if g_peak is not None:
            group["lr"] = float(g_peak) * ratio
        else:
            group["lr"] = float(current_lr)
