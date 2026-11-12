"""Shared uncertainty loss for train and validation (same criterion and scaling)."""
import math
from typing import Optional

import torch

from losses.gaussian_cosine_loss import GaussianCosineLoss
from losses.vmf_loss import VMFLikelihood


def scheduled_uncertainty_lambda(
    epoch: int,
    lambda_: float,
    lambda_final: Optional[float],
    hold_epochs: int,
    ramp_epochs: int,
) -> float:
    """Pull target for this epoch. With no final value, lambda_ is constant.

    Epochs 0 .. hold_epochs-1 stay at lambda_. The next ramp_epochs epochs move
    linearly to lambda_final, and later epochs stay there. The last ramp epoch
    (hold_epochs + ramp_epochs - 1) is lambda_final.
    """
    start = float(lambda_)
    if lambda_final is None:
        return start
    end = float(lambda_final)
    epoch = int(epoch)
    hold = int(hold_epochs)
    ramp = int(ramp_epochs)
    if hold < 0 or ramp < 1:
        raise ValueError(
            "uncertainty lambda schedule needs hold_epochs >= 0 and ramp_epochs >= 1, "
            f"got hold={hold}, ramp={ramp}."
        )
    if epoch < hold:
        return start
    if epoch >= hold + ramp:
        return end
    # First epoch of the ramp takes one step; the last ramp epoch lands on end.
    t = (epoch - hold + 1) / float(ramp)
    return start + (end - start) * t


def curriculum_pull(
    epoch: int,
    warmup_pull: float,
    final_pull: float,
    factor: float,
    step_epochs: int,
    warmup_done_epoch: Optional[int],
) -> float:
    """Baby-step pull. Stay at warmup_pull through warmup_done_epoch.

    The next epoch multiplies by factor, then again every step_epochs, and
    does not go above final_pull. warmup_done_epoch is the epoch whose
    validation first reached the warmup target. None means the warmup is
    still running.
    """
    start = float(warmup_pull)
    if warmup_done_epoch is None or int(epoch) <= int(warmup_done_epoch):
        return start
    end = float(final_pull)
    growth = float(factor)
    span = int(step_epochs)
    if growth <= 1.0 or span < 1:
        raise ValueError(
            "curriculum pull needs factor > 1 and step_epochs >= 1, "
            f"got factor={growth}, step_epochs={span}."
        )
    epochs_after = int(epoch) - int(warmup_done_epoch) - 1
    stage = epochs_after // span
    pull = start * (growth ** (stage + 1))
    if pull > end:
        return end
    return pull


def compute_uncertainty_loss(
    input_norm: torch.Tensor,
    target_norm: torch.Tensor,
    variance: torch.Tensor,
    loss_type: str = "gaussian_nll",
    lambda_: float = 1.0,
    lambda_mode: str = "fixed",
    gnll_mu_scale_mode: str = "sqrt_dim",
    gnll_mu_scale_value: float = 1.0,
) -> torch.Tensor:
    """
    Uncertainty loss used by both training and validation.

    Args:
        input_norm: L2-normalized embeddings [N, D].
        target_norm: L2-normalized target vectors [N, D].
        variance: Predicted variance [N, D].
        loss_type: 'gaussian_nll', 'gaussian_cosine', or 'vmf'.
        lambda_: Scale factor applied to the loss. In kappa mode this is the
            target average cosine pull, not a fixed multiplier.
        lambda_mode: 'fixed' or 'kappa'. kappa divides by the detached batch
            mean of κ (vMF only).
        gnll_mu_scale_mode: GNLL scaling mode for mean vectors: sqrt_dim, none, custom.
        gnll_mu_scale_value: Custom GNLL scale when mode=custom.

    Returns:
        Tensor scalar: lambda_ * mean loss (use .backward() in train, .item() in val).
    """
    loss_type = loss_type.lower()
    lambda_mode = str(lambda_mode or "fixed").lower()
    if lambda_mode not in ("fixed", "kappa"):
        raise ValueError(
            f"Unknown uncertainty_lambda_mode '{lambda_mode}'. Expected 'fixed' or 'kappa'."
        )
    if lambda_mode == "kappa" and loss_type != "vmf":
        raise ValueError("uncertainty_lambda_mode='kappa' is only defined for uncertainty_loss='vmf'.")
    if loss_type == "gaussian_cosine":
        criterion = GaussianCosineLoss()
        loss = criterion(input_norm, target_norm, variance)
    elif loss_type == "vmf":
        kappa = torch.mean(variance, dim=-1, keepdim=True)  # [B, 1]
        criterion = VMFLikelihood()
        loss = criterion(input_norm, kappa, target_norm)
        if lambda_mode == "kappa":
            # Average ∂loss/∂cos stays at lambda_, instead of growing with κ.
            mean_kappa = kappa.detach().mean().clamp(min=1e-6)
            return loss * (float(lambda_) / mean_kappa)
    elif loss_type == "gaussian_nll":
        criterion = torch.nn.GaussianNLLLoss()
        mode = str(gnll_mu_scale_mode).lower()
        if mode == "none":
            scale = 1.0
        elif mode == "custom":
            scale = float(gnll_mu_scale_value)
        else:
            scale = math.sqrt(input_norm.shape[1])
        loss = criterion(input_norm * scale, target_norm * scale, variance)
    else:
        raise ValueError(
            f"Unknown uncertainty_loss '{loss_type}'. "
            "Expected one of: gaussian_nll, gaussian_cosine, vmf."
        )
    return lambda_ * loss


def cosface_positive_grad_magnitude(
    logits: torch.Tensor,
    targets: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Mean |∂CE/∂cos_y| after the CosFace margin.

    ``logits`` are already s * (cos − m) on the target class. The positive-pair
    gradient magnitude is s * (1 − p_y), averaged over the batch.
    """
    probs = torch.softmax(logits, dim=1)
    target_probs = probs.gather(1, targets.reshape(-1, 1))
    return logits.new_tensor(float(scale)) * (1.0 - target_probs).mean()


class DescriptorGradBalancer:
    """Scale only the vMF gradient that flows into the shared descriptor.

    CosFace and the κ head keep the gradients of their own losses. The hook
    on the descriptor makes the vMF descriptor-gradient norm equal
    rho * EMA(||∂L_CosFace / ∂descriptor||).
    """

    def __init__(self, rho: float, ema_decay: float, eps: float = 1e-8):
        if rho < 0:
            raise ValueError(f"vmf_grad_rho must be >= 0, got {rho}.")
        if not 0.0 <= float(ema_decay) < 1.0:
            raise ValueError(f"vmf_grad_ema_decay must be in [0, 1), got {ema_decay}.")
        self.rho = float(rho)
        self.decay = float(ema_decay)
        self.eps = float(eps)
        self.ema_cos = None
        self.ema_vmf = None
        self.last_cos_norm = None
        self.last_vmf_norm = None
        self.last_vmf_scaled_norm = None
        self.last_scale = None

    def _ema(self, previous: Optional[float], value: float) -> float:
        if previous is None:
            return float(value)
        return self.decay * float(previous) + (1.0 - self.decay) * float(value)

    def backward(
        self,
        loss_ce: torch.Tensor,
        loss_vmf: torch.Tensor,
        descriptor: torch.Tensor,
    ) -> None:
        """Backward CosFace unchanged, then vMF with a scaled descriptor gradient."""
        if not descriptor.requires_grad:
            raise RuntimeError("Descriptor gradient balancing needs a descriptor that requires grad.")

        cos_norm = {}

        def record_cosface(grad: torch.Tensor) -> torch.Tensor:
            cos_norm["value"] = grad.detach().float().norm()
            return grad

        cos_hook = descriptor.register_hook(record_cosface)
        try:
            loss_ce.backward(retain_graph=True)
        finally:
            cos_hook.remove()
        if "value" not in cos_norm:
            raise RuntimeError("CosFace loss did not send a gradient to the descriptor.")

        g_cos = cos_norm["value"]
        self.ema_cos = self._ema(self.ema_cos, float(g_cos))
        target = self.rho * self.ema_cos
        vmf_norm = {}

        def scale_vmf(grad: torch.Tensor) -> torch.Tensor:
            current = grad.detach().float().norm()
            vmf_norm["value"] = current
            scale = current.new_tensor(target) / (current + self.eps)
            vmf_norm["scale"] = scale
            return grad * scale.to(dtype=grad.dtype)

        vmf_hook = descriptor.register_hook(scale_vmf)
        try:
            loss_vmf.backward()
        finally:
            vmf_hook.remove()
        if "value" not in vmf_norm:
            raise RuntimeError("vMF loss did not send a gradient to the descriptor.")

        g_vmf = vmf_norm["value"]
        scale = vmf_norm["scale"]
        self.ema_vmf = self._ema(self.ema_vmf, float(g_vmf))
        self.last_cos_norm = float(g_cos)
        self.last_vmf_norm = float(g_vmf)
        self.last_scale = float(scale)
        self.last_vmf_scaled_norm = float(g_vmf) * self.last_scale

