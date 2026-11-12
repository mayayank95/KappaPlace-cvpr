import copy
import math

import torch
import torch.nn as nn

from models.cosplace_uncertainty.cosplace_model.cosplace_network import GeoLocalizationNet
from models.cosplace_uncertainty.cosplace_model.layers import L2Norm
from models.mixvpr import MixVPRModel, mixvpr_agg_config
from models.salad import SaladEncoder, SaladPrenorm


class _SeqVarHead(nn.Module):
    """MixVPR Lightning checkpoints store the head as ``var_head.seq.*``."""

    def __init__(self, layers):
        super().__init__()
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class GeneralModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            return output
        dummy = torch.zeros(output.shape[0], 1, device=output.device, dtype=output.dtype)
        return output, dummy


def _stable_var_init(module, activation="softplus", target_variance=0.1):
    """Initialize variance head bias so activation(0) ≈ target_variance."""
    if activation == "softplus":
        bias_val = math.log(math.exp(target_variance) - 1)
    else:  # sigmoid
        bias_val = -math.log(1.0 / target_variance - 1)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_val)


def _build_var_head(opt, fc_output_dim, aggregation=None):
    """Build the variance head from orthogonal flags."""
    use_agg = bool(opt.get("var_head_agg", False))
    linear_kind = str(opt.get("var_head_linear", "d")).lower()
    act_name = opt.get("variance_activation", "softplus") or "softplus"
    activation = nn.Sigmoid() if act_name == "sigmoid" else nn.Softplus()

    layers = []
    if use_agg:
        assert aggregation is not None, "var_head_agg=True requires the aggregation module"
        copied = copy.deepcopy(aggregation)
        # SALAD's retrieval forward returns the intra-normalized descriptor.
        # The uncertainty copy returns its own prenorm and keeps its own weights.
        if str(opt.get("method", "")).lower() == "salad":
            copied = SaladPrenorm(copied)
        layers.append(copied)

    if linear_kind == "d":
        layers.append(nn.Linear(fc_output_dim, fc_output_dim))
    elif linear_kind == "1":
        layers.append(nn.Linear(fc_output_dim, 1))
    elif linear_kind != "none":
        raise ValueError(
            f"Unknown var_head_linear: {linear_kind!r}. Expected 'none', 'd', or '1'."
        )

    layers.append(activation)
    # MixVPR and SALAD Lightning checkpoints store the head as var_head.seq.*.
    if str(opt.get("method", "")).lower() in ("mixvpr", "salad"):
        head = _SeqVarHead(layers)
    else:
        head = nn.Sequential(*layers)

    if opt.get("var_init"):
        _stable_var_init(head, act_name)

    return head, use_agg


class MixVPREncoder(MixVPRModel):
    """MixVPR method encoder: backbone + aggregator. Init from checkpoint, not ImageNet."""

    def __init__(self, opt):
        raw_size = opt.get("image_size")
        if raw_size is not None:
            image_size = int(raw_size[0] if isinstance(raw_size, (list, tuple)) else raw_size)
            if image_size != 320:
                raise ValueError(
                    f"mixvpr image_size must be 320 (aggregator expects a 20×20 map), got {image_size}"
                )
        super().__init__(
            agg_config=mixvpr_agg_config(opt["descriptors_dimension"]),
            pretrained=False,
        )

    @property
    def aggregation_module(self):
        return self.aggregator


class CosPlaceEncoder(GeoLocalizationNet):
    """CosPlace method encoder: backbone + aggregation."""

    def __init__(self, opt):
        super().__init__(
            backbone=opt.get("backbone", "ResNet18"),
            fc_output_dim=opt.get("descriptors_dimension", 512),
            train_all_layers=opt.get("train_all_layers", False),
            descriptor_activation=opt.get("descriptor_activation", "tanh"),
            descriptor_extra_linear=bool(opt.get("descriptor_extra_linear", False)),
        )

    @property
    def aggregation_module(self):
        return self.aggregation


_METHOD_ENCODERS = {
    "cosplace": CosPlaceEncoder,
    "cosplace_pretrained": CosPlaceEncoder,
    "mixvpr": MixVPREncoder,
    "salad": SaladEncoder,
}


def build_method_encoder(opt):
    method = opt.get("method", "cosplace")
    encoder_cls = _METHOD_ENCODERS.get(method)
    if encoder_cls is None:
        raise ValueError(f"Unknown method: {method}")
    return encoder_cls(opt)


def _attach_encoder(model, encoder):
    """Register encoder weights at top level so checkpoints stay flat."""
    for name, module in encoder.named_children():
        setattr(model, name, module)
    model._encoder_kind = type(encoder).__name__
    model._aggregation_module = encoder.aggregation_module
    model._freeze_base = encoder.freeze_base
    return encoder


def _encode_inputs(model, inputs):
    """One backbone pass. Returns features, retrieval descriptor, and head input.

    SALAD's retrieval descriptor is intra-normalized, so its length is constant.
    The descriptor head reads the pre-L2 token and cluster sum (same width).
    Other methods use the retrieval descriptor for that head as well.
    """
    feat = model.backbone(inputs)
    if getattr(model, "_encoder_kind", None) == "SaladEncoder":
        desc, prenorm = model.aggregator.aggregate(feat)
        return feat, desc, prenorm
    desc = model._aggregation_module(feat)
    return feat, desc, desc


class Basic(nn.Module):
    """Method-agnostic descriptor model with zero variance."""

    def __init__(self, opt):
        super().__init__()
        _attach_encoder(self, build_method_encoder(opt))
        self.id = "basic"
        self.final_l2 = L2Norm()

    def freeze_base(self):
        # Encoder freeze_base only sees backbone/aggregation; collect trainable
        # params from the full wrapper (e.g. final_l2) after freezing.
        self._freeze_base()
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, inputs):
        _, desc, _head_in = _encode_inputs(self, inputs)
        mu = self.final_l2(desc)
        dummy = torch.zeros(mu.shape[0], 1, device=mu.device, dtype=mu.dtype)
        return mu, dummy


class Uncertainty(nn.Module):
    """Method-agnostic descriptor model with learned variance head."""

    def __init__(self, opt):
        super().__init__()
        _attach_encoder(self, build_method_encoder(opt))
        self.id = "uncertainty"
        self.final_l2 = L2Norm()
        fc_output_dim = opt.get("descriptors_dimension", 512)
        self.var_head, self._var_from_feature_map = _build_var_head(
            opt, fc_output_dim, aggregation=self._aggregation_module
        )

    def freeze_base(self):
        # Encoder.freeze_base freezes backbone/aggregation but returns trainable
        # params from the encoder only (misses var_head on this wrapper).
        self._freeze_base()
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, inputs):
        feat, desc, head_in = _encode_inputs(self, inputs)
        mu = self.final_l2(desc)
        if self._var_from_feature_map:
            variance = self.var_head(feat)
        else:
            variance = self.var_head(head_in)
        return mu, variance + 1e-6


def build_model_mode(opt):
    mode = opt["model_mode"]
    if mode == "basic":
        return Basic(opt)
    if mode == "uncertainty":
        return Uncertainty(opt)
    raise ValueError(f"Unknown model_mode: {mode}")
