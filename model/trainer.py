from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional at runtime
    wandb = None  # type: ignore

try:
    from torchvision.utils import make_grid
except ImportError:  # pragma: no cover - torchvision is optional
    make_grid = None  # type: ignore


from .utils import (
    add_gaussian_noise,
    apply_random_brightness_contrast,
    compute_l1_loss,
    estimate_noise_level,
    prepare_image_for_logging,
    sobel_edges,
)


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}")

def _safe_fmean(values: List[float]) -> float:
    return fmean(values) if values else 0.0


@dataclass
class TrainerHistory:
    epoch: int
    total_loss: float
    content_loss: float
    recon_loss: float
    cross_loss: float
    edge_loss: float
    latent_variance: float
    cross_variance: float
    collapsed: bool
    noise_a: float = 0.0
    noise_b: float = 0.0


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"content": 1.0, "recon": 1.0, "cross": 1.0, "edge": 0.0}
    )
    collapse_counter: int = 0


class Contrast2ContrastTrainer:
    """
    Trainer implementing Method #3 (shared encoder + domain-specific decoders).
    Assumes dataloader yields paired batches containing domain A / B tensors.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder_a: nn.Module,
        decoder_b: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        config: Dict[str, Any],
        wandb_run: Optional[Any] = None,
    ) -> None:
        self.encoder = encoder.to(device)
        self.decoder_a = decoder_a.to(device)
        self.decoder_b = decoder_b.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.state = TrainerState()
        self._configure_loss_weights(config.get("loss_weights", {}))
        self.noise_cfg = config.get("noise", {})
        self.augment_cfg = config.get("augmentation", {})
        self.edge_cfg = config.get("edge_loss", {})
        self.anti_collapse_cfg = config.get("anti_collapse", {})
        self.logging_cfg = config.get("logging", {})
        self.scheduler_step = config.get("scheduler_step", "epoch")
        self.grad_clip = config.get("grad_clip_norm")
        self.noise_estimation_cfg = config.get("noise_estimation", {})
        self.encoder_output_index = int(config.get("encoder_output_index", -1))
        self.input_representation = config.get("input_representation")
        self.input_keys = config.get(
            "input_keys", {"domain_a": "xA", "domain_b": "xB"}
        )
        self.cross_domain_strategy = config.get("cross_domain_strategy", "use_source_skip")
        self.mixed_skip_alpha = float(config.get("mixed_skip_alpha", 0.5))
        self.wandb_run = wandb_run
        self.wandb_enabled = bool(
            self.logging_cfg.get("use_wandb", wandb_run is not None)
        )
        if self.wandb_enabled and wandb_run is None and wandb is None:
            raise ImportError("wandb is required for logging but is not installed.")
        self._history: List[TrainerHistory] = []
        self._noise_warning_emitted = False

    def _configure_loss_weights(self, weights: Dict[str, float]) -> None:
        content = float(weights.get("content", weights.get("alpha", 1.0)))
        recon = float(weights.get("recon", weights.get("beta", 1.0)))
        cross = float(weights.get("cross", weights.get("gamma", 1.0)))
        edge = float(weights.get("edge", weights.get("delta", 0.0)))
        self.state.loss_weights.update(
            {"content": content, "recon": recon, "cross": cross, "edge": edge}
        )

    def fit(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        num_epochs: int,
        val_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
        checkpoint_fn: Optional[Any] = None,
    ) -> List[TrainerHistory]:
        for epoch in range(num_epochs):
            self.state.epoch = epoch
            history = self._train_one_epoch(train_loader, epoch)
            if self.scheduler and self.scheduler_step == "epoch":
                self.scheduler.step()
            collapsed = self._maybe_adjust_loss_weights(history)
            history.collapsed = collapsed
            self._history.append(history)
            if checkpoint_fn:
                checkpoint_fn(epoch, history)
            if val_loader is not None:
                self.evaluate(val_loader)
        return self._history

    def evaluate(self, loader: Iterable[Dict[str, torch.Tensor]]) -> None:
        self.encoder.eval()
        self.decoder_a.eval()
        self.decoder_b.eval()
        with torch.no_grad():
            for batch in loader:
                _ = self._prepare_batch(batch)
        self.encoder.train()
        self.decoder_a.train()
        self.decoder_b.train()

    def _prepare_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key_a = self.input_keys["domain_a"]
        key_b = self.input_keys["domain_b"]
        x_a = _to_device(batch[key_a], self.device)
        x_b = _to_device(batch[key_b], self.device)
        scale_a = self._prepare_scale_tensor(batch.get("pd_scale"), x_a.shape[0])
        scale_b = self._prepare_scale_tensor(batch.get("pdfs_scale"), x_b.shape[0])
        return x_a, x_b, scale_a, scale_b

    def _prepare_scale_tensor(
        self, value: Optional[torch.Tensor], batch_size: int
    ) -> torch.Tensor:
        if value is None:
            return torch.ones(batch_size, 1, 1, 1, device=self.device)
        tensor = _to_device(value, self.device).float()
        if tensor.ndim == 0:
            tensor = tensor.view(1)
        if tensor.shape[0] != batch_size:
            tensor = tensor.expand(batch_size)
        return tensor.view(batch_size, 1, 1, 1)

    def _apply_augmentations(
        self, x_a: torch.Tensor, x_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.noise_cfg.get("enabled", False):
            sigma_a = float(self.noise_cfg.get("sigma_a", self.noise_cfg.get("sigma", 0.0)))
            sigma_b = float(self.noise_cfg.get("sigma_b", self.noise_cfg.get("sigma", 0.0)))
            if sigma_a > 0:
                x_a = add_gaussian_noise(x_a, sigma_a)
            if sigma_b > 0:
                x_b = add_gaussian_noise(x_b, sigma_b)
        if self.augment_cfg.get("enable_jitter", False):
            brightness = float(self.augment_cfg.get("brightness", 0.0))
            contrast = float(self.augment_cfg.get("contrast", 0.0))
            if brightness > 0 or contrast > 0:
                x_a = apply_random_brightness_contrast(x_a, brightness, contrast)
                x_b = apply_random_brightness_contrast(x_b, brightness, contrast)
        return x_a, x_b

    def _train_one_epoch(
        self, loader: Iterable[Dict[str, torch.Tensor]], epoch: int
    ) -> TrainerHistory:
        self.encoder.train()
        self.decoder_a.train()
        self.decoder_b.train()

        totals = {
            "total": 0.0,
            "content": 0.0,
            "recon": 0.0,
            "cross": 0.0,
            "edge": 0.0,
            "count": 0,
        }
        latent_vars: List[float] = []
        cross_vars: List[float] = []
        noise_est_a: List[float] = []
        noise_est_b: List[float] = []
        log_interval = int(self.logging_cfg.get("image_interval", 200))

        for step, batch in enumerate(loader):
            x_a, x_b, scale_a, scale_b = self._prepare_batch(batch)
            x_a_noisy, x_b_noisy = self._apply_augmentations(x_a, x_b)

            sigma_a_est = self._estimate_noise_for_tensor(x_a_noisy)
            sigma_b_est = self._estimate_noise_for_tensor(x_b_noisy)

            enc_a = self._run_encoder(x_a_noisy, sigma_a_est, return_full=True)
            enc_b = self._run_encoder(x_b_noisy, sigma_b_est, return_full=True)
            z_a, skips_a, identity_a = self._split_encoder_output(enc_a)
            z_b, skips_b, identity_b = self._split_encoder_output(enc_b)

            x_a_recon = self._run_decoder(self.decoder_a, z_a, skips=skips_a, identity=identity_a)
            x_b_recon = self._run_decoder(self.decoder_b, z_b, skips=skips_b, identity=identity_b)

            # Cross-domain reconstruction with configurable skip strategy
            cross_skips_for_a, cross_identity_for_a = self._prepare_cross_skips(
                skips_a, skips_b, identity_a, identity_b, target_domain="a"
            )
            cross_skips_for_b, cross_identity_for_b = self._prepare_cross_skips(
                skips_b, skips_a, identity_b, identity_a, target_domain="b"
            )
            x_a_from_b = self._run_decoder(self.decoder_a, z_b, skips=cross_skips_for_a, identity=cross_identity_for_a)
            x_b_from_a = self._run_decoder(self.decoder_b, z_a, skips=cross_skips_for_b, identity=cross_identity_for_b)

            # Rescale cross-domain predictions and targets to their original magnitudes.
            x_a_target_scaled = x_a * scale_a
            x_b_target_scaled = x_b * scale_b
            x_a_from_b_scaled = x_a_from_b * scale_a
            x_b_from_a_scaled = x_b_from_a * scale_b

            l_content = compute_l1_loss(z_a, z_b)
            l_recon = compute_l1_loss(x_a_recon, x_a) + compute_l1_loss(x_b_recon, x_b)
            l_cross = compute_l1_loss(x_a_from_b_scaled, x_a_target_scaled) + compute_l1_loss(
                x_b_from_a_scaled, x_b_target_scaled
            )

            if self.edge_cfg.get("enabled", False):
                delta = float(self.edge_cfg.get("weight", self.state.loss_weights["edge"]))
                edges_a = sobel_edges(x_a_from_b_scaled)
                edges_a_gt = sobel_edges(x_a_target_scaled)
                edges_b = sobel_edges(x_b_from_a_scaled)
                edges_b_gt = sobel_edges(x_b_target_scaled)
                l_edge = compute_l1_loss(edges_a, edges_a_gt) + compute_l1_loss(
                    edges_b, edges_b_gt
                )
            else:
                delta = 0.0
                l_edge = torch.tensor(0.0, device=self.device)

            total_loss = (
                self.state.loss_weights["content"] * l_content
                + self.state.loss_weights["recon"] * l_recon
                + self.state.loss_weights["cross"] * l_cross
                + delta * l_edge
            )

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters())
                    + list(self.decoder_a.parameters())
                    + list(self.decoder_b.parameters()),
                    max_norm=float(self.grad_clip),
                )
            self.optimizer.step()
            if self.scheduler and self.scheduler_step == "step":
                self.scheduler.step()

            self.state.global_step += 1
            totals["total"] += float(total_loss.item())
            totals["content"] += float(l_content.item())
            totals["recon"] += float(l_recon.item())
            totals["cross"] += float(l_cross.item())
            totals["edge"] += float(l_edge.item())
            totals["count"] += 1

            latent_var = (
                0.5 * (float(z_a.var(unbiased=False).item()) + float(z_b.var(unbiased=False).item()))
            )
            cross_var = (
                0.5
                * (
                    float(x_a_from_b_scaled.var(unbiased=False).item())
                    + float(x_b_from_a_scaled.var(unbiased=False).item())
                )
            )
            latent_vars.append(latent_var)
            cross_vars.append(cross_var)
            log_extra: Dict[str, float] = {}
            if sigma_a_est is not None:
                mean_a = float(sigma_a_est.mean().item())
                noise_est_a.append(mean_a)
                log_extra["noise/est_a"] = mean_a
            if sigma_b_est is not None:
                mean_b = float(sigma_b_est.mean().item())
                noise_est_b.append(mean_b)
                log_extra["noise/est_b"] = mean_b

            if self.wandb_enabled and self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "loss/total": total_loss.item(),
                        "loss/content": l_content.item(),
                        "loss/recon": l_recon.item(),
                        "loss/cross": l_cross.item(),
                        "loss/edge": l_edge.item(),
                        "variance/latent": latent_var,
                        "variance/cross": cross_var,
                        "weights/alpha": self.state.loss_weights["content"],
                        "weights/beta": self.state.loss_weights["recon"],
                        "weights/gamma": self.state.loss_weights["cross"],
                        "weights/delta": delta,
                        **log_extra,
                    },
                    step=self.state.global_step,
                )

            if (
                self.wandb_enabled
                and self.wandb_run is not None
                and log_interval > 0
                and self.state.global_step % log_interval == 0
            ):
                cross_abs_a = (x_a_from_b_scaled - x_a_target_scaled).abs()
                cross_abs_b = (x_b_from_a_scaled - x_b_target_scaled).abs()
                self._log_wandb_images(
                    x_a_noisy,
                    x_a_recon,
                    x_b_noisy,
                    x_b_recon,
                    cross_abs_a=cross_abs_a,
                    cross_abs_b=cross_abs_b,
                    step=self.state.global_step,
                )

        count = max(1, totals["count"])
        history = TrainerHistory(
            epoch=epoch,
            total_loss=totals["total"] / count,
            content_loss=totals["content"] / count,
            recon_loss=totals["recon"] / count,
            cross_loss=totals["cross"] / count,
            edge_loss=totals["edge"] / count,
            latent_variance=_safe_fmean(latent_vars),
            cross_variance=_safe_fmean(cross_vars),
            collapsed=False,
            noise_a=_safe_fmean(noise_est_a),
            noise_b=_safe_fmean(noise_est_b),
        )
        return history

    def _log_wandb_images(
        self,
        x_a_noisy: torch.Tensor,
        x_a_recon: torch.Tensor,
        x_b_noisy: torch.Tensor,
        x_b_recon: torch.Tensor,
        step: int,
        cross_abs_a: Optional[torch.Tensor] = None,
        cross_abs_b: Optional[torch.Tensor] = None,
    ) -> None:
        visuals = [
            prepare_image_for_logging(x_a_noisy),
            prepare_image_for_logging(x_a_recon),
            prepare_image_for_logging(x_b_noisy),
            prepare_image_for_logging(x_b_recon),
        ]
        if make_grid is not None:
            grid_tensor = make_grid(torch.stack(visuals, dim=0), nrow=4)
        else:
            top_row = torch.cat(visuals[:2], dim=2)
            bottom_row = torch.cat(visuals[2:], dim=2)
            grid_tensor = torch.cat([top_row, bottom_row], dim=1)
        grid_np = (grid_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        caption = "A noisy | A denoised | B noisy | B denoised"
        payload: Dict[str, Any] = {
            "visuals/train": wandb.Image(grid_np, caption=caption)
        }
        if cross_abs_a is not None:
            payload["visuals/cross_abs_a"] = wandb.Image(
                prepare_image_for_logging(cross_abs_a),
                caption="|A_from_B - A|",
            )
        if cross_abs_b is not None:
            payload["visuals/cross_abs_b"] = wandb.Image(
                prepare_image_for_logging(cross_abs_b),
                caption="|B_from_A - B|",
            )
        self.wandb_run.log(payload, step=step)

    def _maybe_adjust_loss_weights(self, history: TrainerHistory) -> bool:
        threshold = float(self.anti_collapse_cfg.get("variance_threshold", 1e-5))
        patience = int(self.anti_collapse_cfg.get("patience", 1))
        adjust = float(self.anti_collapse_cfg.get("adjustment_factor", 0.15))
        max_alpha = float(self.anti_collapse_cfg.get("max_alpha", 10.0))
        min_beta = float(self.anti_collapse_cfg.get("min_beta", 1e-4))

        collapsed = (
            history.latent_variance < threshold or history.cross_variance < threshold
        )
        if collapsed:
            self.state.collapse_counter += 1
        else:
            self.state.collapse_counter = 0

        if collapsed and self.state.collapse_counter >= patience:
            self.state.loss_weights["content"] = min(
                self.state.loss_weights["content"] * (1.0 + adjust), max_alpha
            )
            self.state.loss_weights["recon"] = max(
                self.state.loss_weights["recon"] * (1.0 - adjust), min_beta
            )
            self.state.collapse_counter = 0
            if self.wandb_enabled and self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "weights/alpha": self.state.loss_weights["content"],
                        "weights/beta": self.state.loss_weights["recon"],
                    },
                    step=self.state.global_step,
                )
        return collapsed

    def _prepare_cross_skips(
        self,
        target_skips: Optional[List[torch.Tensor]],
        source_skips: Optional[List[torch.Tensor]],
        target_identity: Optional[torch.Tensor],
        source_identity: Optional[torch.Tensor],
        target_domain: str,
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Prepare skip connections and identity for cross-domain reconstruction.

        Strategies:
        - "use_source_skip": Use skips from source domain (original behavior, problematic)
        - "use_target_skip": Use skips from target domain (matches ground truth structure)
        - "no_skip": Don't use any skips (forces latent to be self-sufficient)
        - "zero_skip": Use zero-valued skips (maintains architecture consistency)
        - "mixed_skip": Weighted mix of source and target skips
        """
        strategy = self.cross_domain_strategy

        if strategy == "use_source_skip":
            # Original behavior: use skips from source domain
            return source_skips, None

        elif strategy == "use_target_skip":
            # Use skips from target domain (better structural alignment)
            return target_skips, target_identity

        elif strategy == "no_skip":
            # No skips at all - latent must encode everything
            return None, None

        elif strategy == "zero_skip":
            # Zero-valued skips to maintain architecture
            if source_skips is not None:
                zero_skips = [torch.zeros_like(s) for s in source_skips]
                return zero_skips, None
            return None, None

        elif strategy == "mixed_skip":
            # Weighted combination of both domains
            if target_skips is not None and source_skips is not None:
                alpha = self.mixed_skip_alpha
                mixed_skips = [
                    alpha * t_skip + (1.0 - alpha) * s_skip
                    for t_skip, s_skip in zip(target_skips, source_skips)
                ]
                return mixed_skips, None
            elif target_skips is not None:
                return target_skips, None
            elif source_skips is not None:
                return source_skips, None
            return None, None

        else:
            raise ValueError(f"Unknown cross_domain_strategy: {strategy}")

    def _estimate_noise_for_tensor(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.noise_estimation_cfg.get("enabled", False):
            return None
        method = self.noise_estimation_cfg.get("method", "auto")
        try:
            return estimate_noise_level(
                tensor,
                method=method,
                representation=self.input_representation,
            )
        except Exception as err:  # pragma: no cover - warn once on failure
            if not self._noise_warning_emitted:
                warnings.warn(f"Noise estimation failed with method '{method}': {err}")
                self._noise_warning_emitted = True
            return None

    def _run_encoder(
        self,
        tensor: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        return_full: bool = False,
    ) -> Any:
        if sigma is not None:
            for attempt in (
                lambda: self.encoder(tensor, sigma=sigma),
                lambda: self.encoder(tensor, sigma),
            ):
                try:
                    output = attempt()
                    break
                except TypeError:
                    output = None
            else:  # pragma: no cover - fallback if both fail
                output = self.encoder(tensor)
        else:
            output = self.encoder(tensor)

        if return_full:
            return output

        if isinstance(output, (list, tuple)):
            idx = self.encoder_output_index
            if -len(output) <= idx < len(output):
                candidate = output[idx]
                if isinstance(candidate, torch.Tensor):
                    return candidate
            for element in reversed(output):
                if isinstance(element, torch.Tensor):
                    return element
            raise TypeError("Encoder returned tuple/list without tensor elements.")
        if isinstance(output, dict):
            for key in ("latent", "z", "embedding", "output"):
                if key in output:
                    return output[key]
            return next(iter(output.values()))
        if isinstance(output, torch.Tensor):
            return output
        raise TypeError(f"Unsupported encoder output type: {type(output)!r}")

    def _split_encoder_output(
        self, output: Any
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        skips: Optional[List[torch.Tensor]] = None
        identity: Optional[torch.Tensor] = None

        def _collect_skips(candidate: Any) -> Optional[List[torch.Tensor]]:
            if candidate is None:
                return None
            if isinstance(candidate, torch.Tensor):
                return [candidate]
            if isinstance(candidate, (list, tuple)):
                tensors = [item for item in candidate if isinstance(item, torch.Tensor)]
                return tensors or None
            return None

        def _select_tensor(values: Iterable[Any]) -> Optional[torch.Tensor]:
            for element in reversed(list(values)):
                if isinstance(element, torch.Tensor):
                    return element
            return None

        def _extract_identity(container: Dict[str, Any]) -> Optional[torch.Tensor]:
            for key in ("identity", "input", "residual", "skip_identity"):
                value = container.get(key)
                if isinstance(value, torch.Tensor):
                    return value
            return None

        if isinstance(output, dict):
            skips = _collect_skips(output.get("skips"))
            latent = output.get("latent")
            identity = _extract_identity(output)
            if latent is None:
                latent = _select_tensor(output.values())
            if latent is None:
                raise TypeError("Encoder dict output did not contain a tensor latent.")
            return latent, skips, identity

        if isinstance(output, (list, tuple)):
            for element in output:
                if isinstance(element, dict) and "skips" in element:
                    skips = _collect_skips(element.get("skips"))
                    if identity is None:
                        identity = _extract_identity(element)
                    break
                if isinstance(element, (list, tuple)):
                    candidate = _collect_skips(element)
                    if candidate:
                        skips = candidate
                        break
            latent = _select_tensor(output)
            if latent is None:
                raise TypeError("Encoder tuple/list output did not contain tensors.")
            return latent, skips, identity

        if isinstance(output, torch.Tensor):
            return output, None, None

        raise TypeError(f"Unsupported encoder output type: {type(output)!r}")

    def _run_decoder(
        self,
        decoder: nn.Module,
        latent: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
        identity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if skips:
            for attempt in (
                lambda: decoder(latent, skips=skips, identity=identity),  # type: ignore[arg-type]
                lambda: decoder(latent, skips, identity),  # positional fall-back
                lambda: decoder(latent, skips=skips),
                lambda: decoder(latent, skips),
            ):
                try:
                    return attempt()
                except TypeError:
                    continue
        for attempt in (
            lambda: decoder(latent, identity=identity),
            lambda: decoder(latent),
        ):
            try:
                return attempt()
            except TypeError:
                continue
        raise TypeError("Decoder invocation failed with provided arguments.")
