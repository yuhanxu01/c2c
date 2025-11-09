import argparse
import importlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from data_loader import build_c2c_dataloaders
from model import Contrast2ContrastTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrast2Contrast Trainer")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to configuration JSON file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (falls back to config).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs"),
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases even if enabled in config.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_channels(sample: torch.Tensor) -> Tuple[int, bool]:
    if torch.is_complex(sample):
        return sample.shape[1], True
    return sample.shape[1], False


def to_real_channels(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return torch.cat([x.real, x.imag], dim=1)
    return x


class SimpleSharedEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        complex_input: bool,
        depth: int = 3,
        hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden = hidden_channels or latent_channels
        effective_in = in_channels * 2 if complex_input else in_channels
        layers = []
        current = effective_in
        for _ in range(depth - 1):
            layers.append(nn.Conv2d(current, hidden, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(1, hidden))
            layers.append(nn.GELU())
            current = hidden
        layers.append(nn.Conv2d(current, latent_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        self.complex_input = complex_input

    def forward(self, x: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        _ = sigma
        if self.complex_input:
            features = to_real_channels(x)
        else:
            features = x.real if torch.is_complex(x) else x
        return self.net(features.float())


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        complex_output: bool,
        depth: int = 3,
        hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden = hidden_channels or latent_channels
        target_channels = out_channels * 2 if complex_output else out_channels
        layers = []
        current = latent_channels
        for _ in range(depth - 1):
            layers.append(nn.Conv2d(current, hidden, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(1, hidden))
            layers.append(nn.GELU())
            current = hidden
        layers.append(nn.Conv2d(current, target_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        self.complex_output = complex_output

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        recon = self.net(z)
        if self.complex_output:
            real, imag = torch.chunk(recon, 2, dim=1)
            return torch.complex(real, imag)
        return recon


from model.unet import UNetEncoder, UNetDecoder

BUILTIN_COMPONENTS = {
    "SharedEncoder": SimpleSharedEncoder,
    "DomainADecoder": SimpleDecoder,
    "DomainBDecoder": SimpleDecoder,
    "UNetEncoder": UNetEncoder,
    "UNetADecoder": UNetDecoder,
    "UNetBDecoder": UNetDecoder,
}


def build_component(
    spec: Dict[str, Any],
    default_kwargs: Dict[str, Any],
) -> nn.Module:
    spec = dict(spec)
    cls_name = spec.get("class")
    kwargs = dict(spec.get("kwargs", {}))
    kwargs = {**default_kwargs, **kwargs}
    module_path = spec.get("module")

    if cls_name is None:
        raise ValueError("Model specification must include 'class'.")

    if module_path:
        module = importlib.import_module(module_path)
        cls = getattr(module, cls_name)
    else:
        cls = BUILTIN_COMPONENTS.get(cls_name)
        if cls is None:
            raise ValueError(
                f"Unknown component '{cls_name}'. Provide a 'module' field or use one of {list(BUILTIN_COMPONENTS)}."
            )
    model = cls(**kwargs)
    checkpoint = spec.get("checkpoint")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        key = spec.get("state_key")
        if key:
            state = state[key]
        model.load_state_dict(state, strict=spec.get("strict", True))
    return model


def build_models(
    cfg: Dict[str, Any],
    sample: Dict[str, torch.Tensor],
    trainer_cfg: Dict[str, Any],
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    key_a = trainer_cfg["input_keys"]["domain_a"]
    x_a = sample[key_a]
    channels, is_complex = infer_channels(x_a)
    cdlnet_cfg = cfg.get("cdlnet", {})
    latent_channels = int(cdlnet_cfg.get("M", 64))
    builtin_defaults = {
        "encoder": {
            "in_channels": channels,
            "latent_channels": latent_channels,
            "complex_input": is_complex,
        },
        "decoder_a": {
            "latent_channels": latent_channels,
            "out_channels": channels,
            "complex_output": is_complex,
        },
        "decoder_b": {
            "latent_channels": latent_channels,
            "out_channels": channels,
            "complex_output": is_complex,
        },
    }

    encoder = build_component(cfg["encoder"], builtin_defaults["encoder"])
    decoder_a = build_component(cfg["decoder_a"], builtin_defaults["decoder_a"])
    decoder_b = build_component(cfg["decoder_b"], builtin_defaults["decoder_b"])
    return encoder, decoder_a, decoder_b


def build_optimizer(
    name: str,
    params: Iterable[nn.Parameter],
    cfg: Dict[str, Any],
) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg["learning_rate"],
            weight_decay=cfg.get("weight_decay", 0.0),
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=cfg["learning_rate"],
            weight_decay=cfg.get("weight_decay", 0.0),
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg["learning_rate"],
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 0.0),
            nesterov=cfg.get("nesterov", False),
        )
    raise ValueError(f"Unsupported optimizer type: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Optional[Dict[str, Any]],
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if not cfg:
        return None
    name = cfg.get("type", "").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["t_max"],
            eta_min=cfg.get("eta_min", 0.0),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["step_size"],
            gamma=cfg.get("gamma", 0.1),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.get("mode", "min"),
            factor=cfg.get("factor", 0.1),
            patience=cfg.get("patience", 5),
            min_lr=cfg.get("min_lr", 0.0),
        )
    raise ValueError(f"Unsupported scheduler type: {name}")


def maybe_init_wandb(cfg: Dict[str, Any]):
    logging.info("Initializing Weights & Biases.")
    import wandb  # local import to keep optional dependency

    return wandb.init(
        project=cfg["project"],
        entity=cfg.get("entity"),
        config=cfg,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    cfg = load_json_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    logging.info("Loaded config from %s", args.config)

    set_seed(int(cfg["seed"]))
    device = torch.device(cfg["device"])

    dataloaders = build_c2c_dataloaders(cfg["data"])
    train_loader = dataloaders["train"]
    val_loader = dataloaders.get("val")

    sample_batch = next(iter(train_loader))
    encoder, decoder_a, decoder_b = build_models(cfg["model"], sample_batch, cfg["trainer"])
    del sample_batch
    encoder.to(device)
    decoder_a.to(device)
    decoder_b.to(device)

    params = list(encoder.parameters()) + list(decoder_a.parameters()) + list(decoder_b.parameters())
    optimizer = build_optimizer(cfg["optimizer"]["type"], params, cfg["optimizer"])
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"))

    wandb_cfg = cfg["trainer"].get("logging", {})
    run = None
    if wandb_cfg.get("use_wandb", False) and not args.no_wandb:
        try:
            run = maybe_init_wandb(wandb_cfg)
        except ImportError:
            logging.warning("wandb is not installed; continuing without logging.")

    cfg["trainer"].setdefault("input_representation", cfg.get("data", {}).get("representation"))

    trainer = Contrast2ContrastTrainer(
        encoder=encoder,
        decoder_a=decoder_a,
        decoder_b=decoder_b,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=cfg["trainer"],
        wandb_run=run,
    )

    num_epochs = args.epochs or cfg["trainer"].get("epochs", 1)
    logging.info("Starting training for %d epochs on %s", num_epochs, device)

    run_dir = args.run_dir
    ensure_dir(run_dir)
    best_loss = math.inf
    checkpoint_path = run_dir / "best.pt"

    def checkpoint_fn(epoch: int, history) -> None:
        nonlocal best_loss
        if history.total_loss < best_loss:
            best_loss = history.total_loss
            payload = {
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder_a": decoder_a.state_dict(),
                "decoder_b": decoder_b.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(payload, checkpoint_path)
            logging.info("Saved new best checkpoint to %s (loss=%.4f)", checkpoint_path, best_loss)

    trainer.fit(train_loader, num_epochs=num_epochs, val_loader=val_loader, checkpoint_fn=checkpoint_fn)
    logging.info("Training complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
