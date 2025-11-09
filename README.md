# Contrast2Contrast

Paired-contrast MRI denoising and translation with a shared encoder and domain-specific decoders (Method #3). The framework enforces a common latent representation while reconstructing each domain and performing cross-domain Noise2Noise supervision to preserve structural content.

## Project Layout
- `data_loader.py` - paired PD/PDFS dataset builder with optional augmentations.
- `model/net.py` - contrastive dictionary learning network components.
- `model/trainer.py` - configurable training loop with anti-collapse heuristics, optional high-frequency loss, and Weights & Biases logging.
- `model/utils.py` - shared tensor utilities used across training and reconstruction.
- `config.json` - starter configuration covering data, optimisation, and trainer behaviour.

## Configuration
Adapt `config.json` before training:
- **data** - update `root`, `split`, and batch settings to match your dataset folders.
- **model** - tune `cdlnet` parameters (`K`, `M`, `P`, `s`, `C`, `adaptive`, `t0`) and set class names or checkpoints for your encoder/decoders.
- **optimizer/scheduler** - tweak learning rate policy as needed.
- **trainer** - adjust loss weights (`alpha`, `beta`, `gamma`, `delta` equivalents), noise injection, brightness/contrast jitter, Sobel edge loss, noise estimation method (`auto` picks `wvlt` for magnitude and `wvlt_c` for RI/complex), encoder output index (`encoder_output_index` selects which tuple element represents the latent code), and anti-collapse thresholds. `input_representation` should mirror the dataset representation string; `input_keys` must match `data_loader` outputs (`noisy_pd`, `noisy_pdfs`).
- **logging** - toggle `use_wandb`, set project/entity, and control the image logging interval (default every 200 steps).

## Training Example
Run the CLI helper for a quick start:

```bash
python train.py --config config.json --epochs 5
```

```python
import json
import torch
import wandb

from data_loader import build_c2c_dataloaders
from model import Contrast2ContrastTrainer
from model.encoder import SharedEncoder  # replace with your implementation
from model.decoder import DomainADecoder, DomainBDecoder  # replace with your implementation
from model.net import CDLNet_C  # optional: instantiate from cfg["model"]["cdlnet"]

with open("config.json") as f:
    cfg = json.load(f)

torch.manual_seed(cfg["seed"])
dataloaders = build_c2c_dataloaders(cfg["data"])

encoder = SharedEncoder(**cfg["model"]["encoder"].get("kwargs", {}))
decoder_a = DomainADecoder(**cfg["model"]["decoder_a"].get("kwargs", {}))
decoder_b = DomainBDecoder(**cfg["model"]["decoder_b"].get("kwargs", {}))
cdlnet = CDLNet_C(**cfg["model"]["cdlnet"])  # example usage if Method #3 uses CDLNet components

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder_a.parameters()) + list(decoder_b.parameters()),
    lr=cfg["optimizer"]["learning_rate"],
    weight_decay=cfg["optimizer"]["weight_decay"],
    betas=tuple(cfg["optimizer"]["betas"]),
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg["scheduler"]["t_max"], eta_min=cfg["scheduler"]["eta_min"]
)

run = wandb.init(project=cfg["trainer"]["logging"]["project"], config=cfg) if cfg["trainer"]["logging"]["use_wandb"] else None
trainer = Contrast2ContrastTrainer(
    encoder=encoder,
    decoder_a=decoder_a,
    decoder_b=decoder_b,
    optimizer=optimizer,
    scheduler=scheduler,
    device=torch.device(cfg["device"]),
    config=cfg["trainer"],
    wandb_run=run,
)

trainer.fit(dataloaders["train"], num_epochs=50, val_loader=dataloaders.get("val"))
```

If collapse is detected (vanishing latent variance), the trainer automatically increases the content weight and relaxes reconstruction pressure. For additional regularisation, raise the noise `sigma` or enable stronger jitter in the config.

## Notes
- Ensure complex tensors remain in `torch.complex64` to match the dataset representation.
- WandB image grids render `[xA, xA_from_B, xB, xB_from_A]` every `image_interval` steps.
- Before launching long runs, dry-test with a small subset to validate configuration and logging.
