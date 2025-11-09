from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dtype=torch.float32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, dtype=dtype)
        self.bn1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, dtype=dtype)
        self.bn2 = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if torch.is_complex(x):
            x = torch.complex(self.bn1(x.real), self.bn1(x.imag))
            x = torch.complex(F.relu(x.real), F.relu(x.imag))
        else:
            x = self.bn1(x)
            x = F.relu(x)

        x = self.conv2(x)
        if torch.is_complex(x):
            x = torch.complex(self.bn2(x.real), self.bn2(x.imag))
            x = torch.complex(F.relu(x.real), F.relu(x.imag))
        else:
            x = self.bn2(x)
            x = F.relu(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dtype=torch.float32):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels, dtype=dtype)
        )

    def forward(self, x):
        if torch.is_complex(x):
            x = torch.complex(F.max_pool2d(x.real, 2), F.max_pool2d(x.imag, 2))
        else:
            x = F.max_pool2d(x, 2)
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dtype=torch.float32):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, dtype=dtype)
        self.conv = DoubleConv(in_channels, out_channels, dtype=dtype)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        else:
            x2 = torch.zeros_like(x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ComplexUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], complex_input=True):
        super().__init__()
        self.complex_input = complex_input
        dtype = torch.complex64 if complex_input else torch.float32

        self.inc = DoubleConv(in_channels, features[0], dtype=dtype)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(len(features) - 1):
            self.downs.append(Down(features[i], features[i + 1], dtype=dtype))

        for i in range(len(features) - 1, 0, -1):
            self.ups.append(Up(features[i], features[i - 1], dtype=dtype))

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1, padding=0, dtype=dtype)

    def forward(self, x, sigma=None):
        skips = []
        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        skips = skips[:-1]

        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)

        x = self.outc(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=256, features=[64, 128, 256], complex_input=True):
        super().__init__()
        self.complex_input = complex_input
        dtype = torch.complex64 if complex_input else torch.float32

        self.inc = DoubleConv(in_channels, features[0], dtype=dtype)
        self.downs = nn.ModuleList()

        for i in range(len(features) - 1):
            self.downs.append(Down(features[i], features[i + 1], dtype=dtype))

        self.bottleneck = nn.Conv2d(features[-1], latent_channels, kernel_size=1, padding=0, dtype=dtype)

    def forward(self, x, sigma=None):
        identity = x
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        latent = self.bottleneck(x)

        if self.complex_input and torch.is_complex(latent):
            latent = torch.cat([latent.real, latent.imag], dim=1)

        return {"latent": latent, "skips": skips[:-1], "identity": identity}


class ComplexUpsample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if torch.is_complex(x):
            real = F.interpolate(x.real, scale_factor=self.scale_factor, mode='nearest')
            imag = F.interpolate(x.imag, scale_factor=self.scale_factor, mode='nearest')
            return torch.complex(real, imag)
        else:
            return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class UNetDecoder(nn.Module):
    def __init__(
        self,
        latent_channels=256,
        out_channels=1,
        features=[256, 128, 64],
        complex_output=True,
        identity_mapping=False,
    ):
        super().__init__()
        self.complex_output = complex_output
        self.identity_mapping = identity_mapping
        dtype = torch.complex64 if complex_output else torch.float32

        self.expand = nn.Conv2d(latent_channels, features[0], kernel_size=1, padding=0, dtype=dtype)

        self.ups = nn.ModuleList()
        for i in range(len(features) - 1):
            self.ups.append(Up(features[i], features[i + 1], dtype=dtype))

        self.outc = nn.Conv2d(features[-1], out_channels, kernel_size=1, padding=0, dtype=dtype)

        if self.identity_mapping:
            self._zero_module_parameters(self.expand)
            self._zero_module_parameters(self.outc)
            for up in self.ups:
                self._zero_module_parameters(up)

    @staticmethod
    def _zero_module_parameters(module: nn.Module) -> None:
        for param in module.parameters():
            if param is not None:
                nn.init.zeros_(param)

    def forward(self, z, skips: Optional[List[torch.Tensor]] = None, identity: Optional[torch.Tensor] = None):
        if self.complex_output and not torch.is_complex(z):
            mid = z.shape[1] // 2
            z = torch.complex(z[:, :mid], z[:, mid:])

        x = self.expand(z)
        skip_stack: List[torch.Tensor] = []
        identity_source: Optional[torch.Tensor] = None
        if skips is not None:
            if isinstance(skips, torch.Tensor):
                skip_stack = [skips]
            else:
                skip_stack = list(skips)
            if skip_stack:
                identity_source = skip_stack[0]

        for up in self.ups:
            skip = skip_stack.pop() if skip_stack else None
            x = up(x, skip)
        x = self.outc(x)

        if self.identity_mapping and identity is not None:
            identity = identity.to(dtype=x.dtype)
            x = x + identity
        elif (
            self.identity_mapping
            and identity_source is not None
            and x.shape[1] == identity_source.shape[1]
        ):
            identity_source = identity_source.to(dtype=x.dtype)
            x = x + identity_source
        return x
