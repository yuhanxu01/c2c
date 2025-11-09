import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.solvers import power_method, uball_project
from model.utils   import pre_process, post_process, calc_pad_2D, unpad, pre_process_3d, post_process_3d

def ST_complex(x, t, eps=1e-16):
    """Complex-valued soft thresholding."""
    return x * F.relu(torch.abs(x) - t) / (torch.abs(x) + eps)

def ST(x, t):
    """Real-valued shrinkage-thresholding operation."""
    return x.sign() * F.relu(x.abs() - t)

class CDLNet_C(nn.Module):
    def __init__(self, K=3, M=64, P=7, s=1, C=1, t0=0, adaptive=False, init=True):
        super(CDLNet_C, self).__init__()
        self.A = nn.ModuleList([
            nn.Conv2d(C, M, P, stride=s, padding=(P-1)//2, 
                      bias=False, dtype=torch.complex64)  
            for _ in range(K)
        ])
        self.B = nn.ModuleList([
            nn.ConvTranspose2d(M, C, P, stride=s, 
                               padding=(P-1)//2, output_padding=s-1, 
                               bias=False, dtype=torch.complex64)
            for _ in range(K)
        ])
        self.D = self.B[0]
        self.t = nn.Parameter(t0*torch.ones(K,2,M,1,1))
        W = torch.randn(M, C, P, P, dtype=torch.complex64)
        for k in range(K):
            self.A[k].weight.data = W.clone()
            self.B[k].weight.data = W.clone().conj() 

        if init:
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0](x))
                L = power_method(DDt, 
                                 torch.rand(1,C,128,128, dtype=torch.complex64), 
                                 num_iter=200, 
                                 verbose=False)[0]
            for k in range(K):
                self.A[k].weight.data /= np.sqrt(L)
                self.B[k].weight.data /= np.sqrt(L)

        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.adaptive = adaptive

    @torch.no_grad()
    def project(self):
        self.t.clamp_(0.0) 
        for k in range(self.K):
            self.A[k].weight.data = uball_project(self.A[k].weight.data)
            self.B[k].weight.data = uball_project(self.B[k].weight.data)

    def _lista_core(self, yp, mask, sigma):
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = ST_complex(self.A[0](yp), self.t[0,:1] + c*self.t[0,1:2])
        for k in range(1, self.K):
            z = z - self.A[k](mask * self.B[k](z) - yp)
            z = ST_complex(z, self.t[k,:1] + c*self.t[k,1:2])
        return z

    def encode(self, y, sigma=None, mask=1):
        yp, params, mask = pre_process(y, self.s, mask=mask)
        z = self._lista_core(yp, mask, sigma)
        return z, params, mask

    def decode(self, z, params):
        xphat = self.D(z)
        return post_process(xphat, params)

    def forward(self, y, sigma=None, mask=1):
        z, params, _ = self.encode(y, sigma=sigma, mask=mask)
        xhat = self.decode(z, params)
        return xhat, z


class CDLNet(nn.Module):
    """Enhanced CDLNet with support for multiple representations.

    Supports:
    - complex: Complex-valued data (default)
    - magnitude: Magnitude-only data
    - ri: Real-Imaginary stacked as 2-channel data
    """
    def __init__(self,
                 K=3,
                 M=64,
                 P=7,
                 s=1,
                 C=1,
                 t0=0,
                 adaptive=False,
                 init=True,
                 representation='complex'):
        super(CDLNet, self).__init__()

        self.representation = representation
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.adaptive = adaptive

        # Determine input channels based on representation
        if representation == 'ri':
            input_channels = C * 2  # Real and Imaginary as separate channels
            self.dtype = torch.float32
            self.use_complex = False
        elif representation == 'magnitude':
            input_channels = C
            self.dtype = torch.float32
            self.use_complex = False
        elif representation == 'complex':
            input_channels = C
            self.dtype = torch.complex64
            self.use_complex = True
        else:
            raise ValueError(f"Unknown representation: {representation}")

        # Initialize operators based on representation type
        if self.use_complex:
            self.A = nn.ModuleList([
                nn.Conv2d(input_channels, M, P, stride=s,
                         padding=(P-1)//2, bias=False, dtype=self.dtype)
                for _ in range(K)
            ])
            self.B = nn.ModuleList([
                nn.ConvTranspose2d(M, input_channels, P, stride=s,
                                  padding=(P-1)//2, output_padding=s-1,
                                  bias=False, dtype=self.dtype)
                for _ in range(K)
            ])
            W = torch.randn(M, input_channels, P, P, dtype=self.dtype)
            for k in range(K):
                self.A[k].weight.data = W.clone()
                self.B[k].weight.data = W.clone().conj()
        else:
            self.A = nn.ModuleList([
                nn.Conv2d(input_channels, M, P, stride=s,
                         padding=(P-1)//2, bias=False)
                for _ in range(K)
            ])
            self.B = nn.ModuleList([
                nn.ConvTranspose2d(M, input_channels, P, stride=s,
                                  padding=(P-1)//2, output_padding=s-1,
                                  bias=False)
                for _ in range(K)
            ])
            W = torch.randn(M, input_channels, P, P)
            for k in range(K):
                self.A[k].weight.data = W.clone()
                self.B[k].weight.data = W.clone()

        self.D = self.B[0]
        self.t = nn.Parameter(t0 * torch.ones(K, 2, M, 1, 1))

        # Power method initialization
        if init:
            print(f"Running power-method for {representation} representation...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0](x))
                if self.use_complex:
                    test_input = torch.rand(1, input_channels, 128, 128,
                                          dtype=self.dtype)
                else:
                    test_input = torch.rand(1, input_channels, 128, 128)

                L = power_method(DDt, test_input, num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}")

                if L < 0:
                    print("STOP: negative spectral norm detected")
                    sys.exit()

                # Spectral normalization
                for k in range(K):
                    self.A[k].weight.data /= np.sqrt(L)
                    self.B[k].weight.data /= np.sqrt(L)

    @torch.no_grad()
    def project(self):
        """Project thresholds and filters to their constraint sets."""
        self.t.clamp_(0.0)
        for k in range(self.K):
            self.A[k].weight.data = uball_project(self.A[k].weight.data)
            self.B[k].weight.data = uball_project(self.B[k].weight.data)

    def _convert_to_internal(self, y):
        """Convert input to internal representation."""
        if self.representation == 'complex':
            return y  # Already in complex format
        elif self.representation == 'magnitude':
            return y.abs() if y.is_complex() else y
        elif self.representation == 'ri':
            if y.is_complex():
                # Stack real and imaginary as channels
                return torch.cat([y.real, y.imag], dim=1)
            else:
                return y
        return y

    def _convert_from_internal(self, x):
        """Convert internal representation back to output format."""
        if self.representation == 'complex':
            return x
        elif self.representation == 'magnitude':
            return x
        elif self.representation == 'ri':
            # Separate channels back into real and imaginary
            if x.shape[1] == 2:
                return torch.complex(x[:, 0:1], x[:, 1:2])
            else:
                mid = x.shape[1] // 2
                return torch.complex(x[:, :mid], x[:, mid:])
        return x

    def _apply_thresholding(self, z, threshold):
        """Apply appropriate thresholding based on representation."""
        if self.use_complex:
            return ST_complex(z, threshold)
        else:
            return ST(z, threshold)

    def _lista_core(self, yp, mask, sigma):
        """Core LISTA iteration."""
        c = 0 if sigma is None or not self.adaptive else sigma / 255.0

        # First iteration
        z = self._apply_thresholding(
            self.A[0](yp),
            self.t[0, :1] + c * self.t[0, 1:2]
        )

        # Subsequent iterations
        for k in range(1, self.K):
            residual = mask * self.B[k](z) - yp
            z = z - self.A[k](residual)
            z = self._apply_thresholding(
                z,
                self.t[k, :1] + c * self.t[k, 1:2]
            )

        return z

    def encode(self, y, sigma=None, mask=1):
        """Encode input to sparse representation."""
        y_internal = self._convert_to_internal(y)
        yp, params, mask = pre_process(y_internal, self.s, mask=mask)
        z = self._lista_core(yp, mask, sigma)
        return z, params, mask

    def decode(self, z, params):
        """Decode sparse representation to output."""
        xphat = self.D(z)
        xhat_internal = post_process(xphat, params)
        return self._convert_from_internal(xhat_internal)

    def forward(self, y, sigma=None, mask=1):
        """Forward pass with encoding and decoding."""
        z, params, _ = self.encode(y, sigma=sigma, mask=mask)
        xhat = self.decode(z, params)
        return xhat, z

    def forward_generator(self, y, sigma=None, mask=1):
        """Generator version that yields intermediate sparse codes."""
        y_internal = self._convert_to_internal(y)
        yp, params, mask = pre_process(y_internal, self.s, mask=mask)

        c = 0 if sigma is None or not self.adaptive else sigma / 255.0

        # First iteration
        z = self._apply_thresholding(
            self.A[0](yp),
            self.t[0, :1] + c * self.t[0, 1:2]
        )
        yield z

        # Subsequent iterations
        for k in range(1, self.K):
            residual = mask * self.B[k](z) - yp
            z = z - self.A[k](residual)
            z = self._apply_thresholding(
                z,
                self.t[k, :1] + c * self.t[k, 1:2]
            )
            yield z

        # Final reconstruction
        xphat = self.D(z)
        xhat_internal = post_process(xphat, params)
        xhat = self._convert_from_internal(xhat_internal)
        yield xhat
