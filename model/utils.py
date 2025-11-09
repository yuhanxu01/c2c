import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from .nle import noise_level as _estimate_noise_level

def pre_process(x, stride, mask=1):
    """ image preprocessing: stride-padding and mean subtraction.
    """
    params = []
    # mean-subtract
    if torch.is_tensor(mask):
        xmean = x.sum(dim=(1,2,3), keepdim=True) / mask.sum(dim=(1,2,3), keepdim=True)
    else:
        xmean = x.mean(dim=(1,2,3), keepdim=True)
    x = mask*(x - xmean)
    params.append(xmean)
    # pad signal for stride
    pad = calc_pad_2D(*x.shape[2:], stride)
    x = F.pad(x, pad, mode='reflect')
    if torch.is_tensor(mask):
        mask = F.pad(mask, pad, mode='reflect')
    params.append(pad)
    return x, params, mask

def post_process(x, params):
    """ undoes image pre-processing given params
    """
    # unpad
    pad = params.pop()
    x = unpad(x, pad)
    # add mean
    xmean = params.pop()
    x = x + xmean
    return x

def calc_pad_1D(L, M):
    """ Return pad sizes for length L 1D signal to be divided by M
    """
    if L%M == 0:
        Lpad = [0,0]
    else:
        Lprime = np.ceil(L/M) * M
        Ldiff  = Lprime - L
        Lpad   = [int(np.floor(Ldiff/2)), int(np.ceil(Ldiff/2))]
    return Lpad

def calc_pad_2D(H, W, M):
    """ Return pad sizes for image (H,W) to be divided by size M
    (H,W): input height, width
    output: (padding_left, padding_right, padding_top, padding_bottom)
    """
    return (*calc_pad_1D(W,M), *calc_pad_1D(H,M))

def conv_pad(x, ks, mode):
    """ Pad a signal for same-sized convolution
    """
    pad = (int(np.floor((ks-1)/2)), int(np.ceil((ks-1)/2)))
    return F.pad(x, (*pad, *pad), mode=mode)

def unpad(I, pad):
    """ Remove padding from 2D signalstack"""
    if pad[3] == 0 and pad[1] > 0:
        return I[..., pad[2]:, pad[0]:-pad[1]]
    elif pad[3] > 0 and pad[1] == 0:
        return I[..., pad[2]:-pad[3], pad[0]:]
    elif pad[3] == 0 and pad[1] == 0:
        return I[..., pad[2]:, pad[0]:]
    else:
        return I[..., pad[2]:-pad[3], pad[0]:-pad[1]]

def pre_process_3d(x, stride, mask=1):
    """ 3D video preprocessing: stride-padding and mean subtraction.
    """
    params = []
    # mean-subtract
    if torch.is_tensor(mask):
        xmean = x.sum(dim=(1,2,3,4), keepdim=True) / mask.sum(dim=(1,2,3,4), keepdim=True)
    else:
        xmean = x.mean(dim=(1,2,3,4), keepdim=True)
    x = mask * (x - xmean)
    params.append(xmean)
    # pad signal for stride
    pad = calc_pad_3D(*x.shape[2:], stride)
    x = F.pad(x, pad, mode='reflect')
    if torch.is_tensor(mask):
        mask = F.pad(mask, pad, mode='reflect')
    params.append(pad)
    return x, params, mask

def calc_pad_3D(D, H, W, M):
    """ Return pad sizes for 3D data (D, H, W) to be divided by size M
    (D, H, W): input depth, height, width
    output: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    """
    pad_w = calc_pad_1D(W, M[1])
    pad_h = calc_pad_1D(H, M[2])
    pad_d = calc_pad_1D(D, M[0])
    return (*pad_w, *pad_h, *pad_d)  # (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)

def post_process_3d(x, params):
    """ Undoes 3D video pre-processing given params
    """
    # unpad
    pad = params.pop()
    x = unpad_3d(x, pad)
    # add mean
    xmean = params.pop()
    x = x + xmean
    return x


def unpad_3d(I, pad):
    """ Remove padding from 3D signal stack
    pad: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    """
    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = pad
    if pad_back == 0 and pad_right > 0:
        return I[..., pad_front:, pad_top:-pad_bottom, pad_left:-pad_right]
    elif pad_back > 0 and pad_right == 0:
        return I[..., pad_front:-pad_back, pad_top:, pad_left:]
    elif pad_back == 0 and pad_right == 0:
        return I[..., pad_front:, pad_top:, pad_left:]
    else:
        return I[..., pad_front:-pad_back, pad_top:-pad_bottom, pad_left:-pad_right]


def compute_l1_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a - b
    if torch.is_complex(diff):
        return diff.abs().mean()
    return diff.abs().mean()


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    if torch.is_complex(x):
        noise_real = torch.randn_like(x.real)
        noise_imag = torch.randn_like(x.imag)
        noise = torch.complex(noise_real, noise_imag)
        return x + noise * sigma
    return x + torch.randn_like(x) * sigma


def apply_random_brightness_contrast(
    x: torch.Tensor, brightness: float, contrast: float
) -> torch.Tensor:
    if brightness <= 0 and contrast <= 0:
        return x
    squeeze = False
    tensor = x
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    if torch.is_complex(tensor):
        magnitude = tensor.abs()
        phase = torch.angle(tensor)
        dtype = magnitude.dtype
        device = magnitude.device
        if contrast > 0:
            factors = torch.empty(
                (magnitude.shape[0], 1, 1, 1), device=device, dtype=dtype
            ).uniform_(1.0 - contrast, 1.0 + contrast)
            magnitude = magnitude * factors
        if brightness > 0:
            offsets = torch.empty(
                (magnitude.shape[0], 1, 1, 1), device=device, dtype=dtype
            ).uniform_(-brightness, brightness)
            magnitude = (magnitude + offsets).clamp_min_(0.0)
        tensor = torch.polar(magnitude, phase)
    else:
        dtype = tensor.dtype if torch.is_floating_point(tensor) else torch.float32
        tensor = tensor.to(dtype=dtype)
        device = tensor.device
        if contrast > 0:
            factors = torch.empty(
                (tensor.shape[0], 1, 1, 1), device=device, dtype=dtype
            ).uniform_(1.0 - contrast, 1.0 + contrast)
            tensor = tensor * factors
        if brightness > 0:
            offsets = torch.empty(
                (tensor.shape[0], 1, 1, 1), device=device, dtype=dtype
            ).uniform_(-brightness, brightness)
            tensor = tensor + offsets
    if squeeze:
        tensor = tensor.squeeze(0)
    return tensor


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    tensor = x
    if torch.is_complex(tensor):
        tensor = tensor.abs()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    device = tensor.device
    dtype = tensor.dtype
    kernel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    channels = tensor.shape[1]
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)
    grad_x = F.conv2d(tensor, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(tensor, kernel_y, padding=1, groups=channels)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12)


def prepare_image_for_logging(tensor: torch.Tensor) -> torch.Tensor:
    image = tensor.detach().cpu()
    if image.ndim == 4:
        image = image[0]
    if torch.is_complex(image):
        image = image.abs()
    image = image.float()
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image = image - image.amin(dim=(-2, -1), keepdim=True)
    denom = image.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (image / denom).clamp(0.0, 1.0)


def _is_complex_representation(rep: Optional[str]) -> bool:
    if not rep:
        return False
    rep = rep.lower()
    return rep in {"complex", "ri", "complex64", "complex128"}


def _is_complex_like_tensor(tensor: torch.Tensor) -> bool:
    if torch.is_complex(tensor):
        return True
    return tensor.ndim >= 2 and tensor.shape[1] == 2 and torch.is_floating_point(tensor)


def estimate_noise_level(
    tensor: torch.Tensor,
    method: str = "auto",
    representation: Optional[str] = None,
) -> torch.Tensor:
    """
    Estimate noise level map/scalar for the tensor using the selected method.

    - magnitude inputs -> wavelet MAD ("wvlt")
    - complex / RI inputs -> complex wavelet estimator ("wvlt_c")
    - explicit method overrides respected unless incompatible.
    """
    method = (method or "auto").lower()
    complex_like = _is_complex_like_tensor(tensor) or _is_complex_representation(representation)

    if method == "auto":
        chosen = "wvlt_c" if complex_like else "wvlt"
    elif "wvlt_c" in method and not complex_like:
        chosen = "wvlt"
    else:
        chosen = method

    return _estimate_noise_level(tensor, method=chosen)
