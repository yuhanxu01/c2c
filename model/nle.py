import torch
import torch.nn.functional as F
import numpy as np
import model.wvlt as wvlt

def nle_mad(y):
    """ Median Absolute Deviation (MAD).
    Robust median estimator for noise standard deviation from 
    image y contaminated by AWGN.
    """
    hh = wvlt.filter_bank_2D('bior4.4')[0][3:4].to(y.device)
    C  = y.shape[1]
    hh = torch.cat([hh]*C) 
    HHy = F.conv2d(y, hh, stride=2, groups=C)
    sigma_hat = torch.median(HHy.abs().reshape(y.shape[0],-1), dim=1)[0] / 0.6745
    return sigma_hat.reshape(-1,1,1,1)
def nle_mad_complex(y):
    """
    Median Absolute Deviation (MAD) for complex-valued data.
    Robust median estimator for noise standard deviation from
    a complex image y contaminated by AWGN.
    """
    hh = wvlt.filter_bank_2D('bior4.4')[0][3:4].to(y.device)
    C = y.shape[1]
    hh = torch.cat([hh] * C)
    
    y_real, y_imag = y.real, y.imag
    
    HHy_real = F.conv2d(y_real, hh, stride=2, groups=C)
    HHy_imag = F.conv2d(y_imag, hh, stride=2, groups=C)
    
    sigma_hat_real = torch.median(HHy_real.abs().reshape(y.shape[0], -1), dim=1)[0] / 0.6745
    sigma_hat_imag = torch.median(HHy_imag.abs().reshape(y.shape[0], -1), dim=1)[0] / 0.6745
    
    sigma_hat = torch.sqrt(sigma_hat_real**2 + sigma_hat_imag**2)
    
    return sigma_hat.reshape(-1, 1, 1, 1)
def nle_mad_complex_avg(y):
    B, C, D, H, W = y.shape
    y = y.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
    return nle_mad_complex(y).mean(dim=0, keepdim=True)

def nle_mad_complex_3dmap(y):
    B, C, D, H, W = y.shape
    map_x = torch.zeros((B, 1, D, H, W), device=y.device)
    map_y = torch.zeros_like(map_x)
    map_z = torch.zeros_like(map_x)
    for i in range(D):
        map_z[:, :, i] = nle_mad_complex(y[:, :, i])[:, :, 0]
    for i in range(H):
        map_y[:, :, :, i] = nle_mad_complex(y.permute(0, 1, 3, 2, 4)[:, :, i])[:, :, 0]
    for i in range(W):
        map_x[:, :, :, :, i] = nle_mad_complex(y.permute(0, 1, 4, 3, 2)[:, :, i])[:, :, 0]
    return (map_x + map_y + map_z) / 3

def nle_mad(y):
    hh = wvlt.filter_bank_2D('bior4.4')[0][3:4].to(y.device)
    C = y.shape[1]
    hh = torch.cat([hh] * C) 
    HHy = F.conv2d(y, hh, stride=2, groups=C)
    sigma_hat = torch.median(HHy.abs().reshape(y.shape[0], -1), dim=1)[0] / 0.6745
    return sigma_hat.reshape(-1, 1, 1, 1)

def noise_level(y, method="MAD", **kwargs):
    if method in [True, "MAD", "wvlt"]:
        return nle_mad(y)
    elif method == "wvlt_c":
        return nle_mad_complex(y)
    elif method == "wvlt_c_avg":
        return nle_mad_complex_avg(y)
    elif method == "wvlt_c_3dmap":
        return nle_mad_complex_3dmap(y)
    else:
        raise NotImplementedError(f"Unknown method: {method}")

def im2col(X, m, n):
    """ image to column.
    """
    return X.unfold(2,m,1).unfold(3,n,1).reshape(-1,m*n).T

def convmtx2(H, m, n):
    """ 2D convolution matrix.
    """
    s = H.shape[2:]
    T = torch.zeros((m-s[0]+1)*(n-s[1]+1), m*n, device=H.device)
    k = 1
    for i in range(1,m-s[0]+2):
        for j in range(1,n-s[1]+2):
            for p in range(1,s[0]+1):
                m1 = (i-1+p-1)*n+(j-1)+1
                m2 = (i-1+p-1)*n+(j-1)+1+s[1]-1
                A  = H[0,0,p-1,:]
                T[k-1,m1-1:m2] = A
            k = k+1
    return T
