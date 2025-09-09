"""Model architectures for ARC Diffusion."""

from .unet import FlatUNet3Shot
from .diffusion import GaussianDiffusion, DiffusionCfg

# Model registry for easy switching
MODEL_REGISTRY = {
    'flat_unet_3shot': FlatUNet3Shot,
    # Future models can be added here
}

DIFFUSION_REGISTRY = {
    'ddpm': GaussianDiffusion,
    # Future: 'ddim': DDIMSampler,
}

__all__ = ['FlatUNet3Shot', 'GaussianDiffusion', 'DiffusionCfg', 'MODEL_REGISTRY', 'DIFFUSION_REGISTRY']
