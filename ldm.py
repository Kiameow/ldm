from typing import List

import torch
import torch.nn as nn

from labml_nn.diffusion.stable_diffusion.model.autoencoder import Autoencoder
from labml_nn.diffusion.stable_diffusion.model.clip_embedder import CLIPTextEmbedder
from labml_nn.diffusion.stable_diffusion.model.unet import UNetModel


class DiffusionWrapper(nn.Module):
    def __init__(self, diffusion_model: UNetModel):
        super().__init__()
        self.diffusion_model = diffusion_model
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor):
        return self.diffusion_model(x, timesteps, context)
    
class LatentDiffusion(nn.Module):
    model: DiffusionWrapper
    first_stage_model: Autoencoder
    cond_stage_model: CLIPTextEmbedder
    
    def __init__(self,
                 unet_model: UNetModel,
                 autoencoder: Autoencoder,
                 clip_embedder: CLIPTextEmbedder,
                 latent_scaling_factor: float,
                 n_steps: int,
                 linear_start: float,
                 linear_end: float,
                ):
        
        super().__init__()
        
        self.model = DiffusionWrapper(unet_model)
        
        self.first_stage_model = autoencoder
        self.latent_scaling_factor = latent_scaling_factor
        
        self.cond_stage_model = clip_embedder
        
        self.n_steps = n_steps
        
        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        
        alpha = 1. - beta
        
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
        
    ## get model device
    @property
    def device(self):
        return next(iter(self.model.parameters())).device

    def get_text_conditioning(self, prompts: List[str]):
        return self.cond_stage_model(prompts)
    
    ## The encoder output is a distribution. We sample from that and multiply by the scaling factor.
    def autoencoder_encode(self, image: torch.Tensor):
        return self.latent_scaling_factor * self.first_stage_model.encode(image).sample()
    
    ## We scale down by the scaling factor and then decode.
    def autoencoder_decode(self, z: torch.Tensor):
        return self.first_stage_model.decode(z / self.latent_scaling_factor)
    
    ## Predict noise given the latent representation x, time step t, and the conditioning context c.
    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        return self.model(x, t, context)
    
    
        

