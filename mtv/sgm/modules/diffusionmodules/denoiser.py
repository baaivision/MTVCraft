from typing import Dict, Union

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config


class Denoiser(nn.Module):
    def __init__(self, weighting_config, scaling_config):
        super().__init__()

        self.weighting = instantiate_from_config(weighting_config)
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)
        
    def get_t0(self, alphas_cumprod_sqrt: torch.Tensor, **additional_model_inputs):

        # import pdb; pdb.set_trace()
        c_skip = alphas_cumprod_sqrt
        c_out = -((1 - alphas_cumprod_sqrt**2) ** 0.5)
        c_in = torch.ones_like(alphas_cumprod_sqrt, device=alphas_cumprod_sqrt.device)
        c_noise = additional_model_inputs["idx_t0"].clone()
        return c_skip, c_out, c_in, c_noise

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        sigma_t0 = None,
        **additional_model_inputs,
    ) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, **additional_model_inputs)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))

        if (sigma_t0 is not None):
            sigma_t0_shape = sigma_t0.shape
            sigma_t0 = append_dims(sigma_t0, input.ndim)
            c_skip_t0, c_out_t0, c_in_t0, c_noise_t0 = self.get_t0(sigma_t0, **additional_model_inputs)
            c_noise_t0 = c_noise_t0.reshape(sigma_t0_shape)

            latent_frame_mask = additional_model_inputs["latent_frame_mask"]

            c_out_t0 = c_out_t0.repeat(1, input.shape[1], 1, 1, 1)
            c_skip_t0 = c_skip_t0.repeat(1, input.shape[1], 1, 1, 1)
            c_out = c_out.repeat(1, input.shape[1], 1, 1, 1)
            c_skip = c_skip.repeat(1, input.shape[1], 1, 1, 1)

            c_out = torch.where(latent_frame_mask[:, :, None, None, None], c_out, c_out_t0)
            c_skip = torch.where(latent_frame_mask[:, :, None, None, None], c_skip, c_skip_t0)
            additional_model_inputs["timesteps_t0"] = c_noise_t0
            
        # c_noise is idx, as well as timestep
        return network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out + input * c_skip


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        weighting_config,
        scaling_config,
        num_idx,
        discretization_config,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
    ):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.sigmas = sigmas
        # self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas.to(sigma.device)[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas.to(idx.device)[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
