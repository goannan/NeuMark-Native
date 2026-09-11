# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:47:55 2023
@author: zhangxin
"""

import random
from models import WMEmbedder
from .modules.seanet import SEANetEncoder, SEANetDecoder
from .quantization import ResidualVectorQuantizer
import torch.nn as nn
from einops import rearrange
import torch
import numpy as np


class SpeechTokenizer(nn.Module):
    def __init__(self, config):
        """

        Parameters
        ----------
        config : json
            Model Config.

        """
        super().__init__()
        self.encoder = SEANetEncoder(
            n_filters=config.get("n_filters"),
            dimension=config.get("dimension"),
            ratios=config.get("strides"),
            lstm=config.get("lstm_layers"),
            bidirectional=config.get("bidirectional"),
            dilation_base=config.get("dilation_base"),
            residual_kernel_size=config.get("residual_kernel_size"),
            n_residual_layers=config.get("n_residual_layers"),
            activation=config.get("activation"),
        )
        self.sample_rate = config.get("sample_rate")
        self.n_q = config.get("n_q")
        self.downsample_rate = np.prod(config.get("strides"))
        if config.get("dimension") != config.get("semantic_dimension"):
            self.transform = nn.Linear(
                config.get("dimension"), config.get("semantic_dimension")
            )
        else:
            self.transform = nn.Identity()
        self.quantizer = ResidualVectorQuantizer(
            dimension=config.get("dimension"),
            n_q=config.get("n_q"),
            bins=config.get("codebook_size"),
        )
        self.decoder = SEANetDecoder(
            n_filters=config.get("n_filters"),
            dimension=config.get("dimension"),
            ratios=config.get("strides"),
            lstm=config.get("lstm_layers"),
            bidirectional=False,
            dilation_base=config.get("dilation_base"),
            residual_kernel_size=config.get("residual_kernel_size"),
            n_residual_layers=config.get("n_residual_layers"),
            activation=config.get("activation"),
        )

    @classmethod
    def load_from_checkpoint(cls, config_path: str, ckpt_path: str):
        """

        Parameters
        ----------
        config_path : str
            Path of model configuration file.
        ckpt_path : str
            Path of model  checkpoint.

        Returns
        -------
        model : SpeechTokenizer
            SpeechTokenizer model.

        """
        import json

        with open(config_path) as f:
            cfg = json.load(f)
        model = cls(cfg)
        params = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(params)
        return model

    def forward(
        self,
        x: torch.tensor,
        n_q: int = None,
        layers: list = [0],
        msg_processor: WMEmbedder = None,
        message: torch.Tensor = None,
        return_features: bool = False,
    ):
        """
        Forward pass of the watermarked VQ-VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input waveforms. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used for encoding. Default is all layers.
        layers : list[int], optional
            Specific RVQ layers to return quantized outputs. Default is [0].
        msg_processor : WMEmbedder, optional
            Module for embedding the watermark message into quantized features.
        message : torch.Tensor, optional
            Binary message or watermark to embed.

        Returns
        -------
        o : torch.Tensor
            Reconstructed audio without watermark. Shape: (batch, channels, timesteps).
        o_wm : torch.Tensor
            Reconstructed audio with embedded watermark. Shape: (batch, channels, timesteps).
        acoustic : torch.Tensor
            Acoustic representation.
        acoustic_wm : torch.Tensor
            Acoustic representation with the watermark embedded.
        """
        if return_features:
            return self.forward_feature(x)
        e = self.encoder(x) # e: [B, 1024, T / 320]
        quantized_full, _, _, quantized_list = self.quantizer(
            e, n_q=n_q, layers=[0, 1, 2, 3, 4, 5, 6, 7], st=0
        )
        o = self.decoder(quantized_full)

        device = x.device  # or self.device
        
        # VQ1-8: Embed in all layers
        subset = [x.to(device) for x in quantized_list]
        acoustic_wm = sum(msg_processor(x, message) for x in subset)
        acoustic = e # Use full latent as reference
        e_wm = acoustic_wm

        o_wm = self.decoder(e_wm)

        return (o, o_wm, acoustic, acoustic_wm)

    def forward_feature(self, x: torch.tensor, layers: list = None):
        """

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape should be (batch, channels, timesteps).
        layers : list[int], optional
            Layers of RVQ should return quantized result. The default is all layers.

        Returns
        -------
        acoustic: encoder(x) - semantic (batch, channels, timesteps)

        """
        e = self.encoder(x)
        return e

    def encode(self, x: torch.tensor, n_q: int = None, st: int = None):
        """

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        codes : torch.tensor
            Output indices for each quantizer. Shape: (n_q, batch, timesteps)

        """
        e = self.encoder(x)
        if st is None:
            st = 0
        n_q = n_q if n_q else self.n_q
        codes = self.quantizer.encode(e, n_q=n_q, st=st)
        return codes

    def decode(self, codes: torch.tensor, st: int = 0):
        """

        Parameters
        ----------
        codes : torch.tensor
            Indices for each quantizer. Shape: (n_q, batch, timesteps).
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        o : torch.tensor
            Reconstruct wavs from codes. Shape: (batch, channels, timesteps)

        """
        quantized = self.quantizer.decode(codes, st=st)
        o = self.decoder(quantized)
        return o
