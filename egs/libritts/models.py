from typing import Optional, Tuple

import torch.nn.functional as F
import torch
import torch.nn as nn


class WMDetector(nn.Module):
    """
    Detect watermarks in an audio signal using a Transformer architecture,
    where the watermark bits are split into bytes (8 bits each).
    We assume nbits is a multiple of 8.
    """

    def __init__(
        self, input_channels: int, nbits: int, nchunk_size: int, d_model: int = 512
    ):
        """
        Args:
            input_channels (int): Number of input channels in the audio feature (e.g., mel channels).
            nbits (int): Total number of bits in the watermark, must be a multiple of 8.
            d_model (int): Embedding dimension for the Transformer.
        """
        super().__init__()
        self.nchunk_size = nchunk_size
        assert nbits % nchunk_size == 0, "nbits must be a multiple of 8!"
        self.nbits = nbits
        self.d_model = d_model
        # Number of bytes
        self.nchunks = nbits // nchunk_size

        # 1D convolution to map the input channels to d_model
        self.embedding = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # Transformer encoder block
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=1,
                dim_feedforward=d_model * 2,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=8,
        )

        # A linear head for watermark presence detection (binary)
        self.watermark_head = nn.Linear(d_model, 1)

        # For each byte, we perform a 16-way classification
        self.message_heads = nn.ModuleList(
            nn.Linear(d_model, 2**nchunk_size) for _ in range(self.nchunks)
        )

        # Learnable embeddings for each byte chunk (instead of per bit)
        # Shape: [nchunks, d_model]
        self.nchunk_embeddings = nn.Parameter(torch.randn(self.nchunks, d_model))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the detector.
        Args:
            x: shape [batch input_channels, time_steps]
        Returns:
            logits (torch.Tensor): Watermark detection logits of shape [batch, seq_len].
            chunk_logits (torch.Tensor): Byte-level classification logits of shape [batch, nchunks, 256].
        """
        batch_size, input_channels, time_steps = x.shape

        # 1) Map [batch, in_channels, time_steps] → [batch, time_steps, d_model]
        x = self.embedding(x).permute(0, 2, 1)  # [batch, time_steps, d_model]

        # 2) Prepend chunk embeddings at the beginning of the sequence
        #    [nchunks, d_model] → [1, nchunks, d_model] → [batch, nchunks, d_model]
        nchunk_embeds = self.nchunk_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        # Concatenate along the time dimension: [batch, nchunks + time_steps, d_model]
        x = torch.cat([nchunk_embeds, x], dim=1)

        # 3) Pass through the Transformer
        x = self.transformer(x)
        # x has shape [batch, nchunks + time_steps, d_model]

        # (a) Watermark presence detection: skip the first nchunks
        detection_part = x[:, self.nchunks :]  # [batch, time_steps, d_model]
        logits = self.watermark_head(detection_part).squeeze(-1)  # [batch, time_steps]

        # (b) Message decoding: use the first nchunks
        message_part = x[:, : self.nchunks]  # [batch, nchunks, d_model]
        chunk_logits_list = []
        for i, head in enumerate(self.message_heads):
            # message_part[:, i, :] has shape [batch, d_model]
            # each head outputs [batch, 16]
            chunk_vec = message_part[:, i, :]
            chunk_logits_list.append(head(chunk_vec).unsqueeze(1))  # [batch, 1, 16]

        # Concatenate along the 'nchunks' dimension → [batch, nchunks, 16]
        chunk_logits = torch.cat(chunk_logits_list, dim=1)

        return logits, chunk_logits

    def detect_watermark(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A convenience function for inference.

        Returns:
            detect_prob (torch.Tensor): Probability that the audio is watermarked, shape [batch].
            binary_message (torch.Tensor): The recovered message of shape [batch, nbits] (binary).
            detected (torch.Tensor): The sigmoid values of the per-timestep watermark detection.
        """
        logits, chunk_logits = self.forward(x)
        # logits: [batch, seq_len] → raw logits for watermark presence detection
        # chunk_logits: [batch, nchunks, 2**nchunk_size] → classification logits for each chunk

        # (1) Compute watermark detection probability
        detected = torch.sigmoid(logits)  # [batch, seq_len]
        detect_prob = detected.mean(dim=-1)  # [batch] Use mean to detect watermark probability

        # (2) Decode the message: chunk_logits has shape [batch, nchunks, 2**nchunk_size]
        chunk_probs = F.softmax(chunk_logits, dim=-1)  # [batch, nchunks, 2**nchunk_size]
        chunk_indices = torch.argmax(
            chunk_probs, dim=-1
        )  # [batch, nchunks], each in [0..255]
        # (3) Convert each chunk back to bits
        #     Finally, assemble into a [batch, nbits] binary tensor
        binary_message = []
        for i in range(self.nchunks):
            chunk_val = chunk_indices[:, i]  # [batch]
            # Extract bits from the integer
            chunk_bits = []
            for b in range(self.nchunk_size):
                bit_b = (chunk_val >> b) & 1  # get bit b
                chunk_bits.append(bit_b.unsqueeze(-1))
            # Concatenate bits to shape [batch, nchunk_size]
            chunk_bits = torch.cat(chunk_bits, dim=-1)
            binary_message.append(chunk_bits)

        # Concatenate all bytes → [batch, nbits]
        binary_message = torch.cat(binary_message, dim=-1)

        return detect_prob, binary_message, detected



class WMEmbedder(nn.Module):
    """
    A class that takes a secret message, processes it into chunk embeddings
    (as a small sequence), and uses a TransformerDecoder to do cross-attention
    between the original hidden (target) and the watermark tokens (memory).
    """

    def __init__(
        self,
        nbits: int,  # total bits in the secret message
        input_dim: int,  # the input dimension (e.g. audio feature dimension)
        nchunk_size: int,
        hidden_dim: int = 256,
        num_heads: int = 1,
        num_layers: int = 4,
    ):
        super().__init__()
        self.nchunk_size = nchunk_size
        assert nbits % nchunk_size == 0, "nbits must be a multiple of nchunk_size!"
        self.nbits = nbits
        self.nchunks = nbits // nchunk_size  # how many chunks

        # Each chunk (0..2^nchunk_size - 1) maps to an embedding of size [hidden_dim]
        self.msg_embeddings = nn.ModuleList(
            nn.Embedding(2**nchunk_size, hidden_dim) for _ in range(self.nchunks)
        )

        # Linear to project [input_dim] -> [hidden_dim]
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # TransformerDecoder for cross-attention
        # d_model=hidden_dim, so the decoder expects [b, seq_len, hidden_dim] as tgt
        # and [b, memory_len, hidden_dim] as memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            activation="gelu",
            batch_first=True,  # so shape is [batch, seq, feature]
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Project [hidden_dim] -> [input_dim]
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, hidden: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [batch, input_dim, seq_len]
            msg: [batch, nbits]
        Returns:
            A tensor [batch, input_dim, seq_len] with watermark injected.
        """
        device = hidden.device  # Device where the embedder resides, ensuring all tensors are on the same GPU/CPU
        b, in_dim, seq_len = hidden.shape

        # 1) Project input features to [b, seq_len, hidden_dim]
        hidden_projected = self.input_projection(hidden.permute(0, 2, 1))  # [b, seq_len, hidden_dim]

        # 2) Convert msg bits into chunk embeddings
        chunk_emb_list = []
        for i in range(self.nchunks):
            chunk_bits = msg[:, i * self.nchunk_size: (i + 1) * self.nchunk_size].to(device)
            chunk_val = torch.zeros(chunk_bits.size(0), device=device, dtype=torch.long)
            for bit_idx in range(self.nchunk_size):
                chunk_val += chunk_bits[:, bit_idx].long() << bit_idx

            # embedding => [b, hidden_dim]
            chunk_emb = self.msg_embeddings[i](chunk_val)
            chunk_emb_list.append(chunk_emb.unsqueeze(1))  # => [b,1,hidden_dim]

        # Concat => [b, nchunks, hidden_dim]
        chunk_emb_seq = torch.cat(chunk_emb_list, dim=1)

        # 3) TransformerDecoder forward
        x_decoded = self.transformer_decoder(tgt=hidden_projected, memory=chunk_emb_seq)

        # 4) Project back to input_dim
        x_output = self.output_projection(x_decoded)

        # 5) permute back to [b, input_dim, seq_len]
        x_output = x_output.permute(0, 2, 1)

        # 6) Residual
        x_output = x_output + hidden.to(device)

        return x_output
