# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Example attacks using different audio effects. 
# For full list of atacks, check 
# https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/utils/audio_effects.py
#
#
import typing as tp

import julius
import torch


def generate_pink_noise(length: int) -> torch.Tensor:
    """
    Generate pink noise using Voss-McCartney algorithm with PyTorch.
    """
    num_rows = 16
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]
    # Normalize
    pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))
    return pink_noise


def audio_effect_return(
    tensor: torch.Tensor, mask: tp.Optional[torch.Tensor]
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the mask if it was in the input otherwise only the output tensor"""
    if mask is None:
        return tensor
    else:
        return tensor, mask


class AudioEffects:
    @staticmethod
    def speed(
        tensor: torch.Tensor,
        speed_range: tuple = (0.5, 1.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Function to change the speed of a batch of audio data.
        The output will have a different length !

        Parameters:
        audio_batch (torch.Tensor): The batch of audio data in torch tensor format.
        speed (float): The speed to change the audio to.

        Returns:
        torch.Tensor: The batch of audio data with the speed changed.
        """
        speed = float(torch.empty(1).uniform_(*speed_range).item())
        target_length = max(1, int(round(tensor.shape[-1] / speed)))
        # ``julius.resample_frac(16000, arbitrary_integer)`` constructs a
        # kernel whose size can explode when the two integers are coprime
        # (minutes for a single short utterance).  Linear time interpolation
        # implements the same random 0.8x--1.2x duration attack without that
        # pathological setup cost.
        resampled_tensor = torch.nn.functional.interpolate(
            tensor, size=target_length, mode="linear", align_corners=False
        )
        if mask is None:
            return resampled_tensor
        else:
            return resampled_tensor, torch.nn.functional.interpolate(
                mask, size=resampled_tensor.size(-1), mode="nearest-exact"
            )

    @staticmethod
    def updownresample(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        intermediate_freq: int = 32000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        orig_shape = tensor.shape
        # upsample
        tensor = julius.resample_frac(tensor, sample_rate, intermediate_freq)
        # downsample
        tensor = julius.resample_frac(tensor, intermediate_freq, sample_rate)

        tensor = tensor[..., : orig_shape[-1]]
        if tensor.shape[-1] < orig_shape[-1]:
            tensor = torch.nn.functional.pad(tensor, (0, orig_shape[-1] - tensor.shape[-1]))
        return audio_effect_return(tensor=tensor, mask=mask)

    @staticmethod
    def echo(
        tensor: torch.Tensor,
        volume_range: tuple = (0.1, 0.5),
        duration_range: tuple = (0.1, 0.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Attenuating the audio volume by a factor of 0.4, delaying it by 100ms,
        and then overlaying it with the original.

        :param tensor: 3D Tensor representing the audio signal [bsz, channels, frames]
        :param echo_volume: volume of the echo signal
        :param sample_rate: Sample rate of the audio signal.
        :return: Audio signal with reverb.
        """

        # Create a simple impulse response
        # Duration of the impulse response in seconds
        duration = torch.FloatTensor(1).uniform_(*duration_range)
        volume = torch.FloatTensor(1).uniform_(*volume_range)

        n_samples = int(sample_rate * duration)
        impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)

        # Define a few reflections with decreasing amplitude
        impulse_response[0] = 1.0  # Direct sound

        impulse_response[int(sample_rate * duration) - 1] = (
            volume  # First reflection after 100ms
        )

        # Add batch and channel dimensions to the impulse response
        impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

        # Match VoiceMark/train/attacks.py: avoid cuFFT failures for short audio.
        pad_len = impulse_response.shape[-1] - 1
        padded = torch.nn.functional.pad(tensor, (pad_len, 0))
        reverbed_signal = torch.nn.functional.conv1d(padded, impulse_response)

        # Normalize to the original amplitude range for stability
        max_abs_reverbed = torch.max(torch.abs(reverbed_signal))
        if max_abs_reverbed > 0:
            reverbed_signal = (
                reverbed_signal
                / max_abs_reverbed
                * torch.max(torch.abs(tensor))
            )

        # Ensure tensor size is not changed
        tmp = torch.zeros_like(tensor)
        tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal[..., : tensor.shape[-1]]
        reverbed_signal = tmp

        return audio_effect_return(tensor=reverbed_signal, mask=mask)

    @staticmethod
    def random_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.001,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Add Gaussian noise to the waveform."""
        noise = torch.randn_like(waveform) * noise_std
        noisy_waveform = waveform + noise
        return audio_effect_return(tensor=noisy_waveform, mask=mask)

    @staticmethod
    def pink_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.01,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Add pink background noise to the waveform."""
        noise = generate_pink_noise(waveform.shape[-1]) * noise_std
        noise = noise.to(waveform.device)
        # Assuming waveform is of shape (bsz, channels, length)
        noisy_waveform = waveform + noise.unsqueeze(0).unsqueeze(0).to(waveform.device)
        return audio_effect_return(tensor=noisy_waveform, mask=mask)

    @staticmethod
    def lowpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 5000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        return audio_effect_return(
            tensor=julius.lowpass_filter(
                waveform, cutoff=cutoff_freq / sample_rate, fft=False
            ),
            mask=mask,
        )

    @staticmethod
    def highpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 500,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        return audio_effect_return(
            tensor=julius.highpass_filter(
                waveform, cutoff=cutoff_freq / sample_rate, fft=False
            ),
            mask=mask,
        )

    @staticmethod
    def bandpass_filter(
        waveform: torch.Tensor,
        cutoff_freq_low: float = 300,
        cutoff_freq_high: float = 8000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Apply a bandpass filter to the waveform by cascading
        a high-pass filter followed by a low-pass filter.

        Parameters:
        - waveform (torch.Tensor): Input audio waveform.
        - low_cutoff (float): Lower cutoff frequency.
        - high_cutoff (float): Higher cutoff frequency.
        - sample_rate (int): The sample rate of the waveform.

        Returns:
        - torch.Tensor: Filtered audio waveform.
        """

        return audio_effect_return(
            tensor=julius.bandpass_filter(
                waveform,
                cutoff_low=cutoff_freq_low / sample_rate,
                cutoff_high=cutoff_freq_high / sample_rate,
                fft=False,
            ),
            mask=mask,
        )

    @staticmethod
    def smooth(
        tensor: torch.Tensor,
        window_size_range: tuple = (2, 10),
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Smooths the input tensor (audio signal) using a moving average filter with the given window size.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor. Assumes tensor shape is (batch_size, channels, time).
        - window_size (int): Size of the moving average window.

        Returns:
        - torch.Tensor: Smoothed audio tensor.
        """

        window_size = int(torch.FloatTensor(1).uniform_(*window_size_range))
        # Create a uniform smoothing kernel
        kernel = torch.ones(1, 1, window_size).type(tensor.type()) / window_size
        kernel = kernel.to(tensor.device)

        # Match VoiceMark/train/attacks.py: avoid cuFFT failures for short audio.
        pad_len = kernel.shape[-1] - 1
        padded = torch.nn.functional.pad(tensor, (pad_len, 0))
        smoothed = torch.nn.functional.conv1d(padded, kernel)
        # Ensure tensor size is not changed
        tmp = torch.zeros_like(tensor)
        tmp[..., : smoothed.shape[-1]] = smoothed[..., : tensor.shape[-1]]
        smoothed = tmp

        return audio_effect_return(tensor=smoothed, mask=mask)

    @staticmethod
    def boost_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor * (1 + amount / 100), mask=mask)

    @staticmethod
    def duck_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor * (1 - amount / 100), mask=mask)

    @staticmethod
    def identity(
        tensor: torch.Tensor, mask: tp.Optional[torch.Tensor] = None
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor, mask=mask)

    @staticmethod
    def shush(
        tensor: torch.Tensor,
        fraction: float = 0.001,
        mask: tp.Optional[torch.Tensor] = None
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Sets a specified chronological fraction of indices of the input tensor (audio signal) to 0.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor. Assumes tensor shape is (batch_size, channels, time).
        - fraction (float): Fraction of indices to be set to 0 (from the start of the tensor) (default: 0.001, i.e, 0.1%)

        Returns:
        - torch.Tensor: Transformed audio tensor.
        """
        time = tensor.size(-1)
        shush_tensor = tensor.detach().clone()
        
        # Set the first `fraction*time` indices of the waveform to 0
        shush_tensor[:, :, :int(fraction*time)] = 0.0
                
        return audio_effect_return(tensor=shush_tensor, mask=mask)

    @staticmethod
    def encodec(
        tensor: torch.Tensor,
        bandwidth: float = 6.0,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        decoded = EncodecAttack.compress(tensor, bandwidth, sample_rate)
        return audio_effect_return(tensor=decoded, mask=mask)


class CodecAttackError(RuntimeError):
    """Raised when a requested codec attack cannot be executed."""


def _match_audio_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    waveform = waveform[..., :target_length]
    if waveform.shape[-1] < target_length:
        waveform = torch.nn.functional.pad(
            waveform, (0, target_length - waveform.shape[-1])
        )
    return waveform


class EncodecAttack:
    """EnCodec 24 kHz compression/reconstruction at a selected bandwidth."""

    _model = None
    _device = None

    @classmethod
    def _get_model(cls, device: torch.device):
        if cls._model is None:
            try:
                from encodec import EncodecModel
            except ImportError as exc:
                raise CodecAttackError(
                    "Encodec attack requires `encodec==0.1.1`."
                ) from exc
            try:
                cls._model = (
                    EncodecModel.encodec_model_24khz().eval().to(device)
                )
            except Exception as exc:
                raise CodecAttackError(
                    f"Unable to load EnCodec 24 kHz checkpoint: {exc}"
                ) from exc
            cls._device = device
            for parameter in cls._model.parameters():
                parameter.requires_grad = False
        elif cls._device != device:
            cls._model.to(device)
            cls._device = device
        return cls._model

    @classmethod
    def compress(
        cls,
        waveform: torch.Tensor,
        bandwidth: float,
        sample_rate: int,
    ) -> torch.Tensor:
        model = cls._get_model(waveform.device)
        # This must be set for every row; otherwise 6/12 kbps reuse 3 kbps.
        model.set_target_bandwidth(float(bandwidth))
        target_sr = int(model.sample_rate)
        original_length = waveform.shape[-1]

        with torch.no_grad():
            codec_input = waveform.detach()
            if sample_rate != target_sr:
                codec_input = julius.resample_frac(
                    codec_input, sample_rate, target_sr
                )
            encoded_frames = model.encode(codec_input)
            decoded = model.decode(encoded_frames)
            if sample_rate != target_sr:
                decoded = julius.resample_frac(decoded, target_sr, sample_rate)
        return _match_audio_length(decoded, original_length)

    @classmethod
    def release(cls):
        cls._model = None
        cls._device = None


class DACAttack:
    """Descript Audio Codec attack using every quantizer, as in valid.py."""

    _models = {}
    _sample_rates = {"16khz": 16000, "24khz": 24000, "44khz": 44100}

    @classmethod
    def _get_model(cls, model_type: str, device: torch.device):
        key = (model_type, str(device))
        if key not in cls._models:
            try:
                from dac.utils import load_model
            except ImportError as exc:
                raise CodecAttackError(
                    "DAC attacks require `descript-audio-codec==1.0.0`."
                ) from exc
            try:
                model = load_model(model_type=model_type, model_bitrate="8kbps")
                model.eval().to(device)
            except Exception as exc:
                raise CodecAttackError(
                    f"Unable to load DAC {model_type} checkpoint: {exc}"
                ) from exc
            for parameter in model.parameters():
                parameter.requires_grad = False
            cls._models[key] = model
        return cls._models[key]

    @classmethod
    def compress(
        cls,
        waveform: torch.Tensor,
        model_type: str,
        sample_rate: int,
    ) -> torch.Tensor:
        if model_type not in cls._sample_rates:
            raise ValueError(f"Unsupported DAC model type: {model_type}")
        model = cls._get_model(model_type, waveform.device)
        target_sr = cls._sample_rates[model_type]
        original_length = waveform.shape[-1]

        with torch.no_grad():
            codec_input = waveform.detach()
            if sample_rate != target_sr:
                codec_input = julius.resample_frac(
                    codec_input, sample_rate, target_sr
                )
            preprocessed = model.preprocess(codec_input, target_sr)
            # n_quantizers=None deliberately matches VoiceMark valid.py.
            quantized, _, _, _, _ = model.encode(
                preprocessed, n_quantizers=None
            )
            decoded = model.decode(quantized)
            if sample_rate != target_sr:
                decoded = julius.resample_frac(decoded, target_sr, sample_rate)
        return _match_audio_length(decoded, original_length)

    @classmethod
    def release(cls):
        cls._models.clear()


def _patch_snac_torch_compat():
    """Provide Python/Torch APIs required by SNAC in older environments."""

    import math
    import torch.nn.functional as functional
    import torch.nn.utils.parametrizations as parametrizations

    if not hasattr(math, "lcm"):
        def lcm(first, second):
            if first == 0 or second == 0:
                return 0
            return abs(first * second) // math.gcd(first, second)

        math.lcm = lcm

    # Modern evaluation environments (for example the AudioSeal environment)
    # already provide both APIs.  Replacing their native weight_norm changes
    # checkpoint parameter names and prevents official SNAC models from loading.
    if (
        hasattr(parametrizations, "weight_norm")
        and hasattr(functional, "scaled_dot_product_attention")
    ):
        return

    from torch import nn
    from torch.nn.utils import parametrize

    class _WeightNorm(nn.Module):
        def __init__(self, dim=0):
            super().__init__()
            self.dim = -1 if dim is None else dim

        def forward(self, weight_g, weight_v):
            return torch._weight_norm(weight_v, weight_g, self.dim)

        def right_inverse(self, weight):
            return torch.norm_except_dim(weight, 2, self.dim), weight

    def weight_norm(module, name="weight", dim=0):
        parametrize.register_parametrization(
            module, name, _WeightNorm(dim), unsafe=True
        )
        return module

    # valle.data.tokenizer installs Torch 1.13's legacy weight_norm here so
    # VoiceMark can import. SNAC checkpoints use the Torch 2.x parametrization
    # key layout, therefore replace the compatibility alias before importing
    # SNAC even when the attribute already exists.
    parametrizations.weight_norm = weight_norm

    if not hasattr(functional, "scaled_dot_product_attention"):
        def scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        ):
            if is_causal:
                raise NotImplementedError("SNAC does not request causal SDPA")
            scores = torch.matmul(query, key.transpose(-2, -1)) * (
                query.shape[-1] ** -0.5
            )
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                else:
                    scores = scores + attn_mask
            probabilities = torch.softmax(scores, dim=-1)
            if dropout_p:
                probabilities = functional.dropout(
                    probabilities, p=dropout_p
                )
            return torch.matmul(probabilities, value)

        functional.scaled_dot_product_attention = scaled_dot_product_attention


class SNACAttack:
    """SNAC compression/reconstruction using the official pretrained models."""

    _models = {}
    _configs = {
        "24khz": ("hubertsiuzdak/snac_24khz", 24000),
        "32khz": ("hubertsiuzdak/snac_32khz", 32000),
        "44khz": ("hubertsiuzdak/snac_44khz", 44100),
    }

    @classmethod
    def _get_model(cls, model_type: str, device: torch.device):
        key = (model_type, str(device))
        if key not in cls._models:
            if model_type not in cls._configs:
                raise ValueError(f"Unsupported SNAC model type: {model_type}")
            _patch_snac_torch_compat()
            try:
                from snac import SNAC
            except ImportError as exc:
                raise CodecAttackError(
                    "SNAC attacks require `snac==1.2.1`."
                ) from exc
            repo_id, _ = cls._configs[model_type]
            try:
                model = SNAC.from_pretrained(repo_id).eval().to(device)
            except Exception as exc:
                raise CodecAttackError(
                    f"Unable to load {repo_id}: {exc}"
                ) from exc
            for parameter in model.parameters():
                parameter.requires_grad = False
            cls._models[key] = model
        return cls._models[key]

    @classmethod
    def compress(
        cls,
        waveform: torch.Tensor,
        model_type: str,
        sample_rate: int,
    ) -> torch.Tensor:
        model = cls._get_model(model_type, waveform.device)
        _, target_sr = cls._configs[model_type]
        original_length = waveform.shape[-1]

        with torch.no_grad():
            codec_input = waveform.detach()
            if sample_rate != target_sr:
                codec_input = julius.resample_frac(
                    codec_input, sample_rate, target_sr
                )
            decoded, _ = model(codec_input)
            if sample_rate != target_sr:
                decoded = julius.resample_frac(decoded, target_sr, sample_rate)
        return _match_audio_length(decoded, original_length)

    @classmethod
    def release(cls):
        cls._models.clear()


CODEC_KEYWORDS = ("Encodec", "DAC", "SNAC")


def build_voicemark_valid_attacks(sample_rate: int):
    """VoiceMark valid.py DSP suite plus the requested neural codecs."""

    return [
        ("Clean (Identity)", lambda wav: AudioEffects.identity(wav), False),
        (
            "Gaussian Noise",
            lambda wav: AudioEffects.random_noise(wav, noise_std=0.001),
            False,
        ),
        (
            "Pink Noise",
            lambda wav: AudioEffects.pink_noise(wav, noise_std=0.01),
            False,
        ),
        (
            "Lowpass Filter (5k)",
            lambda wav: AudioEffects.lowpass_filter(
                wav, cutoff_freq=5000, sample_rate=sample_rate
            ),
            False,
        ),
        (
            "Highpass Filter (500)",
            lambda wav: AudioEffects.highpass_filter(
                wav, cutoff_freq=500, sample_rate=sample_rate
            ),
            False,
        ),
        (
            "Bandpass Filter",
            lambda wav: AudioEffects.bandpass_filter(
                wav,
                cutoff_freq_low=300,
                cutoff_freq_high=8000,
                sample_rate=sample_rate,
            ),
            False,
        ),
        (
            "Echo/Reverb",
            lambda wav: AudioEffects.echo(
                wav,
                volume_range=(0.1, 0.5),
                duration_range=(0.1, 0.5),
                sample_rate=sample_rate,
            ),
            False,
        ),
        (
            "Smooth (Moving Avg)",
            lambda wav: AudioEffects.smooth(wav, window_size_range=(2, 10)),
            False,
        ),
        (
            "Resampling (32k-16k)",
            lambda wav: AudioEffects.updownresample(
                wav, sample_rate=sample_rate, intermediate_freq=32000
            ),
            False,
        ),
        (
            "Volume Boost (+10%)",
            lambda wav: AudioEffects.boost_audio(wav, amount=10),
            False,
        ),
        (
            "Volume Duck (-10%)",
            lambda wav: AudioEffects.duck_audio(wav, amount=10),
            False,
        ),
        (
            "Speed (0.8x-1.2x)",
            lambda wav: AudioEffects.speed(
                wav, speed_range=(0.8, 1.2), sample_rate=sample_rate
            ),
            False,
        ),
        (
            "Encodec 24kHz bandwidth 3kbps",
            lambda wav: EncodecAttack.compress(wav, 3.0, sample_rate),
            True,
        ),
        (
            "Encodec 24kHz bandwidth 6kbps",
            lambda wav: EncodecAttack.compress(wav, 6.0, sample_rate),
            True,
        ),
        (
            "Encodec 24kHz bandwidth 12kbps",
            lambda wav: EncodecAttack.compress(wav, 12.0, sample_rate),
            True,
        ),
        (
            "DAC 16kHz bandwidth 6kbps",
            lambda wav: DACAttack.compress(wav, "16khz", sample_rate),
            True,
        ),
        (
            "DAC 24kHz bandwidth 24kbps",
            lambda wav: DACAttack.compress(wav, "24khz", sample_rate),
            True,
        ),
        (
            "DAC 44.1kHz bandwidth 8kbps",
            lambda wav: DACAttack.compress(wav, "44khz", sample_rate),
            True,
        ),
        (
            "SNAC 24kHz bandwidth 0.98kbps",
            lambda wav: SNACAttack.compress(wav, "24khz", sample_rate),
            True,
        ),
        (
            "SNAC 32kHz bandwidth 1.9kbps",
            lambda wav: SNACAttack.compress(wav, "32khz", sample_rate),
            True,
        ),
        (
            "SNAC 44.1kHz bandwidth 2.6kbps",
            lambda wav: SNACAttack.compress(wav, "44khz", sample_rate),
            True,
        ),
    ]


def release_codec_models():
    """Release attack-only models between codec rows to limit GPU memory."""

    EncodecAttack.release()
    DACAttack.release()
    SNACAttack.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
