"""Standalone mel spectrogram preprocessor — exact match with NeMo's FilterbankFeatures.

Ported from NeMo's nemo.collections.asr.parts.preprocessing.features.FilterbankFeatures
to eliminate the NeMo dependency at inference time. Uses numpy/librosa only.

Key differences from a naive librosa implementation (all fixed here):
- Dither is DISABLED during inference (NeMo only dithers during training)
- STFT uses constant (zero) padding, not reflect padding
- Normalization uses Bessel's correction (ddof=1), not population std (ddof=0)
- log_zero_guard uses the exact NeMo default (2^-24)
- preemph defaults to 0.97 (NeMo default), not 0.0

Reference: nemo.collections.asr.parts.preprocessing.features.FilterbankFeatures
"""

import numpy as np
from typing import Optional


class MelSpectrogramPreprocessor:
    """Exact replica of NeMo's AudioToMelSpectrogramPreprocessor for inference.

    Produces mel spectrograms identical to NeMo's FilterbankFeatures.forward()
    when model.eval() is set (no dither, no augmentation).
    """

    def __init__(self, config: dict):
        import librosa
        self._librosa = librosa

        self.sample_rate = config.get('sample_rate', 16000)
        self.n_fft = config.get('n_fft', 512)

        window_size = config.get('window_size', 0.025)
        window_stride = config.get('window_stride', 0.01)
        self.win_length = config.get('win_length', int(window_size * self.sample_rate))
        self.hop_length = config.get('hop_length', int(window_stride * self.sample_rate))

        self.n_mels = config.get('features', 80)
        self.fmin = config.get('lowfreq', 0)
        self.fmax = config.get('highfreq', None) or self.sample_rate / 2
        self.preemph = config.get('preemph', 0.97)
        self.log = config.get('log', True)
        self.log_zero_guard_type = config.get('log_zero_guard_type', 'add')
        self.log_zero_guard_value = config.get('log_zero_guard_value', 2 ** -24)
        self.normalize = config.get('normalize', 'per_feature')
        self.pad_to = config.get('pad_to', 0)
        self.mag_power = config.get('mag_power', 2.0)
        self.exact_pad = config.get('exact_pad', False)
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if self.exact_pad else None

        self.mel_basis = librosa.filters.mel(
            sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax, norm='slaney',
        )

        # NeMo: torch.hann_window(win_length, periodic=False)
        self.window = np.hanning(self.win_length).astype(np.float32)

    def _get_seq_len(self, audio_len: int) -> int:
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        return int((audio_len + pad_amount - self.n_fft) // self.hop_length)

    def __call__(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = self._librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)

        audio = audio.astype(np.float64)
        audio_len = len(audio)

        # NO dither during inference — NeMo only dithers when self.training is True

        # Exact pad if configured
        if self.stft_pad_amount is not None:
            audio = np.pad(audio, (self.stft_pad_amount, self.stft_pad_amount), mode='constant')

        # Pre-emphasis (NeMo masks beyond original audio length)
        if self.preemph is not None and self.preemph > 0:
            preemphed = np.empty_like(audio)
            preemphed[0] = audio[0]
            preemphed[1:] = audio[1:] - self.preemph * audio[:-1]
            if audio_len < len(preemphed):
                preemphed[audio_len:] = 0.0
            audio = preemphed

        # STFT — NeMo uses torch.stft with center=True and pad_mode="constant" (zeros)
        # librosa center=True uses reflect padding, so we pad manually with zeros instead
        if not self.exact_pad:
            pad_len = self.n_fft // 2
            audio = np.pad(audio, (pad_len, pad_len), mode='constant', constant_values=0)

        stft = self._librosa.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            center=False,  # already padded manually with constant (zero) padding
        )

        # Magnitude
        x = np.abs(stft)

        # Power spectrum (mag_power=2.0 → |STFT|^2)
        if self.mag_power != 1.0:
            x = x ** self.mag_power

        # Mel filterbank
        x = np.dot(self.mel_basis, x)

        # Log
        if self.log:
            if self.log_zero_guard_type == 'add':
                x = np.log(x + self.log_zero_guard_value)
            elif self.log_zero_guard_type == 'clamp':
                x = np.log(np.maximum(x, self.log_zero_guard_value))

        # Sequence length for masking and normalization
        seq_len = self._get_seq_len(audio_len)

        # Per-feature normalization with Bessel's correction (ddof=1)
        # NeMo: std = sqrt(sum((x - mean)^2) / (N - 1)) + 1e-5
        if self.normalize == 'per_feature':
            valid = x[:, :seq_len] if 0 < seq_len < x.shape[1] else x
            mean = valid.mean(axis=1, keepdims=True)
            std = valid.std(axis=1, keepdims=True, ddof=1)
            std = np.where(np.isnan(std), 0.0, std)
            std = std + 1e-5
            x = (x - mean) / std
        elif self.normalize == 'all_features':
            valid = x[:, :seq_len] if 0 < seq_len < x.shape[1] else x
            mean = valid.mean()
            std = valid.std() + 1e-5
            x = (x - mean) / std

        # Mask beyond seq_len
        if 0 < seq_len < x.shape[1]:
            x[:, seq_len:] = 0.0

        # Pad to multiple of pad_to
        if self.pad_to and self.pad_to > 0:
            pad_amt = x.shape[1] % self.pad_to
            if pad_amt != 0:
                x = np.pad(x, ((0, 0), (0, self.pad_to - pad_amt)), mode='constant')

        return x.astype(np.float32)
