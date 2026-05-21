import numpy as np
from typing import Optional


class MelSpectrogramPreprocessor:
    """
    Standalone mel spectrogram preprocessor.
    Replicates NeMo's AudioToMelSpectrogramPreprocessor behavior exactly.
    Uses librosa for mel spectrogram computation.
    """

    def __init__(self, config: dict):
        import librosa
        self.librosa = librosa

        self.sample_rate = config.get('sample_rate', 16000)
        self.n_fft = config.get('n_fft', 512)

        window_size = config.get('window_size', 0.025)
        window_stride = config.get('window_stride', 0.01)
        self.win_length = config.get('win_length', int(window_size * self.sample_rate))
        self.hop_length = config.get('hop_length', int(window_stride * self.sample_rate))

        self.n_mels = config.get('features', 80)
        self.fmin = config.get('lowfreq', 0)
        self.fmax = config.get('highfreq', None)
        self.preemph = config.get('preemph', 0.0)
        self.log = config.get('log', True)
        self.log_zero_guard_value = config.get('log_zero_guard_value', 2**-24)
        self.normalize = config.get('normalize', 'per_feature')
        self.pad_to = config.get('pad_to', 0)

        self.mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax if self.fmax else self.sample_rate / 2,
            norm='slaney'
        )

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        if self.normalize == 'per_feature':
            mean = np.mean(features, axis=1, keepdims=True)
            std = np.std(features, axis=1, keepdims=True)
            std = np.maximum(std, 1e-5)
            features = (features - mean) / std
        elif self.normalize == 'all_features':
            mean = np.mean(features)
            std = np.std(features)
            std = max(std, 1e-5)
            features = (features - mean) / std
        return features

    def _pad_features(self, features: np.ndarray) -> np.ndarray:
        if self.pad_to > 0:
            time_frames = features.shape[1]
            pad_amt = self.pad_to - (time_frames % self.pad_to)
            if pad_amt != self.pad_to:
                features = np.pad(
                    features,
                    ((0, 0), (0, pad_amt)),
                    mode='constant',
                    constant_values=0
                )
        return features

    def __call__(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = self.librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate
            )

        if self.preemph and self.preemph > 0:
            audio = np.concatenate([audio[:1], audio[1:] - self.preemph * audio[:-1]])

        stft = self.librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True,
            pad_mode='reflect'
        )

        power_spec = np.abs(stft) ** 2
        mel_spec = np.dot(self.mel_basis, power_spec)

        if self.log:
            mel_spec = np.log(mel_spec + self.log_zero_guard_value)

        mel_spec = self._normalize_features(mel_spec)
        mel_spec = self._pad_features(mel_spec)

        return mel_spec.astype(np.float32)
