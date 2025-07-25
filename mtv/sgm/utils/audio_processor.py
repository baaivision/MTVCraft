# pylint: disable=C0301
'''
This module contains the AudioProcessor class and related functions for processing audio data.
It utilizes various libraries and models to perform tasks such as preprocessing, feature extraction,
and audio separation. The class is initialized with configuration parameters and can process
audio files using the provided models.
'''
import math
import os

import librosa
import numpy as np
import torch
from audio_separator.separator import Separator
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

from sgm.models.wav2vec import Wav2VecModel
from .util import resample_audio


class AudioProcessor:
    """
    AudioProcessor is a class that handles the processing of audio files.
    It takes care of preprocessing the audio files, extracting features
    using wav2vec models, and separating audio signals if needed.

    :param sample_rate: Sampling rate of the audio file
    :param fps: Frames per second for the extracted features
    :param wav2vec_model_path: Path to the wav2vec model
    :param only_last_features: Whether to only use the last features
    :param audio_separator_model_path: Path to the audio separator model
    :param audio_separator_model_name: Name of the audio separator model
    :param cache_dir: Directory to cache the intermediate results
    :param device: Device to run the processing on
    """
    def __init__(
        self,
        sample_rate,
        wav2vec_model_path,
        only_last_features,
        device="cuda",
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.audio_encoder = Wav2VecModel.from_pretrained(wav2vec_model_path, local_files_only=True).to(device=device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.only_last_features = only_last_features
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path, local_files_only=True)


    def preprocess(self, wav_file: str, clip_length: int=-1, fps: float=24.0):
        """
        Preprocess a WAV audio file by separating the vocals from the background and resampling it to a 16 kHz sample rate.
        The separated vocal track is then converted into wav2vec2 for further processing or analysis.

        Args:
            wav_file (str): The path to the WAV file to be processed. This file should be accessible and in WAV format.

        Raises:
            RuntimeError: Raises an exception if the WAV file cannot be processed. This could be due to issues
                        such as file not found, unsupported file format, or errors during the audio processing steps.

        Returns:
            torch.tensor: Returns an audio embedding as a torch.tensor
        """
        vocal_audio_file=wav_file

        # 2. extract wav2vec features
        speech_array, sampling_rate = librosa.load(vocal_audio_file, sr=self.sample_rate)
        audio_feature = np.squeeze(self.wav2vec_feature_extractor(speech_array, sampling_rate=sampling_rate).input_values)
        seq_len = math.ceil(len(audio_feature) / self.sample_rate * fps)
        audio_length = seq_len

        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)

        if clip_length>0 and (seq_len - 1) % (clip_length - 1) != 0:

            audio_feature = torch.nn.functional.pad(audio_feature, (0, int((clip_length - 1 - (seq_len - 1) % (clip_length - 1)) * (self.sample_rate // fps))), 'constant', 0.0)
            seq_len += clip_length - 1 - (seq_len - 1) % (clip_length - 1)
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=seq_len, output_hidden_states=True)
        assert len(embeddings) > 0, "Fail to extract audio embedding"
        if self.only_last_features:
            audio_emb = embeddings.last_hidden_state.squeeze()
        else:
            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.cpu().detach()

        return audio_emb, audio_length

    def get_embedding(self, wav_file: str, fps: float):
        """preprocess wav audio file convert to embeddings

        Args:
            wav_file (str): The path to the WAV file to be processed. This file should be accessible and in WAV format.

        Returns:
            torch.tensor: Returns an audio embedding as a torch.tensor
        """
        speech_array, sampling_rate = librosa.load(
            wav_file, sr=self.sample_rate)
        assert sampling_rate == 16000, "The audio sample rate must be 16000"
        audio_feature = np.squeeze(self.wav2vec_feature_extractor(
            speech_array, sampling_rate=sampling_rate).input_values)
        seq_len = math.ceil(len(audio_feature) / self.sample_rate * fps)

        audio_feature = torch.from_numpy(
            audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature, seq_len=seq_len, output_hidden_states=True)
        assert len(embeddings) > 0, "Fail to extract audio embedding"

        if self.only_last_features:
            audio_emb = embeddings.last_hidden_state.squeeze()
        else:
            audio_emb = torch.stack(
                embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.cpu().detach()

        return audio_emb

    def close(self):
        """
        TODO: to be implemented
        """
        return self

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
