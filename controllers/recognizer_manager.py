import config as cfg
import numpy as np
import logging

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from models.recognizer.model import background_resnet

import scipy.io as sio
import scipy.io.wavfile
from python_speech_features import *

import itertools

logger = logging.getLogger(__name__)


class ToTensorTestInput(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor.
        Returns:
            Tensor: Converted feature.
        """
        if isinstance(np_feature, np.ndarray):
            if np_feature.ndim == 2:  # B == 1
                np_feature = np.expand_dims(np_feature, axis=0)  # Shape: (1, t, 40)
            np_feature = np.expand_dims(np_feature, axis=1)  # Shape: (B, 1, t, 40)
            assert np_feature.ndim == 4, 'Data is not a 4D tensor. Size: %s' % (np.shape(np_feature),)
            torch_tensor = torch.from_numpy(np_feature.transpose((0,1,3,2))).float()  # Shape: (B, 1, 40, t)
            return torch_tensor

        
class RecognizerManager:

    def __init__(self, device='cpu'):
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            logger.error('Cuda not available. Using CPU instead.')
        self.device = device
        self.load_model()

    def load_model(self):
        self.model = background_resnet(num_classes=5994)
        checkpoint = torch.load(cfg.WEIGHTS_RECOGNIZER, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        logging.debug('Recognizer loaded')

    def compute_embeddings(self, normalized_input):
        TT = ToTensorTestInput()
        normalized_input = TT(normalized_input)
        normalized_input = Variable(normalized_input)
        torch.set_num_threads(1)
        with torch.no_grad():
            activation = self.model(normalized_input)
        return activation  # Shape: (1, 256)

    def compare(self, emb1, emb2, is_numpy=False):
        if is_numpy:
            emb1 = torch.from_numpy(emb1)  # Shape: (B, 256)
            emb2 = torch.from_numpy(emb2)  # Shape: (B, 256)
        
        return F.cosine_similarity(emb1, emb2).data.cpu().numpy()  # Shape: (B, 1)
        
    def preprocess_audio(self, data=None, filepath=None):
        audio_data = sio.wavfile.read(filepath)[1] if filepath is not None else data
        logging.debug('Preprocessing of %s', filepath if filepath is not None else 'buffer')
        return self.compute_features(audio_data)

    def compute_features(self, audio_data):
        features, energies = fbank(audio_data, samplerate=cfg.SAMPLE_RATE, nfilt=cfg.FILTER_BANK, winlen=cfg.WINLEN,
                                   winstep=cfg.WINSTEP, winfunc=np.hamming)

        if cfg.USE_LOGSCALE:
            features = 20 * np.log10(np.maximum(features, 1e-5))

        if cfg.USE_DELTA:
            delta_1 = delta(features, N=1)
            delta_2 = delta(delta_1, N=1)

            features = self._normalize_frames(features, Scale=cfg.USE_SCALE)
            delta_1 = self._normalize_frames(delta_1, Scale=cfg.USE_SCALE)
            delta_2 = self._normalize_frames(delta_2, Scale=cfg.USE_SCALE)
            features = np.hstack([features, delta_1, delta_2])

        features = self._normalize_frames(features, Scale=cfg.USE_SCALE)

        return features.astype('float32')  # Shape: (t, 40)

    def _normalize_frames(self, features, Scale=False):
        if Scale:
            return (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 2e-12)
        else:
            return (features - np.mean(features, axis=0))
