import os
import numpy as np
import logging
import torch
import config as cfg
from models.denoiser.demucs import Demucs
from models.denoiser.utils import deserialize_model
import scipy.io as sio
import scipy.io.wavfile

logger = logging.getLogger(__name__)


class DenoiserManager:
    
    def __init__(self, device='cpu', serialized=False):
        self.weights = cfg.WEIGHTS_DENOISER
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            logger.error('cuda not available. Using CPU instead.')
        self.device = device
        if not os.path.exists(self.weights):
            logger.error('Weights not found!')
        else:
            self.load_model(serialized)
    
    def load_model(self, serialized=False):
        if serialized:
            pkg = torch.load(self.weights, map_location=torch.device(self.device))
            self.model = deserialize_model(pkg)
        else:
            self.model = Demucs(hidden=64)
            state_dict = torch.load(self.weights, map_location=torch.device(self.device))
            self.model.load_state_dict(state_dict)
        logger.debug('Denoiser loaded')
        
    def enhance(self, audio_data, dry=0):
        audio_data = audio_data.astype('float32')
        audio_data = audio_data/32768
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=0)
            audio_data = np.expand_dims(audio_data, axis=1)  # Shape: (1, 1, N)
        elif audio_data.ndim == 2:
            audio_data = np.expand_dims(audio_data, axis=1)  # Shape: (1, 1, N)
        # else shape: (B, 1, N)
        audio_tensor = torch.tensor(audio_data)
        torch.set_num_threads(1)
        with torch.no_grad():
            audio_sign = audio_tensor.to(self.device)
            estimate = self.model(audio_sign)
            estimate = (1 - dry) * estimate + dry * audio_sign
        estimate = estimate.cpu().numpy()*32768
        estimate = estimate.astype(np.int16)
        if estimate.shape[0] == 1:
            estimate = estimate[0][0]  # Shape: (N)
        else:
            estimate = np.squeeze(estimate, axis=1)  # Shape: (B, N)
        return estimate
    
    def enhance_from_files(self, filepaths=[], dry=0, max_length=0):
        audio_ndarr = []
        lengths = []
        for p in filepaths:
            a = sio.wavfile.read(p)[1]
            a = a if max_length == 0 else a[:max_length]
            audio_ndarr.append(a)
            lengths.append(len(a))
        if len(set(lengths)) > 1:
            logger.debug('The files are not the same length')
            enhanced_signals = []
            for a in audio_ndarr:
                enhanced_signals.append(self.enhance(a))
            return enhanced_signals
        else:
            logger.debug('The files are the same length')
            audio_ndarr = np.array(audio_ndarr, dtype=np.int16)
            return self.enhance(audio_ndarr, dry)
    
    def enhance_from_file(self, filepath, dry=0):
        sr, audio_data = sio.wavfile.read(filepath)
        return self.enhance(audio_data, dry)