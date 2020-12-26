import os
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import logging
from .denoiser_manager import DenoiserManager
from .recognizer_manager import RecognizerManager
from .snr_manager import SNRManager
import IPython.display as ipd
import scipy.io as sio
import scipy.io.wavfile
from matplotlib import pyplot as plt
import librosa.display

logger = logging.getLogger(__name__)


class AudioManager:
    
    def __init__(self, device='cpu', load_recognizer=True, load_denoiser=True, load_deepxi=True):
        self.device = device
        if load_recognizer:
            self.recognizer = RecognizerManager(device=device)
        if load_denoiser:
            self.denoiser = DenoiserManager(device=device)
        if load_deepxi:
            self.deepxi = SNRManager(device=device)
        
    def load_audio_data(self, filepath):
        if not os.path.exists(filepath):
            logger.error('This file does not exists: %s', filepath)
            return
        return sio.wavfile.read(filepath)[1]
    
    def load_audio_segment(self, filepath):
        if not os.path.exists(filepath):
            logger.error('This file does not exists: %s', filepath)
            return
        return AudioSegment.from_wav(filepath)
    
    def make_chunks(self, data=None, filepath=None, chunk_length=3000, overlap=0.5, out_numpy=True):
        logger.debug('Obtaining the chunks of %s', filepath if filepath is not None else 'buffer')
        
        step_size = chunk_length*overlap
        audio_segment = self.load_audio_segment(filepath) if filepath is not None else self.convert_numpy_to_pydub(data)
        audio_length = len(audio_segment)

        logger.debug('Audio length: %.2f / Step size: %d / Chunk length: %d',
                     audio_length/1000,
                     step_size/1000,
                     chunk_length/1000)

        if chunk_length >= audio_length:
            while chunk_length > len(audio_segment):
                audio_segment = audio_segment+audio_segment
            logger.debug('Audio length is lower than chunk length. The audio segment has been duplicated.')
            return np.array([self.convert_pydub_to_numpy(audio_segment[:chunk_length])], dtype=np.int16)

        start_points = np.arange(0, audio_length-step_size, step_size)
        logger.debug('Intervals: %s', ['%.1f-%.1f' % (sp/1000, (sp+chunk_length)/1000) for sp in start_points])

        while (start_points[-1]+chunk_length) > len(audio_segment):
            audio_segment = audio_segment+audio_segment
        audio_length = len(audio_segment)   

        chunks = [audio_segment[sp:sp+chunk_length] for sp in start_points]
        logger.debug('Number of chunks: %d', len(chunks))

        if out_numpy:
            chunks = np.array([self.convert_pydub_to_numpy(c) for c in chunks], dtype=np.int16)

        return chunks
    
    def preprocess_and_compute_embeddings(self, data=[], filepaths=[]):
        if len(filepaths) > 0:
            f = [self.recognizer.preprocess_audio(filepath=p) for p in filepaths]
        else:
            f = [self.recognizer.preprocess_audio(data=d) for d in data]
        lengths = [x.shape[0] for x in f]
        if len(set(lengths)) > 1:
            logger.error('The files must have the same length!')
            return
        f = np.array(f, dtype=np.float32)  # Shape: (B, t, 40)
        e = self.recognizer.compute_embeddings(f)  # Shape (B, 256)
        return e
    
    def compute_similarity(self, e1, e2):
        is_numpy = True if type(e1) == np.ndarray else False
        return self.recognizer.compare(e1, e2, is_numpy=is_numpy)
    
    def enhance(self, audio_data):
        return self.denoiser.enhance(audio_data)
    
    def enhance_from_file(self, filepath, dry=0):
        if not os.path.exists(filepath):
            logger.error('This file does not exists: %s', filepath)
            return
        return self.denoiser.enhance_from_file(filepath, dry=dry)
    
    def enhance_from_files(self, filepaths=[], dry=0, max_length=0):
        if len(filepaths) == 0:
            logger.error('No files to process!')
            return
        for f in filepaths:
            if not os.path.exists(f):
                logger.error('This file does not exists: %s', filepath)
                return
        return self.denoiser.enhance_from_files(filepath, dry=dry, max_length=max_length)
    
    def compute_snr(self, filepaths=[]):
        return self.deepxi.compute_snr(filepaths)
    
    def play(self, data=None, filepath=None):
        if data is not None:
            return ipd.Audio(data=data, rate=16000)
        if filepath is not None:
            return ipd.Audio(filename=filepath, rate=16000)
        
    def strip_silence(self, audio_segment, min_len=1000, thresh=25, padding=100):
        chunks = split_on_silence (
            audio_segment, 
            min_silence_len = min_len,
            silence_thresh = audio_segment.dBFS-thresh,
            keep_silence = padding
        )
        if len(chunks) == 0:
            return audio_segment
        else:
            new_audio_segment = chunks[0]
            if len(chunks) > 1:
                for c in chunks[1:]:
                    new_audio_segment += c
            return new_audio_segment
    
    def convert_pydub_to_numpy(self, audio_segment):
        return np.array(audio_segment.get_array_of_samples()).astype(np.int16)
    
    def convert_numpy_to_pydub(self, audio_data):
        audio_data = audio_data.astype(np.int16)
        audio_segment = AudioSegment(
            audio_data.tobytes(), 
            frame_rate=16000,
            sample_width=audio_data.dtype.itemsize, 
            channels=1
        )
        return audio_segment
    
    def show_waveplot(self, data=None, filepath=None):
        fig = plt.figure(figsize=(14,4))
        plt.rcParams["font.size"] = "15"
        audio_data = self.load_audio_data(filepath)/32768 if filepath is not None else data/32768
        librosa.display.waveplot(audio_data, sr=16000)