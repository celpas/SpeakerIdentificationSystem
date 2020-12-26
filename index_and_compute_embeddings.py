from controllers.audio_manager import AudioManager
from web_app.web_app_utils import *
import config as cfg
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

audio_manager = AudioManager(load_denoiser=cfg.ENHANCE)

index_and_compute_embeddings(audio_manager)