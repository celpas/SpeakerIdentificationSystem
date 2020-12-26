import logging
import time
import speech_recognition as sr
import numpy as np
import pickle
from controllers.audio_manager import AudioManager
from web_app.web_app_utils import *

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':
    RATE = 16000

    audio_manager = AudioManager(load_denoiser=False, load_deepxi=False)
    meta_object = load_meta_file()
    logging.info('Loaded %d identities from JSON file', len(meta_object))
    embs_cache = load_embs_cache()
    logging.info('Loaded %d embeddings', len(embs_cache))

    # ****************************************
    # Callback
    def callback(recognizer, audio):
        audio_data = np.frombuffer(audio.get_raw_data(),dtype=np.int16)
        audio_data = audio_data.astype(np.int16)

        embedding = audio_manager.preprocess_and_compute_embeddings(data=[audio_data])
        top3_scores, top3_predictions, T = recognize(meta_object, embs_cache, embedding)

        prediction = meta_object[top3_predictions[0]]['name'] if top3_scores[0] >= T else 'Someone'
        score = top3_scores[0]

        logging.info('%s is speaking (confidence score: %.2f)', prediction, score*100)
    # ****************************************

    # Initialize a Recognizer
    r = sr.Recognizer()

    # Audio source
    m = sr.Microphone(sample_rate=16000, chunk_size=8000)

    # Calibration within the environment
    # we only need to calibrate once, before we start listening
    logging.info("Calibrating...")
    with m as source:
        r.adjust_for_ambient_noise(source)  
    logging.info("Calibration finished")

    # start listening in the background
    # "stop_listening" is now a function that, when called, stops background listening
    logging.info("Recording...")
    stop_listening = r.listen_in_background(m, callback, phrase_time_limit=2)

    # do some unrelated computations for 60 seconds
    for _ in range(600): time.sleep(0.1)  # we're still listening even though the main thread is doing other things

    # calling this function requests that the background listener stop listening
    stop_listening(wait_for_stop=True)