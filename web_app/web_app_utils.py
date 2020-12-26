import os
import pickle
import json
import logging
import subprocess
from pathlib import Path
from glob import glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from controllers.audio_manager import AudioManager
import config as cfg

logger = logging.getLogger(__name__)


def get_refs_path(identity_name: str):
    return os.path.join(cfg.SAMPLES_DIR, 'refs', identity_name)


def get_evls_path(identity_name: str):
    return os.path.join(cfg.SAMPLES_DIR, 'evls', identity_name)


def get_refs(identity_name: str):
    refs_path = get_refs_path(identity_name)
    if not os.path.exists(refs_path):
        return []
    return sorted(glob(os.path.join(refs_path, '*')))


def get_evls(identity_name: str):
    evls_path = get_evls_path(identity_name)
    if not os.path.exists(evls_path):
        return []
    return sorted(glob(os.path.join(evls_path, '*')))


def get_identities():
    if not os.path.exists(os.path.join(cfg.SAMPLES_DIR, 'refs')):
        return []
    else:
        return sorted(os.listdir(os.path.join(cfg.SAMPLES_DIR, 'refs')))


def load_meta_file():
    meta_path = os.path.join(cfg.SAMPLES_DIR, 'meta.json')
    if not os.path.exists(meta_path):
        save_meta_file({})
    with open(meta_path, 'r') as f:
        meta_object = json.load(f)
    return meta_object


def save_meta_file(object: dict):
    if not os.path.exists(cfg.SAMPLES_DIR):
        os.makedirs(cfg.SAMPLES_DIR)
    save_to = os.path.join(cfg.SAMPLES_DIR, 'meta.json')
    with open(save_to, 'w') as f:
        json.dump(object, f, indent = 4)
    logger.info('JSON file saved to: %s', save_to)


def convert_to_wav(input_filepath: str, output_filepath: str):
    if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath))
    if cfg.CONVERTER == 'ffmpeg':
        subprocess.call('ffmpeg -i %s -acodec pcm_s16le -ac 1 -ar 16000 %s' % (input_filepath, output_filepath))
    elif cfg.CONVERTER == 'sox':
        subprocess.call('sox %s -r 16000 -c 1 -b 16 %s' % (input_filepath, output_filepath), shell=True)
    else:
        logger.error('No valid converter...')


def load_embs_cache():
    cache_path = os.path.join(cfg.SAMPLES_DIR, 'embs.pkl')
    if not os.path.exists(cache_path):
        save_embs_cache({})
    with open(cache_path, 'rb') as f:
        embs_object = pickle.load(f)
    return embs_object


def save_embs_cache(object: dict):
    if not os.path.exists(cfg.SAMPLES_DIR):
        os.makedirs(cfg.SAMPLES_DIR)
    save_to = os.path.join(cfg.SAMPLES_DIR, 'embs.pkl')
    with open(save_to, 'wb') as f:
        pickle.dump(object, f)
    logger.info('Embeddings cache saved to: %s', save_to)


def get_short_path(full_path: str):
    parts = Path(full_path).parts[-3:]
    return '%s/%s/%s' % (parts[0], parts[1], parts[2])  # refs/full_name/0000.wav
                                                        # evls/full_name/0000.wav

def recognize(meta_object: dict, embs_cache: dict, e1_embedding: np.ndarray):
    unique_labels = list(meta_object.keys())
    scores = []
    
    enrolled = 0
    for name in unique_labels:
        refs = [os.path.join('refs', name, x) for x in meta_object[name]['refs']]
        if len(refs) >= 1:
            name_scores = []
            for r in refs:
                e2_embedding = embs_cache[get_short_path(r)]
                name_scores.append(cosine_similarity(e1_embedding, e2_embedding)[0][0])
            scores.append(np.mean(name_scores))
            enrolled += 1
        else:
            scores.append(0)
    
    scores = np.array(scores)
    unique_labels = np.array(unique_labels)
    T = (0.7194+(0.0118*np.log(enrolled+1e-12)))  # adaptive threshold
    
    top3_indices = np.argsort(scores)[::-1]
    top3_scores = scores[top3_indices[:3]]
    top3_predictions = unique_labels[top3_indices[:3]]

    return top3_scores, top3_predictions, T


def compute_embedding(audio_manager: AudioManager, full_path: str):
    if cfg.ENHANCE:
        audio_data = audio_manager.enhance_from_file(full_path, cfg.ENHANCE_DRY)
        audio_segment = audio_manager.convert_numpy_to_pydub(audio_data)
    else:
        audio_segment = audio_manager.load_audio_segment(full_path)
    audio_segment = audio_manager.strip_silence(audio_segment, min_len=300, thresh=25, padding=100)
    audio_data = audio_manager.convert_pydub_to_numpy(audio_segment)
    embedding = audio_manager.preprocess_and_compute_embeddings(data=[audio_data])
    return embedding


def index_and_compute_embeddings(audio_manager: AudioManager):
    identities = os.listdir(os.path.join(cfg.SAMPLES_DIR, 'refs'))
    meta_object = load_meta_file()
    embs_cache = {}
    logger.info('JSON file contains %d identities', len(meta_object))
    logger.info('%d performed the enrollment', len(identities))

    for i, identity_name in enumerate(identities):
        logger.info('%d/%d) Computing the embeddings of %s', i, len(identities), identity_name)
        if identity_name not in meta_object:
            meta_object[identity_name] = {}
            meta_object[identity_name]['name'] = identity_name
        refs = sorted(glob(os.path.join(cfg.SAMPLES_DIR, 'refs', identity_name, '*')))
        evls = sorted(glob(os.path.join(cfg.SAMPLES_DIR, 'evls', identity_name, '*')))
        meta_object[identity_name]['refs'] = [os.path.basename(r) for r in refs]
        meta_object[identity_name]['evls'] = [os.path.basename(e) for e in evls]
        for p in refs+evls:
            embedding = compute_embedding(audio_manager, p)
            embs_cache[get_short_path(p)] = embedding

    if len(meta_object) > len(identities):
        remove = meta_object.keys()-set(identities)
        for r in remove:
            del meta_object[r]

    save_meta_file(meta_object)
    save_embs_cache(embs_cache)
