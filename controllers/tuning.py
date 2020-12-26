from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import numpy as np
import sys
import torch.nn.functional as F
import torch
import pickle
from controllers.vox_loader import VoxLoader
import controllers.performance as pf
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)


def init_context(**kwargs):
    global VL, AM, speakers_per_experiment, mapper_refs, mapper_evls
    
    VL = VoxLoader(path=kwargs['vox_path'])

    speakers_file = kwargs['speakers_file']
    speakers_per_experiment = pf.generate_speakers([], [], load_from_file=speakers_file)
    for num_speakers in speakers_per_experiment.copy():
        if num_speakers != 30:
            del speakers_per_experiment[num_speakers]
    speakers_per_experiment.keys()
    
    with open(kwargs['mapper_refs'], 'rb') as f:
        mapper_refs = pickle.load(f)
    with open(kwargs['mapper_evls'], 'rb') as f:
        mapper_evls = pickle.load(f)
       
    pf.VL = VL


def perform_tuning(hyperparams, **kwargs):    
    logger.info('Current hyperparams: %s', hyperparams)
    
    accuracy = _perform_tuning(hyperparams, kwargs['num_speakers'], kwargs['num_repetitions'], kwargs['num_workers'])
    return accuracy
    

def _perform_tuning(hyperparams, num_speakers, num_repetitions, num_workers):
    embs_cache = {'refs': {}, 'evls':{}}
    # embs_cache['refs']['short_path'] = (1,256)
    # embs_cache['evls']['short_path'] = (x,256)

    scores_cache = np.zeros((len(mapper_evls), len(mapper_refs)), dtype=np.float32)
    # scores_cache[i,j] = similarity(r, e)

    performance = []
    
    for repetition in range(num_repetitions):
        # ******************
        # Get speakers
        speakers = speakers_per_experiment[30][repetition][:num_speakers]

        # ******************
        # Get splitting train/test
        ref_dict, evl_dict, experiments, unique_labels = split_train_test(speakers, num_speakers, num_repetitions)
        
        # ******************
        # Compute embeddings
        embs_cache = compute_embeddings(experiments[repetition]['utts'], evl_dict['utts'], hyperparams, embs_cache, num_workers)

        # ******************
        # Adapter
        unique_labels, ref_utts, ref_utts_idxs, evl_utts, evl_utts_idxs, evl_labels, evl_known = transformer(
            experiments[repetition],
            evl_dict,
            unique_labels,
            mapper_refs,
            mapper_evls
        )
        
        # ******************
        # Compute score
        scores_cache = update_scores_cache(ref_utts, evl_utts, mapper_refs, mapper_evls, embs_cache, scores_cache)
        
        # ******************
        # K-Fold
        skf = StratifiedKFold(n_splits=5, random_state=repetition*100, shuffle=True)
        folds = {}
        for fold_id, (indices1, indices2) in enumerate(skf.split(evl_utts, evl_known)):
            folds[fold_id] = {}
            folds[fold_id]['utts'] = evl_utts[indices2]
            folds[fold_id]['utts_idxs'] = evl_utts_idxs[indices2]
            folds[fold_id]['utts_labels_speaker'] = evl_labels[indices2]
            folds[fold_id]['utts_labels_known'] = evl_known[indices2]

        # ******************
        # Compute performance
        for fold_id in folds:
            evl_results_only_known = compute_scores(unique_labels,
                                                    ref_utts_idxs,
                                                    folds[fold_id]['utts'],
                                                    folds[fold_id]['utts_idxs'],
                                                    folds[fold_id]['utts_labels_speaker'],
                                                    folds[fold_id]['utts_labels_known'],
                                                    scores_cache)

            wow1 = compute_performance(evl_results_only_known)

            performance.append(wow1)
        
        if num_repetitions > 1:
            logger.info('Number of repetitions till now: %d', repetition+1)

    return np.mean(performance)
    

def split_train_test(speakers, num_speakers, num_repetitions):
    ref_dict, evl_dict, _, experiments, unique_labels = VL.get_splitting(
        speakers=speakers[:num_speakers],
        num_refs=10,
        snr_min=15,
        snr_max=80,
        num_eval=40,
        evaluate=[],
        num_experiments=num_repetitions,
        reproducibility=True,
        nshot=1
    )
    
    return ref_dict, evl_dict, experiments, unique_labels


def compute_embeddings(ref_utts, evl_utts, hyperparams, embs_cache, num_workers):
    ref_utts_to_process = [u for u in ref_utts if u not in embs_cache['refs']]
    evl_utts_to_process = [u for u in evl_utts if u not in embs_cache['evls']]
    
    snrs = VL.get_snrs()

    embs_cache = compute_embeddings_parallel(ref_utts_to_process, hyperparams, embs_cache, snrs=snrs, key='refs', num_workers=num_workers)
    embs_cache = compute_embeddings_parallel(evl_utts_to_process, hyperparams, embs_cache, snrs=snrs, key='evls', num_workers=num_workers)

    return embs_cache


def transformer(ref_dict, evl_dict, unique_labels, mapper_refs, mapper_evls, nshot=1):
    return pf.adapter(ref_dict, evl_dict, unique_labels, mapper_refs, mapper_evls, nshot)


def update_scores_cache(ref_utts, evl_utts, mapper_refs, mapper_evls, embs_cache, scores_cache):
    for e in evl_utts:
        i = mapper_evls[e]
        
        for r in ref_utts:
            j = mapper_refs[r]
            
            if scores_cache[i,j] == 0:
                ref_emb = embs_cache['refs'][r]
                evl_emb = embs_cache['evls'][e]
                score = compute_similarity(ref_emb, evl_emb)
                scores_cache[i,j] = score
            
    return scores_cache


def compute_similarity(ref_emb, evl_emb):
    e1_embedding = torch.from_numpy(evl_emb)
    e2_embedding = torch.from_numpy(ref_emb)
    e2_embedding = torch.cat(len(e1_embedding)*[e2_embedding])
    assert len(e1_embedding) == len(e2_embedding)
    score = F.cosine_similarity(e1_embedding, e2_embedding).cpu().numpy()
    score = np.mean(score)
    return score


def compute_scores(unique_labels, ref_utts_idxs, evl_utts, evl_utts_idxs, evl_labels_speaker, evl_labels_known, scores_cache):
    # Results
    _, evl_results_only_known = pf.compute_results(
        evl_utts,
        evl_labels_speaker,
        evl_labels_known,
        unique_labels,
        scores_cache,
        ref_utts_idxs,
        evl_utts_idxs
    )

    return evl_results_only_known


def compute_performance(results_only_known):
    total_only_known = len(results_only_known)
    results_only_know_y = np.array([x[0] for x in results_only_known])
    results_only_know_p = np.array([x[1] for x in results_only_known])
    results_only_know_s = np.array([x[2] for x in results_only_known])    
    number_of_correct_preds = len(np.where((results_only_know_y == results_only_know_p))[0])
    wow1 = number_of_correct_preds/total_only_known
    
    return wow1


def compute_embeddings_parallel(utts_to_process, hyperparams, embs_cache, snrs=None, key='refs', num_workers=8):
    global perform_computation
    
    utts_to_process = np.array(utts_to_process)
    
    # ********
    # Adjust the number of workers
    if len(utts_to_process) <= num_workers:
        num_workers = len(utts_to_process)
    if len(utts_to_process)/num_workers < 20 and len(utts_to_process) > 1:
        num_workers = num_workers//2

    # ********
    # Split the utterances
    utts_to_process_splits = np.array_split(utts_to_process, num_workers)

    # ********************************************************
    # Parallel handler
    def perform_computation(worker_id, filepaths, hyperparams, key, snrs):
        from controllers.audio_manager import AudioManager
        AM = AudioManager(load_deepxi=False)
        
        memory = {}
        for i, u in enumerate(filepaths):
            short_path = u.split('/')[-3:]
            short_path = '%s/%s/%s' % (short_path[0], short_path[1], short_path[2])
            
            # *********************
            # Denoiser
            if hyperparams['denoiser']:
                
                snr = snrs[short_path]
                growth = hyperparams['growth']
                translate = hyperparams['translate']
                dry = np.exp(-(snr+translate)/growth)
                
                '''
                snr = snrs[short_path]
                
                growth = hyperparams['growth']
                translate = hyperparams['translate']
                #dry = np.exp(-(snr+40)/growth)
                dry = np.exp(-(snr+translate)/growth)
                #dry = (-0.3/60)*(snr-60)
                '''
                #dry = 0.5

                audio_data = AM.enhance_from_file(u, dry=dry)
                audio_segment = AM.convert_numpy_to_pydub(audio_data)
            else:
                audio_segment = AM.load_audio_segment(u)
            
            '''
            if key == 'refs':
                audio_data = AM.enhance_from_file(u, dry=0)
                audio_segment = AM.convert_numpy_to_pydub(audio_data)
            else:
                audio_segment = AM.load_audio_segment(u)
            '''
            
            # *********************
            # Silence removal
            if hyperparams['silence_removal']:
                audio_segment = AM.strip_silence(audio_segment,
                                                 hyperparams['min_len'],
                                                 hyperparams['thresh'],
                                                 hyperparams['padding'])

            audio_data = AM.convert_pydub_to_numpy(audio_segment)

            # *********************
            # Make chunks
            if key == 'evls':
                chunk_length = hyperparams['chunk_length']
                data = AM.make_chunks(data=audio_data, chunk_length=chunk_length*1000, overlap=0.5, out_numpy=True)  # Shape: (B, 16000*T)
                
                if 'max_num_chunks' in hyperparams:
                    if len(data) > hyperparams['max_num_chunks']:
                        np.random.seed(123)
                        indices = np.random.choice(np.arange(0, len(data)), hyperparams['max_num_chunks'], replace=False)
                        data = np.array(data)[indices]
            else:
                data = np.array([audio_data])  # Shape: (1, x)

            # *********************
            # Compute embeddings
            e = AM.preprocess_and_compute_embeddings(data=data).cpu().numpy()  # Shape: (B, 256)

            # *********************
            # Save
            memory[short_path] = e

            # *********************
            # Logging
            if (i+1) % (len(filepaths)//4) == 0:
                logger.info('Worker: %d - Remaining: %d', worker_id, len(filepaths)-(i+1))
                
        return memory
    # ********************************************************

    # ********
    # Start the workers
    logger.info('Starting %d workers to compute the embeddings of %d utts (%s)', num_workers, len(utts_to_process), key)
    results = Parallel(n_jobs=num_workers, backend='multiprocessing', prefer='processes')(delayed(
        perform_computation)(i+1, utts_to_process_splits[i], hyperparams, key, snrs) for i in range(num_workers))

    # ********
    # Put in cache
    num_processed = 0
    for r in results:
        for u in r:
            embs_cache[key][u] = r[u]
            num_processed += 1

    assert num_processed == len(utts_to_process)
   
    return embs_cache