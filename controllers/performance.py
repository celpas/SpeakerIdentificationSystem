from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
from numba import jit
from tqdm import tqdm
from terminaltables import AsciiTable
from collections import defaultdict
import numpy as np
import time
import logging
import pickle
import os
from controllers.vox_loader import VoxLoader
import gc

logger = logging.getLogger(__name__)


def init_context(**kwargs):
    global VL, CL, mapper_refs, mapper_refs_index, mapper_evls, auto_threshold, thresholds, thresholds_mapper, nshot

    VL = VoxLoader(path=kwargs['vox_path'])
    
    CL = kwargs['chunk_length']
    nshot = kwargs['nshot']
    
    mapper_refs_path = kwargs['mapper_refs']
    mapper_evls_path = kwargs['mapper_evls']
    mapper_refs_index_path = kwargs['mapper_refs_index']
    
    logging.info('kwargs: %s', kwargs)
    
    #used_keys = set()
    
    if not os.path.exists(mapper_refs_path) or \
       not os.path.exists(mapper_evls_path):
        logging.error('There is something of wrong in your paths!')
        return

    with open(mapper_refs_path, 'rb') as f:
        mapper_refs = pickle.load(f)
    with open(mapper_evls_path, 'rb') as f:
        mapper_evls = pickle.load(f)
    
    if mapper_refs_index_path is not None:
        with open(mapper_refs_index_path, 'rb') as f:
            mapper_refs_index = pickle.load(f)
    
    auto_threshold = kwargs['auto_threshold']

    thresholds_mapper = {30:0, 60:1, 90:2, 120:3, 150:4}
    thresholds = kwargs['thresholds']


def generate_speakers(steps=[60, 120, 180, 240, 300], repetitions=[1, 1, 1, 1, 1], seed=None, load_from_file=None):
    global speakers_per_experiment
    
    if load_from_file is None:
        speakers_per_experiment = {}

        speakers = []
        for i in range(repetitions[0]):
            rep_speakers = VL.get_speakers(num=300, sort=False, seed=None)
            speakers.append(rep_speakers)

        for i, num in enumerate(tqdm(steps)):
          speakers_per_experiment[num] = []
          for j in range(repetitions[i]):
              speakers_per_experiment[num].append(speakers[j])
    else:
        if not os.path.exists(load_from_file):
            logging.error('This file does not exists!')
        with open(load_from_file, 'rb') as f:
            speakers_per_experiment = pickle.load(f)
        logging.info('Loaded speakers from file: %s', load_from_file)
        
    return speakers_per_experiment


def show_performance_results(performance):
    t = []
    h = ['#', 'ACC', 'ACC_STD', 'FAR', 'FRR', 'MLR', 'EER', 'EER_STD', 'EER_T']
    t.append(h)
    
    for i, num in enumerate(speakers_per_experiment):
        wow1 = performance[i]['wow1']
        far = performance[i]['far']
        frr = performance[i]['frr']
        mlr = performance[i]['mlr']
        eer = performance[i]['eer']
        eer_threshold = performance[i]['eer_threshold']
        r = [num//2,
             np.round(wow1[0]*100,2),
             np.round(wow1[1]*100,2),
             np.round(far[0]*100,2),
             np.round(frr[0]*100,2),
             np.round(mlr[0]*100,2),
             np.round(eer[0]*100,2),
             np.round(eer[1]*100,2),
             np.round(eer_threshold*100,2)]
        t.append(r)

    table = AsciiTable(t)

    print('***********************************')
    print(table.table)
    print('***********************************')


def perform_experiment(num_speakers, cache):
    perf_per_num_speakers = defaultdict(list)
    num_repetitions = len(speakers_per_experiment[num_speakers])
    
    logging.info('Starting the experiment with %d speakers', num_speakers)
    
    for j in range(num_repetitions):
        # **************************
        # Prepare data
        speakers = speakers_per_experiment[num_speakers][j]

        speakers_per_set = num_speakers//2
        set1_of_known_speakers = speakers[:speakers_per_set]
        set2_of_unknown_speakers = speakers[-speakers_per_set:]
        
        ref, evl, all, exp, unique_labels = VL.get_splitting(
                speakers=set1_of_known_speakers,
                num_refs=10,
                snr_min=15,
                snr_max=80,
                num_eval=130,
                evaluate=set2_of_unknown_speakers,
                num_experiments=num_repetitions,
                reproducibility=True,
                nshot=nshot
        )
        
        unique_labels, ref_utts, ref_utts_idxs, evl_utts, evl_utts_idxs, evl_labels, evl_known = adapter(exp[j],
                                                                                                         evl,
                                                                                                         unique_labels,
                                                                                                         mapper_refs,
                                                                                                         mapper_evls,
                                                                                                         nshot)

        # **************************
        # K-Fold
        skf = StratifiedKFold(n_splits=10, random_state=j*100, shuffle=True)
        folds = {}
        for fold_id, (indices1, indices2) in enumerate(skf.split(evl_utts, evl_known)):
            folds[fold_id] = {}
            folds[fold_id]['utts'] = evl_utts[indices2]
            folds[fold_id]['utts_idxs'] = evl_utts_idxs[indices2]
            folds[fold_id]['utts_labels_speaker'] = evl_labels[indices2]
            folds[fold_id]['utts_labels_known'] = evl_known[indices2]
        
        # **************************
        # Threshold
        T = 0.728
        if speakers_per_set in thresholds_mapper and auto_threshold == False:
            T = thresholds[CL][thresholds_mapper[speakers_per_set]] 

        # **************************
        # Evaluate
        exp_results = []
        wow1_results = []
        far_results = []
        frr_results = []
        
        for fold_id in range(len(folds)):
            # ***********************************
            results, results_only_known = compute_results(folds[fold_id]['utts'],
                                                          folds[fold_id]['utts_labels_speaker'],
                                                          folds[fold_id]['utts_labels_known'],
                                                          unique_labels,
                                                          cache,
                                                          ref_utts_idxs,
                                                          folds[fold_id]['utts_idxs'])
            
            wow1, extra_data = compute_performance(results, results_only_known, T, auto_threshold=auto_threshold)
            # ***********************************
            perf_per_num_speakers['wow1'].append(wow1)
            perf_per_num_speakers['far'].append(extra_data['FAR'])
            perf_per_num_speakers['frr'].append(extra_data['FRR'])
            perf_per_num_speakers['mlr'].append(extra_data['MLR'])
            if auto_threshold:
                perf_per_num_speakers['eer'].append(extra_data['EER'])
                perf_per_num_speakers['eer_threshold'].append(extra_data['EER_threshold'])
            perf_per_num_speakers['results_only_known'].extend(results_only_known)
            perf_per_num_speakers['results'].extend(results)
    
    avg_performance = dict()
    avg_performance['results_y'] = np.array([x[0] for x in perf_per_num_speakers['results']]).astype('uint8')
    avg_performance['results_scores'] = np.array([x[1] for x in perf_per_num_speakers['results']]).astype('float16')
    
    #avg_performance['results_only_known_y'] = np.array([x[0] for x in perf_per_num_speakers['results_only_known']])
    #avg_performance['results_only_known_p'] = np.array([x[1] for x in perf_per_num_speakers['results_only_known']])
    #avg_performance['results_only_known_scores'] = np.array([x[2] for x in perf_per_num_speakers['results_only_known']]).astype('float16')
    
    m1, std1 = np.mean(perf_per_num_speakers['wow1']), np.std(perf_per_num_speakers['wow1'])
    m2, std2 = np.mean(perf_per_num_speakers['far']), np.std(perf_per_num_speakers['far'])
    m3, std3 = np.mean(perf_per_num_speakers['frr']), np.std(perf_per_num_speakers['frr'])
    m4, std4 = np.mean(perf_per_num_speakers['mlr']), np.std(perf_per_num_speakers['mlr'])
    avg_performance['wow1'] = (m1, std1)
    avg_performance['far'] = (m2, std2)
    avg_performance['frr'] = (m3, std3)
    avg_performance['mlr'] = (m4, std4)
    
    if auto_threshold:
        eer, eer_std = np.mean(perf_per_num_speakers['eer']), np.std(perf_per_num_speakers['eer'])
        avg_performance['eer'] = eer, eer_std
        
        eer_threshold = np.mean(perf_per_num_speakers['eer_threshold'])
        avg_performance['eer_threshold'] = eer_threshold
    
    logger.info('Num: %d / Wow1: %.2f / Std: %.2f / FAR: %.2f / FRR: %.2f / MLR: %.2f',
                num_speakers,
                m1,
                std1,
                m2,
                m3,
                m4
               )
    
    del results, results_only_known, perf_per_num_speakers
            
    return avg_performance


def adapter(ref_dict, evl_dict, unique_labels, mapper_refs, mapper_evls, nshot):
    unique_labels = np.array(unique_labels)
    
    ref_utts_per_speaker = ref_dict['uttsxspeaker']
    if nshot == 1:
        ref_utts = [VL.get_short_path(ref_utts_per_speaker[name][0]) for name in unique_labels]
        ref_utts_idxs = [mapper_refs[utt] for utt in ref_utts]
    elif nshot == 3:
        ref_utts = []
        ref_utts_idxs = []
        for name in unique_labels:
            indices = []
            for utt in ref_utts_per_speaker[name]:
                short_path_utt = VL.get_short_path(utt)
                indices.append(mapper_refs_index[short_path_utt])
                ref_utts.append(short_path_utt)
            indices = sorted(indices)
            unique_key = '%s_%d_%d_%d' % (name, indices[0], indices[1], indices[2])

            ref_utts_idxs.append(mapper_refs[unique_key])
            #used_keys.add(unique_key)
            
    ref_utts = np.array(ref_utts)
    ref_utts_idxs = np.array(ref_utts_idxs)
    
    evl_utts = np.array([utt for utt in mapper_evls])
    evl_labels = np.array([utt.split('/')[0] for utt in evl_utts])
    evl_utts_idxs = np.array([mapper_evls[utt] for utt in evl_utts])
    evl_known = np.array([1 if e1_label in unique_labels else 0 for e1_label in evl_labels])
    
    return unique_labels, ref_utts, ref_utts_idxs, evl_utts, evl_utts_idxs, evl_labels, evl_known


@jit(nopython=True, cache=True, parallel=False)
def compute_results(evl_utts,
                    evl_labels_speaker,
                    evl_labels_known,
                    unique_labels,
                    cache,
                    ref_utts_idxs,
                    evl_utts_idxs):

    evl_scores = cache[evl_utts_idxs, :]        # take specific rows (evls)
    evl_scores = evl_scores[:, ref_utts_idxs]   # take specific cols (refs)
    # Shape (num_evls, num_refs) e.g (1000, 30)
    
    evl_results = []
    evl_results_only_known = []
    
    for i in range(len(evl_utts)):
        is_known = evl_labels_known[i]
        label = evl_labels_speaker[i]
        
        argmax_score = np.argmax(evl_scores[i,:])
        score = evl_scores[i, argmax_score]
        predicted_label = unique_labels[argmax_score]

        evl_results.append((is_known, score))
        
        if is_known:
            evl_results_only_known.append((label, predicted_label, score))
            
    return evl_results, evl_results_only_known


def compute_performance(results, results_only_known, T, auto_threshold=True):
    accuracy = 0
    extra_data = dict()
    extra_data['FAR'] = 0.
    extra_data['FRR'] = 0.
    
    results_y = np.array([x[0] for x in results])
    results_s = np.array([x[1] for x in results])
    num_positive_class = len(np.where(results_y == 1)[0])
    num_negative_class = len(np.where(results_y == 0)[0])
    
    if auto_threshold:
        eer, eer_threshold, precision, recall = search_T(results_y, results_s)
        extra_data['EER'] = eer
        extra_data['EER_threshold'] = eer_threshold
        extra_data['precision'] = precision
        extra_data['recall'] = recall
        T = eer_threshold
    
    FAR = len(np.where((results_s >= T) & (results_y == 0))[0])/(num_negative_class)
    FRR = len(np.where((results_s < T) & (results_y == 1))[0])/(num_positive_class)
    extra_data['FAR'] = FAR
    extra_data['FRR'] = FRR
    
    total_only_known = len(results_only_known)
    assert len(results_only_known) == num_positive_class
    results_only_know_y = np.array([x[0] for x in results_only_known])
    results_only_know_p = np.array([x[1] for x in results_only_known])
    results_only_know_s = np.array([x[2] for x in results_only_known])    
    number_of_correct_preds  = len(np.where((results_only_know_y == results_only_know_p) & (results_only_know_s >= T))[0])
    number_of_correct_preds += len(np.where((results_y == 0) & (results_s < T))[0])
    accuracy = number_of_correct_preds/(num_positive_class+num_negative_class)
    
    extra_data['MLR'] = len(np.where((results_only_know_y != results_only_know_p) & (results_only_know_s >= T))[0])/num_positive_class

    return accuracy, extra_data


def search_T(y, p):
    fpr, tpr, thresholds = compute_roc_curve(y, p)
    eer, eer_threshold, precision, recall = compute_eer(fpr, tpr, thresholds)
    return eer, eer_threshold, precision, recall


def compute_roc_curve(y, scores):
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    return fpr, tpr, thresholds


def compute_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr
    
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = np.mean([fpr[eer_index], fpr[eer_index]])
    eer_threshold = thresholds[eer_index]

    precision = tpr[eer_index]/(tpr[eer_index]+fpr[eer_index])
    recall = tpr[eer_index]/(tpr[eer_index]+fnr[eer_index])
    
    return eer, eer_threshold, precision, recall