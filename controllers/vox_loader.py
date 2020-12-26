import os
import numpy as np
import pickle
from glob import glob
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class VoxLoader:
    
    def __init__(self, path='../datasets/vox1/testsetV3'):
        self.path = path
        self.load_meta()
        
    def get_speakers_m(self):
        speakers = sorted(list(self.meta.keys()))
        speakers_m = []
        for s in speakers:
            if self.meta[s]['gender'] == 'm':
                speakers_m.append(s)
        speakers_m = np.array(sorted(speakers_m))
        return speakers_m
    
    def get_speakers_f(self):
        speakers = sorted(list(self.meta.keys()))
        speakers_f = []
        for s in speakers:
            if self.meta[s]['gender'] == 'f':
                speakers_f.append(s)
        speakers_f = np.array(sorted(speakers_f))
        return speakers_f
    
    def get_speakers(self, num=150, seed=None, sort=False):
        speakers = sorted(list(self.meta.keys()))
        np.random.seed(seed)
        speakers = np.random.choice(speakers, num, replace=False)
        if sort:
            speakers = np.sort(speakers)
        return speakers
    
    def get_utts_by_speaker(self, speaker_id, snr_min=-80, snr_max=80, return_snr=False):
        utts = np.array(self.meta[speaker_id]['utts'])
        snrs = np.array(self.meta[speaker_id]['snr'])
        utts_filtered = utts[(snrs >= snr_min) & (snrs < snr_max)]
        snrs_filtered = snrs[(snrs >= snr_min) & (snrs < snr_max)]
        if return_snr:
            return (utts_filtered, snrs_filtered)
        return utts_filtered
    
    def get_snrs(self):
        snrs = dict()
        for speaker_id in self.meta:
            speaker_utts = self.meta[speaker_id]['utts']
            speaker_snrs = self.meta[speaker_id]['snr']
            for i, u in enumerate(speaker_utts):
                short_path = self.get_short_path(u)
                snrs[short_path] = speaker_snrs[i]
        return snrs
    
    def split_utts_by_speaker(self,
                              speaker_id,
                              num_refs=10,
                              snr_min=15,
                              snr_max=80,
                              num_eval=130):
        split_data = {'ref':{}, 'evl':{}}
        
        # ***********************
        # Get utterances
        speaker_utts, speaker_snrs = self.get_utts_by_speaker(speaker_id, return_snr=True)
        speaker_indices = np.arange(0, len(speaker_utts))

        # ***********************
        # References
        ref_indices = speaker_indices[(speaker_snrs >= snr_min) & (speaker_snrs < snr_max)]
        while(len(ref_indices) < num_refs):
            snr_min -= 1
            ref_indices = speaker_indices[(speaker_snrs >= snr_min) & (speaker_snrs < snr_max)]
            if snr_min < 0:
                snr_max += 1
        ref_indices = ref_indices[:num_refs]
        ref_utts = speaker_utts[ref_indices]
        ref_snrs = speaker_snrs[ref_indices]
        
        # ***********************
        # Evaluation
        evl_indices = np.setdiff1d(speaker_indices, ref_indices)
        
        # Sampling (Mode 1)
        evl_by_videos = defaultdict(list)
        for index in evl_indices:
            utterance = speaker_utts[index]
            speaker_id, video_id, _ = utterance.split('/')[-3:]
            evl_by_videos[video_id].append(index)
        
        new_evl_indices = []
        changes_flag = True
        while len(new_evl_indices) < num_eval and changes_flag == True:
            changes_flag = False
            for video_id in evl_by_videos:
                if len(new_evl_indices) == num_eval:
                    break
                if len(evl_by_videos[video_id]) > 0:
                    index = evl_by_videos[video_id][0]
                    evl_by_videos[video_id].remove(index)
                    new_evl_indices.append(index)
                    changes_flag = True
        new_evl_indices = np.array(new_evl_indices)
        evl_indices = new_evl_indices
        
        evl_utts = speaker_utts[evl_indices]
        evl_snrs = speaker_snrs[evl_indices]
        
        '''
        # Sampling (Mode 2)
        np.random.seed(123)
        np.random.shuffle(evl_indices)

        evl_indices = evl_indices[:num_eval]
        evl_utts = speaker_utts[evl_indices][:num_eval]
        evl_snrs = speaker_snrs[evl_indices][:num_eval]
        '''
        
        # ***********************
        # Check
        assert len(ref_utts) == num_refs
        assert len(evl_utts) == num_eval
        
        # ***********************
        # Return data
        split_data['ref']['utts'] = [self.get_full_path_wav(u) for u in ref_utts]
        split_data['ref']['snrs'] = ref_snrs
        split_data['ref']['labels'] = np.array([speaker_id for i in range(len(ref_utts))])
        split_data['evl']['utts'] = [self.get_full_path_wav(u) for u in evl_utts]
        split_data['evl']['snrs'] = evl_snrs
        split_data['evl']['labels'] = np.array([speaker_id for i in range(len(evl_utts))])
        
        return split_data
    
    def get_splitting(self,
                      num_speakers=150,
                      speakers=[],
                      speakers_seed=None,
                      num_refs=10,
                      snr_min=15,
                      snr_max=80,
                      num_eval=130,
                      evaluate=[],
                      num_experiments=100,
                      reproducibility=True,
                      nshot=1):
        ref_samples = {'utts':[], 'labels':[], 'num':{}, 'snrs':[]}
        evl_samples = {'utts':[], 'labels':[], 'num':{}, 'snrs':[]}
        all_samples = {'utts':[], 'labels':[]}
        exp_samples = [{'utts':[], 'labels':[], 'uttsxspeaker':{}} for i in range(num_experiments)]

        # ***********************
        # Get speakers
        if len(speakers) == 0:
            speakers = self.get_speakers(num=num_speakers, seed=speakers_seed, sort=True)
    
        # ***********************
        # Get utterances. Split them. Make esperiments.
        for s in speakers:
            logger.debug('Processing the speaker %s', s)
            
            split_data = self.split_utts_by_speaker(
                s,
                num_refs=num_refs,
                snr_min=snr_min,
                snr_max=snr_max,
                num_eval=num_eval
            )
            
            ref_samples['utts'].extend(split_data['ref']['utts'])
            ref_samples['snrs'].extend(split_data['ref']['snrs'])
            ref_samples['labels'].extend(split_data['ref']['labels'])
            ref_samples['num'][s] = len(split_data['ref']['utts'])
            
            evl_samples['utts'].extend(split_data['evl']['utts'])
            evl_samples['snrs'].extend(split_data['evl']['snrs'])
            evl_samples['labels'].extend(split_data['evl']['labels'])
            evl_samples['num'][s] = len(split_data['evl']['utts'])
            
            for i in range(num_experiments):
                np.random.seed(i*100)
                sampled_utts_indices = np.sort(np.random.choice(np.arange(0,num_refs), nshot, replace=False))
                sampled_utts = np.array(split_data['ref']['utts'])[sampled_utts_indices]
                exp_samples[i]['utts'].extend(sampled_utts)
                exp_samples[i]['labels'].extend([s for i in range(nshot)])
                exp_samples[i]['uttsxspeaker'][s] = sampled_utts
        
        # ***********************
        # Add more speakers to the evaluation set
        if len(evaluate) > 0:
            for s in evaluate:
                if s not in speakers:
                    split_data = self.split_utts_by_speaker(
                        s,
                        num_refs=num_refs,
                        snr_min=snr_min,
                        snr_max=snr_max,
                        num_eval=num_eval
                    )
                    evl_samples['utts'].extend(split_data['evl']['utts'])
                    evl_samples['snrs'].extend(split_data['evl']['snrs'])
                    evl_samples['labels'].extend(split_data['evl']['labels'])
                    evl_samples['num'][s] = len(split_data['evl']['utts'])
            
        # ***********************
        # Merge the references set and the evaluation set
        all_samples['utts'] = np.array(sorted(ref_samples['utts']+evl_samples['utts']))
        all_samples['labels'] = np.array(sorted(ref_samples['labels']+evl_samples['labels']))
        
        # ***********************
        # Return data
        unique_labels = np.sort(speakers)
        
        return ref_samples, evl_samples, all_samples, exp_samples, unique_labels

    def get_full_path_wav(self, short_filepath):
        return self.path+'/wav/'+short_filepath[:-3]+'wav'
    
    def get_short_path(self, filepath):
        p = Path(filepath).parts
        return '{}/{}/{}'.format(p[-3], p[-2], p[-1])
    
    def load_meta(self):
        meta_path = os.path.join(self.path, 'meta.pkl')
        if not os.path.exists(meta_path):
            logger.error('Meta-file not existing: %s', meta_path)
            return
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)
            logger.info('Meta-file loaded: %s', meta_path)
            logger.info('Number of speakers: %d', len(self.meta))
            logger.info('Meta-keys: %s', list(self.meta[list(self.meta.keys())[0]].keys()))
        return self.meta

    def save_meta(self, filename='meta.pkl', obj=None, backup=True):
        meta_path = os.path.join(self.path, filename)
        
        if os.path.exists(meta_path):
            counter = 2
            bck_path = os.path.join(self.path, filename[:-4]+str(counter)+'.pkl')
            while not os.path.exists(bck_path):
                counter += 1
                bck_path = os.path.join(self.path, filename[:-4]+str(counter)+'.pkl')
            copyfile(meta_path, bck_path)
            logger.info('Old meta-file saved to: %s', (bck_path))
        
        if obj is None:
            logger.error('Object is None')
            return
        with open(meta_path, 'wb') as f:
            pickle.dump(obj, f)
            logger.info('New meta-file saved to: %s', (meta_path))


