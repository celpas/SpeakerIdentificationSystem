import json
import config as cfg
import os
import logging
import random
import time

logger = logging.getLogger(__name__)


class SentencesManager:

    def __init__(self, sentences_path, used_sentences_path):
        self.sentences_path = sentences_path
        self.used_sentences_path = used_sentences_path
        self.temp_used_sentences = set()
        self.temp_used_sentences_time = {}
        
        if not os.path.exists(used_sentences_path):
            logger.info('Used sentences file not found... creating a new one!')
            self.used_sentences = {}
            self.used_sentences['indices'] = []
            self.save_used_sentences()
        else:
            self.load_used_sentences()
            logger.info('Load %d used sentences', len(self.used_sentences['indices']))

        if not os.path.exists(sentences_path):
            logger.error('Critical error! This path does not exists: %s', sentences_path)
            return
        else:
            self.load_sentences()

    def get_next_senteces(self, num=1):
        indices = list(range(len(self.sentences)))
        self.clear_temp()
        indices = (set(indices)-set(self.used_sentences['indices']))-self.temp_used_sentences
        if len(indices) < num:
            indices = list(range(len(self.sentences)))
        random.seed(None)
        return random.sample(indices, num)

    def get_sentence_by_index(self, index):
        return self.sentences[index]

    def load_sentences(self):
        with open(self.sentences_path, 'r', encoding='utf-8') as f:
            self.sentences = f.readlines()
            self.sentences = [str(x).rstrip() for x in self.sentences]
        logging.info('Loaded %d sentences' % (len(self.sentences)))
        logging.info('%d sentences not used yet' % (len(set(list(range(len(self.sentences))))-set(self.used_sentences['indices']))))

    def load_used_sentences(self):
        with open(self.used_sentences_path, 'r') as f:
            self.used_sentences = json.load(f)

    def save_used_sentences(self):
        if not os.path.exists(os.path.dirname(self.used_sentences_path)):
            os.makedirs(os.path.dirname(self.used_sentences_path))
        with open(self.used_sentences_path, 'w') as f:
            json.dump(self.used_sentences, f, indent=4, sort_keys=True)

    def mark_as_used(self, index):
        self.used_sentences['indices'].append(index)
        self.save_used_sentences()
        logger.info('Sentence %d marked as used', index)
        
    def clear_temp(self):
        current_time = time.time()
        for k in self.temp_used_sentences_time.copy():
            if (current_time-self.temp_used_sentences_time[k]) > 180:
                self.unmark_as_temp_used(k)
                del self.temp_used_sentences_time[k]
        
    def mark_as_temp_used(self, index):
        self.temp_used_sentences.add(index)
        self.temp_used_sentences_time[index] = time.time()
        logger.info('Sentence %d marked as temporarily used', index)
        
    def unmark_as_temp_used(self, index):
        if index in self.temp_used_sentences:
            self.temp_used_sentences.remove(index)