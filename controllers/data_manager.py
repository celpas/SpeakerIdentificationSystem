import json
import config as cfg
import os
import logging
from utils import *
import random

class DataManager:

    def __init__(self):
        json_path = os.path.join(cfg.TEST_FOLDER, cfg.TEST_JSON)
        if os.path.exists(json_path):
            self.load_json()
            logging.info('Loaded %d identities from JSON file' % (len(self.data)))
        else:
            self.data = dict()
            self.save_json()
            logging.info('Created JSON file: %s' % (json_path))
        self.load_sentences()

    def get_next_senteces(self, num=1):
        indices = list(range(len(self.sentences)))
        indices = set(indices)-set(self.sentences_already_used)
        if len(indices) < num:
            indices = list(range(len(self.sentences)))
        print(len(self.sentences), len(indices), len(self.sentences_already_used))
        return random.sample(indices, num)

    def get_sentence_by_index(self, index):
        return self.sentences[index]

    def load_sentences(self):
        with open(os.path.join(cfg.TEST_FOLDER, cfg.TEST_SENTENCES), 'r', encoding='utf-8') as f:
            self.sentences = f.readlines()
            self.sentences = [str(x).rstrip() for x in self.sentences]
        logging.info('Loaded %d sentences' % (len(self.sentences)))
        self.sentences_already_used = list()
        for speaker in self.data:
            for utterance in self.data[speaker]['u']:
                if 'sentence' in self.data[speaker]['u'][utterance]:
                    self.sentences_already_used.append(self.data[speaker]['u'][utterance]['sentence'])
        logging.info('%d sentences not used yet' % (len(set(list(range(len(self.sentences))))-set(self.sentences_already_used))))

    def load_json(self):
        with open(os.path.join(cfg.TEST_FOLDER, cfg.TEST_JSON), 'r') as f:
            self.data = json.load(f)

    def save_json(self):
        ensure_dir(os.path.join(cfg.TEST_FOLDER, cfg.TEST_JSON))
        with open(os.path.join(cfg.TEST_FOLDER, cfg.TEST_JSON), 'w') as f:
            json.dump(self.data, f, indent=4, sort_keys=True)

    def get_gender(self, user_id):
        if user_id in self.data:
            return self.data[user_id]['g']
        else:
            return None

    def get_identity(self, user_id):
        if user_id in self.data:
            return self.data[user_id]
        else:
            return None

    def get_phone(self, user_id):
        if user_id in self.data:
            if 'p' in self.data[user_id]:
                return self.data[user_id]['p']
            else:
                return None
        else:
            return None

    def add_identity(self, user_id, gender, phone):
        if user_id not in self.data:
            self.data[user_id] = dict()
        self.data[user_id]['g'] = gender
        self.data[user_id]['p'] = phone
        if 'u' not in self.data[user_id]:
            self.data[user_id]['u'] = dict()

        if not os.path.exists(os.path.join(cfg.TEST_FOLDER, user_id)):
            os.mkdir(os.path.join(cfg.TEST_FOLDER, user_id))

        self.save_json()

        logging.info('The following identity has been saved: %s' % (self.data[user_id]))

    def add_utterance(self, user_id, utterance, distance, noise, sentence_index, mode):
        udata = dict()
        udata['distance'] = distance
        udata['noise'] = noise
        udata['sentence'] = sentence_index
        udata['mode'] = mode
        self.sentences_already_used.append(sentence_index)
        self.data[user_id]['u'][utterance] = udata
        self.save_json()

        logging.info('The following utterance sent by %s has been saved: %s' % (user_id, udata))

    def remove_identity(self, user_id):
        del self.data[user_id]
        self.save_json()

    def remove_utterance(self, user_id, utterance):
        del self.data[user_id]['u'][utterance]
        self.save_json()
