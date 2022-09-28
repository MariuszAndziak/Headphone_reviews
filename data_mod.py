import numpy as np
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
import string
import re

class Preprocessing(object):
    '''
    Modified version of preprocessing class
    '''
    def __init__(self, text):
        self.text = text
        self.text = self.text.map(lambda x: str(x))

        self.oryg = text
    
    def __getitem__(self, index):
        return self.text[index]
    
    def __setitem__(self):
        return self.text
    
    def __len__(self):
        return self.text.shape[0]
    
    @property
    def type_(self):
        return type(self.text[0])

    def __repr__(self):
        return f'Dataset of type {self.type_} and length {len(self)}'
    
    def lower(self):
        self.text = self.text.map(lambda x: x.lower())
        return self.text
    
    def remove_digists(self):
        self.text = self.text.map(lambda x: re.sub(r'[\d+]', '', x))
        return self.text
    
    def remove_punctuation(self):
        self.text = self.text.map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
        return self.text
    
    def remove_stop_words(self):
        self.text = self.text.map(lambda x: ' '.join([word for word in x.split() if word not in STOP_WORDS]))
        return self.text
    
    def tokenize(self):
        self.text = self.text.map(lambda x: x.split())
        return self.text
    
    def revert_tokens(self):
        self.text = self.text.map(lambda x: ' '.join([word for word in x]))
        return self.text
    
    def restore_original(self):
        self.text = self.oryg
        self.text = self.text.map(lambda x: str(x))
        return self.text
    

def most_common_tokens(tokens, topn):
    cnt = Counter()
    tokens.map(cnt.update)
    top_tokens = [token[0] for token in cnt.most_common(topn)]

    def retrurn_top_tokens(tokens):
        return [int(token in tokens) for token in top_tokens]
    
    X = retrurn_top_tokens(tokens)
    X.columns = top_tokens

    return X

