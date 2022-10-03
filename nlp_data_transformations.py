import numpy as np
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
import string
import re
from nltk import sent_tokenize


class Preprocessing(object):
    '''
    Preprocessing class for text manipulation
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
    


def most_common_tokens(tokens: pd.Series, topn: int) -> pd.DataFrame:
    '''
    Get most common tokens from the data

    Args:
        tokens: series on lists of tokens
        topn: number of how many most common tokens we want to show
    
    Returns:
        A pandas dataframe with columns names representin topn tokens and
        rows with 0 or 1 correcponding to presence or absence of a particular
        token
    '''
    cnt = Counter()
    tokens.map(cnt.update) 

    # list of top tokens
    top_tokens = [token[0] for token in cnt.most_common(topn)]

    def return_top_tokens(tokens: pd.Series)-> list:
        '''
        Return a list of 0s and 1s corresponding to the fact whether a token
        is a top_token
        '''
        return [int(token in tokens) for token in top_tokens]
    
    X = tokens.apply(return_top_tokens).apply(pd.Series)
    X.columns = top_tokens

    return X


def use_vectorizer(text: pd.Series, vectorizer: type, vectorizer_kwargs: dict):
    '''
    Make a vector representation using inputed vectorizer and its arguments 
    '''
    vec = vectorizer(**vectorizer_kwargs)
    X = vec.fit_transform(text).toarray()

    return X


def use_word2vec_model(model, tokens):
    X = tokens.map(lambda x: np.mean([model.wv[w] for w in x if w in model.wv], axis = 0))
    default_vector = X[False == X.isnull()].mean()
    return np.stack(X.map(lambda x: x if str(x) != 'nan' else default_vector))


def use_doc2vec_model(tokens):
    return TaggedDocument(words = tokens, tags = [tokens.index.tolist()])