import os
import unicodedata
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.utils import murmurhash3_32
from collections import Counter

from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from itertools import product
import pickle
import json

def load_sparse_csr(filename):
    loader = np.load(filename)
    matrix = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    return matrix, loader['context'].item(0) if 'context' in loader else None

def save_sparse_csr(filename, matrix, context=None):
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'context': context
    }
    np.savez(filename, **data)


def load_dict_pickle(dict_path):
    pickle_in = open(dict_path, "rb")
    return pickle.load(pickle_in)


def save_dict_pickle(data_dict, path):
    pickle_out = open(path, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()


def load_dict_json(dict_path):
    file_obj = open(dict_path, "rb")
    return json.loads(file_obj.read())


def save_dict_json(data_dict, path):
    file_obj = open(path, "w+")
    file_obj.write(json.dumps(data_dict, indent=4))
    file_obj.close()


def get_file_path_list(path):
    file_names = []
    for dir_path, _, dir_file_names in os.walk(path):
        file_names = [os.path.join(dir_path, file_name) for file_name in dir_file_names if is_wiki_file(file_name)]
    file_names.sort()
    return file_names


def is_wiki_file(filename):
    return "wiki" in filename


def get_file_batches_from_path(path, num_of_batches):
    file_names = get_file_path_list(path)
    return divide_list_into_batches(file_names, num_of_batches)



def divide_list_into_batches(lst, num_of_batches):
    len_lst = len(lst)
    lst_batches = [lst[index:index+num_of_batches] if index + num_of_batches < len_lst else lst[index:] for index in range(0, len_lst, num_of_batches)]
    return lst_batches



def hash(token, num_buckets):
    return murmurhash3_32(token, positive=True) % num_buckets


def unicode_encode(text):
    return unicodedata.normalize('NFD', text)


STOPWORDS = list(set(stopwords.words('english')))
port_stem = PorterStemmer()
regex_tokenizer = RegexpTokenizer(r'[\w-]{3,}')

def flatten_list(lst):
    return [item for sublist in lst for item in sublist]

def tokenise_string(text):
    return sent_tokenize(text)


def regex_tokenizer_string(text):
    return regex_tokenizer.tokenize(text)


def stopwords_stem(tokens):
    processed_tokens = [port_stem.stem(token.lower()) for token in tokens if token not in STOPWORDS]
    return processed_tokens


def process_tokens(sent_tokens):
    final_tokens = []
    for sent_token in sent_tokens:
        final_tokens += stopwords_stem(sent_token)
    return final_tokens


def get_sentence_matrix(text, hash_size):
    sents = [stopwords_stem(sent) for sent in text]
    counts = [Counter([hash(w, hash_size) for w in sent]) for sent in sents]
    row, col, data = [], [], []
    for i, count in enumerate(counts):
        col.extend(count.keys())
        row.extend([i]*len(count))
        data.extend(count.values())
    vector = sp.csr_matrix(
        (data, (row, col)), shape=(len(counts), hash_size)
    )
    vector = vector.log1p()
    return vector


def closest_sentences(query, text, hash_size, num_of_sents=5):
    scores = get_sentence_matrix(text, hash_size).dot(get_sentence_matrix(query, hash_size).transpose())
    scores = scores.toarray().squeeze()
    if scores.shape == ():
        return {}
    else:
        inds = pd.Series(scores).nlargest(num_of_sents).index
        return {i:text[i] for i in inds}


def unigram_word_pair(claim, evidence):
    sents = process_tokens(evidence)
    processed_duplicates = set([word for word in stopwords_stem(regex_tokenizer.tokenize(claim)) if word in sents])
    return Counter(processed_duplicates)


def bigram_word_pair(claim, evidence):
    sents = process_tokens(evidence)
    return Counter([(w1, w2) for w1, w2 in product(stopwords_stem(regex_tokenizer.tokenize(claim)), sents)])