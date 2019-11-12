#coding:utf-8
import os
import sys
import math
import random
import itertools
from six.moves import zip_longest
#import cPickle
import _pickle as cPickle
import numpy as np
from collections import defaultdict

class Instance:
    def __init__(self):
        self.token_idxs = None
        self.goldLabel = -1
        self.idx = -1

    def _doc_len(self, idx):
        k = len(self.token_idxs)
        return k

    def _max_sent_len(self, idxs):
        k = max([len(sent) for sent in self.token_idxs])
        return k

def get_batches(instances, batch_size, iterator=False):
    iter_instances = iter(instances)
    n_batch = len(instances)//batch_size
    
    batches = [(i_batch, [next(iter_instances) for i_doc in range(batch_size)]) for i_batch in range(n_batch)]
    
    if iterator: batches = iter(batches)
    return batches