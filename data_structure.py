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
    "n_sents_batch: number of sentences in a batch"
    n_sents_all = sum([instance.doc_l for instance in instances])
    n_batch = n_sents_all//batch_size

    n_instances = len(instances)
    n_instances_batch = math.ceil(n_instances/n_batch)

    batch_idxs = np.concatenate([np.random.permutation(np.arange(n_batch)) for i in range(n_instances_batch)])[:n_instances]
    sorted_instances = np.array(sorted(instances, key=lambda x: x.doc_l, reverse=True))
    batches = [(i, list(sorted_instances[np.where(batch_idxs == i)])) for i in range(n_batch)]
    
    assert sum([len(batch) for i, batch in batches]) == n_instances
    assert sum([sum([instance.doc_l for instance in batch]) for i, batch in batches]) == n_sents_all
    
    if iterator: batches = iter(batches)
    
    return batches

def get_test_batches(instances, batch_size, iterator=False):
    n_sents_all = sum([instance.doc_l for instance in instances])
    n_instances = len(instances)
    
    item_idx_instances = defaultdict(list)
    for instance in instances:
        item_idx_instances[instance.item_idx].append(instance)
    
    batches = [(i, instances) for i, instances in enumerate(item_idx_instances.values())]
    
    assert sum([len(batch) for i, batch in batches]) == n_instances
    assert sum([sum([instance.doc_l for instance in batch]) for i, batch in batches]) == n_sents_all
    
    if iterator: batches = iter(batches)
    
    return batches