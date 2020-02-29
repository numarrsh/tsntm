#coding: utf-8
import os
import pdb
import time
from collections import defaultdict, Counter

import numpy as np
from scipy.special import gammaln

# import warnings

# warnings.filterwarnings('error')

class Topic:
    def __init__(self, idx, sibling_idx, parent, depth, config):
        self.idx = idx
        self.sibling_idx = sibling_idx
        self.parent = parent
        self.children = []
        self.depth = depth
        self.cnt_doc = 0
        self.cnt_words = np.zeros(config.n_vocab) # Number of Words over Documents
        self.config=config
        self.set_prob_words()
    
    def sample_child(self, doc, train=True):
        p_child = self.get_probs_child(doc)
        child_index = np.random.multinomial(1, p_child).argmax()
        if self.config.verbose: print('Depth: ', self.depth, 'p_child: ', p_child, 'selected:', child_index)
        
        if child_index < len(self.children):
            child = self.children[child_index]
        else:
            child = self.get_new_child()
            if train: self.children += [child]
        return child
    
    def init_sample_child(self, train=True):
        logits_child_prior = self.get_logits_child_prior()
        s_child = np.exp(logits_child_prior)
        p_child = s_child/np.sum(s_child)
        
        child_index = np.random.multinomial(1, p_child).argmax()
        if self.config.verbose: print('Depth: ', self.depth, 'p_child: ', p_child, 'selected:', child_index)

        if child_index < len(self.children):
            child = self.children[child_index]
        else:
            child = self.get_new_child()
            if train: self.children += [child]
        return child
    
    def get_probs_child(self, doc):
        logits_child_prior = self.get_logits_child_prior()
        logits_child_likelihood = self.get_logits_child_likelihood(doc)
        logits_child = logits_child_prior + logits_child_likelihood

        logits_child -= np.max(logits_child)
        s_child = np.exp(logits_child)

        if np.sum(s_child) > 0:
            p_child = s_child/np.sum(s_child)
            p_child = p_child.astype(np.float64)
        else:
            p_child = np.zeros_like(logits_child, dtype=np.float64)
            p_child[np.argmax(s_child)] = 1.
            
        return p_child
    
    def get_logits_child_prior(self):
        s_child_prior = [child.cnt_doc for child in self.children]
        s_child_prior += [self.config.gam]
        logits_child_prior = np.log(s_child_prior)
        return logits_child_prior
    
    def get_logits_child_likelihood(self, doc):
        if len(self.children) > 0:
            children_cnt_words = np.concatenate([np.array([child.cnt_words for child in self.children]), np.zeros([1, self.config.n_vocab])], 0) # (Children+1) x Vocabulary
        else:
            children_cnt_words = np.zeros([1, self.config.n_vocab]) # (Children+1) x Vocabulary
        
        cnt_words_doc = doc.cnt_words[None, :] # 1 x Vocabulary

        logits_child_likelihood = gammaln(np.sum(children_cnt_words, -1) + self.config.n_vocab*self.config.eta) \
                            - np.sum(gammaln(children_cnt_words + self.config.eta), -1) \
                            - gammaln(np.sum(children_cnt_words + cnt_words_doc, -1) + self.config.n_vocab*self.config.eta) \
                            + np.sum(gammaln(children_cnt_words + cnt_words_doc + self.config.eta), -1)
        return logits_child_likelihood
    
    def get_new_child(self):
        sibling_idx = max([child.sibling_idx for child in self.children]) + 1 if len(self.children) > 0 else 1
        idx = self.idx + '-' + str(sibling_idx)
        depth = self.depth+1
        child = Topic(idx=idx, sibling_idx=sibling_idx, parent=self, depth=depth, config=self.config)        
        return child
        
    def get_children(self):
        child = self.get_new_child()
        children = self.children + [child]
        return children
    
    def delete_topic(self):
        self.parent.children.remove(self)
        
    def set_prob_words(self):
        cnt_words = self.cnt_words + self.config.eta
        self.prob_words = cnt_words / np.sum(cnt_words)
    
class Doc:
    def __init__(self, idx, words, bow, config):
        self.idx = idx
        self.words = words
        self.cnt_words = bow
        self.config = config
        assert len(words) == np.sum(bow)
        
        self.topics = [] # Depth
        self.word_depths = [] # Word Indices
        self.depth_cnt_words = np.zeros([self.config.n_depth, self.config.n_vocab])
                
    def get_probs_depth(self, word_idx):
        s_docs = np.sum(self.depth_cnt_words, -1) + self.config.alp # Depth
        s_words = np.array([topic.cnt_words[word_idx] for topic in self.topics]) + self.config.eta # Depth
        z_words = np.array([np.sum(topic.cnt_words) for topic in self.topics]) + self.config.n_vocab*self.config.eta # Depth
        assert s_docs.shape == s_words.shape == z_words.shape

        s_depths = s_docs*s_words/z_words
        p_depths = s_depths/np.sum(s_depths) # Depth
        return p_depths
    
    def sample_depth(self, word_idx):
        prob_depths = self.get_probs_depth(word_idx)
        word_depth = np.argmax(np.random.multinomial(1, prob_depths))
        return word_depth
    
def init(train_docs, dev_docs, test_docs, topic_root):
    init_doc_topics(docs=train_docs, topic_root=topic_root)
    init_word_topics(docs=train_docs)
    init_doc_topics(docs=dev_docs, topic_root=topic_root, train=False)
    init_word_topics(docs=dev_docs, train=False)        
    init_doc_topics(docs=test_docs, topic_root=topic_root, train=False)
    init_word_topics(docs=test_docs, train=False)    
    assert_sum_cnt_words(topic_root, train_docs)
    
def sample(train_docs, dev_docs, test_docs, topic_root):
    sample_doc_topics(docs=train_docs, topic_root=topic_root)
    sample_word_topics(docs=train_docs)
    sample_doc_topics(docs=dev_docs, topic_root=topic_root, train=False)
    sample_word_topics(docs=dev_docs, train=False)    
    sample_doc_topics(docs=test_docs, topic_root=topic_root, train=False)
    sample_word_topics(docs=test_docs, train=False)    
    assert_sum_cnt_words(topic_root, train_docs)

def sample_each(docs, topic_root, train):
    sample_doc_topics(docs=docs, topic_root=topic_root, train=train)
    sample_word_topics(docs=docs, train=train)
    
def init_doc_topics(docs, topic_root, train=True):
    for doc in docs:
        topic = topic_root
        doc.topics = [topic]
        if train: topic.cnt_doc += 1 # increment count of docs

        for depth in range(1, doc.config.n_depth):
            topic = topic.init_sample_child(train=train)
            doc.topics += [topic]
            if train: topic.cnt_doc += 1 # increment count of docs
                
def sample_doc_topics(docs, topic_root, train=True):
    for doc in docs:
        if train:
            for depth in range(1, doc.config.n_depth):
                topic = doc.topics[depth]
                topic.cnt_doc -= 1 # decrement count of docs
                assert topic.cnt_doc >= 0
                topic.cnt_words -= doc.depth_cnt_words[depth] # decrement count of words
                assert np.min(topic.cnt_words) >= 0

                if topic.cnt_doc == 0: 
                    topic.delete_topic()
                    assert np.sum(topic.cnt_words) == 0

        topic = topic_root
        doc.topics = [topic]
        for depth in range(1, doc.config.n_depth):
            topic = topic.sample_child(doc, train=train)
            doc.topics += [topic]
            if train: topic.cnt_doc += 1 # increment count of docs
            if train: topic.cnt_words += doc.depth_cnt_words[depth] # increment count of words

def init_word_topics(docs, train=True):
    for doc in docs:
        if doc.idx % 10000 == 0: print(doc.idx, end=' ')
        for word_index, word_idx in enumerate(doc.words):
            # sample depth of word
            new_depth = doc.sample_depth(word_idx)
            new_topic = doc.topics[new_depth]
            
            # increment count of words
            doc.depth_cnt_words[new_depth, word_idx] += 1
            if train: new_topic.cnt_words[word_idx] += 1
            doc.word_depths.append(new_depth) # for reference when sampling
            
        assert len(doc.word_depths) == len(doc.words) == np.sum(doc.depth_cnt_words)
        
def sample_word_topics(docs, train=True):
    for doc in docs:
        if doc.idx % 10000 == 0: print(doc.idx, end=' ')
        for word_index, word_idx in enumerate(doc.words):
            # refer depth of word
            old_depth = doc.word_depths[word_index]
            old_topic = doc.topics[old_depth]
            
            # decrement count of words
            doc.depth_cnt_words[old_depth, word_idx] -= 1
            if train: old_topic.cnt_words[word_idx] -= 1
            
            # sample depth of word
            new_depth = doc.sample_depth(word_idx)
            new_topic = doc.topics[new_depth]
            
            # increment count of words
            doc.depth_cnt_words[new_depth, word_idx] += 1
            if train: new_topic.cnt_words[word_idx] += 1
            doc.word_depths[word_index] = new_depth # for sample
            
        assert len(doc.word_depths) == len(doc.words) == np.sum(doc.depth_cnt_words)

def get_perplexity(docs, topic_root, verbose=False):
    def set_prob_words(topic):
        topic.set_prob_words()
        for topic_child in topic.children:
            set_prob_words(topic_child)
            
    # set Probabilty of Words
    set_prob_words(topic_root)

    logit_docs = []
    for doc in docs:
        # Path Probability for each document
        topic = topic_root
        probs_paths= [{topic: 1.}]
        for depth in range(1, topic_root.config.n_depth):
            probs_path = {}
            for topic, prob_path in probs_paths[-1].items():
                topics_child = topic.get_children()
                probs_child = topic.get_probs_child(doc)
                probs_path_child = prob_path * probs_child
                for topic_child, prob_path_child in zip(topics_child, probs_path_child):
                    probs_path[topic_child] = prob_path_child
            probs_paths.append(probs_path)    

        all_topics = []
        for probs_path in probs_paths:
            all_topics += list(probs_path.keys())

        # topic probability for each depth
        probs_depths_topics = np.array([[probs_path[topic] if topic in probs_path else 0. for topic in all_topics] for probs_path in probs_paths])
        # depth probability for each word
        probs_words_depths = np.array([doc.get_probs_depth(word_idx) for word_idx in doc.words])
        # topic probability for each depth
        probs_words_topics = probs_words_depths.dot(probs_depths_topics) # n_doc x n_topic

        probs_topics_words = np.array([topic.prob_words for topic in all_topics]) # n_topic x n_vocab
        probs_words_bow = probs_words_topics.dot(probs_topics_words) # n_doc x n_vocab
        logit_words = np.log(probs_words_bow[np.arange(len(doc.words)), doc.words])
        logit_doc = np.mean(logit_words)
        logit_docs.append(logit_doc)

    perplexity = np.exp(-np.mean(logit_docs))
    if verbose: print('Perplexity= %.1f' % perplexity)
    return perplexity

def get_topic_specialization(docs, topic_root, verbose=False):
    norm_bow = np.sum([doc.cnt_words for doc in docs], 0)
    norm_vec = norm_bow / np.linalg.norm(norm_bow)

    def add_spec(topic, depth_specs=None):
        if depth_specs is None: depth_specs = defaultdict(list)
        topic_vec = topic.prob_words / np.linalg.norm(topic.prob_words)
        topic_spec = 1 - topic_vec.dot(norm_vec)
        depth_specs[topic.depth+1].append(topic_spec)
        for child in topic.children:
            depth_specs = add_spec(child, depth_specs)
        return depth_specs

    depth_specs = add_spec(topic_root)
    depth_spec = {depth: np.mean(specs) for depth, specs in depth_specs.items()}
    
    print('Topic Specialization:', end=' ')
    for depth, spec in depth_spec.items():
        print('depth %i: %.2f' % (depth, spec), end=' ')
    print('')
    
    return depth_spec

def get_hierarchical_affinities(topic_root, verbose=False):
    def get_topics(topic, topics=None):
        if topics is None: topics=[]
        topics.append(topic)
        for child in topic.children:
            topics = get_topics(child, topics)
        return topics
    
    def get_cos_sim(parent_childs):
        parent_child_bows = {parent: np.array([child.prob_words/np.linalg.norm(child.prob_words) for child in childs]) for parent, childs in parent_childs.items()}
        cos_sim = np.mean([np.mean((parent.prob_words/np.linalg.norm(parent.prob_words)).dot(child_bows.T)) for parent, child_bows in parent_child_bows.items()])
        return cos_sim    

    topics = get_topics(topic_root)
    second_parents = [topic for topic in topics if topic.depth==1]
    third_childs = [topic for topic in topics if topic.depth==2]
    second_parent_childs = {parent: parent.children for parent in second_parents}
    second_parent_unchilds = {parent: [child for child in third_childs if child not in second_parent_childs[parent]] for parent in second_parents}
    if sum(len(unchilds) for unchilds in second_parent_unchilds.values()) > 0:
        child_cos_sim = get_cos_sim(second_parent_childs)
        unchild_cos_sim = get_cos_sim(second_parent_unchilds)
    else:
        child_cos_sim = get_cos_sim(second_parent_childs)
        unchild_cos_sim = 0
    if verbose: print('Hierarchical Affinity: child %.2f, non-child %.2f'%(child_cos_sim, unchild_cos_sim))
    return child_cos_sim, unchild_cos_sim

def get_docs(instances, config):
    docs_bow = [instance.bow for instance in instances]
    docs_raw = [[[bow_index]*int(doc_bow[bow_index]) for bow_index in np.where(doc_bow > 0)[0]] for doc_bow in docs_bow]
    docs_words = [[idx for idxs in doc for idx in idxs] for doc in docs_raw]
    docs = [Doc(idx=doc_idx, words=doc_words, bow=doc_bow, config=config) for doc_idx, (doc_words, doc_bow) in enumerate(zip(docs_words, docs_bow))]
    return docs

def get_freq_tokens_ncrp(topic, idx_to_word, bow_idxs, topic_freq_words=None):
    freq_words = [idx_to_word[bow_idxs[bow_index]] for bow_index in np.argsort(topic.cnt_words)[::-1][:10]]
    if topic_freq_words is None: topic_freq_words = {}
    topic_freq_words[topic] = freq_words
    print('  '*topic.depth, topic.idx, topic.cnt_doc, np.sum(topic.cnt_words), ' '.join(freq_words))
    for topic in topic.children:
        topic_freq_words = get_freq_tokens_ncrp(topic, idx_to_word, bow_idxs, topic_freq_words)
    return topic_freq_words

def get_sum_cnt_words(topic):
    cnt_words = np.sum(topic.cnt_words)
    for child in topic.children:
        cnt_words += get_sum_cnt_words(child)
    return cnt_words

def assert_sum_cnt_words(topic_root, docs):
    sum_cnt_words = get_sum_cnt_words(topic_root)
    assert sum_cnt_words == sum([len(doc.words) for doc in docs])
    
def nearly_equal(val, thre):
    return (val > thre-1e-5) and (val < thre+1e-5)