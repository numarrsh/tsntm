#coding: utf-8
import os
import pdb
import time
from collections import defaultdict, Counter

import numpy as np
from scipy.special import gammaln

class Topic:
    def __init__(self, idx, sibling_idx, parent, depth, config):
        self.idx = idx
        self.sibling_idx = sibling_idx
        self.parent = parent
        self.children = []
        self.depth = depth
        
        self.cnt_doc_topic = 0
        self.cum_doc_topic = 0
        
        self.cnt_words = np.zeros(config.n_vocab)
        self.cum_words = np.zeros(config.n_vocab)
        
        self.config=config
    
    def sample_child(self, cnt_words=None, init=False, train=True):
        assert (cnt_words is None) == init
        
        if self.cnt_doc_topic == 0 or self.depth == self.config.n_depth:
            return self
        else:
            probs_child = self.get_probs_child(cnt_words, init=init)
            child_index = np.random.multinomial(1, probs_child).argmax()

            if child_index < len(self.children):
                child = self.children[child_index]
                return child.sample_child(cnt_words, init=init, train=train)
            elif child_index == len(self.children):
                return self
            else:
                assert child_index == len(self.children) + 1
                child = self.get_new_child(train)
                return child.sample_child(cnt_words, init=init, train=train)
        
    def get_probs_child(self, cnt_words, init=False):
        if init:
            logits_child = self.get_logits_child_prior()
        else:
            logits_child_prior = self.get_logits_child_prior()
            logits_child_likelihood = self.get_logits_child_likelihood(cnt_words)
            logits_child = logits_child_prior + logits_child_likelihood

        logits_child -= np.max(logits_child)
        s_child = np.exp(logits_child)
        if np.sum(s_child) > 0:
            probs_child = s_child/np.sum(s_child)
            probs_child = probs_child.astype(np.float64)
        else:
            probs_child = np.zeros_like(logits_child, dtype=np.float64)
            probs_child[np.argmax(s_child)] = 1.
            
        return probs_child
        
    def get_logits_child_prior(self):
        s_child_prior = [child.cum_doc_topic for child in self.children]
        s_child_prior += [self.cnt_doc_topic]
        s_child_prior += [self.config.gam**(self.depth+1)]

        logits_child_prior = np.log(s_child_prior)
        return logits_child_prior
    
    def get_logits_child_likelihood(self, cnt_words_doc_topic):
        if len(self.children) > 0:
            children_cum_words = np.array([child.cum_words for child in self.children]) # (Children) x Vocabulary
            children_cum_words = np.concatenate([children_cum_words, self.cum_words[None, :], np.zeros([1, self.config.n_vocab])], 0) # (Children+Self+NewChildren) x Vocabulary
        else:
            children_cum_words = np.concatenate([self.cnt_words[None, :], np.zeros([1, self.config.n_vocab])], 0) # (Self+NewChildren) x Vocabulary

        logits_child_likelihood = gammaln(np.sum(children_cum_words, -1) + self.config.n_vocab*self.config.eta**(self.depth+1)) \
                            - np.sum(gammaln(children_cum_words + self.config.eta**(self.depth+1)), -1) \
                            - gammaln(np.sum(children_cum_words + cnt_words_doc_topic, -1) + self.config.n_vocab*self.config.eta**(self.depth+1)) \
                            + np.sum(gammaln(children_cum_words + cnt_words_doc_topic + self.config.eta**(self.depth+1)), -1)
        return logits_child_likelihood
    
    def get_new_child(self, train=True):
        sibling_idx = max([child.sibling_idx for child in self.children]) + 1 if len(self.children) > 0 else 1
        idx = self.idx + '-' + str(sibling_idx)
        depth = self.depth+1
        child = Topic(idx=idx, sibling_idx=sibling_idx, parent=self, depth=depth, config=self.config)        
        if train: 
            self.children += [child]
        return child
    
    def increment_cnt(self, cnt_words):
        def increment_cum(topic, cnt_words):
            topic.cum_doc_topic += 1
            topic.cum_words += cnt_words
            if topic.parent is not None: increment_cum(topic.parent, cnt_words=cnt_words)
        
        self.cnt_doc_topic += 1
        self.cnt_words += cnt_words
        increment_cum(self, cnt_words=cnt_words)
            
    def decrement_cnt(self, cnt_words):
        def decrement_cum(topic, cnt_words):
            topic.cum_doc_topic -= 1
            topic.cum_words -= cnt_words
            if topic.parent is not None: decrement_cum(topic.parent, cnt_words=cnt_words)
                
        self.cnt_doc_topic -= 1
        self.cnt_words -= cnt_words
        decrement_cum(self, cnt_words=cnt_words)
            
    def increment_cnt_words(self, word_idx):
        def increment_cum_words(topic, word_idx):
            topic.cum_words[word_idx] += 1
            if topic.parent is not None: increment_cum_words(topic.parent, word_idx=word_idx)
                
        self.cnt_words[word_idx] += 1
        increment_cum_words(self, word_idx=word_idx)
        
    def decrement_cnt_words(self, word_idx):
        def decrement_cum_words(topic, word_idx):
            topic.cum_words[word_idx] -= 1
            if topic.parent is not None: decrement_cum_words(topic.parent, word_idx=word_idx)
                
        self.cnt_words[word_idx] -= 1
        decrement_cum_words(self, word_idx=word_idx)
        
    def increment_cnt_doc(self):
        def increment_cum_doc(topic):
            topic.cum_doc_topic += 1
            if topic.parent is not None: increment_cum_doc(topic.parent)
        
        self.cnt_doc_topic += 1
        increment_cum_doc(self)

    def get_all_topics(self):
        topics = [self]
        for child in self.children:
            topics += child.get_all_topics()
        return topics
    
    def get_probs_all_topics(self, cnt_words_doc_topic):
        all_topics = self.get_all_topics()
        
        s_topic_prior = [topic.cnt_doc_topic for topic in all_topics]
        s_topic_prior += [self.config.gam]
        logits_topic_prior = np.log(s_topic_prior)
        
        topic_cum_words = np.array([topic.cum_words + topic.config.eta**(topic.depth) for topic in all_topics])
        topic_cum_words = np.concatenate([topic_cum_words, np.zeros([1, self.config.n_vocab]) + self.config.eta], 0)
        logits_topic_likelihood = gammaln(np.sum(topic_cum_words, -1)) \
                            - np.sum(gammaln(topic_cum_words), -1) \
                            - gammaln(np.sum(topic_cum_words + cnt_words_doc_topic, -1)) \
                            + np.sum(gammaln(topic_cum_words + cnt_words_doc_topic), -1)
        
        logits_topic = logits_topic_prior + logits_topic_likelihood

        logits_topic -= np.max(logits_topic)
        s_topic = np.exp(logits_topic)
        if np.sum(s_topic) > 0:
            probs_topic = s_topic/np.sum(s_topic)
            probs_topic = probs_topic.astype(np.float64)
        else:
            probs_topic = np.zeros_like(logits_topic, dtype=np.float64)
            probs_topic[np.argmax(s_topic)] = 1.
        
        return probs_topic
        
    def delete_topic(self):
        self.parent.children.remove(self)        
        
    def set_prob_words(self):
        cum_words = self.cum_words + self.config.eta**self.depth
        self.prob_words = cum_words / np.sum(cum_words)
    
class Doc:
    def __init__(self, idx, words, bow, config, topic_root):
        self.idx = idx
        self.words = words
        self.cnt_words = bow
        self.config = config
        assert len(words) == np.sum(bow)
        
        self.index_cnt_words = [] # n_indices_topic x n_vocab
        self.word_indices = [] # n_words
        self.topics = [] # n_indices_topic
        
        self.topic_root = topic_root

    def sample_index(self, word_idx=None, init=False):
        probs_index = self.get_probs_index(word_idx, init=init)
        index = np.random.multinomial(1, probs_index).argmax()     
        return index
    
    def get_probs_index(self, word_idx, init=False):
        if init:
            logits_index = self.get_logits_index_prior()
        else:
            logits_index_prior = self.get_logits_index_prior()
            logits_index_likelihood = self.get_logits_index_likelihood(word_idx)
            logits_index = logits_index_prior + logits_index_likelihood
        
        logits_index -= np.max(logits_index)
        s_index = np.exp(logits_index)
        
        if np.sum(s_index) > 0:
            probs_index = s_index/np.sum(s_index)
            probs_index = probs_index.astype(np.float64)
        else:
            probs_index = np.zeros_like(logits_index, dtype=np.float64)
            probs_index[np.argmax(s_index)] = 1.
        
        return probs_index
                
    def get_logits_index_prior(self):
        if len(self.index_cnt_words) == 0:
            s_index_prior = [self.config.alp]
        else:
            s_index_prior = np.sum(self.index_cnt_words, 1)
            s_index_prior = np.append(s_index_prior, self.config.alp)
        logits_index_prior = np.log(s_index_prior)
        return logits_index_prior
    
    def get_logits_index_likelihood(self, word_idx):
        def get_logit_new_index_likelihood(word_idx):
            all_topics = self.topic_root.get_all_topics()

            s_child_prior = [topic.cnt_doc_topic for topic in all_topics]
            s_child_prior += [self.topic_root.config.gam]
            p_child_prior = s_child_prior / np.sum(s_child_prior)

            s_child_likelihood = np.array([topic.cum_words[word_idx] + topic.config.eta**(topic.depth) for topic in all_topics] + [self.topic_root.config.eta])
            z_child_likelihood = np.array([np.sum(topic.cum_words) + topic.config.n_vocab*(topic.config.eta**(topic.depth)) for topic in all_topics] + [self.topic_root.config.n_vocab*self.topic_root.config.eta])

            p_child_likelihood = s_child_likelihood / z_child_likelihood
            logit_new_index_likelihood = np.log(p_child_prior.dot(p_child_likelihood))

            return logit_new_index_likelihood
    
        s_index_likelihood = np.array([topic.cum_words[word_idx] + topic.config.eta**(topic.depth) if topic is not None else self.topic_root.config.eta for topic in self.topics])
        z_index_likelihood = np.array([np.sum(topic.cum_words) + topic.config.n_vocab*(topic.config.eta**(topic.depth)) if topic is not None else self.topic_root.config.n_vocab*self.topic_root.config.eta for topic in self.topics])

        logits_index_likelihood = np.log(s_index_likelihood/z_index_likelihood)
        logit_new_index_likelihood = get_logit_new_index_likelihood(word_idx)
        logits_index_likelihood = np.append(logits_index_likelihood, logit_new_index_likelihood)

        return logits_index_likelihood
    
def init_word_topics(docs, topic_root, train=True):
    for doc in docs:
        for word_index, word_idx in enumerate(doc.words):
            new_index = doc.sample_index(init=True)

            if new_index == len(doc.index_cnt_words):
                if len(doc.index_cnt_words) == 0:
                    doc.index_cnt_words = np.zeros([1, doc.config.n_vocab])
                else:
                    doc.index_cnt_words = np.concatenate([doc.index_cnt_words, np.zeros([1, doc.config.n_vocab])], 0)

            doc.index_cnt_words[new_index, word_idx] += 1
            doc.word_indices.append(new_index)
            
def init_doc_topics(docs, topic_root, train=True):    
    for doc in docs:
        for index, cnt_words in enumerate(doc.index_cnt_words):
            new_topic = topic_root.sample_child(init=True, train=train)
            doc.topics.append(new_topic)

            if train: new_topic.increment_cnt(cnt_words=cnt_words) # increment count of doc_topic & words
                
def init(train_docs, dev_docs, test_docs, topic_root):
    init_word_topics(docs=train_docs, topic_root=topic_root)
    init_doc_topics(docs=train_docs, topic_root=topic_root)
    init_word_topics(docs=dev_docs, topic_root=topic_root, train=False)
    init_doc_topics(docs=dev_docs, topic_root=topic_root, train=False)
    init_word_topics(docs=test_docs, topic_root=topic_root, train=False)
    init_doc_topics(docs=test_docs, topic_root=topic_root, train=False)
    
    assert_sum_cnt_words(train_docs, topic_root)
    
def sample_word_topics(docs, topic_root, train=True):
    for doc in docs:
        for word_index, word_idx in enumerate(doc.words):
            # refer index of word
            old_index = doc.word_indices[word_index]
            old_topic = doc.topics[old_index]

            # decrement count of words
            doc.index_cnt_words[old_index, word_idx] -= 1
            if train: old_topic.decrement_cnt_words(word_idx=word_idx)
            assert doc.index_cnt_words[old_index, word_idx] >= 0
            assert old_topic.cnt_words[word_idx] >= 0

            # sample topic_index of word
            new_index = doc.sample_index(word_idx, init=False)

            if new_index == len(doc.index_cnt_words):
                cnt_words = np.zeros([1, doc.config.n_vocab])
                doc.index_cnt_words = np.concatenate([doc.index_cnt_words, cnt_words], 0)
                new_topic = topic_root.sample_child(cnt_words, init=False, train=train)
                if train: new_topic.increment_cnt_doc()
                doc.topics.append(new_topic)

            new_topic = doc.topics[new_index]

            # increment count of words
            doc.index_cnt_words[new_index, word_idx] += 1
            if train: new_topic.increment_cnt_words(word_idx=word_idx)
            doc.word_indices[word_index] = new_index
            
def sample_doc_topics(docs, topic_root, train=True):
    for doc in docs:
        for index, cnt_words in enumerate(doc.index_cnt_words):
            old_topic = doc.topics[index]

            # continue if no word is assigned to the index
            if old_topic is None: 
                assert np.sum(cnt_words) == doc.word_indices.count(index) == 0
                continue

            # increment count of docs and words
            if train: 
                old_topic.decrement_cnt(cnt_words)
                assert old_topic.cnt_doc_topic >= 0
                assert np.min(old_topic.cnt_words) >= 0

                # delete topic if no doc is assigned over the descendants
                if old_topic.cum_doc_topic == 0:
                    assert np.sum(old_topic.cum_words) == 0
                    old_topic.delete_topic()

            # delete topic assigned to the index if no word is assigned to the index
            if np.sum(cnt_words) == 0:
                new_topic = None
            else:
                # assign new topic to the index increment count of docs and words
                new_topic = topic_root.sample_child(cnt_words, init=False, train=train)
                if train: new_topic.increment_cnt(cnt_words) # increment count of doc_topic & words

            doc.topics[index] = new_topic
            
    # if topic which has no doc but the descendants has docs, the topic remains. so delete them
    if train:
        all_topics = topic_root.get_all_topics()
        for topic in all_topics:
            if topic.cum_doc_topic == 0:
                assert np.sum(topic.cum_words) == 0
                topic.delete_topic()
            
def sample(train_docs, dev_docs, test_docs, topic_root):
    sample_word_topics(docs=train_docs, topic_root=topic_root)
    sample_doc_topics(docs=train_docs, topic_root=topic_root)
    sample_word_topics(docs=dev_docs, topic_root=topic_root, train=False)
    sample_doc_topics(docs=dev_docs, topic_root=topic_root, train=False)
    sample_word_topics(docs=test_docs, topic_root=topic_root, train=False)
    sample_doc_topics(docs=test_docs, topic_root=topic_root, train=False)
    
#     assert_sum_cnt_words(train_docs, topic_root)
    
def get_perplexity(docs, topic_root, verbose=False):
    topics = topic_root.get_all_topics()
    new_topic = topic_root.get_new_child(train=False)
    all_topics = topics + [new_topic]
    for topic in all_topics:
        topic.set_prob_words()

    logit_docs = []
    for doc in docs:
        # index probability for each word
        probs_words_indices = np.array([doc.get_probs_index(word_idx) for word_idx in doc.words])
        # topic probability for each index
        probs_indices_topics = np.array([topic_root.get_probs_all_topics(cnt_words) for cnt_words in np.concatenate([doc.index_cnt_words, np.zeros([1, doc.config.n_vocab])], 0)])
        # topic probability for each word
        probs_words_topics = probs_words_indices.dot(probs_indices_topics)

        # word probability for each topic
        probs_topics_words = np.array([topic.prob_words for topic in all_topics]) 
        probs_words_bow = probs_words_topics.dot(probs_topics_words)
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
        depth_specs[topic.depth].append(topic_spec)
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
    def get_cos_sim(parent_childs):
        parent_child_bows = {parent: np.array([child.prob_words/np.linalg.norm(child.prob_words) for child in childs]) for parent, childs in parent_childs.items() if len(childs) > 0}
        cos_sim = np.mean([np.mean((parent.prob_words/np.linalg.norm(parent.prob_words)).dot(child_bows.T)) for parent, child_bows in parent_child_bows.items()])
        return cos_sim

    topics = topic_root.get_all_topics()
    second_parents = [topic for topic in topics if topic.depth==2]
    third_childs = [topic for topic in topics if topic.depth==3]
    if len(second_parents) == 0 or len(third_childs) == 0:
        child_cos_sim = unchild_cos_sim = 0
    else:
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

def get_freq_tokens_rcrp(topic, idx_to_word, bow_idxs, topic_freq_words=None):
    freq_words = [idx_to_word[bow_idxs[bow_index]] for bow_index in np.argsort(topic.prob_words)[::-1][:10]]
    if topic_freq_words is None: topic_freq_words = {}
    topic_freq_words[topic] = freq_words
    print('  '*topic.depth, topic.idx, ':', [child.idx for child in topic.children], topic.cnt_doc_topic, np.sum(topic.cnt_words), freq_words)
    for topic in topic.children:
        topic_freq_words = get_freq_tokens_rcrp(topic, idx_to_word, bow_idxs, topic_freq_words)
    return topic_freq_words

def get_docs(instances, config, topic_root):
    docs_bow = [instance.bow for instance in instances]
    docs_raw = [[[bow_index]*int(doc_bow[bow_index]) for bow_index in np.where(doc_bow > 0)[0]] for doc_bow in docs_bow]
    docs_words = [[idx for idxs in doc for idx in idxs] for doc in docs_raw]
    docs = [Doc(idx=doc_idx, words=doc_words, bow=doc_bow, config=config, topic_root=topic_root) for doc_idx, (doc_words, doc_bow) in enumerate(zip(docs_words, docs_bow)) if len(doc_words) > 0]
    return docs

def assert_sum_cnt_words(train_docs, topic_root):
    def get_topic_cnt(docs):
        topic_cnt_words = {}
        topic_cnt_doc_topic = {}
        for doc in docs:
            for index, topic in enumerate(doc.topics):
                if topic not in topic_cnt_words:
                    topic_cnt_words[topic] = np.zeros(doc.config.n_vocab)
                    topic_cnt_doc_topic[topic] = 0
                topic_cnt_words[topic] += doc.index_cnt_words[index]
                topic_cnt_doc_topic[topic] += 1
        return topic_cnt_words, topic_cnt_doc_topic

    def assert_cnt(topic, topic_cnt_words, topic_cnt_doc_topic):
        if np.sum(topic.cnt_words) > 0: assert all(topic_cnt_words[topic] == topic.cnt_words)
        if topic.cnt_doc_topic > 0: assert topic_cnt_doc_topic[topic] == topic.cnt_doc_topic
        for child in topic.children:
            assert_cnt(child, topic_cnt_words, topic_cnt_doc_topic)

    def get_cum_doc_topic(topic):
        cum_doc_topic = np.sum(topic.cnt_doc_topic)
        for child in topic.children:
            cum_doc_topic += get_cum_doc_topic(child)
        return cum_doc_topic

    def get_cum_words(topic):
        cum_words = np.zeros_like(topic.cnt_words)
        cum_words += topic.cnt_words
        for child in topic.children:
            cum_words += get_cum_words(child)
        return cum_words

    def assert_cum(docs, topic_root, cum_doc_topic, cum_words):
        doc_cum_words = np.sum([doc.cnt_words for doc in docs], 0)
        assert topic_root.cum_doc_topic == cum_doc_topic
        assert all(topic_root.cum_words == cum_words)
        assert all(topic_root.cum_words == doc_cum_words)
        
    topic_cnt_words, topic_cnt_doc_topic = get_topic_cnt(train_docs)
    assert_cnt(topic_root, topic_cnt_words, topic_cnt_doc_topic)
    cum_doc_topic = get_cum_doc_topic(topic_root)
    cum_words = get_cum_words(topic_root)
    assert_cum(train_docs, topic_root, cum_doc_topic, cum_words)
