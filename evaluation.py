#coding:utf-8

import numpy as np
import tensorflow as tf
from collections import defaultdict

def validate(sess, batches, model):
    losses = []
    ppl_list = []
    prob_topic_list = []
    n_bow_list = []
    n_topics_list = []
    for ct, batch in batches:
        feed_dict = model.get_feed_dict(batch, mode='test')
        loss_batch, topic_loss_recon_batch, topic_loss_kl_batch, topic_loss_reg_batch, ppls_batch, prob_topic_batch, n_bow_batch, n_topics_batch \
            = sess.run([model.loss, model.topic_loss_recon, model.topic_loss_kl, model.topic_loss_reg, model.topic_ppls, model.prob_topic, model.n_bow, model.n_topics], feed_dict = feed_dict)
        losses += [[loss_batch, topic_loss_recon_batch, topic_loss_kl_batch, topic_loss_reg_batch]]
        ppl_list += list(ppls_batch)
        prob_topic_list.append(prob_topic_batch)
        n_bow_list.append(n_bow_batch)
        n_topics_list.append(n_topics_batch)
    loss_mean, topic_loss_recon_mean, topic_loss_kl_mean, topic_loss_reg_mean = np.mean(losses, 0)
    ppl_mean = np.exp(np.mean(ppl_list))
    
    probs_topic = np.concatenate(prob_topic_list, 0)
    
    n_bow = np.concatenate(n_bow_list, 0)
    n_topics = np.concatenate(n_topics_list, 0)
    probs_topic_mean = np.sum(n_topics, 0) / np.sum(n_bow)
    
    return loss_mean, topic_loss_recon_mean, topic_loss_kl_mean, topic_loss_reg_mean, ppl_mean, probs_topic_mean

def print_topic_sample(sess, model, topic_prob_topic=None, recur_prob_topic=None, topic_freq_tokens=None, parent_idx=0, depth=0):
    if depth == 0: # print root
        assert len(topic_prob_topic) == len(recur_prob_topic) == len(topic_freq_tokens)
        freq_tokens = topic_freq_tokens[parent_idx]
        recur_topic = recur_prob_topic[parent_idx]
        prob_topic = topic_prob_topic[parent_idx]
        print(parent_idx, 'R: %.3f' % recur_topic, 'P: %.3f' % prob_topic, ' '.join(freq_tokens))
    
    child_idxs = model.tree_idxs[parent_idx]
    depth += 1
    for child_idx in child_idxs:
        freq_tokens = topic_freq_tokens[child_idx]
        recur_topic = recur_prob_topic[child_idx]
        prob_topic = topic_prob_topic[child_idx]
        print('  '*depth, child_idx, 'R: %.3f' % recur_topic, 'P: %.3f' % prob_topic, ' '.join(freq_tokens))
        
        if child_idx in model.tree_idxs: 
            print_topic_sample(sess, model, topic_prob_topic=topic_prob_topic, recur_prob_topic=recur_prob_topic, topic_freq_tokens=topic_freq_tokens, parent_idx=child_idx, depth=depth)
            
def print_topic_year(sess, model, topic_freq_tokens=None, topic_year=None, parent_idx=0, depth=0):
    if depth == 0: # print root
        len(topic_freq_tokens) == len(topic_year)
        freq_tokens = topic_freq_tokens[parent_idx]
        year = topic_year[parent_idx]
        print(parent_idx, 'Avg Year: %i' % year, ' '.join(freq_tokens))
    
    child_idxs = model.tree_idxs[parent_idx]
    depth += 1
    for child_idx in child_idxs:
        freq_tokens = topic_freq_tokens[child_idx]
        year = topic_year[child_idx]
        print('  '*depth, child_idx, 'Avg Year: %i' % year, ' '.join(freq_tokens))
        
        if child_idx in model.tree_idxs: 
            print_topic_year(sess, model, topic_freq_tokens=topic_freq_tokens, topic_year=topic_year, parent_idx=child_idx, depth=depth)            

def print_flat_topic_sample(sess, model, topics_freq_tokens):
    for topic_idx, topic_freq_tokens in enumerate(topics_freq_tokens):
        print(topic_idx, ' '.join(topic_freq_tokens))
            
def get_topic_specialization(sess, model, instances, verbose=False):
    if not model.config.prod:
        topics_vec = sess.run(tf.nn.l2_normalize(model.topic_bow, 1))
    else:
        topics_vec = sess.run(tf.nn.l2_normalize(tf.nn.softmax(model.topic_bow, 1), 1))
#     topics_vec = topics_bow / np.linalg.norm(topics_bow, axis=1, keepdims=True)
    
    norm_bow = np.sum([instance.bow for instance in instances], 0)
    norm_vec = norm_bow / np.linalg.norm(norm_bow)

    topics_spec = 1 - topics_vec.dot(norm_vec)

    depth_topic_idxs = defaultdict(list)
    for topic_idx, depth in model.tree_depth.items():
        depth_topic_idxs[depth].append(topic_idx)

    depth_specs = {}
    print('Topic Specialization:', end=' ')
    for depth, topic_idxs in depth_topic_idxs.items():
        topic_indices = np.array([model.topic_idxs.index(topic_idx) for topic_idx in topic_idxs])
        depth_spec = np.mean(topics_spec[topic_indices])
        depth_specs[depth] = depth_spec
        print('depth %i: %.2f' % (depth, depth_spec), end=' ')
    print('')
    return depth_specs

def get_hierarchical_affinity(sess, model, verbose=False):
    def get_cos_sim(parent_to_child_idxs):
        parent_child_bows = {parent_idx: np.concatenate([normed_tree_topic_bow[child_idx] for child_idx in child_idxs], 0) for parent_idx, child_idxs in parent_to_child_idxs.items()}
        cos_sim = np.mean([np.mean(normed_tree_topic_bow[parent_idx].dot(child_bows.T)) for parent_idx, child_bows in parent_child_bows.items()])
        return cos_sim    
    
    if not model.config.prod:
        tree_topic_bow = {topic_idx: tf.nn.l2_normalize(topic_bow) for topic_idx, topic_bow in model.tree_topic_bow.items()}
    else:
        tree_topic_bow = {topic_idx: tf.nn.l2_normalize(tf.nn.softmax(topic_bow)) for topic_idx, topic_bow in model.tree_topic_bow.items()}
    normed_tree_topic_bow = sess.run(tree_topic_bow)

    third_child_idxs = [child_idx for child_idx, depth in model.tree_depth.items() if depth==3]
    second_parent_to_child_idxs = {parent_idx:child_idxs for parent_idx, child_idxs in model.tree_idxs.items() if model.tree_depth[parent_idx] == 2}
    second_parent_to_unchild_idxs = {parent_idx: [child_idx for child_idx in third_child_idxs if child_idx not in child_idxs] for parent_idx, child_idxs in second_parent_to_child_idxs.items()}
    
    if sum(len(unchilds) for unchilds in second_parent_to_unchild_idxs.values()) > 0:
        child_cos_sim = get_cos_sim(second_parent_to_child_idxs)
        unchild_cos_sim = get_cos_sim(second_parent_to_unchild_idxs)
    else:
        child_cos_sim = get_cos_sim(second_parent_to_child_idxs)
        unchild_cos_sim = 0
    
    if verbose: print('Hierarchical Affinity: child %.2f, non-child %.2f'%(child_cos_sim, unchild_cos_sim))
    return child_cos_sim, unchild_cos_sim