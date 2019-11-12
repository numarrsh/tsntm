#coding:utf-8

import numpy as np
from collections import defaultdict

def validate(sess, batches, model):
    losses = []
    ppl_list = []
    rads_bow_list = []
    prob_topic_list = []
    n_bow_list = []
    n_topics_list = []
    for ct, batch in batches:
        feed_dict = model.get_feed_dict(batch, mode='test')
        loss_batch, topic_loss_recon_batch, topic_loss_kl_batch, topic_loss_reg_batch, ppls_batch, rads_bow_batch, prob_topic_batch, n_bow_batch, n_topics_batch \
            = sess.run([model.loss, model.topic_loss_recon, model.topic_loss_kl, model.topic_loss_reg, model.topic_ppls, model.rads_bow, model.prob_topic, model.n_bow, model.n_topics], feed_dict = feed_dict)
        losses += [[loss_batch, topic_loss_recon_batch, topic_loss_kl_batch, topic_loss_reg_batch]]
        ppl_list += list(ppls_batch)
        rads_bow_list.append(rads_bow_batch)
        prob_topic_list.append(prob_topic_batch)
        n_bow_list.append(n_bow_batch)
        n_topics_list.append(n_topics_batch)
    loss_mean, topic_loss_recon_mean, topic_loss_kl_mean, topic_loss_reg_mean = np.mean(losses, 0)
    ppl_mean = np.exp(np.mean(ppl_list))
    
    probs_topic = np.concatenate(prob_topic_list, 0)
    
    n_bow = np.concatenate(n_bow_list, 0)
    n_topics = np.concatenate(n_topics_list, 0)
    probs_topic_mean = np.sum(n_topics, 0) / np.sum(n_bow)
    
    rads_bow = np.concatenate(rads_bow_list, 0)
    rads_bow_mean = np.cos(np.sum(rads_bow, 0) / np.sum(n_topics, 0))
    
    return loss_mean, topic_loss_recon_mean, topic_loss_kl_mean, topic_loss_reg_mean, ppl_mean, rads_bow_mean, probs_topic_mean

def print_topic_sample(sess, model, recur_prob_topic=None, topic_prob_topic=None, topic_freq_token=None, parent_idx=0, depth=0):
    if depth == 0: # print root
        freq_token = topic_freq_token[parent_idx]
        recur_topic = recur_prob_topic[parent_idx]
        prob_topic = topic_prob_topic[parent_idx]
        print(parent_idx, 'R: %.3f' % recur_topic, 'P: %.3f' % prob_topic, freq_token)
    
    child_idxs = model.tree_idxs[parent_idx]
    depth += 1
    for child_idx in child_idxs:
        freq_token = topic_freq_token[child_idx]
        recur_topic = recur_prob_topic[child_idx]
        prob_topic = topic_prob_topic[child_idx]
        print('  '*depth, child_idx, 'R: %.3f' % recur_topic, 'P: %.3f' % prob_topic, freq_token)
        
        if child_idx in model.tree_idxs: 
            print_topic_sample(sess, model, recur_prob_topic=recur_prob_topic, topic_prob_topic=topic_prob_topic, parent_idx=child_idx, topic_freq_token=topic_freq_token, depth=depth)
            
def print_topic_specialization(sess, model, instances):
    topics_bow = sess.run(model.topic_bow)
    norm_bow = np.sum([instance.bow for instance in instances], 0)
    topics_vec = topics_bow / np.linalg.norm(topics_bow, axis=1, keepdims=True)
    norm_vec = norm_bow / np.linalg.norm(norm_bow)

    topics_spec = 1 - topics_vec.dot(norm_vec)

    depth_topic_idxs = defaultdict(list)
    for topic_idx, depth in model.tree_depth.items():
        depth_topic_idxs[depth].append(topic_idx)

    for depth, topic_idxs in depth_topic_idxs.items():
        topic_indices = np.array([model.topic_idxs.index(topic_idx) for topic_idx in topic_idxs])
        depth_spec = np.mean(topics_spec[topic_indices])
        print(depth, depth_spec)    
        
def print_hierarchical_affinity(sess, model):
    def get_cos_sim(parent_to_child_idxs):
        parent_child_bows = {parent_idx: np.concatenate([normed_tree_topic_bow[child_idx] for child_idx in child_idxs], 0) for parent_idx, child_idxs in parent_to_child_idxs.items()}
        cos_sim = np.mean([np.mean(normed_tree_topic_bow[parent_idx].dot(child_bows.T)) for parent_idx, child_bows in parent_child_bows.items()])
        return cos_sim    
    
    tree_topic_bow = sess.run(model.tree_topic_bow)
    normed_tree_topic_bow = {topic_idx: topic_bow/np.linalg.norm(topic_bow) for topic_idx, topic_bow in tree_topic_bow.items()}

    third_child_idxs = [child_idx for child_idx, depth in model.tree_depth.items() if depth==3]
    second_parent_to_child_idxs = {parent_idx:child_idxs for parent_idx, child_idxs in model.tree_idxs.items() if model.tree_depth[parent_idx] == 2}
    second_parent_to_unchild_idxs = {parent_idx: [child_idx for child_idx in third_child_idxs if child_idx not in child_idxs] for parent_idx, child_idxs in second_parent_to_child_idxs.items()}
    
    child_cos_sim = get_cos_sim(second_parent_to_child_idxs)
    unchild_cos_sim = get_cos_sim(second_parent_to_unchild_idxs)
    print('child %.3f, not-child: %.3f' % (child_cos_sim, unchild_cos_sim))