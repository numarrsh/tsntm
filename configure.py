#config: utf-8
import os
import argparse
import numpy as np
import pdb

def get_tree_idxs(tree):
    tree_idxs = {}
    tree_idxs[0] = [i for i in range(1, tree//10 +1)]
    for parent_idx in tree_idxs[0]:
        tree_idxs[parent_idx] = [parent_idx*10+i for i in range(1, tree % 10 +1)]
    return tree_idxs

def get_config(nb_name):
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu')
    parser.add_argument('data')
    parser.add_argument('-m', '--model', default='hntm')

    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-max_to_keep', type=int, default=10)
    
    parser.add_argument('-epoch', '--n_epochs', type=int, default=1000)
    parser.add_argument('-batch', '--batch_size', type=int, default=64)
    parser.add_argument('-log', '--log_period', type=int, default=5000)
    parser.add_argument('-freq', '--n_freq', type=int, default=10)

    parser.add_argument('-opt', default='Adagrad')
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-reg', type=float, default=1.)
    parser.add_argument('-gr', '--grad_clip', type=float, default=5.)
    parser.add_argument('-dr', '--keep_prob', type=float, default=0.8)

    parser.add_argument('-hid', '--dim_hidden_bow', type=int, default=256)
    parser.add_argument('-lat', '--dim_latent_bow', type=int, default=32)
    parser.add_argument('-emb', '--dim_emb', type=int, default=256)

    parser.add_argument('-topic', '--n_topic', type=int, default=10)
    parser.add_argument('-tree', type=int, default=33)
    parser.add_argument('-dep', '--n_depth', type=int, default=3)
    parser.add_argument('-temp', '--depth_temperature', type=float, default=1.)
    parser.add_argument('-min', '--remove_min', action='store_true')
    parser.add_argument('-add', '--add_threshold', type=float, default=0.05)
    parser.add_argument('-rem', '--remove_threshold', type=float, default=0.05)
    parser.add_argument('-static', action='store_true')
    
    parser.add_argument('-tmp', action='store_true')
    
    # for ncrp
    parser.add_argument('-alp', nargs='+', type=float, default=[1, 1, 1])
    parser.add_argument('-eta', type=float, default=1)
    parser.add_argument('-gam', type=float, default=0.01)
    parser.add_argument('-verbose', action='store_true')

    args = nb_name.replace('.ipynb', '').rstrip().split()
    config = parser.parse_args(args=args)
    
    config.tree_idxs = get_tree_idxs(config.tree)
    config.path_data = os.path.join('data', config.data, 'instances.pkl')
    config.dir_model = os.path.join('model', config.data, config.model, ''.join(args[1:]))
    config.dir_corpus = os.path.join('corpus', config.data) 
    config.path_model = os.path.join(config.dir_model, 'model') 
    config.path_config = config.path_model + '-%i.config'
    config.path_log = os.path.join(config.dir_model, 'log')
    config.path_checkpoint = os.path.join(config.dir_model, 'checkpoint')
    
    # for ncrp
    config.alp = np.array(config.alp)
    
    return config