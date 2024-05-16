from smod_wrapper import SMoDWrapper
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import roc_auc_score
import datetime
import time

from eden.util import configure_logging
import logging
logger = logging.getLogger()
configure_logging(logger,verbosity=1)

import random
def random_string(length,alphabet_list):
    rand_str = ''.join(random.choice(alphabet_list) for i in range(length))
    return rand_str

def perturb(seed,alphabet_list,p=0.5):
    seq=''
    for c in seed:
        if random.random() < p: c = random.choice(alphabet_list)
        seq += c
    return seq

def make_artificial_dataset(alphabet='ACGT', motives=None, motif_length=6, 
                            sequence_length=100, n_sequences=1000, n_motives=2, p=0.2,
                           random_state=1):
    random.seed(random_state)

    alphabet_list=[c for c in alphabet]
    
    if motives is None:
        motives=[]
        for i in range(n_motives):
            motives.append(random_string(motif_length,alphabet_list))
    else:
        motif_length = len(motives[0])
        n_motives = len(motives)
    
    sequence_length = sequence_length / len(motives)
    flanking_length = (sequence_length - motif_length ) / 2
    n_seq_per_motif = n_sequences

    counter=0
    seqs=[]
    for i in range(n_seq_per_motif):
        total_seq = ''
        total_binary_seq=''
        for j in range(n_motives):
            left_flanking = random_string(flanking_length,alphabet_list)
            right_flanking = random_string(flanking_length,alphabet_list)
            noisy_motif = perturb(motives[j],alphabet_list,p)
            seq = left_flanking + noisy_motif + right_flanking
            total_seq += seq
        seqs.append(('ID%d'%counter,total_seq))
        counter += 1
    binary_skeleton = '0' * flanking_length + '1' * motif_length + '0' * flanking_length
    binary_seq = binary_skeleton * n_motives
    return motives, seqs, binary_seq

def score_seqs(seqs, n_motives, tool):
    scores = []
    if tool is None:
        return scores
    
    for j in range(len(seqs)):
        seq_scr = []
        iters = tool.nmotifs
        for k in range(iters):
            scr=tool.score(motif_num=k+1, seq=seqs[j][1])
            seq_scr.append(scr)

        # taking average over all motives for a sequence
        if len(seq_scr) > 1:
            x = np.array(seq_scr[0])
            for l in range(1, iters):
                x = np.vstack((x, seq_scr[l]))
            seq_scr = list(np.mean(x, axis=0))
            scores.append(seq_scr)
        elif len(seq_scr) == 1:
            scores.append(np.array(seq_scr[0]))
        else:
            raise ValueError("no sequence score")
    return scores

def get_dataset(sequence_length=200,
                n_sequences=200,
                motif_length=10,
                n_motives=2, 
                p=0.2,
                random_state=1):
    
    motives, pos_seqs, binary_seq = make_artificial_dataset(alphabet='ACGT',
                                                            sequence_length=sequence_length,
                                                            n_sequences=n_sequences,
                                                            motif_length=motif_length,
                                                            n_motives=n_motives,
                                                            p=p, 
                                                            random_state=random_state)

    from eden.modifier.seq import seq_to_seq, shuffle_modifier
    neg_seqs = seq_to_seq(pos_seqs, modifier=shuffle_modifier, times=2, order=2)
    neg_seqs = list(neg_seqs)

    block_size=n_sequences/8

    pos_size = len(pos_seqs)
    train_pos_seqs = pos_seqs[:pos_size/2]
    test_pos_seqs = pos_seqs[pos_size/2:]

    neg_size = len(neg_seqs)
    train_neg_seqs = neg_seqs[:neg_size/2]
    test_neg_seqs = neg_seqs[neg_size/2:]

    true_score = [float(int(i)) for i in binary_seq]
    return (block_size, train_pos_seqs, train_neg_seqs, test_pos_seqs, n_motives, true_score)

def test_on_datasets(n_sets = 5, param_setting=None, p=0.2, max_roc=0.5, std_roc=0.01):
    dataset_score = []
    seeds = [i * 2000 for i in range(1, n_sets + 1)]
    for k in range(n_sets):
        # Generate data set
        seed = seeds[k]
        data = get_dataset(sequence_length=40,
                           n_sequences=50,
                           motif_length=10,
                           n_motives=2,
                           p=p,
                           random_state=seed)
        block_size = data[0]
        train_pos_seqs = data[1]
        train_neg_seqs = data[2]
        test_pos_seqs = data[3]
        n_motives = data[4]
        true_score = data[5]

        smod = SMoDWrapper(alphabet = 'dna',
                           scoring_criteria = 'pwm',

                           complexity = 5,
                           n_clusters = 10,
                           min_subarray_size = 8,
                           max_subarray_size = 12,
                           clusterer = KMeans(),
                           pos_block_size = block_size,
                           neg_block_size = block_size,
                           # sample_size = 300,
                           p_value = param_setting['p_value'],
                           similarity_th = param_setting['similarity_th'],
                           min_score = param_setting['min_score'],
                           min_freq = param_setting['min_freq'],
                           min_cluster_size = param_setting['min_cluster_size'],
                           regex_th = param_setting['regex_th'],
                           freq_th = param_setting['freq_th'],
                           std_th = param_setting['std_th']) 

        

        try:
            smod.fit(train_pos_seqs, train_neg_seqs)
            scores = score_seqs(seqs = test_pos_seqs,
                                n_motives = n_motives,
                                tool = smod)
        except:
            continue

        mean_score = np.mean(scores, axis=0)
        roc_score = roc_auc_score(true_score, mean_score)


        # if a parameter setting performs poorly, don't test on other datasets
        # z-score = (x - mu)/sigma
        # if ((roc_score - max_roc)/std_roc) > 2:
        if roc_score < 0.6:
            break

        dataset_score.append(roc_score)
    return dataset_score

def check_validity(key, value, noise):
    if key == 'min_score':    # atleast greater than (motif_length)/2
        if value >= 5:
            return True, int(round(value))
    elif key == 'min_cluster_size':
        if value >= 3:
            return True, int(round(value))
    elif key == 'min_freq':    # atmost (1 - noise_level)
        if value > 0 and value <= (1 - noise):
            return True, value
    elif key == 'p_value':
        if value <= 1.0 and value >= 0.0:
            return True, value
    elif key == 'similarity_th':
        if value <= 1.0 and value >= 0.8:
            return True, value
    elif key == 'regex_th':
        if value > 0 and value <= 0.3:
            return True, value
    elif key == 'freq_th':
        if value <= 1.0 and value > 0:
            return True, value
    elif key == 'std_th':
        if value <= 1.0 and value > 0:
            return True, value
    else:
        raise ValueError('Invalid key: ', key)
    return False, value

def random_setting(parameters=None, best_config=None, noise=None):
    parameter_setting = {}
    MAX_ITER = 1000
    if not parameters['min_score']:    # use best_configuration of last run as initial setting
        for key in parameters.keys():
            parameters[key].append(best_config[key])
            parameter_setting[key] = best_config[key]
    else:
        for key in parameters.keys():
            success = False
            n_iter = 0
            mu = np.mean(parameters[key])
            sigma = np.mean(parameters[key])
            if sigma == 0:
                sigma == 0.1
            while not success:
                if n_iter == MAX_ITER:    # if max_iterations exceeded, return mean as value
                    value = mu
                    if key in ['min_score', 'min_cluster_size']:
                        value = int(round(value))
                    break
                value = np.random.normal(mu, 2 * sigma)
                n_iter += 1
                success, value = check_validity(key, value, noise)
            parameter_setting[key] = value
    return parameter_setting

get_ipython().run_cell_magic('time', '', '\nprint datetime.datetime.fromtimestamp(time.time()).strftime(\'%H:%M:%S\'),\nprint "Starting experiment...\\n"\n\nbest_config = {\'min_score\':6, # atleast motif_length/2\n               \'min_freq\':0.1, # can not be more than (1- noise level)\n               \'min_cluster_size\':3, # atleast 3\n               \'p_value\':0.1, # atleast 0.1\n               \'similarity_th\':0.8, # 0.8 \n               \'regex_th\':0.3, # max 0.3 \n               \'freq_th\':0.05, # 0.05 \n               \'std_th\':0.2} # 0.2\n\n# Final results\nparam = [0.1, 0.2, 0.3]#, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]\nresults_dic = {}\n\nreps = 30 #0    # different settings to be tried\n\nfor i in param:\n    parameters = {\'min_freq\': [],\n                  \'min_cluster_size\': [],\n                  \'p_value\': [],\n                  \'similarity_th\': [],\n                  \'min_score\': [],\n                  \'regex_th\': [],\n                  \'freq_th\': [],\n                  \'std_th\': []}\n    max_roc = 0.5\n    std_roc = 0.01\n    #parameters = generate_dist(parameters, best_config)\n    for j in range(reps):\n        param_setting = random_setting(parameters, best_config, i)    # Randomize Parameter setting\n        n_sets = 5    # Different data sets\n        dataset_score = test_on_datasets(n_sets=n_sets, \n                                         param_setting=param_setting, \n                                         p=i, \n                                         max_roc=max_roc,\n                                         std_roc=std_roc)\n        mean_roc = np.mean(dataset_score)\n        std = np.std(dataset_score)\n\n        if mean_roc > max_roc:\n            max_roc = mean_roc\n            std_roc = std\n            print datetime.datetime.fromtimestamp(time.time()).strftime(\'%H:%M:%S\'),\n            print "Better Configuration found at perturbation prob = ", i\n            print "ROC: ", mean_roc\n            print "Parameter Configuration: ", param_setting\n            print\n            best_config = param_setting\n            param_setting["ROC"] = mean_roc\n            results_dic[i] = param_setting\n            \nprint datetime.datetime.fromtimestamp(time.time()).strftime(\'%H:%M:%S\'),\nprint " Finished experiment..."')





