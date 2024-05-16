import sys
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io as scio
import bandit_preprocessing as bp
import sys
import os
import matplotlib.pyplot as plt
import calcium_codes as cc
import hmm_on_behavior as hob
get_ipython().magic('matplotlib inline')

record_path = '/Users/celiaberon/GitHub/mouse_bandit/session_record.csv'
ca_data_path = '/Volumes/Neurobio/MICROSCOPE/Celia/data/k7_03142017_test/neuron_results.mat'
#ca_data_path = '/Volumes/Neurobio/MICROSCOPE/Celia/data/q43_03202017_bandit_8020/q43_03202017_neuron_master.mat'
#ca_data_path = '/Volumes/Neurobio/MICROSCOPE/Celia/data/cnmfe_test/neuron_results.mat'

record = pd.read_csv(record_path,index_col=0)
ca_data = scio.loadmat(ca_data_path,squeeze_me = True, struct_as_record = False)
neuron = ca_data['neuron_results'] 

neuron.C_raw.shape

session_name  = '03142017_K7'
mouse_id = 'K7'

#session_name = '03202017_Q43'
#mouse_id = 'Q43'

record[record['Session ID'] == session_name]

'''
load in trial data
'''
columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked',
           'Right Reward Prob','Left Reward Prob','Reward Given',
          'center_frame','decision_frame']

root_dir = '/Users/celiaberon/GitHub/mouse_bandit/data/trial_data'

full_name = session_name + '_trials.csv'

path_name = os.path.join(root_dir,full_name)

trial_df = pd.read_csv(path_name,names=columns)

beliefs = hob.predictBeliefBySession(record_path, session_name, mouse_id)

columns.append('Belief')
trial_df['Belief'] = beliefs
trial_df.head(15)

trial_df.iloc[-1]['decision_frame'] - trial_df.iloc[0]['center_frame']

neuron.C_raw.shape

feature_matrix = bp.create_feature_matrix(trial_df,10,mouse_id,session_name,feature_names='Default',imaging=True)

beliefs_feat_mat = hob.predictBeliefFeatureMat(feature_matrix, 10)

feature_matrix['Belief'] = beliefs_feat_mat

feature_matrix.head(5)

feature_matrix[feature_matrix['Switch']==1]

feature_matrix[feature_matrix['Switch']==1]['0_ITI'].values.mean()

plt.plot(trial_df['Belief'])
plt.plot(feature_matrix[feature_matrix['Reward']==1]['Trial'], feature_matrix[feature_matrix['Reward']==1]['Belief'], alpha=0.5)
#plt.scatter(feature_matrix[feature_matrix['Reward']==1]['Trial'],temp[:,1,0], alpha=0.3)


#plt.scatter(temp[:,0,0], feature_matrix[feature_matrix['Reward']==1]['Belief'])
#temp = aligned_start.sum(axis=1)

plt.scatter(feature_matrix[feature_matrix['Decision']==1]['Port Streak'],
            feature_matrix[feature_matrix['Decision']==1]['Belief'])

def extract_frames(df, cond1_name, cond1=False, cond2_name=False,cond2=False, cond3_name=False,
                   cond3=False, cond1_ops= '=', cond2_ops = '=', cond3_ops = '='):
    
    import operator
    
    # set up operator dictionary
    ops = {'>': operator.gt,
       '<': operator.lt,
       '>=': operator.ge,
       '<=': operator.le,
       '=': operator.eq}
    
    if type(cond3_name)==str:
        frames_c = (df[((ops[cond1_ops](df[cond1_name],cond1)) 
                    & (ops[cond2_ops](df[cond2_name], cond2))
                    & (ops[cond3_ops](df[cond3_name],cond3)))]['center_frame'])
        frames_d = (df[((ops[cond1_ops](df[cond1_name],cond1)) 
                    & (ops[cond2_ops](df[cond2_name], cond2))
                    & (ops[cond3_ops](df[cond3_name],cond3)))]['decision_frame'])
        frames = np.column_stack((frames_c, frames_d))
        return frames
    
    elif type(cond2_name)==str:
        frames_c = (df[((df[cond1_name] == cond1) 
                    & (df[cond2_name] == cond2))]['center_frame'])
        frames_d = (df[((df[cond1_name] == cond1) 
                    & (df[cond2_name] == cond2))]['decision_frame'])
        frames = np.column_stack((frames_c, frames_d))
        return frames
    
    else:
        frames_c =(df[(df[cond1_name] == cond1)]['center_frame'])
        frames_d =(df[(df[cond1_name] == cond1)]['decision_frame'])
        frames = np.column_stack((frames_c, frames_d))
        return frames

"""
Decision: 0=Right, 1=Left
Reward: 0=unrewarded, 1=rewarded
Switch: 0=last trial at same port, 1=last trial at different port-->switched
Belief: 0-1 value where 0 represents right port and 1 represents left port
"""

cond1_name = 'Switch'
cond1_a = 0
cond1_b = 1
cond1_ops = '='
cond2_name = 'Decision'
cond2_a = 0
cond2_b = 1
cond3_name = 'Reward'
cond3_a = 0
cond3_b = 1

conditions = [cond1_name, cond2_name, cond3_name]
n_variables = 2
extension = 30

cond1_ops_b='='
if cond1_ops != '=':
    if cond1_ops == '>':
        cond1_ops_b = '<='
        print(cond1_ops_b)
    elif cond1_ops == '>=':
        cond1_ops_b = '<'
        print(cond1_ops_b)

# center frames in first column, decision frames in second
fr_1a2a3a = extract_frames(feature_matrix, cond1_name, cond1_a, 
                           cond2_name, cond2_a, cond3_name, cond3_a, cond1_ops=cond1_ops)

fr_1b2a3a = extract_frames(feature_matrix, cond1_name, cond1_b, 
                           cond2_name, cond2_a, cond3_name, cond3_a, cond1_ops=cond1_ops_b)

fr_1a2b3a = extract_frames(feature_matrix, cond1_name, cond1_a, 
                           cond2_name, cond2_b, cond3_name, cond3_a, cond1_ops=cond1_ops)

fr_1b2b3a = extract_frames(feature_matrix, cond1_name, cond1_b, 
                           cond2_name, cond2_b, cond3_name, cond3_a, cond1_ops=cond1_ops_b)

fr_1a2b3b = extract_frames(feature_matrix, cond1_name, cond1_a, 
                           cond2_name, cond2_b, cond3_name, cond3_b, cond1_ops=cond1_ops)

fr_1a2a3b = extract_frames(feature_matrix, cond1_name, cond1_a, 
                           cond2_name, cond2_a, cond3_name, cond3_b, cond1_ops=cond1_ops)

fr_1b2a3b = extract_frames(feature_matrix, cond1_name, cond1_b, 
                           cond2_name, cond2_a, cond3_name, cond3_b, cond1_ops=cond1_ops_b)

fr_1b2b3b = extract_frames(feature_matrix, cond1_name, cond1_b, 
                           cond2_name, cond2_b, cond3_name, cond3_b, cond1_ops=cond1_ops_b)

var_keys = '1a2a3a', '1b2a3a', '1a2b3a', '1b2b3a', '1a2b3b', '1a2a3b', '1b2a3b', '1b2b3b'
groupings_2 = np.stack(((0,5), (1,6), (2,4), (3,7)))
groupings_1 = np.stack(((0,2,4,5), (1,3,6,7)))

#start_stop_frames = {var_keys[0]:fr_1a2a3a, var_keys[1]:fr_1b2a3a, var_keys[2]:fr_1a2b3a, var_keys[3]:fr_1b2b3a, 
#          var_keys[4]:fr_1a2b3b, var_keys[5]:fr_1a2a3b, var_keys[6]:fr_1b2a3b, var_keys[7]:fr_1b2b3b}

n_combos = 2**n_variables

for i in range(n_combos):
    if n_variables == 3:
        if i == 0:
            start_stop_frames = {var_keys[i]:eval('fr_%s' %var_keys[i])}
        if i > 0:
            start_stop_frames.update({var_keys[i]:eval('fr_%s' %var_keys[i])})
    if n_variables == 2:
        if i == 0:
            start_stop_frames = {var_keys[i][0:4]: np.transpose(np.column_stack((
                        np.transpose(eval('fr_%s' % var_keys[groupings_2[i][0]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_2[i][1]])))))}
        if i > 0:
            start_stop_frames.update({var_keys[i][0:4]: np.transpose(np.column_stack((
                        np.transpose(eval('fr_%s' % var_keys[groupings_2[i][0]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_2[i][1]])))))})
            if i == np.max(n_combos)-1:
                var_keys = list(start_stop_frames.keys())  

    if n_variables == 1:
        if i == 0:
            start_stop_frames = {var_keys[i][0:2]: np.transpose(np.column_stack((
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][0]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][1]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][2]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][3]])))))}
        if i > 0:
            start_stop_frames.update({var_keys[i][0:2]: np.transpose(np.column_stack((
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][0]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][1]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][2]])),
                        np.transpose(eval('fr_%s' % var_keys[groupings_1[i][3]])))))})
            if i == np.max(n_combos)-1:
                var_keys = list(start_stop_frames.keys())            

                
#if n_variables == 2:
#    var_keys = list(start_stop_frames.keys())     
        
start_stop_frames.keys()

for i in start_stop_frames:
    start_stop_frames[i][:,0] = start_stop_frames[i][:,0] - extension
    start_stop_frames[i][:,1] = start_stop_frames[i][:,1] + extension

events = cc.detectEvents(ca_data_path)

neuron.C_raw = np.copy(events)
nNeurons = neuron.C_raw.shape[0]
nFrames = neuron.C_raw.shape[1]

#Create Gaussian filter and apply to raw trace
sigma = 3;
sz = 10; # total width 

x = np.linspace(-sz / 2, sz / 2, sz);
gaussFilter = np.exp(-x**2 / (2*sigma**2));
gaussFilter = gaussFilter / np.sum(gaussFilter);

smoothed = np.zeros((nNeurons, neuron.C_raw.shape[1]+sz-1));

for i in range(0, nNeurons):
    smoothed[i,:] = np.convolve(neuron.C_raw[i,:], gaussFilter);
    
neuron.C_raw = smoothed[:,0:nFrames]

# This is just used to visualize the effect of the Gaussian filter on each event
plt.plot(neuron.C_raw[0,:])
plt.plot(events[0,:])

nTrials = [start_stop_frames[var_keys[i]].shape[0] for i in range(n_combos)]
max_window = np.zeros(n_combos) 
window_length= np.zeros((np.max(nTrials), n_combos))

    
for i in range(n_combos):
    for iTrial in range(nTrials[i]):
        window_length[iTrial, i] = int(((start_stop_frames[var_keys[i]][iTrial][1]-
                                 start_stop_frames[var_keys[i]][iTrial][0])))
    max_window[i] = np.max(window_length)
    
max_window = int(max_window.max())

med_trial_length = [np.median(window_length[0:nTrials[i],:]) for i in range(n_combos)]
med_trial_length = np.median(med_trial_length) - 2*extension

print(int(start_stop_frames[var_keys[i]][iTrial][0]))
start_stop_frames[var_keys[i]][iTrial][0]+max_window

aligned_start = np.zeros((np.max(nTrials), max_window, nNeurons, n_combos))
mean_center_poke = np.zeros((max_window, nNeurons, n_combos))
during_trial = np.zeros_like((aligned_start))

for i in range(n_combos):

    # create array containing segment of raw trace for each neuron for each trial 
    # aligned to center poke
    for iNeuron in range(nNeurons): # for each neuron
        for iTrial in range(0,nTrials[i]): # and for each trial
            aligned_start[iTrial,:, iNeuron, i] = neuron.C_raw[iNeuron,
                int(start_stop_frames[var_keys[i]][iTrial][0]):
                (int(start_stop_frames[var_keys[i]][iTrial][0])+max_window)]
            during_trial[iTrial,0:int(window_length[iTrial,i]-extension),iNeuron,i] = neuron.C_raw[iNeuron,
                int(start_stop_frames[var_keys[i]][iTrial][0]+extension):
                (int(start_stop_frames[var_keys[i]][iTrial][0] +window_length[iTrial,i]))]
  
    # take mean of fluorescent traces across all trials for each neuron, then normalize
    # for each neuron
    mean_center_poke[:,:,i]= np.mean(aligned_start[0:nTrials[i],:,:,i], axis=0)

pre_trial = aligned_start[:, int(extension-med_trial_length):extension, :, :].sum(axis=1)
during_trial = during_trial.sum(axis=1)

plt.hist(neuron.C_raw.sum(axis=1)/32)

ydim = n_combos/2
plt.figure(figsize=(8,ydim*4))
for i in range(n_combos):

    plt.subplot(ydim,2,i+1)  
    plt.imshow(np.transpose(mean_center_poke[:,:,i]))#, plt.colorbar()
    plt.axvline(x=extension, color='k', linestyle = '--', linewidth=.9)
    plt.axis('tight')
    plt.xlabel('Frame (center poke at %s)' % extension)
    plt.ylabel('Neuron ID')
    if n_variables == 3:
        plt.title("%s = %s\n %s = %s\n%s = %s\nNum trials = %.0f" 
                  %(conditions[int(var_keys[i][0])-1], var_keys[i][1],
                    conditions[int(var_keys[i][2])-1], var_keys[i][3], 
                    conditions[int(var_keys[i][4])-1],
                    var_keys[i][5], nTrials[i])) 
    if n_variables == 2:
        plt.title("%s = %s\n %s = %s\nNum trials = %.0f" 
                  %(conditions[int(var_keys[i][0])-1], var_keys[i][1],
                    conditions[int(var_keys[i][2])-1], var_keys[i][3], nTrials[i]))
    if n_variables == 1:
        plt.title("%s = %s\nNum trials = %.0f" 
                  %(conditions[int(var_keys[i][0])-1], var_keys[i][1], nTrials[i]))
plt.tight_layout()

sample_neuron = 10

#plt.figure(figsize=(10,10))
plt.imshow(aligned_start[0:nTrials[0],:,sample_neuron, 0])
plt.axvline(x=extension, color='white', linestyle = '--', linewidth=.9)
plt.ylabel('Trial Number')
plt.xlabel('Frame (center poke at %s)' %extension)
#plt.scatter((start_stop_times[var_keys[0][:]])-(start_stop_frames[var_keys[0]])+extension,range(nTrials[0]), color='white', marker = '|', s=10)
plt.title('%s = %s\n%s = %s\nNeuron ID = %s' % (cond1_name, conditions[0], cond2_name, cond2_a, sample_neuron))
plt.axis('tight')

aligned_decision = np.zeros((np.max(nTrials), max_window, nNeurons, n_combos))
mean_decision = np.zeros((max_window, nNeurons, n_combos))

for i in range(n_combos):

    # create array containing segment of raw trace for each neuron for each trial 
    # aligned to decision poke
    for iNeuron in range(nNeurons):
        for iTrial in range(nTrials[i]):
            aligned_decision[iTrial,:, iNeuron, i] = neuron.C_raw[iNeuron, 
                int(start_stop_frames[var_keys[i]][iTrial][1])-max_window:
                (int(start_stop_frames[var_keys[i]][iTrial][1]))]

    # take mean of fluorescent traces across all trials for each neuron
    mean_decision[:,:,i]= np.mean(aligned_decision[0:nTrials[i],:,:,i], axis=0)
   

post_trial = aligned_decision[:, int(max_window-extension):int(max_window-extension + 
             med_trial_length), :, :].sum(axis=1)

[plt.scatter(window_length[:,c], during_trial[:,n,c]) for n in range(nNeurons) 
 for c in range(n_combos)]
print('c')

temp = mean_decision.sum(axis=1)
[plt.plot(np.transpose(temp[:,c])) for c in range(n_combos)]
plt.axvline(x=max_window-extension, linestyle='--', color='k', linewidth=.9)
plt.axvline(x=max_window-(med_trial_length+extension), linestyle='--', color='k', linewidth=.9)

plt.figure(figsize=(8,ydim*4))
for i in range(n_combos):
    plt.subplot(ydim,2,i+1)  
    plt.imshow(np.transpose(mean_decision[:,:,i])), plt.colorbar()
    plt.axvline(x=max_window-extension, color='k', linestyle = '--', linewidth=.9)
    plt.xlabel('Frames (decision poke at %s)' % (max_window-extension))
    plt.ylabel('Neuron ID')
    plt.axis('tight')
    if n_variables == 3:
        plt.title("%s = %s\n %s = %s\n%s = %s\nNum trials = %.0f" 
                  %(conditions[int(var_keys[i][0])-1], var_keys[i][1],
                    conditions[int(var_keys[i][2])-1], var_keys[i][3], 
                    conditions[int(var_keys[i][4])-1],
                    var_keys[i][5], nTrials[i])) 
    if n_variables == 2:
        plt.title("%s = %s\n %s = %s\nNum trials = %.0f" 
                  %(conditions[int(var_keys[i][0])-1], var_keys[i][1],
                    conditions[int(var_keys[i][2])-1], var_keys[i][3], nTrials[i]))
    if n_variables == 1:
        plt.title("%s = %s\nNum trials = %.0f" 
                  %(conditions[int(var_keys[i][0])-1], var_keys[i][1], nTrials[i]))
plt.tight_layout()

