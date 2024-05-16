import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import os
from tensorflow.python.client import device_lib
from collections import Counter
import time

VERY_BIG_NUMBER = 1e30

f = open('../../Glove/word_embedding_glove', 'rb')
word_embedding = pickle.load(f)
f.close()

word_embedding = word_embedding[: len(word_embedding)-1]

f = open('../../Glove/vocab_glove', 'rb')
vocab = pickle.load(f)
f.close()

word2id = dict((w, i) for i,w in enumerate(vocab))
id2word = dict((i, w) for i,w in enumerate(vocab))

unknown_token = "UNKNOWN_TOKEN"

# Model Description
model_name = 'model-aw-lex-local-att-fast-v2-9'
model_dir = '../output/all-word/' + model_name
save_dir = os.path.join(model_dir, "save/")
log_dir = os.path.join(model_dir, "log")

if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

with open('../../../dataset/train_val_data_fine/all_word_lex','rb') as f:
    train_data, val_data = pickle.load(f)    
    
# Parameters
mode = 'train'
num_senses = 45
num_pos = 12
batch_size = 128
vocab_size = len(vocab)
unk_vocab_size = 1
word_emb_size = len(word_embedding[0])
max_sent_size = 200
hidden_size = 256
num_filter = 256
window_size = 5
kernel_size = 5
keep_prob = 0.3
l2_lambda = 0.001
init_lr = 0.001
decay_steps = 500
decay_rate = 0.99
clip_norm = 1
clipping = True
moving_avg_deacy = 0.999
num_gpus = 6
width = int(window_size/2)

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, axis=0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

# MODEL
device_num = 0
tower_grads = []
losses = []
predictions = []
predictions_pos = []

x = tf.placeholder('int32', [num_gpus, batch_size, max_sent_size], name="x")
y = tf.placeholder('int32', [num_gpus, batch_size, max_sent_size], name="y")
y_pos = tf.placeholder('int32', [num_gpus, batch_size, max_sent_size], name="y")
x_mask  = tf.placeholder('bool', [num_gpus, batch_size, max_sent_size], name='x_mask') 
sense_mask  = tf.placeholder('bool', [num_gpus, batch_size, max_sent_size], name='sense_mask')
is_train = tf.placeholder('bool', [], name='is_train')
word_emb_mat = tf.placeholder('float', [None, word_emb_size], name='emb_mat')
input_keep_prob = tf.cond(is_train,lambda:keep_prob, lambda:tf.constant(1.0))
pretrain = tf.placeholder('bool', [], name="pretrain")

global_step = tf.Variable(0, trainable=False, name="global_step")
learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate, staircase=True)
summaries = []

def global_attention(input_x, input_mask, W_att):
    h_masked = tf.boolean_mask(input_x, input_mask)
    h_tanh = tf.tanh(h_masked)
    u = tf.matmul(h_tanh, W_att)
    a = tf.nn.softmax(u)
    c = tf.reduce_sum(tf.multiply(h_masked, a), axis=0)  
    return c

with tf.variable_scope("word_embedding"):
    unk_word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[unk_vocab_size, word_emb_size], initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=0, dtype=tf.float32))
    final_word_emb_mat = tf.concat([word_emb_mat, unk_word_emb_mat], 0)

with tf.variable_scope(tf.get_variable_scope()):
    for gpu_idx in range(num_gpus):
        if gpu_idx>=3:
            device_num = 1
        with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device('/gpu:%d' % device_num):

            if gpu_idx > 0:
                    tf.get_variable_scope().reuse_variables()

            with tf.name_scope("word"):
                Wx = tf.nn.embedding_lookup(final_word_emb_mat, x[gpu_idx])  

            float_x_mask = tf.cast(x_mask[gpu_idx], 'float')
            x_len = tf.reduce_sum(tf.cast(x_mask[gpu_idx], 'int32'), axis=1)
            
            with tf.variable_scope("lstm1"):
                cell_fw1 = tf.contrib.rnn.BasicLSTMCell(hidden_size,state_is_tuple=True)
                cell_bw1 = tf.contrib.rnn.BasicLSTMCell(hidden_size,state_is_tuple=True)

                d_cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell_fw1, input_keep_prob=input_keep_prob)
                d_cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell_bw1, input_keep_prob=input_keep_prob)

                (fw_h1, bw_h1), _ = tf.nn.bidirectional_dynamic_rnn(d_cell_fw1, d_cell_bw1, Wx, sequence_length=x_len, dtype='float', scope='lstm1')
                h1 = tf.concat([fw_h1, bw_h1], 2)

            with tf.variable_scope("lstm2"):
                cell_fw2 = tf.contrib.rnn.BasicLSTMCell(hidden_size,state_is_tuple=True)
                cell_bw2 = tf.contrib.rnn.BasicLSTMCell(hidden_size,state_is_tuple=True)

                d_cell_fw2 = tf.contrib.rnn.DropoutWrapper(cell_fw2, input_keep_prob=input_keep_prob)
                d_cell_bw2 = tf.contrib.rnn.DropoutWrapper(cell_bw2, input_keep_prob=input_keep_prob)

                (fw_h2, bw_h2), _ = tf.nn.bidirectional_dynamic_rnn(d_cell_fw2, d_cell_bw2, h1, sequence_length=x_len, dtype='float', scope='lstm2')
                h = tf.concat([fw_h2, bw_h2], 2)
            
            with tf.variable_scope("local_attention"):
                W_att_local = tf.get_variable("W_att_local", shape=[2*hidden_size, 1], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=gpu_idx*10))
                flat_h = tf.reshape(h, [batch_size*max_sent_size, tf.shape(h)[2]])
                h_tanh = tf.tanh(flat_h)
                u_flat = tf.matmul(h_tanh, W_att_local)
                u_local = tf.reshape(u_flat, [batch_size, max_sent_size])
                final_u = (tf.cast(x_mask[gpu_idx], 'float') -1)*VERY_BIG_NUMBER + u_local
                c_local = tf.map_fn(lambda i:tf.reduce_sum(tf.multiply(h[:, tf.maximum(0, i-width-1):tf.minimum(1+width+i, max_sent_size)],
                    tf.expand_dims(tf.nn.softmax(final_u[:, tf.maximum(0, i-width-1):tf.minimum(1+width+i, max_sent_size)], 1), 2)), axis=1),
                    tf.range(max_sent_size), dtype=tf.float32)  
            
            c_local = tf.transpose(c_local, perm=[1,0,2]) 
            h_final = tf.concat([tf.multiply(c_local, tf.expand_dims(float_x_mask, 2)), h], 2)
            flat_h_final = tf.reshape(h_final, [-1, tf.shape(h_final)[2]])
            with tf.variable_scope("hidden_layer"):
                W = tf.get_variable("W", shape=[4*hidden_size, 2*hidden_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=gpu_idx*20))
                b = tf.get_variable("b", shape=[2*hidden_size], initializer=tf.zeros_initializer())
                drop_flat_h_final = tf.nn.dropout(flat_h_final, input_keep_prob)
                flat_hl = tf.matmul(drop_flat_h_final, W) + b

            with tf.variable_scope("softmax_layer"):
                W = tf.get_variable("W", shape=[2*hidden_size, num_senses], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=gpu_idx*20))
                b = tf.get_variable("b", shape=[num_senses], initializer=tf.zeros_initializer())
                drop_flat_hl = tf.nn.dropout(flat_hl, input_keep_prob)
                flat_logits_sense = tf.matmul(drop_flat_hl, W) + b
                logits = tf.reshape(flat_logits_sense, [batch_size, max_sent_size, num_senses])
                predictions.append(tf.argmax(logits, 2))

            with tf.variable_scope("softmax_layer_pos"):
                W = tf.get_variable("W", shape=[2*hidden_size, num_pos], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=gpu_idx*30))
                b = tf.get_variable("b", shape=[num_pos], initializer=tf.zeros_initializer())
                flat_h1 = tf.reshape(h1, [-1, tf.shape(h1)[2]])
                drop_flat_hl = tf.nn.dropout(flat_hl, input_keep_prob)
                flat_logits_pos = tf.matmul(drop_flat_hl, W) + b
                logits_pos = tf.reshape(flat_logits_pos, [batch_size, max_sent_size, num_pos])
                predictions_pos.append(tf.argmax(logits_pos, 2))


            float_sense_mask = tf.cast(sense_mask[gpu_idx], 'float')

            loss = tf.contrib.seq2seq.sequence_loss(logits, y[gpu_idx], float_sense_mask, name="loss")
            loss_pos = tf.contrib.seq2seq.sequence_loss(logits_pos, y_pos[gpu_idx], float_x_mask, name="loss_")

            l2_loss = l2_lambda * tf.losses.get_regularization_loss()

            total_loss = tf.cond(pretrain, lambda:loss_pos, lambda:loss + loss_pos + l2_loss)

            summaries.append(tf.summary.scalar("loss_{}".format(gpu_idx), loss))
            summaries.append(tf.summary.scalar("loss_pos_{}".format(gpu_idx), loss_pos))
            summaries.append(tf.summary.scalar("total_loss_{}".format(gpu_idx), total_loss))


            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_vars = optimizer.compute_gradients(total_loss)

            clipped_grads = grads_vars
            if(clipping == True):
                clipped_grads = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in clipped_grads]

            tower_grads.append(clipped_grads)
            losses.append(total_loss)

with tf.device('/gpu:0'):
    tower_grads = average_gradients(tower_grads)
    losses = tf.add_n(losses)/len(losses)
    apply_grad_op = optimizer.apply_gradients(tower_grads, global_step=global_step)
    summaries.append(tf.summary.scalar('total_loss', losses))
    summaries.append(tf.summary.scalar('learning_rate', learning_rate))

    variable_averages = tf.train.ExponentialMovingAverage(moving_avg_deacy, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(apply_grad_op, variables_averages_op)
    saver = tf.train.Saver(tf.global_variables())
    summary = tf.summary.merge(summaries)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
# print (device_lib.list_local_devices())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())                          # For initializing all the variables
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)          # For writing Summaries

save_period = 100
log_period = 100

def model(xx, yy, yy_pos, mask, smask, train_cond=True, pretrain_cond=False):
    num_batches = int(len(xx)/(batch_size*num_gpus))
    _losses = 0
    temp_loss = 0
    preds_sense = []
    true_sense = []
    preds_pos = []
    true_pos = []
    
    for j in range(num_batches): 
        
        s = j * batch_size * num_gpus
        e = (j+1) * batch_size * num_gpus
        xx_re = xx[s:e].reshape([num_gpus, batch_size, -1])
        yy_re = yy[s:e].reshape([num_gpus, batch_size, -1])
        yy_pos_re = yy_pos[s:e].reshape([num_gpus, batch_size, -1])
        mask_re = mask[s:e].reshape([num_gpus, batch_size, -1])
        smask_re = smask[s:e].reshape([num_gpus, batch_size, -1])
 
        feed_dict = {x:xx_re, y:yy_re, y_pos:yy_pos_re, x_mask:mask_re, sense_mask:smask_re, pretrain:pretrain_cond, is_train:train_cond, input_keep_prob:keep_prob, word_emb_mat:word_embedding}
        
        if(train_cond==True):
            _, _loss, step, _summary = sess.run([train_op, losses, global_step, summary], feed_dict)
            summary_writer.add_summary(_summary, step)
            
            temp_loss += _loss
            if((j+1)%log_period==0):
                print("Steps: {}".format(step), "Loss:{0:.4f}".format(temp_loss/log_period), ", Current Loss: {0:.4f}".format(_loss))
                temp_loss = 0
            if((j+1)%save_period==0):
                saver.save(sess, save_path=save_dir)                         
                
        else:
            _loss, pred, pred_pos = sess.run([total_loss, predictions, predictions_pos], feed_dict)
            for i in range(num_gpus):
                preds_sense.append(pred[i][smask_re[i]])
                true_sense.append(yy_re[i][smask_re[i]])
                preds_pos.append(pred_pos[i][mask_re[i]])
                true_pos.append(yy_pos_re[i][mask_re[i]])

        _losses +=_loss

    if(train_cond==False): 
        sense_preds = []
        sense_true = []
        pos_preds = []
        pos_true = []
        
        for preds in preds_sense:
            for ps in preds:      
                sense_preds.append(ps)  
        for trues in true_sense:
            for ts in trues:
                sense_true.append(ts)
        
        for preds in preds_pos:
            for ps in preds:      
                pos_preds.append(ps)      
        for trues in true_pos:
            for ts in trues:
                pos_true.append(ts)
                
        return _losses/num_batches, sense_preds, sense_true, pos_preds, pos_true

    return _losses/num_batches, step

def eval_score(yy, pred, yy_pos, pred_pos):
    f1 = f1_score(yy, pred, average='macro')
    accu = accuracy_score(yy, pred)
    f1_pos = f1_score(yy_pos, pred_pos, average='macro')
    accu_pos = accuracy_score(yy_pos, pred_pos)
    return f1*100, accu*100, f1_pos*100, accu_pos*100

x_id_train = train_data['x']
mask_train = train_data['x_mask']
sense_mask_train = train_data['sense_mask']
y_train = train_data['y']
y_pos_train = train_data['pos']

x_id_val = val_data['x']
mask_val = val_data['x_mask']
sense_mask_val = val_data['sense_mask']
y_val = val_data['y']
y_pos_val = val_data['pos']

def testing():
    start_time = time.time()
    val_loss, val_pred, val_true, val_pred_pos, val_true_pos = model(x_id_val, y_val, y_pos_val, mask_val, sense_mask_val, train_cond=False)        
    f1_, accu_, f1_pos_, accu_pos_ = eval_score(val_true, val_pred, val_true_pos, val_pred_pos)
    time_taken = time.time() - start_time
    print("Val: F1 Score:{0:.2f}".format(f1_), "Accuracy:{0:.2f}".format(accu_), " POS: F1 Score:{0:.2f}".format(f1_pos_), "Accuracy:{0:.2f}".format(accu_pos_), "Loss:{0:.4f}".format(val_loss), ", Time: {0:.1f}".format(time_taken))
    return f1_, accu_, f1_pos_, accu_pos_

def training(current_epoch, pre_train_cond):
        random = np.random.choice(len(y_train), size=(len(y_train)), replace=False)
        x_id_train_tmp = x_id_train[random]
        y_train_tmp = y_train[random]
        mask_train_tmp = mask_train[random]    
        sense_mask_train_tmp = sense_mask_train[random]
        y_pos_train_tmp = y_pos_train[random]

        start_time = time.time()
        train_loss, step = model(x_id_train_tmp, y_train_tmp, y_pos_train_tmp, mask_train_tmp, sense_mask_train_tmp, pretrain_cond=pre_train_cond)
        time_taken = time.time() - start_time
        print("Epoch: {}".format(current_epoch+1),", Step: {}".format(step), ", loss: {0:.4f}".format(train_loss), ", Time: {0:.1f}".format(time_taken))
        saver.save(sess, save_path=save_dir)                         
        print("Model Saved")
        return [step, train_loss]

loss_collection = []
val_collection = []
num_epochs = 20
val_period = 2

for i in range(num_epochs):
    loss_collection.append(training(i, False))
    if((i+1)%val_period==0):
        val_collection.append(testing())

loss_collection = []
val_collection = []
num_epochs = 20
val_period = 2

for i in range(num_epochs):
    loss_collection.append(training(i, False))
    if((i+1)%val_period==0):
        val_collection.append(testing())

loss_collection = []
val_collection = []
num_epochs = 20
val_period = 2

for i in range(num_epochs):
    loss_collection.append(training(i, False))
    if((i+1)%val_period==0):
        val_collection.append(testing())

loss_collection = []
val_collection = []
num_epochs = 20
val_period = 2

for i in range(num_epochs):
    loss_collection.append(training(i, False))
    if((i+1)%val_period==0):
        val_collection.append(testing())



start_time = time.time()
train_loss, train_pred, train_true, train_pred_pos, train_true_pos = model(x_id_train, y_train, y_pos_train, mask_train, sense_mask_train, train_cond=False)        
f1_, accu_, f1_pos_, accu_pos_ = etrain_score(train_true, train_pred, train_true_pos, train_pred_pos)
time_taken = time.time() - start_time
print("train: F1 Score:{0:.2f}".format(f1_), "Accuracy:{0:.2f}".format(accu_), " POS: F1 Score:{0:.2f}".format(f1_pos_), "Accuracy:{0:.2f}".format(accu_pos_), "Loss:{0:.4f}".format(train_loss), ", Time: {0:.1f}".format(time_taken))

saver.restore(sess, save_dir)

