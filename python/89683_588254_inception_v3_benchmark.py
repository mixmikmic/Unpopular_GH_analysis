from __future__ import print_function
from datetime import datetime
import time
import math
import tensorflow as tf
slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def inception_v3_arg_scope(weight_decay=0.00004, stddev=0.1, 
                                   batch_norm_var_collection='moving_vars'):
    
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }
    
    # slim.arg_scope automatically assigns default values to parameters
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                       weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d],
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           activation_fn=tf.nn.relu,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=batch_norm_params) as sc:
            return sc

def inception_v3(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 is_training=True,
                 restore_logits=True,
                 reuse=None,
                 scope='inceptionV3'):
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.variable_scope(scope, 'inceptionV3', [inputs, num_classes], reuse=reuse) as scope:
    with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.dropout]):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
        # 299 x 299 x 3
        end_points['conv0'] = slim.conv2d(inputs, 32, [3, 3], stride=2,
                                         scope='conv0')
        # 149 x 149 x 32
        end_points['conv1'] = slim.conv2d(end_points['conv0'], 32, [3, 3],
                                         scope='conv1')
        # 147 x 147 x 32
        end_points['conv2'] = slim.conv2d(end_points['conv1'], 64, [3, 3],
                                         padding='SAME', scope='conv2')
        # 147 x 147 x 64
        end_points['pool1'] = slim.max_pool2d(end_points['conv2'], [3, 3],
                                           stride=2, scope='pool1')
        # 73 x 73 x 64
        end_points['conv3'] = slim.conv2d(end_points['pool1'], 80, [1, 1],
                                         scope='conv3')
        # 73 x 73 x 80.
        end_points['conv4'] = slim.conv2d(end_points['conv3'], 192, [3, 3],
                                         scope='conv4')
        # 71 x 71 x 192.
        end_points['pool2'] = slim.max_pool2d(end_points['conv4'], [3, 3],
                                           stride=2, scope='pool2')
        # 35 x 35 x 192.
        net = end_points['pool2']
      
     # Inception blocks
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
        
        # mixed: 35 x 35 x 256.
        with tf.variable_scope('mixed_35x35x256a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = slim.conv2d(net, 48, [1, 1])
            branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 64, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 32, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x256a'] = net
        
        # mixed_1: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = slim.conv2d(net, 48, [1, 1])
            branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 64, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 64, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x288a'] = net
        
        # mixed_2: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = slim.conv2d(net, 48, [1, 1])
            branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 64, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 64, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x288b'] = net
        
        # mixed_3: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768a'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 64, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3],
                                      stride=2, padding='VALID')
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID')
          net = tf.concat(axis=3, values=[branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_17x17x768a'] = net
        
        # mixed4: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = slim.conv2d(net, 128, [1, 1])
            branch7x7 = slim.conv2d(branch7x7, 128, [1, 7])
            branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = slim.conv2d(net, 128, [1, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [1, 7])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768b'] = net
        
        # mixed_5: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768c'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = slim.conv2d(net, 160, [1, 1])
            branch7x7 = slim.conv2d(branch7x7, 160, [1, 7])
            branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = slim.conv2d(net, 160, [1, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [1, 7])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768c'] = net
        
        # mixed_6: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768d'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = slim.conv2d(net, 160, [1, 1])
            branch7x7 = slim.conv2d(branch7x7, 160, [1, 7])
            branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = slim.conv2d(net, 160, [1, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [1, 7])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768d'] = net
        
        # mixed_7: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768e'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = slim.conv2d(net, 192, [1, 1])
            branch7x7 = slim.conv2d(branch7x7, 192, [1, 7])
            branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = slim.conv2d(net, 192, [1, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768e'] = net
        
        # Auxiliary Head logits
        aux_logits = tf.identity(end_points['mixed_17x17x768e'])
        with tf.variable_scope('aux_logits'):
          aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
                                    padding='VALID')
          aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='proj')
          # Shape of feature map before the final layer.
          shape = aux_logits.get_shape()
          aux_logits = slim.conv2d(aux_logits, 768, shape[1:3], 
                               weights_initializer=trunc_normal(0.01), padding='VALID')
          aux_logits = slim.flatten(aux_logits)
          aux_logits = slim.fully_connected(aux_logits, num_classes, activation_fn=None,
                               weights_initializer=trunc_normal(0.01))
          end_points['aux_logits'] = aux_logits
        
        # mixed_8: 8 x 8 x 1280.
        # Note that the scope below is not changed to not void previous
        # checkpoints.
        # (TODO) Fix the scope when appropriate.
        with tf.variable_scope('mixed_17x17x1280a'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = slim.conv2d(net, 192, [1, 1])
            branch3x3 = slim.conv2d(branch3x3, 320, [3, 3], stride=2,
                                   padding='VALID')
          with tf.variable_scope('branch7x7x3'):
            branch7x7x3 = slim.conv2d(net, 192, [1, 1])
            branch7x7x3 = slim.conv2d(branch7x7x3, 192, [1, 7])
            branch7x7x3 = slim.conv2d(branch7x7x3, 192, [7, 1])
            branch7x7x3 = slim.conv2d(branch7x7x3, 192, [3, 3],
                                     stride=2, padding='VALID')
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID')
          net = tf.concat(axis=3, values=[branch3x3, branch7x7x3, branch_pool])
          end_points['mixed_17x17x1280a'] = net
        
        # mixed_9: 8 x 8 x 2048.
        with tf.variable_scope('mixed_8x8x2048a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 320, [1, 1])
          with tf.variable_scope('branch3x3'):
            branch3x3 = slim.conv2d(net, 384, [1, 1])
            branch3x3 = tf.concat(axis=3, values=[slim.conv2d(branch3x3, 384, [1, 3]),
                                                  slim.conv2d(branch3x3, 384, [3, 1])])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 448, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 384, [3, 3])
            branch3x3dbl = tf.concat(axis=3, values=[slim.conv2d(branch3x3dbl, 384, [1, 3]),
                                                     slim.conv2d(branch3x3dbl, 384, [3, 1])])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_8x8x2048a'] = net
        
        # mixed_10: 8 x 8 x 2048.
        with tf.variable_scope('mixed_8x8x2048b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 320, [1, 1])
          with tf.variable_scope('branch3x3'):
            branch3x3 = slim.conv2d(net, 384, [1, 1])
            branch3x3 = tf.concat(axis=3, values=[slim.conv2d(branch3x3, 384, [1, 3]),
                                                  slim.conv2d(branch3x3, 384, [3, 1])])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 448, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 384, [3, 3])
            branch3x3dbl = tf.concat(axis=3, values=[slim.conv2d(branch3x3dbl, 384, [1, 3]),
                                                     slim.conv2d(branch3x3dbl, 384, [3, 1])])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_8x8x2048b'] = net
        
        # Final pooling and prediction
        with tf.variable_scope('logits'):
          shape = net.get_shape()
          net = slim.avg_pool2d(net, shape[1:3], padding='VALID', scope='pool')
          # 1 x 1 x 2048
          net = slim.dropout(net, dropout_keep_prob, scope='dropout')
          net = slim.flatten(net, scope='flatten')
          # 2048
          logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='logits')
          # 1000
          end_points['logits'] = logits
          end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
      return logits, end_points

def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                     (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch' %
         (datetime.now(), info_string, num_batches, mn, sd))

batch_size = 32
height, width = 299, 299
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(inception_v3_arg_scope()):
    logits, end_points =  inception_v3(inputs, is_training=False)
    
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches=100
time_tensorflow_run(sess, logits, 'Forward')

