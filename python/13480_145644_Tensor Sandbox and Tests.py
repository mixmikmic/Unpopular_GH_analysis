
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

b = 10
h = 20
n = 5
x = tf.ones([b, h])
W = tf.ones([h,h,n])
y = tf.ones([b,h])

x = tf.expand_dims(x, [1])
# print x.get_shape()
x = tf.tile(x, [n,1,1])
# print x.get_shape()

y = tf.expand_dims(y, [1])
y = tf.tile(y, [n,1,1])
print y.get_shape()
W = tf.tile(tf.reshape(W, [-1, h, h]), [b, 1, 1])
print W.get_shape()

r = tf.batch_matmul(x, W)
print r.get_shape()
f = tf.squeeze(tf.batch_matmul(r, y, adj_y=True), [1,2])
print f.get_shape() 
f = tf.reshape(f, [b,n])
print f.get_shape()

a = tf.pack([1,2,3,4,5,6,7])
l = tf.pack([0,3,6])
print a[l].eval()

def one_hot(dense_labels, num_classes):
    sparse_labels = tf.reshape(dense_labels, [-1, 1])
    derived_size = tf.shape(dense_labels)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    outshape = tf.pack([derived_size, num_classes])
    labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
    return labels

def classification_loss(scores, class_labels, label_mask, margin=1.0, num_classes=3):
    """Calculate the classification loss of the network
    
    Args:
        - scores (Tensor[batch_size, num_classes]): 
            The matrix of predicted scores for all examples
            
        - class_labels (Tensor[batch_size, 1]):
            The list lof class labels with an expanded 2nd dimension
            
        - label_mask (Tensor[batch_size, num_classes], dtype=bool): 
            The boolean masked encoding of the class labels for the score tensor.
            Done this way because sparse indicator masks in tensorflow have unknown shapes...
    
    Returns:
        avg_class_loss: the average loss over all of the scores
    """
    # get the true values
    true_scores = tf.expand_dims(tf.boolean_mask(scores, label_mask), [1])
    # set true values for 'Other' class to zero (we don't actually model that class)
    others = (num_classes-1)*tf.ones_like(class_labels)
    true_scores = tf.select(tf.equal(class_labels, others),
                    tf.zeros_like(class_labels, dtype=tf.float32), 
                    true_scores, name="other_replace")
    # repeat the true score across columns for each row
    tile_true_scores = tf.tile(true_scores, [1, num_classes])

    # create margins same size as scores
    margins = margin*tf.ones_like(scores)
    
    # calculate the intermediate loss value inside the real loss function
    raw_loss = margins - tile_true_scores + scores
    
    # set the loss for true labels to 0
    raw_loss = tf.select(label_mask, tf.zeros_like(raw_loss), raw_loss)
        
    # SOFT PLUS LOSS
#     rank_loss = tf.nn.softplus(raw_loss)
    # HINGE LOSS
    rank_loss = tf.maximum(tf.zeros_like(scores, dtype=tf.float32), raw_loss)
    return tf.reduce_mean(rank_loss)
    

scores = tf.to_float(tf.pack([[1,2,3],[3,4,5], [5,6,7]]))
class_labels = tf.to_int64(tf.pack([[0],[1],[2]]))
sparse_class_labels = tf.SparseTensor(tf.transpose(tf.pack(
                                       [tf.to_int64(tf.range(tf.shape(class_labels)[0])), 
                                        tf.squeeze(class_labels)])), 
                                        tf.squeeze(class_labels), 
                                        tf.to_int64(class_labels.get_shape()))
true_mask = tf.squeeze(tf.pack([true_bool]))
margin = 1
num_classes = 3
classification_loss(scores, class_labels, true_mask).eval()

print"scores: "
print scores.eval()
flat_scores = tf.reshape(scores, [-1])
flat_true_indices = ( tf.squeeze(class_labels, [1]) 
                + tf.to_int64(tf.range(tf.shape(scores)[0])*tf.shape(scores)[1]))
# set true scores to 0 for 'Other'
flat_true_scores = tf.expand_dims(tf.gather(flat_scores, flat_true_indices), [1])
others = (num_classes-1)*tf.ones_like(class_labels)
true_scores = tf.select(tf.equal(class_labels, others),
                    tf.zeros_like(class_labels, dtype=tf.float32), 
                    flat_true_scores, name="other_replace")
# tile it to match size of all scores (ie, same true score along all columns for each row)
tile_true_scores = tf.tile(true_scores, [1, num_classes]) # [batch_size, num_classes]

# subtract margin from all scores where the score is the true class
# at these we'll have the loss is (margin - true_score + true_score) - margin = 0 
# print("Sparse Labels: ", tf.sparse_tensor_to_dense(sparse_class_labels).eval())
true_indicators = tf.sparse_to_indicator(sparse_class_labels, num_classes)
true_bool = true_indicators.eval()
print"Flat label indicator: ", true_indicators.get_shape()
scores = tf.select(true_indicators, (scores - margin), scores)
print"Augmented Scores: "
print scores.eval()
# now calculate the component-wise rank losses
# SOFT PLUS LOSS
#     rank_loss = tf.nn.softplus(self._margin*tf.ones_like(scores) - tile_true_scores + scores)
# HINGE LOSS
print "Rank Hinge Loss: "
rank_loss = tf.maximum(tf.zeros_like(scores, dtype=tf.float32), 
                   margin*tf.ones_like(scores) - tile_true_scores + scores)
print rank_loss.eval()

print scores.get_shape(), true_indicators.get_shape()
true_mask = tf.squeeze(tf.pack([true_bool]))
true_scores = tf.boolean_mask(scores, true_mask)
print true_scores.eval()

labels = class_labels.eval()

print labels

mask = np.zeros([len(labels), num_classes], dtype=np.bool)

for i in range(len(mask)):
    mask[i, labels[i]] = True

print mask

tf.bool

w = tf.ones([3,3,3])
x = 2*tf.ones([100,3])
y = 3*tf.ones([100,3])
z = 4*tf.ones([100,3])

score = tf.zeros([100])
for i in xrange(3):
    for j in xrange(3):
        for k in xrange(3):
            score += w[i,j,k]*x[:,i]*y[:,j]*z[:,k]

print score.get_shape()

seqs = tf.pack([[1., 1., 0., 0.],
                 [1., 1., 0., 0.],
                 [1., 1., 0., 0.]])
lens = tf.pack([2, 3, 4])
batch_size = lens.get_shape()[0]
print lens.get_shape()

tf.reshape(lens, [-1,1])

inputs = [ tf.select(tf.less(i, lens), seq, tf.zeros_like(seq)).eval()
            for i, seq in enumerate(tf.split(1, 4, seqs)) ]
print inputs[0].shape

avg = tf.truediv(tf.reshape(tf.add_n(inputs), [-1]), tf.to_float(lens))
print avg.eval()



