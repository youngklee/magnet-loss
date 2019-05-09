import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from utils import *
from magnet_ops import *
from magnet_tools import *

def _pdist(a, b=None):
    sq_sum_a = tf.reduce_sum(tf.square(a), reduction_indices=[1])
    if b is None:
        return -2 * tf.matmul(a, tf.transpose(a)) + \
            tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_a, (1, -1))
    sq_sum_b = tf.reduce_sum(tf.square(b), reduction_indices=[1])
    return -2 * tf.matmul(a, tf.transpose(b)) + \
        tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_b, (1, -1))

def unimodal_magnet_loss(features, labels, margin=1.0, unique_labels=None):
    """Simple unimodal magnet loss.
    See::
        Rippel, Paluri, Dollar, Bourdev: Metric Learning With Adaptive
        Density Discrimination. ICLR, 2016.
    Parameters
    ----------
    features : tf.Tensor
        A matrix of shape NxM that contains the M-dimensional feature vectors
        of N objects (floating type).
    labels : tf.Tensor
        The one-dimensional array of length N that contains for each feature
        the associated class label (integer type).
    margin : float
        A scalar margin hyperparameter.
    unique_labels : Optional[tf.Tensor]
        Optional tensor of unique values in `labels`. If None given, computed
        from data.
    Returns
    -------
    tf.Tensor
        A scalar loss tensor.
    """
    nil = tf.constant(0., tf.float32)
    one = tf.constant(1., tf.float32)
    minus_two = tf.constant(-2., tf.float32)
    eps = tf.constant(1e-4, tf.float32)
    margin = tf.constant(margin, tf.float32)

    num_per_class = None
    if unique_labels is None:
        unique_labels, sample_to_unique_y, num_per_class = tf.unique_with_counts(labels)
        num_per_class = tf.cast(num_per_class, tf.float32)

    y_mat = tf.cast(tf.equal(
        tf.reshape(labels, (-1, 1)), tf.reshape(unique_labels, (1, -1))),
        dtype=tf.float32)

    # If class_means is None, compute from batch data.
    if num_per_class is None:
        num_per_class = tf.reduce_sum(y_mat, reduction_indices=[0])
    class_means = tf.reduce_sum(
        tf.expand_dims(tf.transpose(y_mat), -1) * tf.expand_dims(features, 0),
        reduction_indices=[1]) / tf.expand_dims(num_per_class, -1)

    squared_distance = _pdist(features, class_means)

    num_samples = tf.cast(tf.shape(labels)[0], tf.float32)
    variance = tf.reduce_sum(
        y_mat * squared_distance) / (num_samples - one)

    const = one / (minus_two * (variance + eps))
    linear = const * squared_distance - y_mat * margin

    maxi = tf.reduce_max(linear, reduction_indices=[1], keepdims=True)
    loss_mat = tf.exp(linear - maxi)

    a = tf.reduce_sum(y_mat * loss_mat, reduction_indices=[1])
    b = tf.reduce_sum((one - y_mat) * loss_mat, reduction_indices=[1])
    loss = tf.maximum(nil, -tf.log(eps + a / (eps + b)))
    return tf.reduce_mean(loss), loss

mnist = input_data.read_data_sets('MNIST_data')
x_train, y_train = mnist.train.images, mnist.train.labels

learning_rate = 0.01
n_epochs = 200
batch_size = 100
feature_size = 784
embed_size = 2

n_train = len(x_train)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(1000)
train_data = train_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                          train_data.output_shapes)

features, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)

w1 = tf.Variable(tf.random_normal(shape=[feature_size, 393], stddev=0.01), name="weights1")
b1 = tf.Variable(tf.zeros([1, 393]), name="bias1")
h1 = tf.add(tf.matmul(tf.cast(features, tf.float32), w1), b1)
h1 = tf.layers.batch_normalization(h1, training=True)
h1 = tf.nn.relu(h1)

w2 = tf.Variable(tf.random_normal(shape=[393, embed_size], stddev=0.01), name="weights2")
b2 = tf.Variable(tf.zeros([1, embed_size]), name="bias2")
h2 = tf.add(tf.matmul(tf.cast(h1, tf.float32), w2), b2)
h2 = tf.layers.batch_normalization(h2, training=True)
emb = tf.nn.sigmoid(h2)

train_loss, losses = unimodal_magnet_loss(emb, label)
h2
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    extract = lambda x: sess.run(emb, feed_dict={features: x})
    initial_reps = compute_reps(extract, x_train, 400)
    
    batch_losses = []
    for i in range(n_epochs):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, train_loss])
                batch_losses.append(l)
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        
        if i % 10 == 0:
            print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

    final_reps = compute_reps(extract, x_train, 400)