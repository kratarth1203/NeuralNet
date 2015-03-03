
# Author: Kratarth Goel
# BITS Pilani Goa Campus (2014)
# RNN-DBN for polyphonic music generation
# for any further clarifications visit
# for the ICANN 2014 paper or email me @ kratarthgoel@gmail.com
# This code is based on the one writen by Nicolas Boulanger-Lewandowski
# University of Montreal (2012)
# RNN-RBM deep learning tutorial
# More information at http://deeplearning.net/tutorial/rnnrbm.html

import os
import sys
import tables
import tarfile
import fnmatch
import random
import numpy
import numpy as np
from scipy.io import wavfile
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from midify import melspec, invmelspec, soundsc, stft, istft
from sklearn.cluster import KMeans

from theano.compat.python2x import OrderedDict


import cPickle
#Don't use python long as this doesn't work on 32 bit computers.
numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False

from midify import lpc_analysis, lpc_to_lsf, lpc_synthesis
from midify import lsf_to_lpc, soundsc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_clip_rmsprop_updates(params, cost, gparams, 
                                 learning_rate, momentum, rescale=5. ):

    updates = OrderedDict()

    #if not hasattr(self, "running_average_"):
    running_square_ = [0.] * len(gparams)
    running_avg_ = [0.] * len(gparams)
    updates_storage_ = [0.] * len(gparams)

    #if not hasattr(self, "momentum_velocity_"):
    momentum_velocity_ = [0.] * len(gparams)

    # Gradient clipping
    grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), gparams)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    grad_norm = T.sqrt(grad_norm)
    scaling_num = rescale
    scaling_den = T.maximum(rescale, grad_norm)
    for n, (param, gparam) in enumerate(zip(params, gparams)):
        gparam = T.switch(not_finite, 0.1 * param,
                          gparam * (scaling_num / scaling_den))
        combination_coeff = 0.9
        minimum_grad = 1e-4
        old_square = running_square_[n]
        new_square = combination_coeff * old_square + (
            1. - combination_coeff) * T.sqr(gparam)
        old_avg = running_avg_[n]
        new_avg = combination_coeff * old_avg + (
            1. - combination_coeff) * gparam
        rms_grad = T.sqrt(new_square - new_avg ** 2)
        rms_grad = T.maximum(rms_grad, minimum_grad)
        velocity = momentum_velocity_[n]
        update_step = momentum * velocity - learning_rate * (
            gparam / rms_grad)
        running_square_[n] = new_square
        running_avg_[n] = new_avg
        updates_storage_[n] = update_step
        momentum_velocity_[n] = update_step
        updates[param] = param + update_step

    return updates


def load_fruitspeech(fruit_list = ['apple', 'pineapple']):
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'audio.tar.gz'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if not os.path.isfile(data_file):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
            url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
        except AttributeError:
            import urllib.request as urllib
            url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    if not os.path.exists(os.path.join(data_path, "audio")):
        tar = tarfile.open(data_file)
        os.chdir(data_path)
        tar.extractall()
        tar.close()
    
    h5_file_path = os.path.join(data_path, "saved_fruit.h5")
    if not os.path.exists(h5_file_path):
        audio_matches = []
        
        data_path = os.path.join(data_path, "audio")
        for root, dirnames, filenames in os.walk(data_path):
            for fruit in fruit_list:
                for filename in fnmatch.filter(filenames, fruit + '*.wav'):
                    audio_matches.append(os.path.join(root, filename))

        random.seed(1999)
        random.shuffle(audio_matches)

        # http://mail.scipy.org/pipermail/numpy-discussion/2011-March/055219.html
        h5_file = tables.openFile(h5_file_path, mode='w')
        data_x = h5_file.createVLArray(h5_file.root, 'data_x',
                                       tables.Float32Atom(shape=()),
                                       filters=tables.Filters(1))
        data_y = h5_file.createVLArray(h5_file.root, 'data_y',
                                       tables.Int32Atom(shape=()),
                                       filters=tables.Filters(1))
        for wav_path in audio_matches:
            # Convert chars to int classes
            word = wav_path.split(os.sep)[-1][:-6]
            chars = [ord(c) - 97 for c in word]
            data_y.append(np.array(chars, dtype='int32'))
            fs, d = wavfile.read(wav_path)
            # Preprocessing from A. Graves "Towards End-to-End Speech
            # Recognition"
            data_x.append(d.astype('float32'))
        h5_file.close()

    h5_file = tables.openFile(h5_file_path, mode='r')
    data_x = h5_file.root.data_x
    data_y = h5_file.root.data_y

    # FIXME: HACKING
    train_x = data_x
    train_y = data_y
    valid_x = data_x
    valid_y = data_y
    test_x = data_x
    test_y = data_y
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def build_rbm(v, W, bv, bh, k):
    '''Construct a k-step Gibbs chain starting at v for an RBM.

v : Theano vector or matrix
  If a matrix, multiple chains will be run in parallel (batch).
W : Theano matrix
  Weight matrix of the RBM.
bv : Theano vector
  Visible bias vector of the RBM.
bh : Theano vector
  Hidden bias vector of the RBM.
k : scalar or Theano scalar
  Length of the Gibbs chain.

Return a (v_sample, cost, monitor, updates) tuple:

v_sample : Theano vector or matrix with the same shape as `v`
  Corresponds to the generated sample(s).
cost : Theano scalar
  Expression whose gradient with respect to W, bv, bh is the CD-k approximation
  to the log-likelihood of `v` (training example) under the RBM.
  The cost is averaged in the batch case.
monitor: Theano scalar
  Pseudo log-likelihood (also averaged in the batch case).
updates: dictionary of Theano variable -> Theano variable
  The `updates` object returned by scan.'''

    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
        v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)
        return mean_v, v

    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                 n_steps=k)
    v_sample = chain[-1]
    mean_v = gibbs_step(v_sample)[0]
    monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates


def shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
elements.'''
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))


def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


def build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent):
    '''Construct a symbolic RNN-RBM and initialize parameters.

n_visible : integer
  Number of visible units.
n_hidden : integer
  Number of hidden units of the conditional RBMs.
n_hidden_recurrent : integer
  Number of hidden units of the RNN.

Return a (v, v_sample, cost1, monitor1, params1, updates_train1,cost2, monitor2, params2, updates_train2, v_t,
          updates_generate) tuple:

v : Theano matrix
  Symbolic variable holding an input sequence (used during training)
v_sample : Theano matrix
  Symbolic variable holding the negative particles for CD log-likelihood
  gradient estimation (used during training)
cost1(2) : Theano scalar
  Expression whose gradient (considering v_sample constant) corresponds to the
  LL gradient of the RNN-RBM1(2) i.e. the visible layer and the first hidden layer of the DBN
  (used during training)
monitor1(2) : Theano scalar
  Frame-level pseudo-likelihood (useful for monitoring during training) for RNN_RBM1(2)
params1(2) : tuple of Theano shared variables
  The parameters of the RNN-RBM1(2) model to be optimized during training.
updates_train1(2) : dictionary of Theano variable -> Theano variable
  Update object that should be passed to theano.function when compiling the
  training function for the RNN-RBM1(2).
v_t : Theano matrix
  Symbolic variable holding a generated sequence (used during sampling)
updates_generate : dictionary of Theano variable -> Theano variable
  Update object that should be passed to theano.function when compiling the
  generation function.'''

    W1 = shared_normal(n_visible, n_hidden, 0.01)
    bv = shared_zeros(n_visible)
    bh1 = shared_zeros(n_hidden)
    Wuh1 = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
    Wuv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
    Wvu = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
    Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bu = shared_zeros(n_hidden_recurrent)

    params1 = W1, bv, bh1, Wuh1, Wuv, Wvu, Wuu, bu  # learned parameters as shared
                                                    # variables for RNN_RBM1
    W2 = shared_normal(n_hidden, n_hidden, 0.01)
    bh2 = shared_zeros(n_hidden)
    Wuh2 = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)

    params2 = W2, bh2, bh1, Wuh2, Wuh1 # learned parameters as shared
                                                # variables for RNN-RBM2

    v = T.matrix()  # a training sequence
    lin_output = T.dot(v, W1) + bh1
    activation = theano.tensor.nnet.sigmoid
    h = activation(lin_output)
    u0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden
                                         # units

    # deterministic recurrence to compute the variable
    # biases bv_t , bh1_t at each time step.
    def recurrence1(v_t, u_tm1):
        bv_t = bv + T.dot(u_tm1, Wuv)
        bh1_t = bh1 + T.dot(u_tm1, Wuh1)
        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))
        return [u_t, bv_t, bh1_t]

    # If `h_t` is given, deterministic recurrence to compute the variable
    # biases bh1_t, bh2_t at each time step. If `h_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # of the top layer RBM from the RNN-DBN. The resulting sample v_t is returned
    # in order to be passed down to the sequence history.
    def recurrence2(v_t,h_t, u_tm1):
        bh1_t = bh1 + T.dot(u_tm1, Wuh1)
        bh2_t = bh2 + T.dot(u_tm1, Wuh2)
        generate = h_t is None
        if generate:
            h_t, _, _, updates = build_rbm(T.zeros((n_hidden,)), W2, bh1_t,
                                           bh2_t, k=25)
        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))
        return ([u_t, h_t], updates) if generate else [u_t, bh1_t, bh2_t]

    # function used for generation of a sample from the RNN_DBN.
    # Starting with the sampling if the first hidden layer by
    # Gibbs Sampling in the top layer RBM of the RNN_DBN, which involves
    # generation of the RBM parameters that depend upon the RNN.
    # This is followed by generation of the visible layer sample.
    def generate(u_tm1):
        bh1_t = bh1 + T.dot(u_tm1, Wuh1)
        bh2_t = bh2 + T.dot(u_tm1, Wuh2)
        bv_t = bv + T.dot(u_tm1, Wuv)
        h_t, _, _, updates = build_rbm(T.zeros((n_hidden,)), W2, bh1_t,
                                           bh2_t, k=25)
        lin_v_t = T.dot(h_t, W1.T) + bv_t
        mean_v = activation(lin_v_t)
        v_t = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)
        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))
        return ([u_t,v_t],updates)
    # For training, the deterministic recurrence is used to compute all the
    # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
    # in batches using those parameters.
    (u_t, bv_t, bh1_t), updates_train1 = theano.scan(
        lambda v_t, u_tm1, *_: recurrence1(v_t, u_tm1),
        sequences=v, outputs_info=[u0, None, None], non_sequences=params1)
    v_sample, cost1, monitor1, updates_rbm1 = build_rbm(v, W1, bv_t[:], bh1_t[:],
                                                     k=15)
    updates_train1.update(updates_rbm1)


    (u_t, bh1_t, bh2_t), updates_train2 = theano.scan(
        lambda v_t, h_t, u_tm1, *_: recurrence2(v_t , h_t, u_tm1),
        sequences=[v,h], outputs_info=[u0, None, None], non_sequences=params2)

    h1_sample, cost2, monitor2, updates_rbm2 = build_rbm(h, W2, bh1_t[:], bh2_t[:],
                                                     k=15)

    updates_train2.update(updates_rbm2)

    # symbolic loop for sequence generation
    (u_t,v_t), updates_generate = theano.scan(
        lambda u_tm1,*_ : generate(u_tm1), outputs_info = [u0,None],
        non_sequences = params2, n_steps=50)


    return (v, v_sample, cost1, monitor1, params1, updates_train1, h, h1_sample,cost2, monitor2, params2, updates_train2, v_t,
            updates_generate)
    '''
    return (v, v_sample, cost1, monitor1, params1, updates_train1, v_t,
            updates_generate)
    '''


def Adam(cost, params, consider_constant, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params, consider_constant)
    i = theano.shared(np.array([0.,]).astype(theano.config.floatX))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates


class RnnRbm:
    '''Simple class to train an RNN-RBM from MIDI files and to generate sample
                 r=(21, 109), dt=0.3)
sequences.'''

    def __init__(self, n_vis, n_hidden=300, n_hidden_recurrent=200, lr=0.0005,
                 r=(1, 401), dt=0.3, momentum=0.9999):
        '''Constructs and compiles Theano functions for training and sequence
generation.

n_vis : integer
n_hidden : integer
  Number of hidden units of the conditional RBMs.
n_hidden_recurrent : integer
  Number of hidden units of the RNN.
lr : float
  Learning rate
       '''

        (v, v_sample, cost1, monitor1, params1, updates_train1, h, h1_sample , cost2, monitor2, params2, updates_train2, v_t,
         updates_generate) = build_rnnrbm(n_vis, n_hidden,
                                           n_hidden_recurrent)
        '''
        (v, v_sample, cost1, monitor1, params1, updates_train1,v_t,
         updates_generate) = build_rnnrbm(n_vis, n_hidden,
                                           n_hidden_recurrent)
        '''
        gradient1 = T.grad(cost1, params1, consider_constant=[v_sample])
        #updates_train1.update(((p, p - lr * g) for p, g in zip(params1,
        #                                                        gradient1)))
        
        updates_rmsprop1 = get_clip_rmsprop_updates(params1, cost1, gradient1,  lr, momentum )
        updates_train1.update(updates_rmsprop1)
        
        gradient2 = T.grad(cost2, params2, consider_constant=[h1_sample])
        #updates_train2.update(((p, p - lr * g) for p, g in zip(params2,
        #                                                        gradient2)))
        
        updates_rmsprop2 = get_clip_rmsprop_updates(params2, cost2, gradient2,  lr, momentum )
        updates_train2.update(updates_rmsprop2)
        
        self.train_function1 = theano.function([v], monitor1,
                                               updates=updates_train1)

        self.train_function2 = theano.function([v], monitor2,
                                               updates=updates_train2)
        
        self.generate_function = theano.function([], v_t,
                                                 updates=updates_generate)

    def train_RNNRBM1(self, dataset, batch_size=100, num_epochs=200):
        print len(dataset)
        print len(dataset[0])
        a = []
        for item in dataset:
            a.append(len(item))
        print np.mean(a)
        print (dataset[0])
        print len(dataset[0][0])
        try:
            for epoch in xrange(num_epochs):
                costs1 = []
                np.random.shuffle(dataset)
                for s, sequence in enumerate(dataset):
                    for i in xrange(0, len(sequence), batch_size):
                        cost1 = self.train_function1(sequence[i:i + batch_size])
                        costs1.append(cost1)
                print 'Epoch %i/%i' % (epoch + 1, num_epochs),
                print numpy.mean(costs1)
                sys.stdout.flush()

        except KeyboardInterrupt:
            print 'Interrupted by user.'

    def train_RNNRBM2(self, dataset, batch_size=100, num_epochs=200):
        try:
            for epoch in xrange(num_epochs):
                costs2 = []
                for s, sequence in enumerate(dataset):
                    for i in xrange(0, len(sequence), batch_size):
                        cost2 = self.train_function2(sequence[i:i + batch_size])
                        costs2.append(cost2)
                print 'For 2nd layer Epoch %i/%i' % (epoch + 1, num_epochs),
                print numpy.mean(costs2)
                sys.stdout.flush()

        except KeyboardInterrupt:
            print 'Interrupted by user.'


    def generate(self):
        generated = self.generate_function()
        return generated


def test_rnnrbm(X, batch_size=100, num_epochs=200):
    model = RnnRbm(X[0].shape[1])
    model.train_RNNRBM1(X,
                batch_size=batch_size, num_epochs=47)
    model.train_RNNRBM2(X,
                batch_size=batch_size, num_epochs=200)
    return model

if __name__ == '__main__':
    train, valid, test = load_fruitspeech()
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test
    # load into main memory and normalize between -1 and 1
    train_x = [x / (2. ** 15) for x in train_x[:]]
    new_x = []
    for x in train_x:
        for i in [0,1,-1,2,-2,3,-3,4,-4, 5]:
            x_hat = np.roll(x, i)
            new_x.append(x_hat)
    train_x = new_x    
    fs = 8000
    lpc_order = 8
    window_size = 128
    window_step = window_size
    lsf_kmeans_samples = []
    gain_kmeans_samples = []
    for n, d in enumerate(train_x):
        a, gain, e = lpc_analysis(d, order=lpc_order,
                                  window_step=window_step,
                                  window_size=window_size, emphasis=0.9,
                                  copy=True)
        print a.shape
        lsf = lpc_to_lsf(a)
        length, n_features = lsf.shape
        slice_sz = 10
        lo = slice_sz // 2
        hi = len(lsf) - slice_sz // 2
        if lo > hi:
            # ??? Shouldn't happen unless sequence is too short
            continue
        r = np.random.randint(lo, hi)
        lsf_kmeans_samples.append(
            lsf[r - slice_sz // 2:r + slice_sz // 2])
        gain_kmeans_samples.append(
            gain[r - slice_sz // 2:r + slice_sz // 2])

    lsf_kmeans_samples = np.array(lsf_kmeans_samples)
    lsf_kmeans_samples = lsf_kmeans_samples.reshape(
        lsf_kmeans_samples.shape[0] * lsf_kmeans_samples.shape[1], -1)
    gain_kmeans_samples = np.array(gain_kmeans_samples)
    gain_kmeans_samples = gain_kmeans_samples.reshape(
        gain_kmeans_samples.shape[0] * gain_kmeans_samples.shape[1], -1)

    print("Fitting Kmeans...")
    n_lsf_clusters = 100
    lsf_tf = KMeans(n_clusters=n_lsf_clusters, random_state=1999)
    lsf_tf.fit(lsf_kmeans_samples)
    n_gain_clusters = 100
    gain_tf = KMeans(n_clusters=n_gain_clusters, random_state=1999)
    gain_tf.fit(gain_kmeans_samples)

    print("Generating dataset")
    X = []
    for n, d in enumerate(train_x):
        a, gain, e = lpc_analysis(d, order=lpc_order,
                                  window_step=window_step,
                                  window_size=window_size, emphasis=0.9,
                                  copy=True)
        lsf = lpc_to_lsf(a)
        length, n_features = lsf.shape
        lsf_labels = lsf_tf.predict(lsf)
        lsf_one_hot = np.zeros((length, n_lsf_clusters)).astype(
            theano.config.floatX)
        gain_labels = gain_tf.predict(gain)
        gain_one_hot = np.zeros((length, n_gain_clusters)).astype(
            theano.config.floatX)
        for i in np.arange(length):
            lsf_one_hot[i, lsf_labels[i]] = 1.
            gain_one_hot[i, gain_labels[i]] = 1.
        one_hot = np.concatenate((lsf_one_hot, gain_one_hot), axis=1)
        X.append(one_hot)

    def reconstruct(codebook, one_hot, n_features):
        arr = np.zeros((len(one_hot), n_features))
        prev_code = None
        for i in range(len(arr)):
            idx = np.where(one_hot[i])[0]
            code = codebook[idx]
            if len(idx) > 1 and prev_code is not None:
                if prev_code in code:
                    print("Using previous code")
                    code = prev_code
                else:
                    code = code.mean(axis = 0 )
            elif len(idx) == 1:
                code = codebook[idx]
            else:
                if prev_code is not None:
                    code = prev_code
                else:
                    code = codebook[0]
            arr[i] = code
            prev_code = code
        return arr   
    print len(X)
    for i in [0, 3, 5, 8, 13]:
        lsf = reconstruct(lsf_tf.cluster_centers_, X[i][:, :n_lsf_clusters],
                          n_features)
        a = lsf_to_lpc(lsf)
        gain = reconstruct(gain_tf.cluster_centers_, X[i][:, -n_gain_clusters:],
                           1)
        X_s = lpc_synthesis(a, gain, window_step=window_step)
        wavfile.write('orig_%i.wav' % i, fs, soundsc(X_s))

    model = test_rnnrbm(X, num_epochs=200)
    n_samples_to_gen = 5
    for i in range(n_samples_to_gen):
        g = model.generate()
        #f = open('sample_peach_pickle_rnndbn_%i.pkl' % i, 'w')
        #cPickle.dump(g,f)
        #f.close()
        lsf = reconstruct(lsf_tf.cluster_centers_, g[:, :n_lsf_clusters],
                          n_features)
        a = lsf_to_lpc(lsf)
        gain = reconstruct(gain_tf.cluster_centers_, g[:, -n_gain_clusters:],
                           1)
        X_s = lpc_synthesis(a, gain, window_step=window_step)
        wavfile.write('sample_apple_pineapple_rnndbn_%i.wav' % i, fs, soundsc(X_s))


'''
Good configurations:-
1) For 'peach/kiwi' with 10X jittering;
    def __init__(self, n_vis, n_hidden=300, n_hidden_recurrent=200, lr=0.0005,
                 r=(1, 401), dt=0.3, momentum=0.9999):

Num epochs for RBM1 : 89 / 159
Num epochs for RBM2 : 127 / 127

'''
