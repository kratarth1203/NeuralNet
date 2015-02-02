 
# Author: Kratarth Goel
# BITS Pilani Goa Campus (2014)
# RNN-DBN for polyphonic music generation
# for any further clarifications visit 
# for the ICANN 2014 paper or email me @ kratarthgoel@gmail.com
# This code is based on the one writen by Nicolas Boulanger-Lewandowski
# University of Montreal (2012)
# RNN-RBM deep learning tutorial
# More information at http://deeplearning.net/tutorial/rnnrbm.html

import glob
import os
import sys

import numpy
try:
    import pylab
except ImportError:
    print "pylab isn't available, if you use their fonctionality, it will crash"
    print "It can be installed with 'pip install -q Pillow'"

from midi.utils import midiread, midiwrite
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

#Don't use python long as this doesn't work on 32 bit computers.
numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False


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
        h_t, _, _, updates = build_rbm(T.zeros((n_hidden,)), W2, bh1_t,
                                           bh2_t, k=25)
        lin_v_t = T.dot(h_t, W1.T) + bv
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
        non_sequences = params2, n_steps=200)

    
    return (v, v_sample, cost1, monitor1, params1, updates_train1, h, h1_sample,cost2, monitor2, params2, updates_train2, v_t,
            updates_generate)
    '''
    return (v, v_sample, cost1, monitor1, params1, updates_train1, v_t,
            updates_generate)
    '''

class RnnRbm:
    '''Simple class to train an RNN-RBM from MIDI files and to generate sample
sequences.'''

    def __init__(self, n_hidden=150, n_hidden_recurrent=100, lr=0.001,
                 r=(21, 109), dt=0.3):
        '''Constructs and compiles Theano functions for training and sequence
generation.

n_hidden : integer
  Number of hidden units of the conditional RBMs.
n_hidden_recurrent : integer
  Number of hidden units of the RNN.
lr : float
  Learning rate
r : (integer, integer) tuple
  Specifies the pitch range of the piano-roll in MIDI note numbers, including
  r[0] but not r[1], such that r[1]-r[0] is the number of visible units of the
  RBM at a given time step. The default (21, 109) corresponds to the full range
  of piano (88 notes).
dt : float
  Sampling period when converting the MIDI files into piano-rolls, or
  equivalently the time difference between consecutive time steps.'''

        self.r = r
        self.dt = dt
        
        (v, v_sample, cost1, monitor1, params1, updates_train1, h, h1_sample , cost2, monitor2, params2, updates_train2, v_t,
         updates_generate) = build_rnnrbm(r[1] - r[0], n_hidden,
                                           n_hidden_recurrent)
        '''
        (v, v_sample, cost1, monitor1, params1, updates_train1,v_t,
         updates_generate) = build_rnnrbm(r[1] - r[0], n_hidden,
                                           n_hidden_recurrent)
        '''
        gradient1 = T.grad(cost1, params1, consider_constant=[v_sample])
        updates_train1.update(((p, p - lr * g) for p, g in zip(params1,
                                                                gradient1)))
        
        gradient2 = T.grad(cost2, params2, consider_constant=[h1_sample])
        updates_train2.update(((p, p - lr * g) for p, g in zip(params2,
                                                                gradient2)))
        
        self.train_function1 = theano.function([v], monitor1,
                                               updates=updates_train1)

        self.train_function2 = theano.function([v], monitor2,
                                               updates=updates_train2)
        
        self.generate_function = theano.function([], v_t,
                                                 updates=updates_generate)

    def train_RNNRBM1(self, files, batch_size=100, num_epochs=200):
        '''Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
files converted to piano-rolls.

files : list of strings
  List of MIDI files that will be loaded as piano-rolls for training.
batch_size : integer
  Training sequences will be split into subsequences of at most this size
  before applying the SGD updates.
num_epochs : integer
  Number of epochs (pass over the training set) performed. The user can
  safely interrupt training with Ctrl+C at any time.'''

        assert len(files) > 0, 'Training set is empty!' \
                               ' (did you download the data files?)'
        dataset = [midiread(f, self.r,
                            self.dt).piano_roll.astype(theano.config.floatX)
                   for f in files]

        try:
            for epoch in xrange(num_epochs):
                numpy.random.shuffle(dataset)
                costs1 = []
                for s, sequence in enumerate(dataset):
                    for i in xrange(0, len(sequence), batch_size):
                        cost1 = self.train_function1(sequence[i:i + batch_size])
                        costs1.append(cost1)
                print 'Epoch %i/%i' % (epoch + 1, num_epochs),
                print numpy.mean(costs1)
                sys.stdout.flush()

        except KeyboardInterrupt:
            print 'Interrupted by user.'
        
    def train_RNNRBM2(self, files, batch_size=100, num_epochs=200):
        '''Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
files converted to piano-rolls.

files : list of strings
  List of MIDI files that will be loaded as piano-rolls for training.
batch_size : integer
  Training sequences will be split into subsequences of at most this size
  before applying the SGD updates.
num_epochs : integer
  Number of epochs (pass over the training set) performed. The user can
  safely interrupt training with Ctrl+C at any time.'''

        assert len(files) > 0, 'Training set is empty!' \
                               ' (did you download the data files?)'
        dataset = [midiread(f, self.r,
                            self.dt).piano_roll.astype(theano.config.floatX)
                   for f in files]
        try:
            for epoch in xrange(num_epochs):
                numpy.random.shuffle(dataset)
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


    def generate(self, filename, show=True):
        '''Generate a sample sequence, plot the resulting piano-roll and save
it as a MIDI file.

filename : string
  A MIDI file will be created at this location.
show : boolean
  If True, a piano-roll of the generated sequence will be shown.'''

        piano_roll = self.generate_function()
        midiwrite(filename, piano_roll, self.r, self.dt)
        if show:
            extent = (0, self.dt * len(piano_roll)) + self.r
            pylab.figure()
            pylab.imshow(piano_roll.T, origin='lower', aspect='auto',
                         interpolation='nearest', cmap=pylab.cm.gray_r,
                         extent=extent)
            pylab.xlabel('time (s)')
            pylab.ylabel('MIDI note number')
            pylab.title('generated piano-roll')


def test_rnnrbm(batch_size=100, num_epochs=200):
    model = RnnRbm()
    re = os.path.join(os.path.split(os.path.dirname(__file__))[0],
                      'data', 'Nottingham', 'train', '*.mid')
    model.train_RNNRBM1(glob.glob(re),
                batch_size=batch_size, num_epochs=num_epochs)
    model.train_RNNRBM2(glob.glob(re),
                batch_size=batch_size, num_epochs=num_epochs)
    return model

if __name__ == '__main__':
    model = test_rnnrbm()
    model.generate('sample1.mid')
    model.generate('sample2.mid')
    pylab.show()
