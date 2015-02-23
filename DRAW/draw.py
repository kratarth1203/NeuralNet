# Author: Kratarth Goel
# LISA Lab (2015)
# Implementation of DRAW : A recurrent Neural Network For Image Generation
# http://arxiv-web3.library.cornell.edu/abs/1502.04623v1

import glob
import os
import sys

import gzip
import cPickle
import numpy
try:
    import pylab
except ImportError:
    print (
        "pylab isn't available. If you use its functionality, it will crash."
    )
    print "It can be installed with 'pip install -q Pillow'"

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data

#Don't use a python long as this don't work on 32 bits computers.
numpy.random.seed(0xbeef)
theano_rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False

from theano.compat.python2x import OrderedDict

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


class _ElemwiseNoGradient(theano.tensor.Elemwise):
    """
    A Theano Op that applies an elementwise transformation and reports
    having no gradient.
    """

    def connection_pattern(self, node):
        """
        Report being disconnected to all inputs in order to have no gradient
        at all.

        Parameters
        ----------
        node : WRITEME
        """
        return [[False]]

    def grad(self, inputs, output_gradients):
        """
        Report being disconnected to all inputs in order to have no gradient
        at all.

        Parameters
        ----------
        inputs : WRITEME
        output_gradients : WRITEME
        """
        return [theano.gradient.DisconnectedType()()]

# Call this on a theano variable to make a copy of that variable
# No gradient passes through the copying operation
# This is equivalent to making my_copy = var.copy() and passing
# my_copy in as part of consider_constant to tensor.grad
# However, this version doesn't require as much long range
# communication between parts of the code
block_gradient = _ElemwiseNoGradient(theano.scalar.identity)

def shared_normal(num_rows, num_cols, scale=1, name = 'W'):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX), name = name)


def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))

n_visible = 784
n_z = 784
n_hidden_recurrent = 784
T_ = 10



bi_enc = shared_zeros(n_hidden_recurrent)
Wci_enc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wci_enc')
Whi_enc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Whi_enc')
Wvi_enc = shared_normal((2*n_visible) + n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wvi_enc')

bf_enc = shared_zeros(n_hidden_recurrent)
Wcf_enc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wcf_enc')
Whf_enc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Whf_enc')
Wvf_enc = shared_normal((2*n_visible) + n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wvf_enc')
    
Wvc_enc = shared_normal((2*n_visible) + n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wvc_enc') 
Whc_enc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Whc_enc')
bc_enc = shared_zeros(n_hidden_recurrent)
    
bo_enc = shared_zeros(n_hidden_recurrent)
Wco_enc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wco_enc') 
Who_enc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Who_enc')
Wvo_enc = shared_normal((2*n_visible) + n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wvo_enc')

bi_dec = shared_zeros(n_hidden_recurrent)
Wci_dec = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wci_dec')
Whi_dec = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Whi_dec')
Wzi_dec = shared_normal(n_visible, n_hidden_recurrent, 0.0001, 'Wzi_dec')

bf_dec = shared_zeros(n_hidden_recurrent)
Wcf_dec = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wcf_dec')
Whf_dec = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Whf_dec')
Wzf_dec = shared_normal(n_visible, n_hidden_recurrent, 0.0001, 'Wzf_dec')
    
Wzc_dec = shared_normal(n_visible, n_hidden_recurrent, 0.0001, 'Wzc_dec')
Whc_dec = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Whc_dec')
bc_dec = shared_zeros(n_hidden_recurrent)
   
bo_dec = shared_zeros(n_hidden_recurrent)
Wco_dec = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wco_dec') 
Who_dec = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Who_dec')
Wzo_dec = shared_normal(n_visible, n_hidden_recurrent, 0.0001, 'Wzo_dec')

Wh_enc_mew = shared_normal(n_hidden_recurrent, n_z, 0.0001, 'Wh_enc_mew') 
Www = shared_normal(n_visible, n_visible, 0.0001, 'Www') 

params = [bi_enc, Wci_enc,  Whi_enc, Wvi_enc, bf_enc, Wcf_enc, Whf_enc, Wvc_enc, Wvf_enc,
Whc_enc, bc_enc, bo_enc, Wco_enc, Who_enc, Wvo_enc, bi_dec, Wci_dec, Whi_dec,
Wzi_dec, bf_dec, Wcf_dec, Whf_dec, Wzf_dec, Wzc_dec, Whc_dec, bc_dec, bo_dec,
Wco_dec, Who_dec, Wzo_dec, Wh_enc_mew, Www]


def build_rnn_step(v_dig):
    '''Construct a symbolic RNN-RBM and initialize parameters.

    n_visible : integer
        Number of visible units.
    n_hidden : integer
        Number of hidden units of the conditional RBMs.
    n_hidden_recurrent : integer
        Number of hidden units of the RNN.

    Return a (v, v_sample, cost, monitor, params, updates_train, v_t,
    updates_generate) tuple:

    v : Theano matrix
        Symbolic variable holding an input sequence (used during training)
    v_sample : Theano matrix
        Symbolic variable holding the negative particles for CD log-likelihood
        gradient estimation (used during training)
    cost : Theano scalar
        Expression whose gradient (considering v_sample constant) corresponds
        to the LL gradient of the RNN-RBM (used during training)
    monitor : Theano scalar
        Frame-level pseudo-likelihood (useful for monitoring during training)
    params : tuple of Theano shared variables
        The parameters of the model to be optimized during training.
    updates_train : dictionary of Theano variable -> Theano variable
        Update object that should be passed to theano.function when compiling
        the training function.
    v_t : Theano matrix
        Symbolic variable holding a generated sequence (used during sampling)
    updates_generate : dictionary of Theano variable -> Theano variable
        Update object that should be passed to theano.function when compiling
        the generation function.'''

    generate = v_dig is None
    #z_t = T.vector('z_t')
    # initial value for the RNN_enc hidden units
    h0_enc = T.zeros((n_hidden_recurrent,))
    c0_enc = T.zeros((n_hidden_recurrent,))
    # initial value for the RNN_dec hidden units
    h0_dec = T.zeros((n_hidden_recurrent,))
    c0_dec = T.zeros((n_hidden_recurrent,))

    w0 = T.zeros((n_visible, ))         

    
    # For training, the deterministic recurrence is used to compute all the
    # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
    # in batches using those parameters.
    if not generate:

        # If `v_t` is given, deterministic recurrence to compute the variable
        # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
        # but with a separate Gibbs chain at each time step to sample (generate)
        # from the RNN-RBM. The resulting sample v_t is returned in order to be
        # passed down to the sequence history.
        def recurrence(v_t, h_tm1_enc, h_tm1_dec, c_tm1_enc, c_tm1_dec, w_tm1):
            v_t_hat = v_t - T.nnet.sigmoid(w_tm1)  #error input
            r_t = T.concatenate( [v_t , v_t_hat] ) 
        
             # v_enc = [r_t, h_tm1_dec]
            v_t_enc = T.concatenate( [r_t, h_tm1_dec] )
        
            #Generate h_t_enc = RNN_enc(h_tm1_enc, v_enc)
            i_t_enc = T.nnet.sigmoid(bi_enc + T.dot(c_tm1_enc, Wci_enc) + T.dot(h_tm1_enc, Whi_enc) + T.dot(v_t_enc, Wvi_enc))
            f_t_enc = T.nnet.sigmoid(bf_enc + T.dot(c_tm1_enc, Wcf_enc) + T.dot(h_tm1_enc, Whf_enc) + T.dot(v_t_enc, Wvf_enc))
            c_t_enc = (f_t_enc * c_tm1_enc) + ( i_t_enc * T.tanh( T.dot(v_t_enc, Wvc_enc) + T.dot( h_tm1_enc, Whc_enc) + bc_enc ))
            o_t_enc = T.nnet.sigmoid(bo_enc + T.dot(c_t_enc, Wco_enc) + T.dot(h_tm1_enc, Who_enc) + T.dot(v_t_enc, Wvo_enc))
            h_t_enc = o_t_enc * T.tanh( c_t_enc )
        
            # Get z_t
            mew_t = T.dot(h_t_enc, Wh_enc_mew )
            sigma_t = T.exp(mew_t)
            z_t = mew_t + sigma_t# * theano_rng.normal(size=mew_t.shape, avg = 0, std = 1, dtype=theano.config.floatX))

            # Generate h_t_dec = RNN_dec(h_tm1_dec, z_t) 
            i_t_dec = T.nnet.sigmoid(bi_dec + T.dot(c_tm1_dec, Wci_dec) + T.dot(h_tm1_dec, Whi_dec) + T.dot(z_t, Wzi_dec))
            f_t_dec = T.nnet.sigmoid(bf_dec + T.dot(c_tm1_dec, Wcf_dec) + T.dot(h_tm1_dec, Whf_dec) + T.dot(z_t , Wzf_dec))
            c_t_dec = (f_t_dec * c_tm1_dec) + ( i_t_dec * T.tanh( T.dot(z_t, Wzc_dec) + T.dot( h_tm1_dec, Whc_dec) + bc_dec ))
            o_t_dec = T.nnet.sigmoid(bo_dec + T.dot(c_t_dec, Wco_dec) + T.dot(h_tm1_dec, Who_dec) + T.dot(z_t, Wzo_dec))
            h_t_dec = o_t_dec * T.tanh( c_t_dec )

            # Get w_t
            w_t = w_tm1 + T.dot(w_tm1, Www)
            return [ h_t_enc, h_t_dec, c_t_enc, c_t_dec, w_t, mew_t, sigma_t]



        (h_t_enc, h_t_dec, c_t_enc, c_t_dec, w_t, mew_t, sigma_t), updates_train = theano.scan(
            lambda v_t, h_tm1_enc, h_tm1_dec, c_tm1_enc, c_tm1_dec, w_tm1, *_: recurrence(v_t, h_tm1_enc, h_tm1_dec, c_tm1_enc, c_tm1_dec, w_tm1),
            sequences=v_dig , outputs_info=[h0_enc, h0_dec, c0_enc, c0_dec, None, None, None], non_sequences=params)


        L_z = -(T.mean( 0.5 * ( (mew_t ** 2).sum() + (sigma_t ** 2).sum() - ( T.log(sigma_t ** 2) ).sum() ) ) - ( T_ / 2 ))

        #TODO
        #L_x
        L_x = T.mean(
            T.sum(
                v_dig * T.log(T.nnet.sigmoid(w_t[-1])) +
                (1 - v_dig) * T.log(1 - T.nnet.sigmoid(w_t[-1])),
                axis=1
            )
        )

        cost = L_z + L_x
        monitor = L_x
        return ([cost, monitor], updates_train)

    else:

        def generate( h_tm1_dec, c_tm1_dec, w_tm1):
   
            mew_t = T.dot(h_tm1_dec, Wh_enc_mew )
            z_t = theano_rng.normal(size=mew_t.shape, avg = 0, std = 1, dtype=theano.config.floatX)

            # Generate h_t_dec = RNN_dec(h_tm1_dec, z_t) 
            i_t_dec = T.nnet.sigmoid(bi_dec + T.dot(c_tm1_dec, Wci_dec) + T.dot(h_tm1_dec, Whi_dec) + T.dot(z_t, Wzi_dec))
            f_t_dec = T.nnet.sigmoid(bf_dec + T.dot(c_tm1_dec, Wcf_dec) + T.dot(h_tm1_dec, Whf_dec) + T.dot(z_t , Wzf_dec))
            c_t_dec = (f_t_dec * c_tm1_dec) + ( i_t_dec * T.tanh( T.dot(z_t, Wzc_dec) + T.dot( h_tm1_dec, Whc_dec) + bc_dec ))
            o_t_dec = T.nnet.sigmoid(bo_dec + T.dot(c_t_dec, Wco_dec) + T.dot(h_tm1_dec, Who_dec) + T.dot(z_t, Wzo_dec))
            h_t_dec = o_t_dec * T.tanh( c_t_dec )

            # Get w_t
            w_t = w_tm1 + T.dot(w_tm1, Www)

            return [ h_t_dec, c_t_dec, w_t ]


        # symbolic loop for sequence generation
        (h_t, c_t, w_t_g), updates_generate = theano.scan(
            lambda h_tm1, c_tm1, w_tm1, *_: generate(h_tm1, c_tm1, w_tm1),
            outputs_info=[ h0_dec, c0_dec, w0], non_sequences=params, n_steps=30)

        return ([ w_t_g[-1]], updates_generate)


def build_rnn():

    #Generate h_t_enc = RNN_enc(h_tm1_enc, v_enc)
    v = T.tensor3('v_tensor')  # a training sequence
    (cost, monitor), updates_train = theano.scan(
           lambda v_dig, *_ : build_rnn_step(v_dig),
           sequences=v, outputs_info=[None, None], non_sequences = params)

    (w_t), updates_generate = theano.scan(
           lambda *_ : build_rnn_step(None),
           outputs_info = [None], non_sequences = params, n_steps = 1 )
    return (v, T.mean(cost[-1]), T.mean(monitor[-1]), updates_train, w_t,
            updates_generate)

class RnnRbm:
    '''Simple class to train an RNN-RBM from MIDI files and to generate sample
    sequences.'''

    def __init__(
        self,
        n_z = 784,
        n_hidden_recurrent=784,
        T_ = 10,
        lr=0.001,
        r=(1, 785),
        dt=0.3,
        momentum=0.99999
    ):
        '''Constructs and compiles Theano functions for training and sequence
        generation.

        n_hidden : integer
            Number of hidden units of the conditional RBMs.
        n_hidden_recurrent : integer
            Number of hidden units of the RNN.
        lr : float
            Learning rate
        r : (integer, integer) tuple
            Specifies the pitch range of the piano-roll in MIDI note numbers,
            including r[0] but not r[1], such that r[1]-r[0] is the number of
            visible units of the RBM at a given time step. The default (21,
            109) corresponds to the full range of piano (88 notes).
        dt : float
            Sampling period when converting the MIDI files into piano-rolls, or
            equivalently the time difference between consecutive time steps.'''

        self.r = r
        self.dt = dt
        self.T_ = T_
        (v, cost, monitor, updates_train, w_t,
            updates_generate) = build_rnn()



        gradient = T.grad(cost, params)        
        updates_rmsprop = get_clip_rmsprop_updates(params, cost, gradient,  lr, momentum )
        updates_train.update(updates_rmsprop)
        
        self.train_function = theano.function([v], monitor,
                                               updates=updates_train)

        self.generate_function = theano.function(
            [],
            w_t,
            updates=updates_generate
        )

    def train(self,  batch_size=100, num_epochs=200):
        '''Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
        files converted to piano-rolls.

        batch_size : integer
            Training sequences will be split into subsequences of at most this
            size before applying the SGD updates.
        num_epochs : integer
            Number of epochs (pass over the training set) performed. The user
            can safely interrupt training with Ctrl+C at any time.'''

        f = gzip.open('../data/mnist.pkl.gz' ,'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        #train_set, valid_set, test_set = load_data()
        train_set_x = train_set[0]
        try:
            for epoch in xrange(num_epochs):
                numpy.random.shuffle(train_set[0])
                costs = []

                for i in range(0, len(train_set_x), batch_size):
                    training_set = []
                    to_train = train_set_x[ i : i+ batch_size]
                    for image in to_train:
                        training_set.append( [image for i in range(self.T_)] )
                    if len(training_set) == 0:
                        pass
                    else:
                        cost = self.train_function( training_set)
                        costs.append(cost)

                    print 'Epoch %i/%i' % (epoch + 1, num_epochs),
                    print numpy.mean(costs)
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print 'Interrupted by user.'

    def generate(self, filename, show=True):

        n_samples_to_gen = 5
        for i in range(n_samples_to_gen):
            g = self.generate_function()
            f = open('sample_rnn_%i.pkl' % i, 'w')
            cPickle.dump(g,f)
            f.close()

def test_rnnrbm(batch_size=5000, num_epochs=200):
    model = RnnRbm()
    model.train(batch_size=batch_size, num_epochs=num_epochs)
    return model

if __name__ == '__main__':
    model = test_rnnrbm()
    model.generate('sample1.mid')
    model.generate('sample2.mid')
    pylab.show()
 
