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

from theano.compat.python2x import OrderedDict


import cPickle
#Don't use python long as this doesn't work on 32 bit computers.
numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
theano_rng = RandomStreams(seed=numpy.random.randint(1 << 30))

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import copy
import time
floatX='float32'

from theano.compat.python2x import OrderedDict
# some utilities
def constantX(value, float_dtype='float32'):
    return theano.tensor.constant(numpy.asarray(value, dtype=float_dtype))

def sharedX(value):
    return theano.shared(value)

def build_updates(
        cost, params, 
        clip_c=0,clip_idx=None,
        shrink_grad=None, choice=None):
    updates = OrderedDict()
    grads = T.grad(cost, params)
    def apply_clip(g):
        g2 = 0.
        g2 += (g**2).sum()
        new_grad = T.switch(g2 > (clip_c**2), 
                        g / T.sqrt(g2) * clip_c,
                        g)
        return new_grad
    if clip_c > 0. and clip_idx is not None:
        for idx in clip_idx:
            grads[idx] = apply_clip(grads[idx])
    if shrink_grad is not None:
        for idx in shrink_grad:
            grads[idx] *= 0.001
    def get_updates_adadelta(grads,params,decay=0.95):
        decay = constantX(decay)
        print 'build updates with adadelta'
        for param, grad in zip(params, grads):
            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = sharedX(numpy.zeros(param.get_value().shape, dtype=floatX))
            # mean_square_dx := E[(\Delta x)^2]_{t-1}
            mean_square_dx = sharedX(numpy.zeros(param.get_value().shape, dtype=floatX))
            if param.name is not None:
                mean_square_grad.name = 'mean_square_grad_' + param.name
                mean_square_dx.name = 'mean_square_dx_' + param.name

            # Accumulate gradient
            new_mean_squared_grad = \
                    decay * mean_square_grad +\
                    (1. - decay) * T.sqr(grad)
            # Compute update
            epsilon = constantX(1e-7)
            rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
            rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
            delta_x_t = - rms_dx_tm1 / rms_grad_t * grad

            # Accumulate updates
            new_mean_square_dx = \
                    decay * mean_square_dx + \
                    (1. - decay) * T.sqr(delta_x_t)

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[mean_square_dx] = new_mean_square_dx
            updates[param] = param + delta_x_t
    def get_updates_grads_momentum(gparams, params, lr=0.1, momentum=0.5):
        print 'building updates with momentum'
        # build momentum
        gparams_mom = []
        for param in params:
            gparam_mom = theano.shared(
                numpy.zeros(param.get_value(borrow=True).shape,
                dtype=floatX))
            gparams_mom.append(gparam_mom)

        for gparam, gparam_mom, param in zip(gparams, gparams_mom, params):
            inc = momentum * gparam_mom - (constantX(1) - momentum) * lr * gparam
            updates[gparam_mom] = inc
            updates[param] = param + inc
    def get_updates_rmsprop(grads, params, lr=0.1, decay=0.95):
        for param,grad in zip(params,grads):
            mean_square_grad = sharedX(numpy.zeros(param.get_value().shape, dtype=floatX))
            new_mean_squared_grad = (decay * mean_square_grad +
                                     (1. - decay) * T.sqr(grad))
            rms_grad_t = T.sqrt(new_mean_squared_grad)
            delta_x_t = constantX(-1) * lr * grad / rms_grad_t
            updates[mean_square_grad] = new_mean_squared_grad
            updates[param] = param + delta_x_t
    get_updates_adadelta(grads, params)
    #get_updates_grads_momentum(grads, params)
    #get_updates_rmsprop(grads, params)
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
    
    h5_file_path = os.path.join(data_path, "saved_all_fruit.h5")
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


def shared_normal(num_rows, num_cols, scale=1, name = 'W'):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX), name = name)


def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))
def logsumexp(x, axis=None):
    x_max = T.max(x, axis=axis, keepdims=True)
    z = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)



def GMM(y, mu, logvar, coeff, tol=0.):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y      : TensorVariable
    mu     : FullyConnected (Linear)
    logvar : FullyConnected (Linear)
    coeff  : FullyConnected (Softmax)
    """
    y = y.dimshuffle(0, 1, 'x')
    mu = mu.reshape((mu.shape[0],
                     mu.shape[1]/coeff.shape[-1],
                     coeff.shape[-1]))
    logvar = logvar.reshape((logvar.shape[0],
                             logvar.shape[1] / coeff.shape[-1],
                             coeff.shape[-1]))
    logvar = T.log(T.exp(logvar) + tol)
    inner = -0.5 * T.sum(T.sqr(y - mu) * T.exp(-logvar) + logvar +
                        T.log(2 * np.pi), axis=1)
    nll = -logsumexp(T.log(coeff) + inner, axis=1)
   
    return nll



def build_rnn(n_visible = 784, n_z = 100, n_hidden_recurrent = 200, T_ = 10, batch_size = 15, n_gmm = 3,sigma_bound = 1e-3):
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

        
        
    #Generate h_t_enc = RNN_enc(h_tm1_enc, v_enc)
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
    Wzi_dec = shared_normal(n_z, n_hidden_recurrent, 0.0001, 'Wzi_dec')

    bf_dec = shared_zeros(n_hidden_recurrent)
    Wcf_dec = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wcf_dec')
    Whf_dec = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Whf_dec')
    Wzf_dec = shared_normal(n_z, n_hidden_recurrent, 0.0001, 'Wzf_dec')
    
    Wzc_dec = shared_normal(n_z, n_hidden_recurrent, 0.0001, 'Wzc_dec')
    Whc_dec = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Whc_dec')
    bc_dec = shared_zeros(n_hidden_recurrent)
    
    bo_dec = shared_zeros(n_hidden_recurrent)
    Wco_dec = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wco_dec') 
    Who_dec = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Who_dec')
    Wzo_dec = shared_normal(n_z, n_hidden_recurrent, 0.0001, 'Wzo_dec')

    Wh_enc_mu_z = shared_normal(n_hidden_recurrent, n_z, 0.0001, 'Wh_enc_mew') 
    #Wh_enc_sig_z = shared_normal(n_hidden_recurrent, n_z, 0.0001, 'Wh_enc_sig') 

    b_mu_z = shared_zeros(n_z)
    b_sig_z = shared_zeros(batch_size*n_z)
    
    Wh_dec_mu_x = shared_normal(n_hidden_recurrent, n_visible* n_gmm, 0.0001, 'Wh_dec_mu_x') 
    Wh_dec_coeff_x = shared_normal(n_hidden_recurrent, n_gmm, 0.0001, 'Wh_dec_mu_x') 
    #Wh_dec_sigma_x = shared_normal(n_hidden_recurrent, n_visible, 0.0001, 'Wh_dec_sigma_x') 

    b_mu_x = shared_zeros(n_visible * n_gmm)
    b_sig_x = shared_zeros(batch_size*n_visible * n_gmm)
    b_coeff_x = shared_zeros(n_gmm)
    
    params = [bi_enc, Wci_enc,  Whi_enc, Wvi_enc, bf_enc, Wcf_enc, Whf_enc, Wvc_enc, Wvf_enc,
    Whc_enc, bc_enc, bo_enc, Wco_enc, Who_enc, Wvo_enc, bi_dec, Wci_dec, Whi_dec,
    Wzi_dec, bf_dec, Wcf_dec, Whf_dec, Wzf_dec, Wzc_dec, Whc_dec, bc_dec, bo_dec,
    Wco_dec, Who_dec, Wzo_dec, Wh_enc_mu_z,  Wh_dec_mu_x,Wh_dec_coeff_x,  b_mu_z, b_sig_z, b_mu_x, b_sig_x, b_coeff_x]
   

    # learned parameters as shared
    # variables

    v = T.matrix()  # a training sequencei
    
    non_seq = [v, bi_enc, Wci_enc,  Whi_enc, Wvi_enc, bf_enc, Wcf_enc, Whf_enc, Wvc_enc, Wvf_enc,
    Whc_enc, bc_enc, bo_enc, Wco_enc, Who_enc, Wvo_enc, bi_dec, Wci_dec, Whi_dec,
    Wzi_dec, bf_dec, Wcf_dec, Whf_dec, Wzf_dec, Wzc_dec, Whc_dec, bc_dec, bo_dec,
    Wco_dec, Who_dec, Wzo_dec, Wh_enc_mu_z,  Wh_dec_mu_x, Wh_dec_coeff_x, b_mu_z, b_sig_z, b_mu_x, b_sig_x, b_coeff_x]


    #z_t = T.vector('z_t')
    # initial value for the RNN_enc hidden units
    h0_enc = T.zeros((batch_size , n_hidden_recurrent))
    c0_enc = T.zeros((batch_size , n_hidden_recurrent))
    # initial value for the RNN_dec hidden units
    h0_dec = T.zeros((batch_size, n_hidden_recurrent))
    c0_dec = T.zeros((batch_size ,n_hidden_recurrent))
    mu_z_0 = T.zeros((batch_size, n_z))
    #sigma_z_0 = T.zeros((batch_size, n_z))
    mu_x_0 = T.zeros((batch_size, n_visible*n_gmm))
    coeff_x_0 = T.zeros((batch_size, n_gmm))
    #sigma_x_0 = T.zeros(( batch_size,n_visible))
    #prior_log_sigma_z = T.zeros((n_z,))
    #prior_log_sigma_x = T.zeros((n_visible,))
    #prior_mu_x = T.zeros((n_visible,))
    #prior_mu_z = T.zeros((n_z,))
    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    def recurrence( sample_z_t, sample_x_t, h_tm1_enc, h_tm1_dec, c_tm1_enc, c_tm1_dec,  mu_z_t,  mu_x_tm1, coeff_x_tm1,  v):
        v_hat = v - T.sum(( coeff_x_tm1.dimshuffle(0,'x',1) *  ( mu_x_tm1 + (T.exp(b_sig_x) * sample_x_t).reshape((batch_size, n_visible*n_gmm)) ).reshape((batch_size, n_visible, n_gmm)) ), axis = -1 ) #error input
        r_t = T.concatenate( [v , v_hat], axis = 1 ) 
        
        # v_enc = [r_t, h_tm1_dec]
        v_enc = T.concatenate( [r_t, h_tm1_dec] , axis = 1)
        
        #Generate h_t_enc = RNN_enc(h_tm1_enc, v_enc)
        i_t_enc = T.nnet.sigmoid(bi_enc + T.dot(c_tm1_enc, Wci_enc) + T.dot(h_tm1_enc, Whi_enc) + T.dot(v_enc, Wvi_enc))
        f_t_enc = T.nnet.sigmoid(bf_enc + T.dot(c_tm1_enc, Wcf_enc) + T.dot(h_tm1_enc, Whf_enc) + T.dot(v_enc, Wvf_enc))
        c_t_enc = (f_t_enc * c_tm1_enc) + ( i_t_enc * T.tanh( T.dot(v_enc, Wvc_enc) + T.dot( h_tm1_enc, Whc_enc) + bc_enc ))
        o_t_enc = T.nnet.sigmoid(bo_enc + T.dot(c_t_enc, Wco_enc) + T.dot(h_tm1_enc, Who_enc) + T.dot(v_enc, Wvo_enc))
        h_t_enc = o_t_enc * T.tanh( c_t_enc )
        
        # Get z_t
        mu_z_t = T.dot(h_t_enc, Wh_enc_mu_z ) + b_mu_z
        #sigma_z_t = T.dot(h_t_enc, Wh_enc_sig_z ) + b_sig_z
        #sample =  theano_rng.normal(size=mew_t.shape, avg = 0, std = 1, dtype=theano.config.floatX)
        z_t = mu_z_t + (T.exp(b_sig_z) * sample_z_t).reshape((batch_size,n_z)) 
        # Generate h_t_dec = RNN_dec(h_tm1_dec, z_t) 
        i_t_dec = T.nnet.sigmoid(bi_dec + T.dot(c_tm1_dec, Wci_dec) + T.dot(h_tm1_dec, Whi_dec) + T.dot(z_t, Wzi_dec))
        f_t_dec = T.nnet.sigmoid(bf_dec + T.dot(c_tm1_dec, Wcf_dec) + T.dot(h_tm1_dec, Whf_dec) + T.dot(z_t , Wzf_dec))
        c_t_dec = (f_t_dec * c_tm1_dec) + ( i_t_dec * T.tanh( T.dot(z_t, Wzc_dec) + T.dot( h_tm1_dec, Whc_dec) + bc_dec ))
        o_t_dec = T.nnet.sigmoid(bo_dec + T.dot(c_t_dec, Wco_dec) + T.dot(h_tm1_dec, Who_dec) + T.dot(z_t, Wzo_dec))
        h_t_dec = o_t_dec * T.tanh( c_t_dec )

        # Get w_t
        mu_x_t = mu_x_tm1 + T.dot(h_t_dec, Wh_dec_mu_x) + b_mu_x
        coeff_x_t = T.nnet.softmax( T.dot(h_t_dec, Wh_dec_coeff_x) + b_coeff_x)
        #sigma_x_t = sigma_x_tm1 + T.dot(h_t_dec, Wh_dec_sigma_x) + b_sig_x

        return [ h_t_enc, h_t_dec, c_t_enc, c_t_dec,  mu_z_t,  mu_x_t , coeff_x_t]


    def generate( h_tm1_dec, c_tm1_dec, mu_x_tm1 ):

        #mew_t = T.dot(h_tm1_dec, Wh_enc_mew )
        z_t = theano_rng.normal(size=(n_z,), avg = 0, std = 1, dtype=theano.config.floatX)

        # Generate h_t_dec = RNN_dec(h_tm1_dec, z_t) 
        i_t_dec = T.nnet.sigmoid(bi_dec + T.dot(c_tm1_dec, Wci_dec) + T.dot(h_tm1_dec, Whi_dec) + T.dot(z_t, Wzi_dec))
        f_t_dec = T.nnet.sigmoid(bf_dec + T.dot(c_tm1_dec, Wcf_dec) + T.dot(h_tm1_dec, Whf_dec) + T.dot(z_t , Wzf_dec))
        c_t_dec = (f_t_dec * c_tm1_dec) + ( i_t_dec * T.tanh( T.dot(z_t, Wzc_dec) + T.dot( h_tm1_dec, Whc_dec) + bc_dec ))
        o_t_dec = T.nnet.sigmoid(bo_dec + T.dot(c_t_dec, Wco_dec) + T.dot(h_tm1_dec, Who_dec) + T.dot(z_t, Wzo_dec))
        h_t_dec = o_t_dec * T.tanh( c_t_dec )

        # Get w_t
        mu_x_t = mu_x_tm1 + T.dot(h_t_dec, Wh_dec_mu_x) + b_mu_x
        #sigma_x_t =  b_sig_x

        return [ h_t_dec, c_t_dec, mu_x_t ]
    
    # For training, the deterministic recurrence is used to compute all the
    # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
    # in batches using those parameters.
    rand_samples_z =  theano_rng.normal(size=(T_, n_z*batch_size), avg = 0, std = 1, dtype=theano.config.floatX)
    rand_samples_x =  theano_rng.normal(size=(T_,n_visible*batch_size*n_gmm), avg = 0, std = 1, dtype=theano.config.floatX)
  
    (h_t_enc, h_t_dec, c_t_enc, c_t_dec,  mu_z_t, mu_x_t, coeff_x_t  ), updates_train = theano.scan(
        lambda sample_z_t, sample_x_t, h_tm1_enc, h_tm1_dec, c_tm1_enc, c_tm1_dec, mu_z_t, mu_x_tm1,coeff_x_tm1,  v, *_: recurrence( sample_z_t, sample_x_t, h_tm1_enc, h_tm1_dec, c_tm1_enc, c_tm1_dec,  mu_z_t, mu_x_tm1, coeff_x_tm1,  v),
        sequences=[rand_samples_z, rand_samples_x], outputs_info=[h0_enc, h0_dec, c0_enc, c0_dec,  mu_z_0, mu_x_0, coeff_x_0], non_sequences=non_seq)

   
    L_z = (
        - b_sig_z.reshape((batch_size, n_z)).dimshuffle('x',0,1)
        + 0.5 * (
        T.exp(2 * b_sig_z).reshape((batch_size, n_z)).dimshuffle('x',0,1) + (mu_z_t ) ** 2
        )
        - 0.5
        ).sum(axis=-1).sum(axis = 0)

    #L_z = -(T.mean( 0.5 * ( (mew_t ** 2).sum() + (sigma_t ** 2).sum() - ( T.log(sigma_t ** 2) ).sum() ) ) - ( T_ / 2 ))

    #TODO
    #L_x
    '''
    L_x = (
            T.sum(
                v * T.log(T.nnet.sigmoid(w_t[-1])) +
                (1 - v) * T.log(1 - T.nnet.sigmoid(w_t[-1])),
                axis=1
            )
        )
    '''
    #L_x = T.nnet.binary_crossentropy(  T.nnet.sigmoid(w_t[-1,:,:]), v).sum( axis = 1)
    #L_x =  ( (  (mu_x_t[-1,:,:] - v ) ** 2 )/ (  (sigma_x_t[-1,:,:]**2) + 1e-4 ) ).sum(axis = 1)
    #L_x  = 0.5 * T.sum( (T.sqr(v - mu_x_t[-1,:,:]) * T.exp(-2*sigma_x_t[-1,:])) + (2*sigma_x_t[-1,:]) + T.log(2 * numpy.pi) , axis=1)
    #diff = T.sqr(v - mu_x_t[-1,:,:])
    #L_x  = 0.5 * T.sum( (diff * T.exp(-2*b_sig_x).reshape((batch_size,n_visible))) + (2*b_sig_x).reshape((batch_size, n_visible)) + T.log(2 * numpy.pi) , axis=1)
     
    L_x = GMM(v, mu_x_t[-1,:,:], b_sig_x.reshape(( batch_size, n_visible * n_gmm)), coeff_x_t[-1,:,:], 1e-4 )
    cost = (L_z + L_x).mean()
    monitor = L_x.mean()
    # symbolic loop for sequence generation
    (h_t, c_t, g_mu_x_t ), updates_generate = theano.scan(
        lambda h_tm1, c_tm1, mu_x_tm1,  *_: generate(h_tm1, c_tm1, mu_x_tm1 ),
        outputs_info=[ h0_dec, c0_dec, mu_x_0 ], non_sequences=params, n_steps=30)

    return (v, cost, monitor, params, updates_train, mu_x_t[-1,:,:], b_sig_x, g_mu_x_t[-1,:,:],
            updates_generate)


class Rnn:
    '''Simple class to train an DRAW.'''
    ''' last known good configuration
    n_z = 100
    n_hidden_recurrent - 200,
    T = 8,
    lr = 0.01,
    batch_size =1000
    ''' 
    def __init__(
        self,
        n_z = 500,
        n_hidden_recurrent=1200,
        T_ = 5,
        lr=0.01,
        r=(1, 6967), 
        batch_size = 15,
        momentum=0.99999
    ):
        #r = (1,2881) for apple[0]
        '''Constructs and compiles Theano functions for training and sequence
        generation.

        n_hidden : integer
            Number of hidden units of the conditional RBMs.
        n_hidden_recurrent : integer
            Number of hidden units of the RNN.
        lr : float
            Learning rate
        '''

        self.r = r
        self.T_ = T_
        (v, cost, monitor, params, updates_train, mu_x_t, b_sig_x, g_mu_x_t,
            updates_generate) = build_rnn(
                r[1] - r[0],
                n_z,
                n_hidden_recurrent,
                T_,
                batch_size
            )


        #gradient = T.grad(cost, params)        
        #updates_rmsprop = get_clip_rmsprop_updates(params, cost, gradient,  lr, momentum )
        #updates_train.update(updates_rmsprop)

        #gradient = T.grad(cost, params)
        #updates_train.update(
        #    ((p, p - lr * g) for p, g in zip(params, gradient))
        #)

        #sum_squared_grad = shared_zeros(1)
        #for g in gradient:
        #    sum_squared_grad = sum_squared_grad + T.sum(T.sqr(g))
    

        updates_new = build_updates(cost, params)
        updates_train.update(updates_new)
        
        self.train_function = theano.function([v], [monitor,cost, mu_x_t, b_sig_x],
                                               updates=updates_train)
        
        self.test_function = theano.function( [v], monitor )
        self.generate_function = theano.function(
            [],
            [g_mu_x_t,g_mu_x_t],
            updates=updates_generate
        )

    def train(self,  batch_size=15, num_epochs=4000):
        '''Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
        files converted to piano-rolls.

        batch_size : integer
            Training sequences will be split into subsequences of at most this
            size before applying the SGD updates.
        num_epochs : integer
            Number of epochs (pass over the training set) performed. The user
            can safely interrupt training with Ctrl+C at any time.'''

        #f = gzip.open('../data/mnist.pkl.gz' ,'rb')
        #train_set, valid_set, test_set = cPickle.load(f)
        #f.close()
        #train_set_x = train_set[0]
        #test_set_x = test_set[0]

        #train_set_x = numpy.load('binarized_mnist_train.npy')
        #test_set_x = numpy.load('binarized_mnist_test.npy')
        '''
        train, valid, test = load_fruitspeech(['peach'])
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test
        #new_x = []
        #for item in train_x:
        #    new_x.append(item)
        #train_x = numpy.array(new_x, dtype=theano.config.floatX)
        #print train_x
        # load into main memory and normalize between -1 and 1
        #train_x = [x[:4923]  for x in train_x[:]]
        train_x = train_x[:]
        new_x = []
        for i in range(15):
            #new_x.append(train_x[i][:4900])
            new_x.append(train_x[9][:4900])
        train_x = new_x
        a= []
        b =[]
        for item in train_x:
            a.append(max(item))
            b.append(min(item))
        max_ = max(a)
        min_ = min(b)
        print 'max = ' , max_
        print 'min =' , min_
        train_x = (train_x + numpy.abs(min_)) / ( max_ + numpy.abs(min_) )
        print train_x
        print len(train_x)
        print len(train_x[0])
        '''
        '''
        train_set_x = numpy.load('data/TIMIT/train_aa_1000-FL_100-OL_X_normalized.npy')
        train_set_x = train_set_x[:3600]
        train_set_x = numpy.array(train_set_x, dtype=numpy.float32)

        test_set_x = numpy.load('data/TIMIT/test_aa_1000-FL_100-OL_X_normalized.npy')
        test_set_x = test_set_x[:450]
        test_set_x = numpy.array(test_set_x, dtype=numpy.float32)
        '''

        #x = numpy.load('/data/lisa/data/timit/readable/per_phone/wav_aa.npy')
        
        train, valid, test = load_fruitspeech(['peach'])
        x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test
        new_x = []
        for item in x:                                  
            if item.shape[0] < 6966:
                item = numpy.append(item, [0 for i in range(6966 - item.shape[0])])
                new_x.append(item)
            else:
                new_x.append(item)
        new_x = new_x[:105]
        new_x = numpy.array(new_x, dtype= numpy.float32)
        means = numpy.mean(new_x, axis = 0)
        std = numpy.std(new_x, axis = 0)
        train_set_x = (new_x - means ) / std
        try:
            for epoch in xrange(num_epochs):
                numpy.random.shuffle(train_set_x)
                costs = []
                monitors=  []
                #diffs = []
                test_cost = []

                for i in range(0, len(train_set_x), batch_size):
                    to_train = train_set_x[ i : i+ batch_size]
                    monitor, cost,  mu_x_t, sigma_x_t = self.train_function( to_train )
                    if epoch%20 == 0 and epoch != 0:
                         f = open('recons_%i.pkl'%i,'w')
                         cPickle.dump([mu_x_t, sigma_x_t], f)
                         f.close()
                    #print gradient
                    #diffs.append(diff)
                    monitors.append(monitor)
                    costs.append(cost)
                #if epoch%10 == 0:
                #    test_costs = []
                #    for j in range(0, len(test_set_x), batch_size):
                #        to_test = test_set_x[j : j + batch_size]
                #        test_cost = self.test_function( to_test  )
                #        test_costs.append(test_cost)
                #    print 'test_set_KL = %f' , numpy.mean(test_costs)

                print 'Epoch %i/%i' % (epoch + 1, num_epochs),
                print numpy.mean(costs), ' monitor = ' , numpy.mean(monitors)  # , ' and diff = ', numpy.mean(diffs)
                sys.stdout.flush()


        except KeyboardInterrupt:
            print 'Interrupted by user.'

    def generate(self, filename, show=True):

        n_samples_to_gen = 50
        for i in range(n_samples_to_gen):
            g = self.generate_function()
            f = open('sample_rnn_%i.pkl' % i, 'w')
            cPickle.dump(g,f)
            f.close()

def test_rnnrbm(batch_size=15, num_epochs=4000):
    model = Rnn(batch_size = batch_size)
    model.train(batch_size=batch_size, num_epochs=num_epochs)
    return model

if __name__ == '__main__':
    model = test_rnnrbm()
    model.generate('sample1.mid')
    pylab.show()
 

