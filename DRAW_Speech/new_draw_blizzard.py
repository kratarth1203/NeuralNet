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


def shared_normal(num_rows, num_cols, scale=1, name = 'W'):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX), name = name)


def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


def build_rnn(n_visible = 784, n_z = 100, n_hidden_recurrent = 200, T_ = 10, batch_size = 5):
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

    Wh_enc_mew = shared_normal(n_hidden_recurrent, n_z, 0.0001, 'Wh_enc_mew') 
    Wh_enc_sig = shared_normal(n_hidden_recurrent, n_z, 0.0001, 'Wh_enc_sig') 

    Wh_dec_w = shared_normal(n_hidden_recurrent, n_visible, 0.0001, 'Www') 

    params = [bi_enc, Wci_enc,  Whi_enc, Wvi_enc, bf_enc, Wcf_enc, Whf_enc, Wvc_enc, Wvf_enc,
    Whc_enc, bc_enc, bo_enc, Wco_enc, Who_enc, Wvo_enc, bi_dec, Wci_dec, Whi_dec,
    Wzi_dec, bf_dec, Wcf_dec, Whf_dec, Wzf_dec, Wzc_dec, Whc_dec, bc_dec, bo_dec,
    Wco_dec, Who_dec, Wzo_dec, Wh_enc_mew, Wh_enc_sig, Wh_dec_w]
   

    # learned parameters as shared
    # variables

    v = T.matrix()  # a training sequencei
    
    non_seq = [v, bi_enc, Wci_enc,  Whi_enc, Wvi_enc, bf_enc, Wcf_enc, Whf_enc, Wvc_enc, Wvf_enc,
    Whc_enc, bc_enc, bo_enc, Wco_enc, Who_enc, Wvo_enc, bi_dec, Wci_dec, Whi_dec,
    Wzi_dec, bf_dec, Wcf_dec, Whf_dec, Wzf_dec, Wzc_dec, Whc_dec, bc_dec, bo_dec,
    Wco_dec, Who_dec, Wzo_dec, Wh_enc_mew, Wh_enc_sig, Wh_dec_w]


    #z_t = T.vector('z_t')
    # initial value for the RNN_enc hidden units
    h0_enc = T.zeros((batch_size , n_hidden_recurrent))
    c0_enc = T.zeros((batch_size , n_hidden_recurrent))
    # initial value for the RNN_dec hidden units
    h0_dec = T.zeros((batch_size, n_hidden_recurrent))
    c0_dec = T.zeros((batch_size ,n_hidden_recurrent))
    mu_0 = T.zeros((batch_size, n_z))
    sigma_0 = T.zeros((batch_size, n_z))
    w0 = T.zeros((batch_size, n_visible)) 
    prior_log_sigma = T.zeros((n_z,))
    prior_mu = T.zeros((n_z,))
    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    def recurrence( sample_t, h_tm1_enc, h_tm1_dec, c_tm1_enc, c_tm1_dec, w_tm1, mew_t, sigma_t, v):
        v_hat = v - T.nnet.sigmoid(w_tm1) #error input
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
        mew_t = T.dot(h_t_enc, Wh_enc_mew )
        sigma_t = T.dot(h_t_enc, Wh_enc_sig )
        #sample =  theano_rng.normal(size=mew_t.shape, avg = 0, std = 1, dtype=theano.config.floatX)
        z_t = mew_t + (T.exp(sigma_t) * sample_t )
        # Generate h_t_dec = RNN_dec(h_tm1_dec, z_t) 
        i_t_dec = T.nnet.sigmoid(bi_dec + T.dot(c_tm1_dec, Wci_dec) + T.dot(h_tm1_dec, Whi_dec) + T.dot(z_t, Wzi_dec))
        f_t_dec = T.nnet.sigmoid(bf_dec + T.dot(c_tm1_dec, Wcf_dec) + T.dot(h_tm1_dec, Whf_dec) + T.dot(z_t , Wzf_dec))
        c_t_dec = (f_t_dec * c_tm1_dec) + ( i_t_dec * T.tanh( T.dot(z_t, Wzc_dec) + T.dot( h_tm1_dec, Whc_dec) + bc_dec ))
        o_t_dec = T.nnet.sigmoid(bo_dec + T.dot(c_t_dec, Wco_dec) + T.dot(h_tm1_dec, Who_dec) + T.dot(z_t, Wzo_dec))
        h_t_dec = o_t_dec * T.tanh( c_t_dec )

        # Get w_t
        w_t = w_tm1 + T.dot(h_t_dec, Wh_dec_w)
        return [ h_t_enc, h_t_dec, c_t_enc, c_t_dec, w_t, mew_t, sigma_t]


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
        w_t = w_tm1 + T.dot(h_t_dec, Wh_dec_w)

        return [ h_t_dec, c_t_dec, w_t ]
    
    # For training, the deterministic recurrence is used to compute all the
    # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
    # in batches using those parameters.
    rand_samples =  theano_rng.normal(size=(T_, n_z), avg = 0, std = 1, dtype=theano.config.floatX)

  
    (h_t_enc, h_t_dec, c_t_enc, c_t_dec, w_t, mew_t, sigma_t ), updates_train = theano.scan(
        lambda sample_t, h_tm1_enc, h_tm1_dec, c_tm1_enc, c_tm1_dec, w_tm1, mew_t, sigma_t, v, *_: recurrence( sample_t, h_tm1_enc, h_tm1_dec, c_tm1_enc, c_tm1_dec, w_tm1, mew_t, sigma_t, v),
        sequences=[rand_samples], outputs_info=[h0_enc, h0_dec, c0_enc, c0_dec, w0, mu_0, sigma_0], non_sequences=non_seq)

   
    L_z = (
        prior_log_sigma - sigma_t
        + 0.5 * (
        T.exp(2 * sigma_t) + (mew_t - prior_mu) ** 2
        ) / T.exp(2 * prior_log_sigma)
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
    L_x =   (  (T.nnet.sigmoid(w_t[-1,:,:]) - v ) ** 2 ).sum(axis = 1)
    cost = (L_z + L_x).mean()
    monitor = L_x.mean()
    # symbolic loop for sequence generation
    (h_t, c_t, wg_t), updates_generate = theano.scan(
        lambda h_tm1, c_tm1, w_tm1, *_: generate(h_tm1, c_tm1, w_tm1),
        outputs_info=[ h0_dec, c0_dec, w0], non_sequences=params, n_steps=30)

    return (v, cost, monitor, params, updates_train, T.nnet.sigmoid(wg_t[-1,:,:]),
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
        n_hidden_recurrent=1000,
        T_ = 10,
        lr=0.01,
        r=(1, 8001),
        batch_size = 10,
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
        '''

        self.r = r
        self.T_ = T_
        (v, cost, monitor, params, updates_train, w_t,
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

        gradient = T.grad(cost, params)
        updates_train.update(
            ((p, p - lr * g) for p, g in zip(params, gradient))
        )

        sum_squared_grad = shared_zeros(1)
        for g in gradient:
            sum_squared_grad = sum_squared_grad + T.sum(T.sqr(g))
                    
        self.train_function = theano.function([v], [monitor, sum_squared_grad],
                                               updates=updates_train)
        
        self.test_function = theano.function( [v], monitor )
        self.generate_function = theano.function(
            [],
            w_t,
            updates=updates_generate
        )

    def train(self,  batch_size=10, num_epochs=4000):
        '''Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
        files converted to piano-rolls.

        batch_size : integer
            Training sequences will be split into subsequences of at most this
            size before applying the SGD updates.
        num_epochs : integer
            Number of epochs (pass over the training set) performed. The user
            can safely interrupt training with Ctrl+C at any time.'''

        #MNIST
        #f = gzip.open('../data/mnist.pkl.gz' ,'rb')
        #train_set, valid_set, test_set = cPickle.load(f)
        #f.close()
        #train_set_x = train_set[0]
        #test_set_x = test_set[0]

        # Binarized MNIST
        #train_set_x = numpy.load('binarized_mnist_train.npy')
        #test_set_x = numpy.load('binarized_mnist_test.npy')

        # FruitSpeech
        #train, valid, test = load_fruitspeech(['apple'])
        #train_x, train_y = train
        #valid_x, valid_y = valid
        #test_x, test_y = test
        # load into main memory and normalize between 0 and 1
        #train_x = [x[:2500]  for x in train_x[:]]
        
        #Blizzard

        train_x = np.load('/data/lisatmp3/Lessac_Blizzard2013_segmented/train/sf_train_segmented_0downsampled_by4.npy')
        train_x = np.array(train_x, dtype=np.float32)        
        max = numpy.max(train_x)
        min = numpy.min(train_x)
        print 'max = ' , max
        print 'min =' , min
        train_x = (train_x + numpy.abs(min)) / ( max + numpy.abs(min) )
        train_x = train_x[:50]
        print train_x
        print len(train_x)
        try:
            for epoch in xrange(num_epochs):
                numpy.random.shuffle(train_x)
                costs = []
                test_cost = []

                for i in range(0, len(train_x), batch_size):
                    to_train = train_x[ i : i+ batch_size]
                    cost, gradient = self.train_function( to_train )
                    #print gradient
                    costs.append(cost)
                #if epoch%10 == 0:
                #    test_costs = []
                #    for j in range(0, len(test_set_x), batch_size):
                #        to_test = test_set_x[j : j + batch_size]
                #        test_cost = self.test_function( to_test  )
                #        test_costs.append(test_cost)
                #    print 'test_set_KL = %f' , numpy.mean(test_costs)

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

def test_rnnrbm(batch_size=10, num_epochs=4000):
    model = Rnn(batch_size = batch_size)
    model.train(batch_size=batch_size, num_epochs=num_epochs)
    return model

if __name__ == '__main__':
    model = test_rnnrbm()
    model.generate('sample1.mid')
    pylab.show()
 

