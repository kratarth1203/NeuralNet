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

#Don't use a python long as this don't work on 32 bits computers.
numpy.random.seed(0xbeef)
theano_rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False

from theano.compat.python2x import OrderedDict

img_height, img_width = (28, 28)


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


def build_rnn(n_visible = 784, n_z = 100, n_hidden_recurrent = 200, T_ = 10, batch_size = 1000, N_READ = 4, N_WRITE = 7):
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
    Wvi_enc = shared_normal((2*N_READ*N_READ) + n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wvi_enc')

    bf_enc = shared_zeros(n_hidden_recurrent)
    Wcf_enc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wcf_enc')
    Whf_enc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Whf_enc')
    Wvf_enc = shared_normal((2*N_READ*N_READ) + n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wvf_enc')
    
    Wvc_enc = shared_normal((2*N_READ*N_READ) + n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wvc_enc') 
    Whc_enc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Whc_enc')
    bc_enc = shared_zeros(n_hidden_recurrent)
    
    bo_enc = shared_zeros(n_hidden_recurrent)
    Wco_enc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wco_enc') 
    Who_enc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Who_enc')
    Wvo_enc = shared_normal((2*N_READ*N_READ) + n_hidden_recurrent, n_hidden_recurrent, 0.0001, 'Wvo_enc')

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

    
    Wh_dec_read = shared_normal(n_hidden_recurrent, 5, 0.0001, 'Wh_dec_read') 
    Wh_dec_write = shared_normal(n_hidden_recurrent, 5, 0.0001, 'Wh_dec_write') 
    Wh_dec_w = shared_normal(n_hidden_recurrent, N_WRITE*N_WRITE, 0.0001, 'Wh_dec_w') 


    params = [bi_enc, Wci_enc,  Whi_enc, Wvi_enc, bf_enc, Wcf_enc, Whf_enc, Wvc_enc, Wvf_enc,
    Whc_enc, bc_enc, bo_enc, Wco_enc, Who_enc, Wvo_enc, bi_dec, Wci_dec, Whi_dec,
    Wzi_dec, bf_dec, Wcf_dec, Whf_dec, Wzf_dec, Wzc_dec, Whc_dec, bc_dec, bo_dec,
    Wco_dec, Who_dec, Wzo_dec, Wh_enc_mew, Wh_enc_sig, Wh_dec_read, Wh_dec_write, Wh_dec_w]
   

    # learned parameters as shared
    # variables

    v = T.matrix()  # a training sequencei
    
    non_seq = [v, bi_enc, Wci_enc,  Whi_enc, Wvi_enc, bf_enc, Wcf_enc, Whf_enc, Wvc_enc, Wvf_enc,
    Whc_enc, bc_enc, bo_enc, Wco_enc, Who_enc, Wvo_enc, bi_dec, Wci_dec, Whi_dec,
    Wzi_dec, bf_dec, Wcf_dec, Whf_dec, Wzf_dec, Wzc_dec, Whc_dec, bc_dec, bo_dec,
    Wco_dec, Who_dec, Wzo_dec, Wh_enc_mew, Wh_enc_sig, Wh_dec_read, Wh_dec_write, Wh_dec_w]


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
    
    def batched_dot(A, B):
        """Batched version of dot-product.

        For A[dim_1, dim_2, dim_3] and B[dim_1, dim_3, dim_4] this
        is \approx equal to:

        for i in range(dim_1):
        C[i] = tensor.dot(A, B)
        Returns
        -------
        C : shape (dim_1 \times dim_2 \times dim_4)
        """
        C = A.dimshuffle([0,1,2,'x']) * B.dimshuffle([0,'x',1,2])
        return C.sum(axis=-2)

    def filterbank_matrices(N,  center_y, center_x, delta, sigma):
        """Create a Fy and a Fx
        
        Parameters
        ----------
        center_y : T.vector (shape: batch_size)
        center_x : T.vector (shape: batch_size)
            Y and X center coordinates for the attention window
        delta : T.vector (shape: batch_size)
        sigma : T.vector (shape: batch_size)
        
        Returns
        -------
            FY, FX 
        """
        tol = 1e-4

        muX = center_x.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*(T.arange(N)-N/2-0.5)
        muY = center_y.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*(T.arange(N)-N/2-0.5)

        a = T.arange(img_width)
        b = T.arange(img_height)
        
        FX = T.exp( -(a-muX.dimshuffle([0,1,'x']))**2 / 2. / sigma.dimshuffle([0,'x','x'])**2 )
        FY = T.exp( -(b-muY.dimshuffle([0,1,'x']))**2 / 2. / sigma.dimshuffle([0,'x','x'])**2 )
        #FX = FX / (T.sum(FX,axis=-1).dimshuffle(0, 1, 'x'))
        #FX = FX / (T.sum(FY,axis=-1).dimshuffle(0, 1, 'x'))
        FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)

        return FY, FX


    def read(N,  images, center_y, center_x, delta, sigma):
        """
        Parameters
        ----------
        images : T.matrix    (shape: batch_size x img_size)
            Batch of images. Internally it will be reshaped to be a 
            (batch_size, img_height, img_width)-shaped stack of images.
        center_y : T.vector (shape: batch_size)
        center_x : T.vector (shape: batch_size)
        delta : T.vector    (shape: batch_size)
        sigma : T.vector    (shape: batch_size)

        Returns
        -------
        window : T.matrix   (shape: batch_size x N**2)
        """
        batch_size = images.shape[0]

        # Reshape input into proper 2d images
        I = images.reshape( (batch_size, img_height, img_width) )

        # Get separable filterbank
        FY, FX = filterbank_matrices(N, center_y, center_x, delta, sigma)

        # apply to the batch of images
        W = batched_dot(batched_dot(FY, I), FX.transpose([0,2,1]))

        return W.reshape((batch_size, N*N))


    def write(N,  windows, center_y, center_x, delta, sigma):

        batch_size = windows.shape[0]
        # Reshape input into proper 2d windows
        W = windows.reshape( (batch_size, N, N) )

        # Get separable filterbank
        FY, FX = filterbank_matrices(N, center_y, center_x, delta, sigma)

        # apply...
        I = batched_dot(batched_dot(FY.transpose([0,2,1]), W), FX)

        return I.reshape( (batch_size, img_height*img_width) )


    def nn2att( N, l ):
        """Convert neural-net outputs to attention parameters
    
        Parameters
        ----------
        l : tensor (batch_size x 5)
    
        Returns
        -------
        center_y : vector (batch_size)
        center_x : vector (batch_size)
        delta : vector (batch_size)
        sigma : vector (batch_size)
        gamma : vector (batch_size)
        """
        center_y  = l[:,0]
        center_x  = l[:,1]
        log_delta = l[:,2]
        log_sigma = l[:,3]
        log_gamma = l[:,4]
    
        delta = T.exp(log_delta)
        sigma = T.exp(log_sigma/2.)
        gamma = T.exp(log_gamma).dimshuffle(0, 'x')
    
        # normalize coordinates
        center_x = ((center_x+1.)/2.) * (img_width+1)
        center_y = ((center_y+1.)/2.) * (img_height+1)
        delta = ( (img_width-1) / (N-1) ) * delta
    
        return center_y, center_x, delta, sigma, gamma


    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    def recurrence( sample_t, h_tm1_enc, h_tm1_dec, c_tm1_enc, c_tm1_dec, w_tm1, mew_t, sigma_t, v):
        
        v_hat = v - T.nnet.sigmoid(w_tm1) #error input
        att_l_r = T.dot( h_tm1_dec, Wh_dec_read) 
        center_y, center_x, delta, sigma, gamma = nn2att(N_READ, att_l_r)        
        w = gamma * read(N_READ, v , center_y, center_x, delta, sigma)
        w_hat = gamma * read(N_READ, v_hat , center_y, center_x, delta, sigma)
        
        # v_enc = [r_t, h_tm1_dec]
        r_t = T.concatenate([w, w_hat], axis = 1)
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
        
        att_l_w = T.dot( h_t_dec, Wh_dec_write) 
        center_y, center_x, delta, sigma, gamma = nn2att(N_WRITE,att_l_w)        
        
        # Get w_t
        m_t = T.dot(h_t_dec, Wh_dec_w)
        m_update = write(N_WRITE, m_t, center_y, center_x, delta, sigma)
        w_t = w_tm1 + ( (1./gamma) * m_update)
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
    L_x = T.nnet.binary_crossentropy(  T.nnet.sigmoid(w_t[-1,:,:]), v).sum( axis = 1)

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
        n_z = 100,
        n_hidden_recurrent=200,
        T_ = 8,
        lr=0.01,
        r=(1, 785),
        batch_size = 1000,
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

        #sum_squared_grad = shared_zeros(1)
        #for g in gradient:
        #    sum_squared_grad = sum_squared_grad + T.sum(T.sqr(g))
                    
        self.train_function = theano.function([v], monitor,
                                               updates=updates_train)
        
        self.test_function = theano.function( [v], monitor )
        self.generate_function = theano.function(
            [],
            w_t,
            updates=updates_generate
        )

    def train(self,  batch_size=1000, num_epochs=4000):
        '''Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
        files converted to piano-rolls.

        batch_size : integer
            Training sequences will be split into subsequences of at most this
            size before applying the SGD updates.
        num_epochs : integer
            Number of epochs (pass over the training set) performed. The user
            can safely interrupt training with Ctrl+C at any time.'''

        #f = gzip.open('../../../../data/mnist.pkl.gz' ,'rb')
        #train_set, valid_set, test_set = cPickle.load(f)
        #f.close()
        #train_set_x = train_set[0]
        #test_set_x = test_set[0]

        train_set_x = numpy.load('binarized_mnist_train.npy')
        train_set_x = numpy.array(train_set_x, dtype=numpy.float32)
        test_set_x = numpy.load('binarized_mnist_test.npy')
        test_set_x = numpy.array(test_set_x, dtype=numpy.float32)

        try:
            for epoch in xrange(num_epochs):
                numpy.random.shuffle(train_set_x)
                costs = []
                test_cost = []

                for i in range(0, len(train_set_x), batch_size):
                    to_train = train_set_x[ i : i+ batch_size]
                    cost = self.train_function( to_train )
                    #print gradient
                    costs.append(cost)
                if epoch%10 == 0:
                    test_costs = []
                    for j in range(0, len(test_set_x), batch_size):
                        to_test = test_set_x[j : j + batch_size]
                        test_cost = self.test_function( to_test  )
                        test_costs.append(test_cost)
                    print 'test_set_KL = %f' , numpy.mean(test_costs)

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

def test_rnnrbm(batch_size=1000, num_epochs=4000):
    model = Rnn(batch_size = batch_size)
    model.train(batch_size=batch_size, num_epochs=num_epochs)
    return model

if __name__ == '__main__':
    model = test_rnnrbm()
    model.generate('sample1.mid')
    pylab.show()
 

