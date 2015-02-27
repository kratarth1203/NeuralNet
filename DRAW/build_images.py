import cPickle
import theano
from theano import tensor as T
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


v = T.matrix('v')

fn = theano.function( [v], T.nnet.sigmoid(v) )

f = open('Samples/sample_rnn_1.pkl', 'r')
x = cPickle.load(f)

f.close()

x_hat = fn(x)

x_hat = x_hat*256
img = x_hat[0]
img = numpy.array(img,dtype=numpy.int16).reshape((28,28))
plt.imshow(img)
plt.savefig('img.jpg')
