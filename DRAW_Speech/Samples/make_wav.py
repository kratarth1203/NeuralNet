'''
apple
max =  17926.0
min = -18192.0

blizzard downsamples by 4
max =  34628.3
min = -32365.4

peach

max = 8695.0
min = -7221.0
'''

import cPickle
f = open('sample_rnn_0.pkl','r')

x = cPickle.load(f)

f.close()

x_hat = x[0]

numpy.min(x_hat)
#Out[93]: 0.00090053852

numpy.max(x_hat)
#Out[94]: 0.9990989

max = 8695.0

min = -7221.0

x_hat = (x_hat * (numpy.abs(min) + max)) - numpy.abs(min)

numpy.max(x_hat)
#Out[98]: 8680.6582

x_hat = numpy.array(x_hat, dtype=numpy.int16)

wv.write('my_peach.wav', 8000, x_hat)

