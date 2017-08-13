import numpy as np
import os
import gzip
import cPickle

def load_data():
       if not os.path.exists(os.path.join(os.curdir, 'data')):
           os.mkdir(os.path.join(os.curdir, 'data'))
       f = gzip.open(os.path.join(os.curdir, 'data', 'mnist.pkl.gz'), 'rb')
       tr_d, va_d, te_d = cPickle.load(f)
       f.close()
       tr_in=[np.reshape(x,(784,1)) for x in tr_d[0]]
       tr_out=[vectorized_res(y) for y in tr_d[1]]
       training_data=zip(tr_in,tr_out)
       va_in=[np.reshape(x,(784,1)) for x in va_d[0]]
       validation_data=zip(va_in,va_d[1])
       te_in=[np.reshape(x,(784,1)) for x in te_d[0]]
       test_data=zip(te_in,te_d[1])
       return (training_data,validation_data,test_data)

def vectorized_res(i):
       y=np.zeros((10,1))
       y[i]=1.0
       return y

