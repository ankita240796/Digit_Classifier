import numpy as np
import random

class Network(object):
          def __init__(self,sizes):
	        #sizes is the list of number of neurons in layers of the neural network  
	       self.num_of_layers=len(sizes) 
	       self.layers=sizes
	       self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
	        #list of weights or theta matrix for each layer
	       self.biases=[np.random.randn(y,1) for y in sizes[1:]] 
                  #list of (-threshold) for each layer except the input layer

                   
          def mbgd(self,training_data,epoch,eta,mini_batch_size,test_data=None):
                   if test_data: n_test=len(test_data)
                   n=len(training_data)
                   for i in xrange(epoch):
                         random.shuffle(training_data)
                         mini_batches=[training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
                         for mini_batch in mini_batches:
                         	self.update(mini_batch,eta)
                         if test_data:
                             print "Epoch {0}: {1}/{2}".format(i,self.evaluate(test_data),n_test)

          
          def forward_prop(self,x):
                   #to return output of network for input layer x 
                   y=x;        
                   for w,b in zip(self.weights,self.biases):
                          y=sigmoid(np.dot(w,y)+b)
                   return y                   

          def update(self,mini_batch,eta):
                   n=len(mini_batch)
                   grad_w=[np.zeros(w.shape) for w in self.weights]
                   grad_b=[np.zeros(b.shape)  for b in self.biases]
                   for (x,y) in mini_batch:
                   	     delta_w,delta_b=self.back_prop(x,y)
                   	     grad_w=[grad+delta for grad,delta in zip(grad_w,delta_w)]
                   	     grad_b=[grad+delta for grad,delta in zip(grad_b,delta_b)]
                   self.weights=[w-(eta/n)*grad for w,grad in zip(self.weights,grad_w)]
                   self.biases = [b-(eta/n)*grad for b,grad in zip(self.biases,grad_b)]	     
            
          def back_prop(self,x,y):
                   grad_w=[np.zeros(w.shape) for w in self.weights]
                   grad_b=[np.zeros(b.shape)  for b in self.biases]
                   activation=x
                   activations=[x]
                   zl=[]
                   for w,b in zip(self.weights,self.biases):
                   	     z=np.dot(w,activation)+b
                   	     zl.append(z)
                   	     activation=sigmoid(z)
                   	     activations.append(activation)
                   delta=(activations[-1]-y)*sigmoid_prime(zl[-1])
                   grad_w[-1]=np.dot(delta,activations[-2].transpose())
                   grad_b[-1]=delta	     
                   for i in xrange(2,self.num_of_layers):
                   	    z=zl[-i]
                   	    delta=np.dot(self.weights[-i+1].transpose(),delta)*sigmoid_prime(z)
                   	    grad_w[-i]=np.dot(delta,activations[-i-1].transpose())
                   	    grad_b[-i]=delta
                   return (grad_w,grad_b)

          def evaluate(self,test_data):
                   result=[(np.argmax(self.forward_prop(x)),y) for (x,y) in test_data]
                   return sum(int(x==y) for (x,y) in result) 

def sigmoid(z):
                   return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
                   return sigmoid(z)*(1-sigmoid(z))                   	    

          





