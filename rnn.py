import numpy as np
import itertools,sys

START_TOKEN = "START_TOKEN"
END_TOKEN = "END_TOKEN"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"

def error(y_hat,y):
    return float(np.sum(np.argmax(y_hat,axis=1) != 
                        np.argmax(y,axis=1)))/y.shape[0]

softmax_loss = lambda yp,y : (np.log(np.sum(np.exp(yp))) - yp.dot(y), 
                                  np.exp(yp)/np.sum(np.exp(yp)) - y)
f_tanh = lambda x : (np.tanh(x), 1./np.cosh(x)**2)
f_relu = lambda x : (x>=0).astype(np.float64) #np.maximum(0,x)
f_lin = lambda x : (x, np.ones(x.shape))
#http://stackoverflow.com/questions/34968722/softmax-function-python
softmax = lambda x : np.divide(np.exp(x - np.max(x)),np.sum(np.exp(x - np.max(x)), axis=0))
    
#code written with extensive help from 
#http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
class RNN:
    def __init__(self, vocab_size = -1, activ_size=100, bptt_max=5, U=None, V=None, W=None):
        if(U is None or V is None or W is None):
            self.vocab_size = vocab_size
            self.activ_size = activ_size
            self.bptt_max = bptt_max
            np.random.seed(0)
            self.U = 0.1*np.random.randn(self.activ_size,self.vocab_size)
            self.V = 0.1*np.random.randn(self.vocab_size,self.activ_size)
            self.W = np.identity(self.activ_size) #0.1*np.random.randn(self.activ_size,self.activ_size)
            #self.U = np.random.uniform(-np.sqrt(1.0/vocab_size),np.sqrt(1.0/vocab_size),(self.activ_size,self.vocab_size))
            #self.V = np.random.uniform(-np.sqrt(1.0/activ_size),np.sqrt(1.0/activ_size),(self.vocab_size,self.activ_size))
            #self.W = np.random.uniform(-np.sqrt(1.0/activ_size),np.sqrt(1.0/activ_size),(self.activ_size,self.activ_size))
        else:
            self.vocab_size = U.shape[1]
            self.activ_size = U.shape[0]
            self.bptt_max = bptt_max
            np.random.seed(0)
            self.U = U
            self.V = V
            self.W = W

    def rnn_fprop(self, x):
        activ = np.zeros((len(x)+1,self.activ_size))
        activ[-1] = np.zeros(self.activ_size)
        output = np.zeros((len(x),self.vocab_size))
        for t in xrange(len(x)):
            activ[t] = f_relu(self.U[:,x[t]] + self.W.dot(activ[t-1]))
            output[t] = softmax(self.V.dot(activ[t]))
        return (output,activ)
    
    def rnn_loss(self, x, y):
        loss = 0
        for i in xrange(len(y)):
            output, activ = self.rnn_fprop(x[i])
            how_correct = output[xrange(len(y[i])),y[i]]
            loss -= np.sum(np.log(how_correct))
        totalwords = np.sum((len(yi) for yi in y))
        loss /= totalwords
        return loss

    def rnn_bproptt(self, x, y):
        output, activ = self.rnn_fprop(x)
        dU, dV, dW = np.zeros(self.U.shape),np.zeros(self.V.shape),np.zeros(self.W.shape)
        d_output = output
        d_output[xrange(len(y)),y] -= 1
        for i in xrange(len(y)-1,-1,-1):
            dV += np.outer(d_output[i], activ[i])
            d_t = self.V.transpose().dot(d_output[i]) * (1 - (activ[i]**2))
            for t in xrange(i,max(-1, i-self.bptt_max)-1,-1):
                dW += np.outer(d_t, activ[t-1])
                dU[:,x[t]] += d_t
                d_t = self.W.transpose().dot(d_t) * (1 - (activ[t-1]**2))
                normthresh = 1
                '''if np.linalg.norm(d_t) > normthresh:
                    d_t = (normthresh/np.linalg.norm(d_t))*d_t
                if np.linalg.norm(dU) > normthresh:                   
                    dU = (normthresh/np.linalg.norm(dU))*dU
                if np.linalg.norm(dV) > normthresh:                   
                    dV = (normthresh/np.linalg.norm(dV))*dV
                if np.linalg.norm(dW) > normthresh:                   
                    dW = (normthresh/np.linalg.norm(dW))*dW
                '''#print "self.W.transpose().dot(d_t)",d_t, self.W, self.W.transpose().dot(d_t),activ[t-1]
        return (dU,dV,dW)
    
    def rnn_sgd(self, X, y, epochs=20, alpha=0.01, loss_epoch=5):
        for t in xrange(epochs):
            if t % loss_epoch == 0:
                print "Loss at "+str(t)+"/"+str(epochs)+": "+str(self.rnn_loss(X,y))
            for i in xrange(len(y)):
                dU, dV, dW = self.rnn_bproptt(X[i],y[i])
                self.U -= alpha*dU
                self.V -= alpha*dV
                self.W -= alpha*dW
        return (self.U,self.V,self.W)
