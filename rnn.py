import numpy as np
import itertools,sys
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator

START_TOKEN = "START_TOKEN"
END_TOKENS = ["END_TOKEN_1","END_TOKEN_2","END_TOKEN_3","END_TOKEN_4","END_TOKEN_5"]
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"

def error(y_hat,y):
    return float(np.sum(np.argmax(y_hat,axis=1) != 
                        np.argmax(y,axis=1)))/y.shape[0]

softmax_loss = lambda yp,y : (np.log(np.sum(np.exp(yp))) - yp.dot(y), 
                                  np.exp(yp)/np.sum(np.exp(yp)) - y)
f_tanh = lambda x : (np.tanh(x), 1./np.cosh(x)**2)
f_relu = lambda x : np.maximum(0,x)
f_lin = lambda x : (x, np.ones(x.shape))
#http://stackoverflow.com/questions/34968722/softmax-function-python
softmax = lambda x : np.divide(np.exp(x - np.max(x)),np.sum(np.exp(x - np.max(x)), axis=0))
    
#code written with extensive help from 
#http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
class RNN:
    def __init__(self, vocab_size = -1, activ_size=100, bptt_max=5, E=None, U=None, V=None, W=None, b=None, c=None):
        np.random.seed(0)
        if(U is None or V is None or W is None):
            self.vocab_size = vocab_size
            self.activ_size = activ_size
            self.bptt_max = bptt_max
            
            # Initialize the network parameters
            E = np.random.uniform(-np.sqrt(1./self.vocab_size), np.sqrt(1./self.vocab_size), (self.activ_size, self.vocab_size))
            U = np.random.uniform(-np.sqrt(1./self.activ_size), np.sqrt(1./self.activ_size), (6, self.activ_size, self.activ_size))
            W = np.random.uniform(-np.sqrt(1./self.activ_size), np.sqrt(1./self.activ_size), (6, self.activ_size, self.activ_size))
            V = np.random.uniform(-np.sqrt(1./self.activ_size), np.sqrt(1./self.activ_size), (self.vocab_size, self.activ_size))
            b = np.zeros((6, self.activ_size))
            c = np.zeros(self.vocab_size)
        else:
            self.vocab_size = E.shape[1]
            self.activ_size = E.shape[0]
            self.bptt_max = bptt_max
        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    def __theano_build__(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c
        
        x = T.ivector('x')
        y = T.ivector('y')

        def rnn_fprop(x_t, s_t1_prev, s_t2_prev):
            x_e = E[:,x_t]
            
            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            
            # GRU Layer 2
            z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
            r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
            c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
            s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
            
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

            return [o_t, s_t1, s_t2]
            
        [o, s, s2], updates = theano.scan(
            rnn_fprop,
            sequences=x,
            truncate_gradient=self.bptt_max,
            outputs_info=[None, 
                          dict(initial=T.zeros(self.activ_size)),
                          dict(initial=T.zeros(self.activ_size))])
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Total cost (could add regularization here)
        cost = o_error
        
        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)
        
        # Assign functions
        self.predict = theano.function([x], o)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.rnn_bproptt = theano.function([x, y], [dE, dU, dW, db, dV, dc])
        
        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        
        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2
        
        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.In(decay, value=0.9)],
            [], 
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])
        
    def rnn_loss(self, X, Y):
        loss = np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
        totalwords = np.sum((len(y) for y in Y))
        loss /= totalwords
        return loss
    
    def rnn_sgd(self, X, y, epochs=20, alpha=0.001, loss_epoch=5, decay=0.9):
        lastloss = self.rnn_loss(X,y)+1
        start_time = time.time()
        start_interval_time = time.time()
        for t in xrange(epochs):
            if t % loss_epoch == 0:
                currloss = self.rnn_loss(X,y)
                interval_time = time.time()-start_interval_time
                start_interval_time = time.time()
                time_so_far = time.time()-start_time
                adjstring = ""
                #if currloss > lastloss:
                #    alpha *= 0.5
                #    adjstring = " | Adj learn to "+str(alpha)
                ETR = (interval_time*((epochs-t)/loss_epoch))
                print "%3.0f/%d | L:%7.4f | T:%3.0fh%2.0fm%2.0fs | ETR:%3.0fh%2.0fm%2.0fs"%(t,epochs,currloss,np.floor(time_so_far/3600),np.floor((time_so_far%3600)/60),(time_so_far%60),np.floor(ETR/3600),np.floor((ETR%3600)/60),(ETR%60))+adjstring
                lastloss = currloss
            for i in xrange(len(y)):
                self.sgd_step(X[i],y[i],alpha,decay)
        currloss = self.rnn_loss(X,y)
        interval_time = time.time()-interval_time
        time_so_far = time.time()-start_time
        ETR = 0
        print "%3.0f/%d | L:%7.4f | T:%3.0fh%2.0fm%2.0fs | ETR:%3.0fh%2.0fm%2.0fs"%(epochs,epochs,currloss,np.floor(time_so_far/3600),np.floor((time_so_far%3600)/60),(time_so_far%60),np.floor(ETR/3600),np.floor((ETR%3600)/60),(ETR%60))
        return (self.E.get_value(),self.U.get_value(),self.V.get_value(),self.W.get_value(),self.b.get_value(),self.c.get_value())
