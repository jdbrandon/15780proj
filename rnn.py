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
f_relu = lambda x : np.maximum(0,x)
f_lin = lambda x : (x, np.ones(x.shape))
sigmoid = lambda x : 1. / (1. + np.exp(-x))

#http://stackoverflow.com/questions/34968722/softmax-function-python
softmax = lambda x : np.divide(np.exp(x - np.max(x)),np.sum(np.exp(x - np.max(x)), axis=0))
softsign = lambda x : x / (1 + np.absolute(x))    

#code written with extensive help from 
#http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
class RNN:
    def __init__(self, vocab_size = -1, activ_size=100, bptt_max=5, Wg=None,Wi=None,Wf=None,Wo=None,bg=None,bi=None,bf=None,bo=None):
        if(Wg is None or Wi is None or Wf is None):
            self.vocab_size = vocab_size
            self.activ_size = activ_size
            self.concat_size = vocab_size + activ_size
            self.bptt_max = bptt_max
            np.random.seed(0)

            self.Wg = 0.1*np.random.randn(self.activ_size,self.concat_size)
            self.Wi = 0.1*np.random.randn(self.activ_size,self.concat_size)
            self.Wf = 0.1*np.random.randn(self.activ_size,self.concat_size)
            self.Wo = 0.1*np.random.randn(self.activ_size,self.concat_size)
            
            self.bg = np.zeros(self.activ_size)
            self.bi = np.zeros(self.activ_size)
            self.bf = np.zeros(self.activ_size)
            self.bo = np.zeros(self.activ_size)
            
            
            self.U = np.random.uniform(-np.sqrt(1.0/vocab_size),np.sqrt(1.0/vocab_size),(self.activ_size,self.vocab_size))
            self.V = np.random.uniform(-np.sqrt(1.0/activ_size),np.sqrt(1.0/activ_size),(self.vocab_size,self.activ_size))
            #self.W = np.random.uniform(-np.sqrt(1.0/activ_size),np.sqrt(1.0/activ_size),(self.activ_size,self.activ_size))
        else:
            self.vocab_size = Wg.shape[1]-Wg.shape[0]
            self.activ_size = Wg.shape[0]
            self.concat_size = vocab_size + activ_size
            self.bptt_max = bptt_max
            np.random.seed(0)
            self.Wg = Wg
            self.Wi = Wi
            self.Wf = Wf
            self.Wo = Wo
                      
            self.bg = bg
            self.bi = bi
            self.bf = bf
            self.bo = bo

    def rnn_fprop(self, x):
        activ = np.zeros((len(x)+1,self.activ_size))
        g = np.zeros((len(x)+1,self.activ_size))
        i = np.zeros((len(x)+1,self.activ_size))
        f = np.zeros((len(x)+1,self.activ_size))
        o = np.zeros((len(x)+1,self.activ_size))
        s = np.zeros((len(x)+1,self.activ_size))
        h = np.zeros((len(x)+1,self.activ_size))
        bdh = np.zeros((len(x)+1,self.activ_size))
        bds = np.zeros((len(x)+1,self.activ_size))
        bdx = np.zeros((len(x)+1,len(x)))
        
        h_prev = np.zeros(self.activ_size)
        s_prev = np.zeros(self.activ_size)
        
        for t in xrange(len(x)):
            xc = np.hstack
            g[t] = np.tanh(self.Wg[:,x[t]] + np.dot(self.Wg[:,-h.shape[1]:],h[t-1]) + self.bg)
            i[t] = sigmoid(self.Wi[:,x[t]] + np.dot(self.Wi[:,-h.shape[1]:],h[t-1]) + self.bi)
            f[t] = sigmoid(self.Wf[:,x[t]] + np.dot(self.Wf[:,-h.shape[1]:],h[t-1]) + self.bf)
            o[t] = sigmoid(self.Wo[:,x[t]] + np.dot(self.Wo[:,-h.shape[1]:],h[t-1]) + self.bo)
            s[t] = g[t] * i[t] + s[t-1] * f[t]
            h[t] = s[t] * o[t]
        return (g,i,f,o,s,h)
    
    def rnn_loss(self, x, y):
        loss = 0
        for idx in xrange(len(y)):
            (g,i,f,o,s,h) = self.rnn_fprop(x[idx])
            output = softmax(np.dot(h,self.Wg)[:,:self.vocab_size])
            how_correct = output[xrange(len(y[idx])),y[idx]]
            loss -= np.sum(np.log(how_correct))
        totalwords = np.sum((len(yi) for yi in y))
        loss /= totalwords
        return loss

    def rnn_bproptt(self, x, y):
        (g,i,f,o,s,h) = self.rnn_fprop(x)
        dWg = np.zeros((self.activ_size,self.concat_size))
        dWi = np.zeros((self.activ_size,self.concat_size))
        dWf = np.zeros((self.activ_size,self.concat_size))
        dWo = np.zeros((self.activ_size,self.concat_size))
        dbg = np.zeros(self.activ_size)
        dbi = np.zeros(self.activ_size)
        dbf = np.zeros(self.activ_size)
        dbo = np.zeros(self.activ_size)
        diff_s = np.zeros(self.activ_size)
        output = softmax(np.dot(h,self.Wg[:,:self.vocab_size]))
        d_output = output
        d_output[xrange(len(y)),y] -= 1
        d_outputc = np.dot(d_output,self.Wg[:,:self.vocab_size].transpose())
        
        bdx = np.zeros(self.vocab_size)
        bdh = np.zeros(self.activ_size)
        
        for idx in xrange(len(y)-1,-1,-1):
            ds = o[idx] * ((h[idx]-d_outputc[idx]) + bdh) + diff_s
            do = s[idx] * ((h[idx] - d_outputc[idx]) + bdh)
            di = g[idx] * ds
            dg = i[idx] * ds
            df = s[idx-1] * ds
            
            di_input = (1. - i[idx]) * i[idx] * di
            df_input = (1. - f[idx]) * f[idx] * df
            do_input = (1. - o[idx]) * o[idx] * do
            dg_input = (1. - g[idx] ** 2) * dg
            
            onehotx = np.zeros(self.vocab_size)
            onehotx[x[idx]] = 1
            xc = np.hstack((onehotx, h[idx-1]))
            
            dWi += np.outer(di_input, xc)
            dWf += np.outer(df_input, xc)
            dWo += np.outer(do_input, xc)
            dWg += np.outer(dg_input, xc)
            dbi += di_input
            dbf += df_input
            dbo += do_input
            dbg += dg_input
            
            dxc = np.zeros(xc.shape[0])
            dxc += np.dot(self.Wi.transpose(), di_input)
            dxc += np.dot(self.Wf.transpose(), df_input)
            dxc += np.dot(self.Wo.transpose(), do_input)
            dxc += np.dot(self.Wg.transpose(), dg_input)
            
            diff_s = ds * f[idx]
            bdx = dxc[:self.vocab_size]
            bdh = dxc[self.vocab_size:]
            
            '''dV += np.outer(d_output[idx], activ[idx])
            d_t = self.V.transpose().dot(d_output[idx]) * (1 - (activ[idx]**2))
            for t in xrange(idx,max(0, idx-self.bptt_max)-1,-1):
                dW += np.outer(d_t, activ[t-1])
                dU[:,x[t]] += d_t
                a = self.W.transpose().dot(d_t)
                b = activ[t-1]**2
                d_t = a * (1 - b)'''
        return (dWg, dWi, dWf, dWo, dbg, dbi, dbf, dbg, dbo)
    
    def rnn_sgd(self, X, y, epochs=20, alpha=0.01, loss_epoch=5):
        lastloss = self.rnn_loss(X,y)+1
        for t in xrange(epochs):
            if t % loss_epoch == 0:
                currloss = self.rnn_loss(X,y)
                print "Loss at "+str(t)+"/"+str(epochs)+": "+str(currloss)
                if currloss > lastloss:
                    alpha *= 0.5
                    print "Adjusting learning rate to "+str(alpha)
                lastloss = currloss
            for i in xrange(len(y)):
                dWg, dWi, dWf, dWo, dbg, dbi, dbf, dbg, dbo = self.rnn_bproptt(X[i],y[i])
                self.Wg -= alpha*dWg
                self.Wi -= alpha*dWi
                self.Wf -= alpha*dWf
                self.Wo -= alpha*dWo
                self.bg -= alpha*dbg
                self.bi -= alpha*dbi
                self.bf -= alpha*dbf
                self.bo -= alpha*dbo
        return (self.Wg,self.Wi,self.Wf,self.Wo,self.bg,self.bi,self.bf,self.bo)
