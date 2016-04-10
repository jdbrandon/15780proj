import numpy as np
import itertools,sys,nltk
from nltk.tokenize import TweetTokenizer

MAX_VOCAB_SIZE = 3000
START_TOKEN = "START_TOKEN"
END_TOKEN = "END_TOKEN"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"

def error(y_hat,y):
    return float(np.sum(np.argmax(y_hat,axis=1) != 
                        np.argmax(y,axis=1)))/y.shape[0]

def createDataset(filename):
    yaks = []
    tokenizer = TweetTokenizer()
    ids = set()
    for line in open(filename).readlines():
        stuff = line.split(":::")
        id = stuff[0]
        if len(stuff) > 3 and id not in ids:
            sentence = stuff[3]
            ids.add(id)
            tokens = [START_TOKEN]
            tokens.extend(tokenizer.tokenize(sentence.lower()))
            tokens.append(END_TOKEN)
            yaks.append(tokens)
    token_frequency = nltk.FreqDist(itertools.chain(*yaks))
    vocab = token_frequency.most_common(MAX_VOCAB_SIZE-1)
    i2t = [token[0] for token in vocab]
    i2t.append(UNKNOWN_TOKEN)
    t2i = dict()
    for i,t in enumerate(i2t):
        t2i[t] = i
    
    yaks = [[t if t in t2i else UNKNOWN_TOKEN for t in yak] for yak in yaks]
    
    Xtrain = np.asarray([[t2i[token] for token in yak[:-1]] for yak in yaks])
    Ytrain = np.asarray([[t2i[token] for token in yak[1:]] for yak in yaks])
    return (Xtrain, Ytrain, i2t, t2i)

def gen_yak(t2i, i2t, model):
    yak = [t2i[START_TOKEN]]
    while yak[-1] != t2i[END_TOKEN]:
        yak.append(np.argmax(model.rnn_fprop(yak)[0][-1]))
    return " ".join(i2t[i] for i in yak[1:-1])

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
    def __init__(self, vocab_size, activ_size=100, bptt_max=4):
        self.vocab_size = vocab_size
        self.activ_size = activ_size
        self.bptt_max = bptt_max
        np.random.seed(0)
        self.U = 0.1*np.random.randn(self.activ_size,self.vocab_size)
        self.V = 0.1*np.random.randn(self.vocab_size,self.activ_size)
        self.W = 0.1*np.random.randn(self.activ_size,self.activ_size)

    def rnn_fprop(self, x):
        activ = np.zeros((len(x)+1,self.activ_size))
        activ[-1] = np.zeros(self.activ_size)
        output = np.zeros((len(x),self.vocab_size))
        for t in xrange(len(x)):
            activ[t] = np.tanh(self.U[:,x[t]] + self.W.dot(activ[t-1]))
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
        for i in range(len(y))[::-1]:
            dV += np.outer(d_output[i], activ[i].transpose())
            d_t = self.V.transpose().dot(d_output[i]) * (1 - (activ[i]**2))
            for t in xrange(i,max(0, i-self.bptt_max)-1,-1):
                dW += np.outer(d_t, activ[t-1])
                dU[:,x[t]] += d_t
                d_t = self.W.transpose().dot(d_t) * (1 - activ[t-1]**2)
        return (dU,dV,dW)
    
    def rnn_sgd(self, X, y, epochs=10, alpha=0.01):
        for t in xrange(epochs):
            for i in xrange(len(y)):
                dU, dV, dW = self.rnn_bproptt(X[i],y[i])
                self.U -= alpha*dU
                self.V -= alpha*dV
                self.W -= alpha*dW

#train the model and print out a sentence
(Xtrain, Ytrain, i2t, t2i) = createDataset(sys.argv[1])
model = RNN(len(i2t),activ_size=100)
print "Vocab size: "+str(len(i2t))
model.rnn_sgd(Xtrain,Ytrain)
print gen_yak(t2i,i2t,model)
