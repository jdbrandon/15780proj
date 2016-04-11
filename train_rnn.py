import numpy as np
import itertools,sys,nltk
from nltk.tokenize import TweetTokenizer
from rnn import *
from optparse import OptionParser

def createDataset(filename, MAX_VOCAB_SIZE):
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

#train the model and print out a sentence
parser = OptionParser()
parser.add_option("-t","--trainingfile",dest="trainingfile", help="log file to train from")
parser.add_option("-o","--modeloutput",dest="modeloutput", help="file to output trained model to", default="yakmodel")
parser.add_option("-a","--hiddensize",type="int",dest="activ_size", help="size of hidden layer", default=50)
parser.add_option("-e","--numepochs",type="int",dest="numepochs", help="number of epochs to train", default=50)
parser.add_option("-v","--vocabsize",type="int",dest="vocabsize", help="max size of vocab to train with",default=3000)
parser.add_option("-b","--bpttmax",type="int",dest="bpttmax", help="max number of times to unroll loop",default=4)
(options, args) = parser.parse_args()
if options.trainingfile == None:
    parser.error("must specify training file")

(Xtrain, Ytrain, i2t, t2i) = createDataset(options.trainingfile, options.vocabsize)
model = RNN(vocab_size=len(i2t),activ_size=options.activ_size, bptt_max = options.bpttmax)
print "Vocab size: "+str(len(i2t))
U,V,W = model.rnn_sgd(Xtrain,Ytrain,epochs=options.numepochs)
np.savez(options.modeloutput,U=U,V=V,W=W,i2t=i2t,t2i=t2i)