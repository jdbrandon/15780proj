import numpy as np
import itertools,sys,nltk
from nltk.tokenize import TweetTokenizer
from rnn import *
from optparse import OptionParser

def createDataset(filename, MAX_VOCAB_SIZE):
    yaks = []
    tokenizer = TweetTokenizer()
    ids = set()
    numyaks = 0
    for line in open(filename).readlines():
        stuff = line.split(":::")
        id = stuff[0]
        if len(stuff) > 3 and id not in ids:
            numyaks+=1
            sentence = stuff[3]
            ids.add(id)
            tokens = [START_TOKEN]
            tokens.extend(tokenizer.tokenize(sentence.lower()))
            if int(stuff[2]) < 0:
                tokens.append(END_TOKENS[0])
            elif int(stuff[2]) < 10:
                tokens.append(END_TOKENS[1])
            elif int(stuff[2]) < 40:
                tokens.append(END_TOKENS[2])
            elif int(stuff[2]) < 100:
                tokens.append(END_TOKENS[3])
            else:
                tokens.append(END_TOKENS[4])
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
    print "Num unique Yaks: "+str(numyaks)
    return (Xtrain, Ytrain, i2t, t2i)

#train the model and print out a sentence
parser = OptionParser()
parser.add_option("-t","--trainingfile",dest="trainingfile", help="log file to train from")
parser.add_option("-o","--modeloutput",dest="modeloutput", help="file to output trained model to", default="yakmodel")
parser.add_option("-z","--hiddensize",type="int",dest="activ_size", help="size of hidden layer", default=100)
parser.add_option("-e","--numepochs",type="int",dest="numepochs", help="number of epochs to train", default=50)
parser.add_option("-i","--epochinterval",type="int",dest="epochinterval", help="interval of epochs to evaluate learning rate and print on", default=5)
parser.add_option("-v","--vocabsize",type="int",dest="vocabsize", help="max size of vocab to train with",default=3000)
parser.add_option("-b","--bpttmax",type="int",dest="bpttmax", help="max number of times to unroll loop",default=-1)
parser.add_option("-a","--alpha",type="float",dest="alpha", help="initial learning rate for SGD",default=0.001)
parser.add_option("-d","--decay",type="float",dest="decay", help="decay rate for caching gradients for SGD",default=0.9)
(options, args) = parser.parse_args()
if options.trainingfile == None:
    parser.error("must specify training file")

(Xtrain, Ytrain, i2t, t2i) = createDataset(options.trainingfile, options.vocabsize)
model = RNN(vocab_size=len(i2t),activ_size=options.activ_size, bptt_max = options.bpttmax)
print "Vocab size: "+str(len(i2t))
(E,U,V,W,b,c) = model.rnn_sgd(Xtrain,Ytrain,epochs=options.numepochs,loss_epoch=options.epochinterval,alpha=options.alpha,decay=options.decay)
np.savez_compressed(options.modeloutput,E=E,U=U,V=V,W=W,b=b,c=c,i2t=i2t,t2i=t2i)