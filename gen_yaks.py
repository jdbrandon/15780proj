import numpy as np
import itertools,sys,codecs,locale
from rnn import *
from optparse import OptionParser

def gen_most_likely_yak(t2i, i2t, model):
    yak = [t2i[START_TOKEN]]
    while yak[-1] != t2i[END_TOKEN]:
        yak.append(np.argmax(model.rnn_fprop(yak)[0][-1]))
    return " ".join(i2t[i] for i in yak[1:-1])

def gen_yak(t2i, i2t, model):
    yak = [t2i[START_TOKEN]]
    END_TOKEN_IDS = [t2i[END_TOKEN] for END_TOKEN in END_TOKENS]
    while yak[-1] not in END_TOKEN_IDS:
        token_p = model.predict(yak)[-1]
        token_p[token_p<10**-8] = 0
        nexttoken = t2i[UNKNOWN_TOKEN]
        while nexttoken == t2i[UNKNOWN_TOKEN]:
            nexttoken = np.argmax(np.random.multinomial(1,token_p))
        yak.append(nexttoken)
    return " ".join(i2t[i].split(DELIMITER)[0].encode(locale.getpreferredencoding(),'xmlcharrefreplace') for i in yak[1:-1]),END_TOKEN_IDS.index(yak[-1])

    
def gen_yak_pos(t2i, i2t, post2i, posi2t, model, posmodel):
    yak = [t2i[START_TOKEN]]
    posyak = [post2i[START_TOKEN]]
    END_TOKEN_IDS = [t2i[END_TOKEN] for END_TOKEN in END_TOKENS]
    POS_END_TOKEN_IDS = [post2i[END_TOKEN] for END_TOKEN in END_TOKENS]
    while posyak[-1] not in POS_END_TOKEN_IDS:
        pos_token_p = posmodel.predict(posyak)[-1]
        token_p = model.predict(yak)[-1]
        pos_token_p[pos_token_p<10**-8] = 0
        token_p[token_p<10**-8] = 0
        postoken = np.argmax(np.random.multinomial(1,pos_token_p))
        if postoken in POS_END_TOKEN_IDS:
            posyak.append(postoken)
            continue
        posstr = posi2t[postoken]
        nexttoken = t2i[UNKNOWN_TOKEN]
        while nexttoken == t2i[UNKNOWN_TOKEN] or nexttoken in END_TOKEN_IDS or len(i2t[nexttoken].split(DELIMITER))< 2 or i2t[nexttoken].split(DELIMITER)[1] != posstr:
            nexttoken = np.argmax(np.random.multinomial(1,token_p))
        posyak.append(postoken)
        yak.append(nexttoken)
    return " ".join(i2t[i].split(DELIMITER)[0].encode(locale.getpreferredencoding(),'xmlcharrefreplace') for i in yak[1:]),POS_END_TOKEN_IDS.index(posyak[-1])

#train the model and print out a sentence
parser = OptionParser()
parser.add_option("-m","--wordmodel",dest="wordmodelfile", help="trained word model")
parser.add_option("-p","--posmodel",dest="posmodelfile", help="trained pos model", default=None)
parser.add_option("-n","--numyaks",type="int",dest="num_yaks", help="number of yaks to generate", default=5)
(options, args) = parser.parse_args()
if options.wordmodelfile == None:
    parser.error("must specify word model file")

weights = np.load(options.wordmodelfile)
model = RNN(U=weights["U"], V=weights["V"], W=weights["W"], E=weights["E"], b=weights["b"], c=weights["c"])

posweights = None
posmodel = None
if options.posmodelfile is not None:
    posweights = np.load(options.posmodelfile)
    posmodel = RNN(U=posweights["U"], V=posweights["V"], W=posweights["W"], E=posweights["E"], b=posweights["b"], c=posweights["c"])

for i in xrange(options.num_yaks):
    yak,grade = "",-1
    if posmodel is None:
        yak,grade = gen_yak(weights["t2i"][()],weights["i2t"],model)
    else:
        yak,grade = gen_yak_pos(weights["t2i"][()],weights["i2t"],posweights["t2i"][()],posweights["i2t"],model,posmodel)
    print str(grade)+" -- "+yak+"\n----------------"
