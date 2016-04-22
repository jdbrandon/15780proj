import numpy as np
import itertools,sys,codecs,locale
from rnn import *

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
        nexttoken = t2i[UNKNOWN_TOKEN]
        while nexttoken == t2i[UNKNOWN_TOKEN]:
            nexttoken = np.argmax(np.random.multinomial(1,token_p))
        yak.append(nexttoken)
    return " ".join(i2t[i].encode(locale.getpreferredencoding(),'xmlcharrefreplace') for i in yak[1:-1]),END_TOKEN_IDS.index(yak[-1])

#train the model and print out a sentence
if len(sys.argv) < 2 or len(sys.argv) > 3:
    print "usage: gen_yaks.py <model_file> [num_yaks]"
    quit()
weights = np.load(sys.argv[1])
model = RNN(U=weights["U"], V=weights["V"], W=weights["W"], E=weights["E"], b=weights["b"], c=weights["c"])
#sys.stdout = codecs.getwriter(locale.getpreferredencoding())(sys.stdout)
if len(sys.argv) == 3:
    for i in xrange(int(sys.argv[2])):
        yak,grade = gen_yak(weights["t2i"][()],weights["i2t"],model)
        print str(grade)+" -- "+yak+"\n----------------"
else:
    for i in xrange(5):
        yak,grade = gen_yak(weights["t2i"][()],weights["i2t"],model)
        print str(grade)+" -- "+yak+"\n----------------"
