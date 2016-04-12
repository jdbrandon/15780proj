import sys

def condenseDataset(filename):
    yaks = dict()
    ids = set()
    numyaks = 0
    logfile = open(filename)
    for line in logfile.readlines():
        stuff = line.split(":::")
        if len(stuff) > 3:
            id = stuff[0]
            if id not in yaks:
                yaks[id] = line
                numyaks+=1
            elif (int(stuff[2]) > int(yaks[id].split(":::")[2])):
                yaks[id] = line
    logfile.close()
    logfile = open(filename,"w")
    for k,v in yaks.iteritems():
        logfile.write(v+"\n")
    logfile.close()
    return numyaks

if len(sys.argv) < 2:
    print "must specify logfile to condense"
    quit()
print "Num unique Yaks: "+str(condenseDataset(sys.argv[1]))