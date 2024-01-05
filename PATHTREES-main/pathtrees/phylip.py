# functions to read a phylip sequence file and to read a NEWICK tree string

DEBUG = False

def readData(myfile, type='STANDARD'):     #testdata.phy : sequences
    '''
    read the phylip file and returns labels and sequences as lists
    '''
    f = open(myfile,'r')
    label =[]
    sequence=[]
    data = f.readlines()
    f.close()
    numind,numsites = (data.pop(0)).split()
    for i in data:
        if i=='':
            continue
        if type=='STANDARD':
            l = i[:10]    #this assumes standard phylip format
            s = i[11:]    #
            #l = i[:34]    #this assumes standard phylip format
            #s = i[35:]    #
        else:
            index = i.rfind('  ')
            l = i[:index]
            s = i[index+1:]
        label.append(l.strip().replace(' ','_'))
        if DEBUG:
            print("myread()",l.replace(' ','_').strip())
        sequence.append(s.strip())
    if DEBUG:
        print ("Phylip file:", myfile, file=sys.stderr)
        print ("    species:", numind, file=sys.stderr)
        print ("    sites:  ", numsites, file=sys.stderr)
        print ("first label:",label[0], file=sys.stderr)
        print ("last  label:",label[-1], file=sys.stderr)
    varsites = [list(si) for si in sequence if len(si)>0]
    if DEBUG:
        print(len(varsites),len(varsites[0]))
    varsites = [len([i for i in list(set(si)) if i!='-']) for si in zip(*varsites)]
    varsites = [sum([vi>1 for vi in varsites]),len(varsites)]
    return label,sequence,varsites

def readTreeString(myfile):
    '''
    read NEWICK formated trees from a file and returns a list of strings
    '''
    f = open(myfile,'r')
    data = f.readlines()
    f.close()
    return data
    

def guess_totaltreelength(sites):
    return sites[0]/sites[1]  #aka p-distance this may need a lot of improvement



