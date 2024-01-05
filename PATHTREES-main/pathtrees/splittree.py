#!/usr/bin/env python

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
sys.path.append(str(file.parent))

import tree
from io import StringIO

DEBUG=False

#from the internet, needs reference
class Redirectedstdout:
    def __init__(self):
        self._stdout = None
        self._string_io = None

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._string_io = StringIO()
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout

    def __str__(self):
        return self._string_io.getvalue()
    
def print_newick_string(tips,edges,tiplen,edgelen):
    if DEBUG:
        print(f"\n~~~~~~~~~~~~~~~~~~~~~~ print_newick_string ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    treenodes = {}
    interior={}
    finished={}
    for name,blen in zip(tips,tiplen):
        p = tree.Node()
        p.name = name
        p.blength = blen
        treenodes[name]=p
    tmp = zip(edges,edgelen)
    sorted_edges = sorted(tmp,key=lambda x: len(x[0]))
    count = 0
    for edge,elen in sorted_edges:
        if DEBUG:
            print(f"\n\n~~~~~~~~~~~~~~~~~~~~~~~ edge = {edge} ~~~~~~~~~~~~~~~~~~~~~~~ \n")
        count += 1
        p = tree.Node()
        p.blength = elen
        ledge = len(edge)
        x = sorted(edge)
        z ='|'.join(x)
        p.name = z
        treenodes[z] = p
        pick=[]
        for e in x:
            pick.append(e)  # find all atoms [=tips]
        
        if len(pick)==2: # if there are only 2, this is easy
            if DEBUG:
                print(f"\n##########  interior node connects TWO TIPS   ###########")
            # because the interior node connects "TWO TIPS"
            q = treenodes[pick[0]]
            p.left = q
            p.right_seq= []
            q.ancestor = p
            q = treenodes[pick[1]]
            p.right = q
            p.right_seq.append(q)
            q.ancestor = p
            del treenodes[pick[0]] #delete these tips from dict
            del treenodes[pick[1]] #
            finished[pick[0]]=z
            finished[pick[1]]=z
            interior[z]=[pick[:1]]
            if DEBUG:
                print(f"\ntreenodes.keys() = {treenodes.keys()}")
                print(f"\nfinished = {finished}")
                print(f"\ninterior = {interior}")
        else:
            # if pick contains more than 3 atoms then either this is
            # - tip + interior
            # - interior + interior, find first tip + interior
            #...
            candidate=[]
            name=[]
            counter = 0
            for key in pick:
                if key in treenodes:       #counter means TIPS
                    name.append(key)  #we found a tip
                    counter +=1
                else:         #candi means INTERIORS
                    interiorkeys = interior.keys()
                    for inter in interiorkeys:
                        if key in inter.split('|'):
                            candidate.append([inter,key])
            candi = list(set([c[0] for c in candidate]))
            
            if DEBUG:
                print(f"candidate = {candidate}")
                print(f"\ncandi = {candi}")
                print(f"\nname = {name}\n\n")
            
            
            if len(candi)==1 and counter==1:
                # because the interior node connects a "1tip + 1interior"
                if DEBUG:
                    print(f"\n##########  interior node connects  1TIP + 1INTERIOR  ###########")
                q = treenodes[name[0]]
                p.left = q
                p.right_seq= []
                q.ancestor = p
                q = treenodes[candi[0]]
                p.right = q
                p.right_seq.append(q)
                q.ancestor = p
                del treenodes[name[0]] #delete these tips from dict:
                del treenodes[candi[0]]
                finished[name[0]]=z
                finished[candi[0]]=z
                del interior[candi[0]]
                interior[z]=[[name[0],candi[0]]]
                if DEBUG:
                    print(f"\nlen(candi)= {len(candi)} & counter = {counter}")
                    print(f"\ntreenodes.keys() = {treenodes.keys()}")
                    print(f"\nfinished = {finished}")
                    print(f"\ninterior = {interior}")
                
            elif len(candi)==2 and counter==0:
                # because the interior node connects  "1interior + 1interior"
                if DEBUG:
                    print(f"\n##########  interior node connects  1INTERIOR + 1INTERIOR  ###########")
                q = treenodes[candi[0]]
                p.left = q
                p.right_seq= []
                q.ancestor = p
                q = treenodes[candi[1]]
                p.right = q
                p.right_seq.append(q)
                q.ancestor = p
                del treenodes[candi[0]] #delete these tips from dict
                del treenodes[candi[1]]
                finished[candi[0]]=z
                finished[candi[1]]=z
                del interior[candi[0]]
                del interior[candi[1]]
                interior[z]=[[candi[0],candi[1]]]
                if DEBUG:
                    print(f"\nlen(candi)=={len(candi)} & counter == {counter}")
                    print(f"\ntreenodes.keys() = {treenodes.keys()}")
                    print(f"\nfinished = {finished}")
                    print(f"\ninterior = {interior}")
                    
            #~~~~~~~~~~~~~multi cases ~~~~~~~~~~~~~
            elif len(candi)==0 and counter > 2:
                # because the interior node connects "more than two tips"
                if DEBUG:
                    print(f"\n##########  interior node connects  MORE THAN TWO TIPS  ###########")
                q = treenodes[name[0]]
                p.left = q
                q.ancestor = p
                p.right_seq= []
                for i in range(1,len(name)):
                    q = treenodes[name[i]]
                    p.right = q
                    p.right_seq.append(q)
                    q.ancestor = p
                for i in range(len(name)):
                    del treenodes[name[i]] #delete these tips from dict
                    finished[name[i]]=z
                interior[z]=[name[:]]
                if DEBUG:
                    print(f"\nlen(candi)=={len(candi)} and counter == {counter} ")
                    print(f"\ntreenodes.keys() = {treenodes.keys()}")
                    print(f"\nfinished = {finished}")
                    print(f"\ninterior = {interior}")
                
            elif len(candi)>2 and counter==0:
                # because the interior node connects "more than two interiors"
                if DEBUG:
                    print(f"\n##########  interior node connects  MORE THAN TWO INTERIORS  ###########")
                q = treenodes[candi[0]]
                p.left = q
                q.ancestor = p
                p.right_seq= []
                for i in range(1,len(candi)):
                    q = treenodes[candi[i]]
                    p.right = q
                    p.right_seq.append(q)
                    q.ancestor = p

                for a1 in candi:
                    del treenodes[a1] #delete these tips from dict
                    del interior[a1]
                    finished[a1]=z
                interior[z]=[candi[:]]
                if DEBUG:
                    print(f"\nlen(candi)=={len(candi)} and counter == {counter} ")
                    print(f"\ntreenodes.keys() = {treenodes.keys()}")
                    print(f"\nfinished = {finished}")
                    print(f"\ninterior = {interior}")
                
            elif len(candi)!=0 and counter != 0 and len(candi)+counter >2:
                # because the interior node connects multiple tips and multiple interiors
                if DEBUG:
                    print(f"\n##########  interior node connects  MULTIPLE TIPS AND INTERIORS  ###########")
                q = treenodes[candi[0]]
                p.left = q
                q.ancestor = p
                p.right_seq= []
                for i in range(1,len(candi)):
                    q = treenodes[candi[i]]
                    p.right = q
                    p.right_seq.append(q)
                    q.ancestor = p
                for i in range(counter):
                    q = treenodes[name[i]]
                    p.right = q
                    p.right_seq.append(q)
                    q.ancestor = p

                for a1 in candi:
                    del treenodes[a1] #delete these tips from dict
                    del interior[a1]
                    finished[a1]=z
                for a2 in name:
                    del treenodes[a2] #delete these tips from dict
                    finished[a2]=z
   
                interior[z]=[name[:],candi[:]]
                if DEBUG:
                    print(f"\nlen(candi)=={len(candi)} and counter == {counter} ")
                    print(f"\ntreenodes.keys() = {treenodes.keys()}")
                    print(f"\nfinished = {finished}")
                    print(f"\ninterior = {interior}")

    subtrees = list(treenodes.values())
    
    q = tree.Node()
    q.name = 'root'
    q.left= subtrees[0]
    q.right_seq= []

    q.left.ancestor = q
    q.blength=0.0
    for a in subtrees[1:]:
        q.right= a
        q.right_seq.append(a)
        a.ancestor = q
    if DEBUG:
        print(f"\n\nsubtrees.name = {[a.name for a in subtrees]}")
        print(f"q.left.name = {q.left.name}")
        print(f"q.right_seq.name = {[a.name for a in q.right_seq]}\n\n")

    t = tree.Tree()
    t.root = q
    t.remove_internal_labels_multi(t.root)
        
    with Redirectedstdout() as newick:
        t.myprint_multi(t.root, file=sys.stdout)
    if DEBUG:
        print("Newick",str(newick))
        
    return str(newick)






def print_newick_string_obsolete(tips,edges,tiplen,edgelen):
    if len(edges)==0:     #Tara did this for debuging
        newick  = '('+str(tips[0])+':'+str(tiplen[0])+','+ str(tips[1])+':'+str(tiplen[1])+')'+':0.0'
        sys.exit()
        return  newick  # this should probably abort!
    treenodes = {}
    for name,blen in zip(tips,tiplen):
        p = tree.Node()
        p.name = name
        p.blength = blen
        treenodes[name] = p

    numelem = [len(xi) for xi in edges]
    x = list(zip(edges, edgelen, numelem))
    x = sorted(x,key=lambda x: x[2] )
    for e, el, elem in x:
        if elem == 2:
            i = tree.Node()
            i.name = "|".join(list(sorted(e)))
            i.blength = el
            i.left = treenodes[e[0]]
            i.right = treenodes[e[1]]
            treenodes[e[0]].ancestor = i
            treenodes[e[1]].ancestor = i
            treenodes[i.name] = i
            del treenodes[e[1]]
            del treenodes[e[0]]
            continue
        elif elem > 2:
            i = tree.Node()
            i.name = "|".join(list(sorted(e)))
            i.blength = el
            pick =[]
            #print("i.name",i.name)
            for key in treenodes:
                if key in i.name:
                    pick.append(key)
            if len(pick)==0:
                for key in treenodes:
                    keylist = key.split('|')
                    keylen = len(keylist)
                    c=0
                    for k in keylist:
                        if k in i.name:
                            c += 1
                    if c==keylen:     
                        pick.append(key)
            if len(pick)==1:
                tempstr = i.name
                tempstr = tempstr.replace(pick[0],"").replace("||","|")
                for key in treenodes:
                    if key in tempstr:
                        pick.append(key)
            if len(pick)==2:
                i.left = treenodes[pick[0]]
                i.right = treenodes[pick[1]]
                treenodes[pick[0]].ancestor = i
                treenodes[pick[1]].ancestor = i
                treenodes[i.name] = i
                del treenodes[pick[1]]
                del treenodes[pick[0]]
            else:
                print("works only for bifurcating trees")
                for di in treenodes:
                    print("DICT",di)
                sys.exit()
    # we should be back down to two treenodes entries
    q = tree.Node()
    q.name = 'root'
    keys = list(treenodes.keys())
    n1 = treenodes[keys[0]]
    n2 = treenodes[keys[1]]
    q.left = n1
    q.right = n2
    n1.ancestor = q
    n2.ancestor = q
    q.blength=0.0
    t = tree.Tree()
    t.root=q
    t.remove_internal_labels(t.root)
    with Redirectedstdout() as newick:
        #t.myprint(t.root,file=sys.stdout)
        #print(';',file=sys.stdout)
        t.treeprint(file=sys.stdout)
    return str(newick)
    
    

if __name__ == "__main__":

    #~~~~~~~~~~~~  Example1  ~~~~~~~~~~~~~
    edges = [['Microcebus_murinus', 'Cheirogaleus_major'], ['Propithecus_coquereli', 'Lemur_catta', 'Varecia_variegata_variegata'], ['Lemur_catta', 'Varecia_variegata_variegata']]
    edgelengths = [0.050346, 0.023932, 0.042509]
    tips = ['Propithecus_coquereli', 'Lemur_catta', 'Varecia_variegata_variegata', 'Microcebus_murinus', 'Cheirogaleus_major']
    tipslength = [0.085616, 0.075789, 0.093902, 0.104756, 0.06542]

    newick = print_newick_string(tips,edges,tipslength, edgelengths)
    print(newick)

    #~~~~~~~~~~~~  Example2  ~~~~~~~~~~~~~
    edges = [['Cheirogaleus_major', 'Microcebus_murinus', 'Lemur_catta', 'Varecia_variegata_variegata'], ['Lemur_catta', 'Varecia_variegata_variegata'], ['Cheirogaleus_major', 'Microcebus_murinus']]
    edgelengths = [0.022699, 0.03823, 0.032412]
    tips = ['Cheirogaleus_major', 'Microcebus_murinus', 'Lemur_catta', 'Varecia_variegata_variegata', 'Propithecus_coquereli']
    tipslength = [0.087437, 0.118768, 0.094438, 0.119288, 0.105318]
    
    newick = print_newick_string(tips,edges,tipslength, edgelengths)
    print(newick)


    
